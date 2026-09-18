// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Main scheduler implementation, ported from `vllm/v1/core/sched/scheduler.py`.
//!
//! The scheduler is responsible for deciding which requests to process at each
//! scheduling step and how many tokens to allocate to each request. It manages
//! the request lifecycle through waiting, running, and finished states.
//!
//! This initial Rust port simplifies the KV cache interaction by tracking a
//! simple block counter rather than full block allocation. A trait
//! [`KVCacheManagerOps`] is defined so that the real KV cache manager (built
//! by another agent) can be plugged in later.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use scratchy_core_common::multimodal::MultimodalData;
use scratchy_core_common::{BlockKind, Request, RequestStatus};
use scratchy_core_config::{SchedulerConfig, SchedulerPolicy, SpansConfig};
use tracing::warn;

use super::interface::{PauseState, SchedulerInterface};
use super::output::{CachedRequestData, NewRequestData, SchedulerOutput};
use super::request_queue::{RequestQueue, SchedulingPolicy, create_request_queue};

// ---------------------------------------------------------------------------
// KVCacheManagerOps -- trait for KV cache interaction
// ---------------------------------------------------------------------------

/// Trait that abstracts the KV cache manager operations needed by the
/// scheduler.
///
/// The real `KVCacheManager` (in the `kv_cache_manager` module) will
/// implement this trait. For testing and initial bring-up, a simple
/// block-counting implementation is provided.
pub trait KVCacheManagerOps: Send {
    /// Try to allocate blocks for a request.
    ///
    /// Returns the block IDs (one `Vec<usize>` per KV cache group) if
    /// allocation succeeded, or `None` if there are insufficient free
    /// blocks.
    fn allocate_slots(
        &mut self,
        request: &Request,
        num_new_tokens: usize,
        num_lookahead_tokens: usize,
        // Spans: per-GROUP per-logical-block cached block ids from
        // `get_computed_blocks`, with `usize::MAX` for misses. Lets a new
        // request SHARE cached span blocks even past a gap (not just a
        // contiguous prefix). Empty for the decode/running path
        // (currently_held > 0). `matched_block_ids[g][i]` = cached block id for
        // group `g`, logical block `i` (or `usize::MAX` = miss → alloc fresh).
        matched_block_ids: &[Vec<usize>],
    ) -> Option<Vec<Vec<usize>>>;

    /// Free all blocks held by a request.
    fn free(&mut self, request_id: &str);

    /// 🦭 Seal: pad the request's token sequence to the next block boundary,
    /// hash the final block, and register it in the cache so future requests
    /// can hit it.
    fn seal(&mut self, request: &Request);

    /// Free blocks in volatile mode: push them to the front of the free queue
    /// so they are evicted first.
    fn free_volatile(&mut self, request_id: &str);

    /// Get the block IDs currently assigned to a request.
    fn get_blocks(&self, request_id: &str) -> Vec<Vec<usize>>;

    /// Get the number of computed tokens from prefix cache for a new
    /// request.
    ///
    /// Returns `(num_computed_tokens, block_ids)` where `block_ids` are
    /// the cached blocks.
    fn get_computed_blocks(&self, request: &Request) -> (u32, Vec<Vec<usize>>);

    /// Notify the KV cache manager that a new scheduling step is starting.
    fn new_step_starts(&mut self);

    /// THE WORKER'S POOL-WIDE KV REACH, for the allocator that sizes a request nothing has run yet.
    ///
    /// Defaulted to a no-op so a tracker that does not need it says nothing — the value only means anything
    /// where a batched step appends every row at one shared slot.
    fn set_kv_pool_reach(&mut self, _reach: scratchy_core_common::KvSlotSpan) {}

    /// THIS STEP'S SHARED WRITE-SLOT DEPTH — the max over every request about to run, set once before any of
    /// them is allocated. Defaulted to a no-op for the same reason as above: it means nothing where KV is
    /// addressed by token.
    fn set_step_reach(&mut self, _reach: scratchy_core_common::KvSlotSpan) {}

    /// Reset the prefix cache. Returns `true` if successful.
    fn reset_prefix_cache(&mut self) -> bool;

    /// Number of free blocks available.
    fn num_free_blocks(&self) -> usize;

    /// Total number of blocks.
    fn num_total_blocks(&self) -> usize;

    /// Block size (tokens per block).
    fn block_size(&self) -> usize;

    /// KV cache usage as a fraction in `[0.0, 1.0]`.
    fn usage(&self) -> f64;

    /// Number of blocks retained in the prefix cache.
    fn num_cached_blocks(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// SimpleBlockTracker -- a minimal KV cache manager for initial bring-up
// ---------------------------------------------------------------------------

/// A block tracker that satisfies [`KVCacheManagerOps`].
///
/// Full Python parity with `vllm/v1/core/block_pool.py`:
///
/// - **Reference counting**: Each block has a `ref_cnt`. Multiple requests
///   sharing a cached prefix share the same block (ref_cnt > 1).
/// - **Free queue**: Doubly-linked list emulated via `VecDeque`. Blocks enter
///   the free queue only when ref_cnt drops to 0. Stale entries (blocks
///   reclaimed via `touch()`) are skipped lazily on pop.
/// - **Lazy eviction**: Hash→block mappings persist after `free()` and are
///   only removed when the block is popped from the free queue for a new
///   allocation (`_maybe_evict_cached_block`).
/// - **`touch()`**: Increments ref_cnt. If ref_cnt was 0 (block in free queue),
///   the block is logically removed from the free queue (stale entry skipped
///   lazily).
/// - **`free_blocks()`**: Decrements ref_cnt. Blocks with ref_cnt == 0 are
///   appended to the free queue.
/// - Blocks are freed in reverse order so tail (decode) blocks are evicted
///   first and prefix blocks survive longest (matching Python's
///   `reversed(req_blocks)` in `SingleTypeKVCacheManager.free()`).
pub struct SimpleBlockTracker {
    total_blocks: usize,
    block_size: usize,

    /// THE WORKER'S LAST POOL-WIDE KV REACH — the deepest slot any live request will occupy, and therefore
    /// the slot a batched step's shared write lands at or before. `None` on every backend whose keys sit at
    /// their token positions, which makes every line that reads it inert there.
    ///
    /// ⛔ IT SIZES THE REQUESTS `Request::kv_extent` CANNOT: that one is per request, so a request being
    /// ADMITTED has none and is sized by its token count alone.
    kv_pool_reach: Option<scratchy_core_common::KvSlotSpan>,
    /// HOW DEEP THIS STEP'S SHARED WRITE SLOT GOES — the max over every request the scheduler is about to run,
    /// computed BEFORE any of them is allocated.
    ///
    /// ⛔ SET ONCE PER `schedule()`, NOT DERIVED PER REQUEST. A value derived while allocating would be read by
    /// the requests allocated before the one that raised it, which is the very window this closes.
    /// [`KvSlotSpan::NONE`] unless [`Self::addressing`] says this backend shares one write slot.
    step_reach: scratchy_core_common::KvSlotSpan,
    /// HOW THIS BACKEND ADDRESSES ITS KV — declared when the tracker is built.
    ///
    /// ⛔ IT WAS INFERRED FROM `kv_pool_reach.is_some()`, WHICH IS FALSE UNTIL THE WORKER'S FIRST REPORT, so
    /// every allocation before that got the token-addressed arithmetic on a backend that shares a write slot.
    /// See [`KvAddressing`] for the measurement.
    addressing: scratchy_core_common::KvAddressing,

    /// Per-block reference count. ref_cnt > 0 means allocated (possibly shared).
    /// ref_cnt == 0 means in free queue (eviction candidate).
    /// Matches Python's `KVCacheBlock.ref_cnt`.
    ref_cnt: Vec<usize>,

    /// Number of blocks with ref_cnt == 0. Maintained incrementally to avoid
    /// O(n) scans.
    num_free_blocks: usize,

    /// Eviction queue: blocks with ref_cnt == 0. Pop from front (LRU).
    /// May contain stale entries (blocks whose ref_cnt > 0 due to `touch()`);
    /// these are skipped during `allocate_fresh_blocks`.
    free_queue: VecDeque<usize>,

    /// request_id -> (block_ids, num_blocks_held)
    allocations: HashMap<String, (Vec<Vec<usize>>, usize)>,

    // --- Prefix caching fields ---
    /// Whether prefix caching is enabled.
    enable_caching: bool,

    /// Hash of a full block's token content → block ID.
    /// Persists after free(). Only removed when the block is popped
    /// from the free queue and given to a new allocation
    /// (lazy eviction, matching Python's `_maybe_evict_cached_block`).
    block_hash_to_id: HashMap<u64, usize>,

    /// Reverse mapping: block ID → hash. For O(1) hash cleanup when
    /// a block is evicted during allocation.
    block_id_to_hash: HashMap<usize, u64>,

    /// Request ID → ordered list of block hashes.
    req_to_hashes: HashMap<String, Vec<u64>>,

    /// Spans (relocatable KV cache block) configuration.
    spans_config: SpansConfig,

    /// Per-group SWA specs for hybrid models (gemma4). `Some` switches
    /// `allocate_slots` to vLLM's hybrid KV path: one **null-padded block
    /// table per group** over the ONE shared pool, where full groups never
    /// free and sliding groups free out-of-window blocks each step
    /// (`remove_skipped_blocks`). `None` = single uniform full-attention group
    /// (every non-SWA model — byte-identical to the pre-SWA behavior).
    swa_groups: Option<Vec<SwaGroup>>,

    /// Lowest-free-first allocation order for the hybrid path (a min-heap of
    /// free block IDs). Full groups take the lowest IDs and hold them; sliding
    /// groups recycle their just-freed low IDs immediately, so the **max
    /// allocated block ID stays ~`full_blocks + Σ window_blocks`**. This is the
    /// load-bearing adaptation to scratchy's substrate: lockstep
    /// `grow_to_cover` residency is keyed on the max block ID, so an unbounded
    /// (FIFO, free-to-back) ID would commit the whole pool resident and
    /// re-create the OOM this project exists to fix. Empty on the non-SWA path.
    free_min: BinaryHeap<Reverse<usize>>,

    /// Hybrid prefix cache: namespaced per group. `(group_id, block_hash) ->
    /// block_id`. A full group and a sliding group with identical token content
    /// MUST NOT collide (different KV semantics: full keeps everything, sliding
    /// stores unrotated relocatable spans), so the group id is part of the key.
    /// Only used on the hybrid path; empty otherwise.
    hybrid_hash_to_id: HashMap<(u32, u64), usize>,

    /// Reverse map for lazy eviction on the hybrid path: `block_id ->
    /// (group_id, block_hash)`. A block belongs to exactly one group's table at
    /// a time, so a single reverse entry suffices.
    hybrid_id_to_key: HashMap<usize, (u32, u64)>,

    /// Hybrid per-request full-block hashes per group: `request_id ->
    /// Vec_by_group<Vec_by_logical_block<hash>>`. Sliding groups only ever
    /// register hashes for blocks still in-window at seal time.
    hybrid_req_hashes: HashMap<String, Vec<Vec<u64>>>,
}

/// One KV-cache group's allocation spec on the hybrid (SWA) path.
#[derive(Clone, Copy, Debug)]
struct SwaGroup {
    /// `true` for a sliding-window group (frees out-of-window blocks each
    /// step), `false` for a full-attention group (never frees).
    is_sliding: bool,
    /// Sliding window in tokens; `0` for full groups.
    window: usize,
    /// This group's KV block size. Page-unified across groups (gemma4: full
    /// group = 32, sliding groups = 16) so every group draws from the one
    /// shared pool.
    block_size: usize,
}

impl SimpleBlockTracker {
    /// Create a new block tracker with the given number of GPU blocks and
    /// block size.
    pub fn new(num_gpu_blocks: usize, block_size: usize) -> Self {
        Self {
            kv_pool_reach: None,
            step_reach: scratchy_core_common::KvSlotSpan::NONE,
            addressing: scratchy_core_common::KvAddressing::ByToken,
            total_blocks: num_gpu_blocks,
            block_size,
            ref_cnt: vec![0; num_gpu_blocks],
            num_free_blocks: num_gpu_blocks,
            free_queue: (0..num_gpu_blocks).collect(),
            allocations: HashMap::new(),
            enable_caching: false,
            block_hash_to_id: HashMap::new(),
            block_id_to_hash: HashMap::new(),
            req_to_hashes: HashMap::new(),
            spans_config: SpansConfig::from_env(),
            swa_groups: None,
            free_min: BinaryHeap::new(),
            hybrid_hash_to_id: HashMap::new(),
            hybrid_id_to_key: HashMap::new(),
            hybrid_req_hashes: HashMap::new(),
        }
    }

    /// Shared null block (vLLM `_null_block`): held out of the free pool, used
    /// to pad the head of a sliding group's block table once blocks fall out of
    /// the window. Its KV bytes are never read (those logical positions are
    /// always masked out-of-window).
    const NULL_BLOCK: usize = 0;

    /// Enable vLLM's hybrid KV path: one null-padded block table per group over
    /// the ONE shared pool. `num_blocks` is the page-unified pool size; `groups`
    /// gives each group's `(is_sliding, window, block_size)` (gemma4: 1 full
    /// `(false, 0, 32)` + 5 sliding `(true, 1024, 16)`).
    ///
    /// Block 0 is reserved as the shared null block (out-of-window table slots
    /// point at it). Allocation is lowest-free-first so the max block ID — and
    /// thus lockstep `grow_to_cover` residency — stays bounded near the working
    /// set. Prefix caching is HONORED (the `with_caching` flag is preserved):
    /// `get_computed_blocks` computes a joint-P-correct cached prefix per group
    /// (full group fully cached + each sliding group's in-window suffix cached,
    /// min P across groups), and relocatable spans store K/V unrotated so a
    /// sliding group's reuse is valid (re-rope on read at the kernel).
    pub fn enable_hybrid(&mut self, num_blocks: usize, groups: Vec<(bool, usize, usize)>) {
        assert!(
            num_blocks >= 2,
            "enable_hybrid: need >= 2 blocks (null + data)"
        );
        assert!(!groups.is_empty(), "enable_hybrid: at least one group");
        for &(is_sliding, window, block_size) in &groups {
            assert!(block_size > 0, "enable_hybrid: block_size must be > 0");
            assert!(
                !is_sliding || window > 0,
                "enable_hybrid: sliding group needs window > 0"
            );
        }
        self.total_blocks = num_blocks;
        self.ref_cnt = vec![0; num_blocks];
        // Block 0 = null (held out of the free set); data blocks are 1..num_blocks.
        self.free_min = (1..num_blocks).map(Reverse).collect();
        self.num_free_blocks = num_blocks - 1;
        self.free_queue.clear();
        self.swa_groups = Some(
            groups
                .into_iter()
                .map(|(is_sliding, window, block_size)| SwaGroup {
                    is_sliding,
                    window,
                    block_size,
                })
                .collect(),
        );
        // NB: prefix caching is HONORED on the hybrid path (do NOT force off).
        // `with_caching` sets `enable_caching`; the hybrid get_computed_blocks /
        // allocate_slots compute a per-group, joint-P-correct cached prefix
        // (full group fully cached + each sliding group's in-window suffix
        // cached). When caching is off this is a no-op (the alloc tail is all
        // fresh), preserving the prior force-off behavior.
    }

    /// Highest currently-allocated (`ref_cnt > 0`) block ID, or 0 if none.
    /// The hybrid residency invariant: after any number of decode steps of a
    /// sliding-only sequence this must stay near the window, so lockstep
    /// `grow_to_cover` (keyed on the max block ID) never commits the pool.
    pub fn max_allocated_block_id(&self) -> usize {
        (1..self.total_blocks)
            .rev()
            .find(|&i| self.ref_cnt[i] > 0)
            .unwrap_or(0)
    }

    /// Create a new block tracker with prefix caching enabled.
    pub fn with_caching(num_gpu_blocks: usize, block_size: usize) -> Self {
        let mut tracker = Self::new(num_gpu_blocks, block_size);
        tracker.enable_caching = true;
        tracker
    }

    /// ⭐⭐⭐ DECLARE THAT THIS BACKEND APPENDS EVERY ROW AT ONE SHARED SLOT — the spyre paged pool.
    ///
    /// The allocator then sizes every request by the STEP'S REACH rather than by its own token count, which
    /// is what lets the HOST own every page a batched launch writes. Without it the worker has to cover
    /// those pages from a reserve of its own, and that reserve is the last thing keeping a per-request KV
    /// row below the host.
    ///
    /// ⛔ IT MUST BE DECLARED, NOT INFERRED. See [`scratchy_core_common::KvAddressing`]: inferring it from
    /// the worker's first report left every earlier allocation on the token-addressed rule.
    pub fn sharing_one_write_slot(mut self) -> Self {
        self.addressing = scratchy_core_common::KvAddressing::OneSharedWriteSlot;
        self
    }

    /// Create a new block tracker with a specific spans configuration.
    #[cfg(test)]
    pub fn with_spans_config(
        num_gpu_blocks: usize,
        block_size: usize,
        spans_config: SpansConfig,
    ) -> Self {
        let mut tracker = Self::new(num_gpu_blocks, block_size);
        tracker.enable_caching = true;
        tracker.spans_config = spans_config;
        tracker
    }

    /// Sentinel hash representing "no parent" — used for the first block
    /// in a sequence and for relocatable span blocks.
    const NONE_HASH: u64 = 0;

    /// Stable per-image hash over (height, width, raw pixel bytes).
    ///
    /// Process-local determinism is sufficient: the prefix cache lives only
    /// inside the engine process, and `std::hash::DefaultHasher` is SipHash
    /// with a fixed key, so two `MultimodalData::images[i]` with identical
    /// contents produce identical hashes within the run.
    fn hash_image(img: &scratchy_core_common::multimodal::ImageData) -> u64 {
        let mut h = std::hash::DefaultHasher::new();
        img.height.hash(&mut h);
        img.width.hash(&mut h);
        let bytes = unsafe {
            std::slice::from_raw_parts(
                img.pixels.as_ptr() as *const u8,
                std::mem::size_of_val(img.pixels.as_slice()),
            )
        };
        bytes.hash(&mut h);
        h.finish()
    }

    /// Build a per-block list of image hashes whose placeholder range
    /// overlaps that block's token range. Returns `None` for text-only
    /// requests so the hash is byte-identical to the pre-MM behavior.
    fn build_mm_extras(
        mm: Option<&MultimodalData>,
        num_full_blocks: usize,
        block_size: usize,
    ) -> Option<Vec<Vec<u64>>> {
        let mm = mm?;
        if mm.images.is_empty() || num_full_blocks == 0 {
            return None;
        }
        let mut per_block = vec![Vec::<u64>::new(); num_full_blocks];
        for (i, ph) in mm.image_placeholders.iter().enumerate() {
            let img = match mm.images.get(i) {
                Some(img) => img,
                None => continue,
            };
            let img_hash = Self::hash_image(img);
            let start_blk = ph.offset / block_size;
            let end_blk = (ph.offset + ph.length)
                .div_ceil(block_size)
                .min(num_full_blocks);
            for slot in per_block.iter_mut().take(end_blk).skip(start_blk) {
                slot.push(img_hash);
            }
        }
        Some(per_block)
    }

    /// Hash a block with parent-chain awareness for spans.
    ///
    /// `kind` is looked up from the request's `block_annotations` map:
    /// - `Some(Relocatable)`: parent hash is reset to `NONE_HASH`, making
    ///   this block cacheable independently of preceding context.
    /// - `Some(Prefixed)`: all tokens before this block are folded into
    ///   the hash, forcing recomputation when context differs.
    /// - `None`: normal parent-chained hashing.
    ///
    /// `mm_extras` are image hashes whose placeholder range overlaps this
    /// block (empty slice for text-only blocks → no hasher writes →
    /// byte-identical to the pre-MM hash).
    fn hash_block_with_parent(
        &self,
        parent_hash: u64,
        block_tokens: &[u32],
        all_tokens_before_block: &[u32],
        kind: Option<BlockKind>,
        mm_extras: &[u64],
    ) -> u64 {
        let effective_parent = if let Some(k) = kind.filter(|k| k.is_relocatable()) {
            // Relocatable span: reset the parent chain to NONE_HASH ONLY at the
            // span's FIRST block (making the SPAN position-independent), then
            // CHAIN normally within the span so a block matches only when the
            // whole span-prefix is identical.
            //
            // The old code reset EVERY relocatable block to NONE_HASH, so a
            // block hashed on its own 16 tokens alone — ignoring the span's
            // earlier tokens. Two DIFFERENT tools that share a 16-token schema
            // sub-block (`"type":"string","description":...`) then collided and
            // reused each other's KV. A block's K/V depends on its span's
            // preceding tokens (the span attends itself), so that reuse is
            // wrong → coherent on a fresh prefill but garbage once the false
            // hits fire. Chaining within the span makes reuse fire only for a
            // genuinely identical tool (the launch-claude case), regardless of
            // where in the sequence it sits.
            let block_start = all_tokens_before_block.len();
            let first_token = k.first_token() as usize;
            let is_span_first_block =
                first_token >= block_start && first_token < block_start + block_tokens.len();
            if self.spans_config.debug {
                tracing::debug!(
                    "[SPANS] Relocatable block (span_first_block={is_span_first_block}): \
                     parent {}",
                    if is_span_first_block {
                        "reset→NONE"
                    } else {
                        "chained"
                    }
                );
            }
            if is_span_first_block {
                Self::NONE_HASH
            } else {
                parent_hash
            }
        } else {
            parent_hash
        };

        let mut hasher = std::hash::DefaultHasher::new();
        effective_parent.hash(&mut hasher);
        block_tokens.hash(&mut hasher);

        if matches!(kind, Some(BlockKind::Prefixed { .. })) {
            if self.spans_config.debug {
                tracing::debug!(
                    "[SPANS] Prefixed: including {} previous tokens in block hash",
                    all_tokens_before_block.len()
                );
            }
            all_tokens_before_block.hash(&mut hasher);
        }

        for k in mm_extras {
            k.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Compute block hashes for an entire token sequence, returning one hash
    /// per full block. When `annotations` is provided, uses span-aware
    /// hashing for annotated blocks. When `mm` is `Some`, blocks whose token
    /// range overlaps an image placeholder fold the corresponding image
    /// hashes into their per-block hash so different images break the prefix.
    fn hash_all_blocks(
        &self,
        all_tokens: &[u32],
        annotations: Option<&scratchy_core_common::BlockAnnotations>,
        mm: Option<&MultimodalData>,
    ) -> Vec<u64> {
        self.hash_all_blocks_bs(all_tokens, annotations, mm, self.block_size)
    }

    /// Like [`hash_all_blocks`] but with an explicit `block_size` — needed on
    /// the hybrid path where each group has its own block size (gemma4: full
    /// group 32, sliding groups 16).
    fn hash_all_blocks_bs(
        &self,
        all_tokens: &[u32],
        annotations: Option<&scratchy_core_common::BlockAnnotations>,
        mm: Option<&MultimodalData>,
        block_size: usize,
    ) -> Vec<u64> {
        let num_full_blocks = all_tokens.len() / block_size;
        let mm_extras = Self::build_mm_extras(mm, num_full_blocks, block_size);
        let mut hashes = Vec::with_capacity(num_full_blocks);
        let mut parent_hash = Self::NONE_HASH;
        for i in 0..num_full_blocks {
            let start = i * block_size;
            let end = start + block_size;
            let block_tokens = &all_tokens[start..end];
            let tokens_before = &all_tokens[..start];
            // Annotations are keyed by block index at the CONFIG block size
            // (`self.block_size`), but a hybrid group may hash at a DIFFERENT
            // size (gemma4: global group 32, config 16). Indexing by this
            // group's block index `i` reads the wrong annotation for the global
            // group, so relocatable spans mis-hash and never content-match.
            // Index by the block's TOKEN offset instead — block-size-independent
            // (`first_token` in the annotation is a token index too, so
            // `is_span_first_block` stays exact).
            let ann_idx = start / self.block_size;
            let kind = annotations.and_then(|a| a.get(&ann_idx).copied());
            let extras: &[u64] = mm_extras.as_ref().map(|v| v[i].as_slice()).unwrap_or(&[]);
            let hash =
                self.hash_block_with_parent(parent_hash, block_tokens, tokens_before, kind, extras);
            parent_hash = hash;
            hashes.push(hash);
        }
        hashes
    }

    /// Try to allocate `count` fresh block IDs from the free queue.
    ///
    /// Pops blocks from the front of `free_queue`, skipping stale entries
    /// (blocks with ref_cnt > 0, already reclaimed via `touch()`). When a
    /// popped block has a hash mapping, the mapping is removed (lazy eviction
    /// matching Python's `_maybe_evict_cached_block`).
    fn allocate_fresh_blocks(&mut self, count: usize) -> Option<Vec<usize>> {
        if self.num_free_blocks < count {
            return None;
        }

        let mut ids = Vec::with_capacity(count);
        while ids.len() < count {
            let block_id = self.free_queue.pop_front()?;

            // Skip stale entries: blocks reclaimed via touch() have ref_cnt > 0.
            if self.ref_cnt[block_id] > 0 {
                continue;
            }

            // Set ref_cnt to 1 (allocated).
            self.ref_cnt[block_id] = 1;
            self.num_free_blocks -= 1;

            // Lazy eviction: remove hash mapping if this block was cached.
            // Matches Python's `_maybe_evict_cached_block`.
            if let Some(hash) = self.block_id_to_hash.remove(&block_id) {
                self.block_hash_to_id.remove(&hash);
            }

            ids.push(block_id);
        }
        Some(ids)
    }

    /// Touch a block: increment ref_cnt. If ref_cnt was 0 (block in free
    /// queue), the block is logically removed — the stale entry in
    /// `free_queue` is skipped lazily on the next `allocate_fresh_blocks`.
    ///
    /// Matches Python's `BlockPool.touch()`.
    fn touch_block(&mut self, block_id: usize) {
        if self.ref_cnt[block_id] == 0 {
            // Block was free, now allocated — decrement free count.
            // The stale free_queue entry will be skipped lazily.
            self.num_free_blocks -= 1;
        }
        self.ref_cnt[block_id] += 1;
    }

    /// Pop `count` lowest free block IDs (hybrid path). Returns `None` if the
    /// shared pool can't satisfy the request (callers capacity-check first; this
    /// is the defensive guard).
    fn alloc_hybrid_fresh(&mut self, count: usize) -> Option<Vec<usize>> {
        if self.num_free_blocks < count {
            return None;
        }
        let mut ids = Vec::with_capacity(count);
        while ids.len() < count {
            let Reverse(id) = self.free_min.pop()?;
            // Skip STALE heap entries: a block touched for cache reuse
            // (`touch_hybrid_block`) keeps its old heap slot but has ref_cnt > 0.
            // Mirrors `allocate_fresh_blocks`'s lazy free-queue skip.
            if self.ref_cnt[id] > 0 {
                continue;
            }
            self.ref_cnt[id] = 1;
            self.num_free_blocks -= 1;
            // Lazy eviction: if this recycled block was still hash-mapped, drop
            // both directions so it can't be handed out as a stale cache hit
            // (mirrors `allocate_fresh_blocks` / Python `_maybe_evict_cached_block`).
            if let Some(key) = self.hybrid_id_to_key.remove(&id) {
                self.hybrid_hash_to_id.remove(&key);
            }
            ids.push(id);
        }
        Some(ids)
    }

    /// Return `blocks` to the shared pool (hybrid path), skipping the null
    /// block. Freed IDs go back into the lowest-free-first heap so a sliding
    /// group reuses its own low IDs immediately — the bound on the max block ID.
    fn free_hybrid_blocks(&mut self, blocks: &[usize]) {
        for &b in blocks {
            if b == Self::NULL_BLOCK {
                continue;
            }
            debug_assert!(self.ref_cnt[b] > 0, "double-free of block {b}");
            self.ref_cnt[b] -= 1;
            if self.ref_cnt[b] == 0 {
                self.num_free_blocks += 1;
                self.free_min.push(Reverse(b));
            }
        }
    }

    /// vLLM `SlidingWindowManager.remove_skipped_blocks`: free the blocks that
    /// have fallen out of group `g`'s window, overwriting their table slots with
    /// the null block. `total_computed` = `request.num_computed_tokens` (tokens
    /// already cached BEFORE this step's new tokens). Idempotent — the reverse
    /// scan stops at the first null, so re-calling with a larger skip only frees
    /// the newly-exposed lower-index blocks. No-op for full groups.
    fn remove_skipped_blocks(
        &mut self,
        request_id: &str,
        g: usize,
        grp: SwaGroup,
        total_computed: usize,
    ) {
        if !grp.is_sliding {
            return; // full attention never frees
        }
        // skipped_tokens = max(0, num_computed - window + 1)
        let num_skipped_tokens = (total_computed + 1).saturating_sub(grp.window);
        if num_skipped_tokens == 0 {
            return;
        }
        let mut removed: Vec<usize> = Vec::new();
        if let Some((tables, _)) = self.allocations.get_mut(request_id)
            && let Some(blocks) = tables.get_mut(g)
        {
            // Only WHOLE skipped blocks are freed (// block_size floor); the
            // partial block straddling the window edge is retained.
            let num_skipped_blocks = (num_skipped_tokens / grp.block_size).min(blocks.len());
            for i in (0..num_skipped_blocks).rev() {
                if blocks[i] == Self::NULL_BLOCK {
                    // Already nulled — usually by a prior call (contiguous
                    // nulled head, could `break`), but spans credits also NULL
                    // interior sliding ranges, so keep scanning or the fresh
                    // blocks BELOW a credited run would be stranded until
                    // request free.
                    continue;
                }
                removed.push(blocks[i]);
                blocks[i] = Self::NULL_BLOCK;
            }
        }
        self.free_hybrid_blocks(&removed);
    }

    /// vLLM hybrid allocation: one null-padded block table per group over the
    /// ONE shared pool. Sliding groups first free out-of-window blocks
    /// (`remove_skipped_blocks`, BEFORE the capacity check — vLLM's order so an
    /// admissible request isn't blocked by blocks it's about to release), then
    /// every group grows at the tail to cover `total_tokens`. Lowest-free-first
    /// keeps the max block ID — and thus lockstep `grow_to_cover` residency —
    /// bounded near the working set (the lazy-commit memory win), with ONE
    /// shared pool (no separate / statically-sized sliding pool).
    fn allocate_slots_hybrid(
        &mut self,
        request: &Request,
        num_new_tokens: usize,
        num_lookahead_tokens: usize,
        matched_block_ids: &[Vec<usize>],
    ) -> Option<Vec<Vec<usize>>> {
        let groups = self.swa_groups.clone().expect("hybrid path without groups");
        let num_groups = groups.len();
        let total_tokens =
            request.num_computed_tokens as usize + num_new_tokens + num_lookahead_tokens;
        let total_computed = request.num_computed_tokens as usize;

        // Fresh request with cache hits: seed each group's table from the
        // per-group matched ids (cached block ids reused; usize::MAX = miss →
        // alloc fresh; NULL_BLOCK = out-of-window head → null-pad). Reused
        // cached blocks are touched (ref-counted) so a concurrent free can't
        // reclaim them. The tail past the matched prefix is filled fresh below.
        let currently_held = self
            .allocations
            .get(&request.request_id)
            .map(|(t, _)| t.iter().any(|g| !g.is_empty()))
            .unwrap_or(false);
        if !currently_held && self.enable_caching && !matched_block_ids.is_empty() {
            return self.allocate_slots_hybrid_with_reuse(
                request,
                total_tokens,
                &groups,
                matched_block_ids,
            );
        }

        // Ensure a per-request entry with one (null-padded) table per group.
        self.allocations
            .entry(request.request_id.clone())
            .or_insert_with(|| (vec![Vec::new(); num_groups], 0));

        // 1. Free out-of-window blocks per sliding group BEFORE the capacity
        //    check (vLLM: "call before allocating to reduce evicted blocks").
        for (g, grp) in groups.iter().enumerate() {
            self.remove_skipped_blocks(&request.request_id, g, *grp, total_computed);
        }

        // 2. New tail blocks needed per group + ONE atomic cross-group check.
        let mut new_per_group = vec![0usize; num_groups];
        let mut total_new = 0usize;
        {
            let (tables, _) = self
                .allocations
                .get(&request.request_id)
                .expect("entry just inserted");
            for (g, grp) in groups.iter().enumerate() {
                let needed = if total_tokens == 0 {
                    0
                } else {
                    total_tokens.div_ceil(grp.block_size)
                };
                new_per_group[g] = needed.saturating_sub(tables[g].len());
                total_new += new_per_group[g];
            }
        }
        if total_new > self.num_free_blocks {
            return None; // single atomic capacity check (vLLM)
        }

        // 3. Allocate + append at the tail (lowest-free-first).
        for (g, &n) in new_per_group.iter().enumerate() {
            if n > 0 {
                let fresh = self.alloc_hybrid_fresh(n)?;
                self.allocations
                    .get_mut(&request.request_id)
                    .expect("entry just inserted")
                    .0[g]
                    .extend(fresh);
            }
        }

        let tables_clone = {
            let (tables, held) = self
                .allocations
                .get_mut(&request.request_id)
                .expect("entry just inserted");
            *held = total_tokens.div_ceil(groups[0].block_size);
            tables.clone()
        };
        // Register full-block hashes (group-namespaced) for any NEWLY full,
        // in-window blocks so subsequent requests can hit them. Sliding NULL
        // slots are skipped, so an out-of-window block is never registered.
        self.register_hybrid_hashes(request, &groups, &tables_clone);
        Some(tables_clone)
    }

    /// Hybrid allocation for a FRESH request that has cache hits. `matched[g]`
    /// is the per-logical-block plan for group `g`: a cached block id to reuse,
    /// `usize::MAX` for a miss (alloc fresh), or `NULL_BLOCK` for an
    /// out-of-window head slot (null-pad). Builds each group's full table from
    /// the plan + a fresh tail, ref-counting reused blocks and registering
    /// fresh full-block hashes (group-namespaced) so later requests can hit.
    fn allocate_slots_hybrid_with_reuse(
        &mut self,
        request: &Request,
        total_tokens: usize,
        groups: &[SwaGroup],
        matched: &[Vec<usize>],
    ) -> Option<Vec<Vec<usize>>> {
        let num_groups = groups.len();

        // Per-group target block count for total_tokens (group-local bs).
        let needed_per_group: Vec<usize> = groups
            .iter()
            .map(|g| {
                if total_tokens == 0 {
                    0
                } else {
                    total_tokens.div_ceil(g.block_size)
                }
            })
            .collect();

        // Capacity pre-check: every slot that is NOT a reuse of a live (ref>0)
        // block consumes a free block (a fresh alloc OR a touch of a ref==0
        // cached block). NULL slots are free (the shared null block).
        let mut demand = 0usize;
        for (g, need) in needed_per_group.iter().enumerate() {
            let plan = matched.get(g).map(|v| v.as_slice()).unwrap_or(&[]);
            for i in 0..*need {
                match plan.get(i).copied() {
                    Some(b) if b == Self::NULL_BLOCK => {}
                    Some(b) if b != usize::MAX => {
                        if self.ref_cnt[b] == 0 {
                            demand += 1; // reviving a cached (free) block
                        }
                    }
                    _ => demand += 1, // miss → fresh
                }
            }
        }
        if demand > self.num_free_blocks {
            return None;
        }

        // Touch all reused (non-null, non-miss) blocks FIRST so they leave the
        // free pool before any fresh pop can recycle (and evict) them.
        for (g, need) in needed_per_group.iter().enumerate() {
            let plan = matched.get(g).map(|v| v.as_slice()).unwrap_or(&[]);
            for i in 0..*need {
                if let Some(b) = plan.get(i).copied()
                    && b != usize::MAX
                    && b != Self::NULL_BLOCK
                {
                    self.touch_hybrid_block(b);
                }
            }
        }

        // Build each group's table: reuse / null / fresh per slot.
        let mut tables: Vec<Vec<usize>> = vec![Vec::new(); num_groups];
        for (g, need) in needed_per_group.iter().enumerate() {
            let plan = matched.get(g).map(|v| v.as_slice()).unwrap_or(&[]).to_vec();
            let mut table = Vec::with_capacity(*need);
            for i in 0..*need {
                match plan.get(i).copied() {
                    Some(b) if b == Self::NULL_BLOCK => table.push(Self::NULL_BLOCK),
                    Some(b) if b != usize::MAX => table.push(b), // already touched
                    _ => {
                        let fresh = self.alloc_hybrid_fresh(1)?;
                        table.push(fresh[0]);
                    }
                }
            }
            tables[g] = table;
        }

        let held = total_tokens.div_ceil(groups[0].block_size);
        self.allocations
            .insert(request.request_id.clone(), (tables.clone(), held));

        // Register fresh FULL-block hashes (group-namespaced) so later requests
        // can hit them. Sliding groups register only in-window blocks: a NULL
        // (out-of-window) slot is skipped, so no stale hash is recorded.
        self.register_hybrid_hashes(request, groups, &tables);

        Some(tables)
    }

    /// Touch a hybrid block (ref-count), accounting for free-pool membership.
    /// The block leaves the lowest-free-first heap lazily (stale heap entries
    /// are skipped on pop since ref_cnt > 0).
    fn touch_hybrid_block(&mut self, block_id: usize) {
        if self.ref_cnt[block_id] == 0 {
            self.num_free_blocks -= 1;
        }
        self.ref_cnt[block_id] += 1;
    }

    /// Register group-namespaced full-block hashes for the request's current
    /// tables. Full groups register every full block; sliding groups register
    /// only blocks still in-window (NULL slots skipped). Idempotent per logical
    /// block (re-registers the same hash→id, harmless).
    fn register_hybrid_hashes(
        &mut self,
        request: &Request,
        groups: &[SwaGroup],
        tables: &[Vec<usize>],
    ) {
        if !self.enable_caching {
            return;
        }
        let all_tokens = &request.all_token_ids;
        let annotations = request.block_annotations.as_ref();
        let mm = request.mm_data.as_ref();
        let mut req_group_hashes: Vec<Vec<u64>> = vec![Vec::new(); groups.len()];
        for (g, grp) in groups.iter().enumerate() {
            let hashes = self.hash_all_blocks_bs(all_tokens, annotations, mm, grp.block_size);
            let table = &tables[g];
            let num_full = all_tokens.len() / grp.block_size;
            for i in 0..num_full {
                if i >= table.len() || i >= hashes.len() {
                    break;
                }
                let bid = table[i];
                if bid == Self::NULL_BLOCK {
                    continue; // out-of-window: never register (KV is gone)
                }
                let key = (g as u32, hashes[i]);
                self.hybrid_hash_to_id.insert(key, bid);
                self.hybrid_id_to_key.insert(bid, key);
            }
            req_group_hashes[g] = hashes.into_iter().take(num_full).collect();
        }
        self.hybrid_req_hashes
            .insert(request.request_id.clone(), req_group_hashes);
    }

    /// Hybrid prefix-cache lookup. Returns `(num_computed_tokens, per_group
    /// cached tables)`.
    ///
    /// Correctness: for a reused prefix of length P with sliding window W, no
    /// token >= P attends below P-W. So num_computed=P is valid for ALL groups
    /// ONLY IF the full/global group has all blocks [0,P) cached AND every
    /// sliding group has the blocks covering its in-window suffix [P-W, P)
    /// cached. We take the max admissible P (block-aligned to the global group).
    ///
    /// Per-group return tables (fed to allocate_slots as `matched_block_ids`):
    /// - full group: cached block id for every block in [0,P), MAX past P.
    /// - sliding group: NULL_BLOCK for [0,P-W) (out-of-window, KV never read),
    ///   the cached id for each in-window block in [P-W,P), MAX past P.
    fn get_computed_blocks_hybrid(&self, request: &Request) -> (u32, Vec<Vec<usize>>) {
        let groups = match &self.swa_groups {
            Some(g) => g.clone(),
            None => return (0, vec![Vec::new()]),
        };
        let all_tokens = &request.all_token_ids;
        let annotations = request.block_annotations.as_ref();
        let mm = request.mm_data.as_ref();

        // Per-group block hashes + hit ids (None = miss).
        let mut group_hits: Vec<Vec<Option<usize>>> = Vec::with_capacity(groups.len());
        for (g, grp) in groups.iter().enumerate() {
            let hashes = self.hash_all_blocks_bs(all_tokens, annotations, mm, grp.block_size);
            let hits: Vec<Option<usize>> = hashes
                .iter()
                .map(|h| self.hybrid_hash_to_id.get(&(g as u32, *h)).copied())
                .collect();
            group_hits.push(hits);
        }

        // Global group = the (first) non-sliding group; its contiguous hit
        // prefix block-aligns the candidate P. If all groups are sliding, use
        // group 0's block size for alignment.
        let global_g = groups.iter().position(|g| !g.is_sliding).unwrap_or(0);
        let global_bs = groups[global_g].block_size;
        let global_hits = &group_hits[global_g];
        let global_contig = global_hits.iter().take_while(|h| h.is_some()).count();

        // Helper: is group g admissible at token-count P?
        // - full group: all blocks [0,P) cached (i.e. contiguous hits cover P).
        // - sliding group: blocks covering [P-W, P) all cached; keep
        //   ceil(W/bs)+1 blocks (the partial block straddling the window edge).
        let admissible = |g: usize, p: usize| -> bool {
            let grp = &groups[g];
            let hits = &group_hits[g];
            if p == 0 {
                return true;
            }
            if !grp.is_sliding {
                // need blocks [0, ceil(P/bs)) all hit
                let need = p.div_ceil(grp.block_size);
                (0..need).all(|i| hits.get(i).map(|h| h.is_some()).unwrap_or(false))
            } else {
                let w = grp.window;
                let lo_tok = p.saturating_sub(w);
                // in-window blocks: floor(lo/bs) .. ceil(P/bs); keeping the
                // straddling partial block gives ceil(W/bs)+1 coverage.
                let lo_blk = lo_tok / grp.block_size;
                let hi_blk = p.div_ceil(grp.block_size);
                (lo_blk..hi_blk).all(|i| hits.get(i).map(|h| h.is_some()).unwrap_or(false))
            }
        };

        // Candidate P walks DOWN from the global contiguous prefix, block by
        // (global) block, until every group is admissible. Max admissible wins.
        let mut chosen_p = 0usize;
        for blocks in (0..=global_contig).rev() {
            let p = blocks * global_bs;
            if p > all_tokens.len() {
                continue;
            }
            if (0..groups.len()).all(|g| admissible(g, p)) {
                chosen_p = p;
                break;
            }
        }

        // Relocatable spans: a span block is cacheable independent of preceding
        // context, so hits PAST the contiguous gap are still valid to SHARE
        // (their unrotated KV re-ropes on read). Beyond `chosen_p` we therefore
        // emit each hit's real block id (shared, KV not recomputed) and
        // `usize::MAX` for misses (alloc fresh). num_computed stays `chosen_p`
        // (those post-gap tokens are still "recomputed" for the global mask),
        // matching the non-hybrid span path. Without annotations, post-gap slots
        // are MAX (no reuse), so the contiguous-prefix behavior is unchanged.
        // Scattered post-gap span reuse is DISABLED (contiguous-only) — see the
        // long note in `get_computed_blocks`. Sharing a relocatable block at a
        // different position (relocation) poisons the cache for non-block-aligned
        // spans; every group reuses only its contiguous prefix.
        let mut tables: Vec<Vec<usize>> = Vec::with_capacity(groups.len());
        for (g, grp) in groups.iter().enumerate() {
            let hits = &group_hits[g];
            let computed_blocks = chosen_p.div_ceil(grp.block_size);
            // Post-gap span reuse is restricted to the GLOBAL (full) group: the
            // worker's reused_block_idxs path is global-group-indexed, and a
            // sliding block reused past its window would be out-of-window at the
            // new position (a correctness hazard). Sliding groups stay
            // contiguous-only (post-gap slots → fresh, recomputed).
            let table_len = computed_blocks;
            let mut table = Vec::with_capacity(table_len);
            let lo_blk = if grp.is_sliding {
                chosen_p.saturating_sub(grp.window) / grp.block_size
            } else {
                0
            };
            for i in 0..table_len {
                if i < computed_blocks {
                    // Within the joint-admissible prefix.
                    if grp.is_sliding && i < lo_blk {
                        table.push(Self::NULL_BLOCK); // out-of-window head
                    } else {
                        table.push(hits[i].expect("admissible prefix ⇒ hit"));
                    }
                } else {
                    // Post-gap (span reuse only): share real hits, else MAX.
                    match hits.get(i).and_then(|h| *h) {
                        Some(bid) => table.push(bid),
                        None => table.push(usize::MAX),
                    }
                }
            }
            tables.push(table);
        }

        tracing::info!(
            "[CACHE-HYBRID] req={} P={} tokens ({} global blocks of {} hit)",
            request.request_id,
            chosen_p,
            chosen_p / global_bs.max(1),
            global_contig,
        );

        (chosen_p as u32, tables)
    }
}

impl KVCacheManagerOps for SimpleBlockTracker {
    /// ⭐ THE ONE WRITE OF THE STEP'S REACH, and the allocator that reads it is right below.
    ///
    /// ⛔ THE GATE IS THE DECLARED CAPABILITY, NOT A REPORT. It was `self.kv_pool_reach.is_some()`, which is
    /// false until the worker's FIRST REPORT — so every allocation before that used the token-addressed rule on
    /// a backend that shares one write slot. Found by printing this decision:
    /// `[step-reach] asked=517 applied=0 pool_reach=None`. The print is gone (it fired every step); the
    /// property it established is guarded by
    /// `a_new_request_is_allocated_for_the_step_reach_before_any_pool_reach_exists`, which sets NO pool reach.
    fn set_step_reach(&mut self, reach: scratchy_core_common::KvSlotSpan) {
        self.step_reach = if self.addressing.shares_one_write_slot() {
            reach
        } else {
            scratchy_core_common::KvSlotSpan::NONE
        };
    }

    fn set_kv_pool_reach(&mut self, reach: scratchy_core_common::KvSlotSpan) {
        self.kv_pool_reach = Some(reach);
    }

    fn allocate_slots(
        &mut self,
        request: &Request,
        num_new_tokens: usize,
        num_lookahead_tokens: usize,
        matched_block_ids: &[Vec<usize>],
    ) -> Option<Vec<Vec<usize>>> {
        let total_tokens =
            request.num_computed_tokens as usize + num_new_tokens + num_lookahead_tokens;
        // ⭐ BLOCKS ARE SIZED BY SLOTS, HASHES BY FULL TOKEN-ADDRESSED BLOCKS, AND ON SOME BACKENDS THOSE
        // ARE DIFFERENT NUMBERS.
        //
        // A worker that appends a whole batch at ONE KV slot per step leaves a
        // masked hole in every request shorter than its batch-mates: its keys then occupy MORE slots than
        // it has tokens, and past the hole token `t` no longer lives at slot `t`. Both rules live on
        // [`KvExtent`] rather than being spelled here, because "spelled here" is how the second of the
        // three sites that need them ends up sizing by tokens while the first sizes by slots.
        //
        // `kv_extent` is `None` for every backend whose KV is token-addressed (cuda/metal), and both
        // lines are then exactly the arithmetic they always were.
        let block = scratchy_core_common::KvBlockTokens::new(self.block_size)
            .expect("a KV block holds at least one token");
        let writing = num_new_tokens + num_lookahead_tokens;
        // ⭐ THE STEP'S REACH APPLIES TO **RUNNING** REQUESTS TOO — the window a per-request report cannot close.
        // A request's `kv_extent` was reported by the worker at the END of an earlier step, so it cannot know
        // about a request ADMITTED THIS STEP, and that admission is exactly what deepens the shared write slot
        // for everyone already running. Measured before this: 2-4 hole pages on every batched gate axis, served
        // by the worker's own reserve.
        //
        // ⛔ AFFORDABLE ONLY BECAUSE THE POOL IS SIZED FROM THE DECLARATION (c26a38ac). Allocating every live
        // request out to the step's reach costs `rows * pages_for(reach)`, which is precisely what
        // `PoolDemand::of_declaration(--max-model-len, --max-num-seqs)` already bought; against the old
        // hardcoded 8 GiB it would have exhausted the pool.
        let reach = self.step_reach;
        let needed = match request.kv_extent {
            // ⛔ AGED BY THE STEPS IN FLIGHT. The report is from the last FINALIZED step, which under async
            // scheduling is not the step before this one — see `Request::kv_inflight_slots` and
            // [`scratchy_core_common::InflightSlots`] for the card measurement this closes.
            Some(e) => e.blocks_needed(
                block,
                total_tokens,
                writing,
                reach,
                request.kv_inflight_slots(),
            ),
            // ⭐ NO REPORT YET — SO USE THE POOL-WIDE ONE, IF THE BACKEND GAVE US ONE.
            //
            // A request being ADMITTED has no `kv_extent`: nothing has run it. Sizing it by its own token
            // count is right on a backend whose keys sit at their token positions, and WRONG on one whose
            // batched step appends every row at a shared slot — there its very first write lands at the
            // batch's slot, which can be far past its own tokens, and the page for it was never allocated.
            // The worker has been covering that page from a reserve of its own, which is the last reason a
            // per-request KV row exists down there. `kv_pool_reach` is `None` on cuda/metal, so this is the
            // same arithmetic it always was for them.
            // ⛔⛔⛔ ONE ARM, BECAUSE THE NESTED MATCH HAD A HOLE IN IT AND THE HOLE WAS THE BUG.
            //
            // It used to read `match self.kv_pool_reach { Some(pool) => …reach…, None => sized by tokens }`
            // — so a request admitted BEFORE the worker's first report was sized by its token count and the
            // step's reach was skipped entirely, even though the scheduler had computed it correctly and the
            // tracker had stored it. Measured: `[step-reach] asked=698 applied=698 pool_reach=None`, then
            // `[hole-page] … history ends at slot 441 … host granted 2 block(s)` — 2 being ceil(441/256).
            //
            // The pool reach and the step reach are the same KIND of fact (how deep the shared write slot
            // goes), so they belong in one `max`, and the absent one is `NONE` rather than a separate branch
            // that forgets the other. `reach` is `NONE` on every token-addressed backend, so this is the same
            // arithmetic cuda/metal always had.
            None => scratchy_core_common::KvSlotSpan::NONE.blocks_this_step(
                block,
                self.kv_pool_reach
                    .unwrap_or(scratchy_core_common::KvSlotSpan::NONE)
                    .max(reach),
                total_tokens,
                writing,
            ),
        };
        // Resolved once, not as a closure over `self`: this function mutates `self` below, and the rule
        // must not be re-read after the mutation either.
        let num_hashable_full_blocks = match request.kv_extent {
            Some(e) => e.hashable_full_blocks(block, total_tokens),
            None => total_tokens / self.block_size,
        };

        // Hybrid sliding-window models: one null-padded block table per group
        // over the one shared pool. With caching on, reused cached blocks are
        // shared per group (matched_block_ids[g]); fresh tail filled otherwise.
        if self.swa_groups.is_some() {
            return self.allocate_slots_hybrid(
                request,
                num_new_tokens,
                num_lookahead_tokens,
                matched_block_ids,
            );
        }

        // Non-hybrid: single group 0.
        let matched_block_ids: &[usize] = matched_block_ids
            .first()
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        // How many blocks does this request already hold?
        let currently_held = self
            .allocations
            .get(&request.request_id)
            .map(|(_, n)| *n)
            .unwrap_or(0);

        // New/re-admitted request: build a DENSE, ordinal block table that
        // SHARES cached blocks (`matched_block_ids[i] != usize::MAX`) and
        // allocates fresh ones for misses — for EVERY slot `0..needed`, not
        // just the contiguous prefix. This is what makes spans reuse a cached
        // block past a gap (reordered docs); the old loop stopped at the
        // contiguous prefix and discarded post-gap hits. Works for the
        // non-span path too (its `matched` is a compacted contiguous prefix,
        // so slots `>= matched.len()` are simply fresh).
        if currently_held == 0 && self.enable_caching {
            let matched = matched_block_ids;
            let reuse_at = |i: usize| -> Option<usize> {
                matched.get(i).copied().filter(|&bid| bid != usize::MAX)
            };
            let any_reuse = (0..needed).any(|i| reuse_at(i).is_some());
            if any_reuse {
                let all_hashes = self.hash_all_blocks(
                    &request.all_token_ids,
                    request.block_annotations.as_ref(),
                    request.mm_data.as_ref(),
                );
                // Only FULL blocks are content-addressable (hashable + shareable) — and only the ones
                // whose keys are at their own token positions.
                let num_full_blocks = num_hashable_full_blocks;

                // 1. Capacity pre-check BEFORE any mutation (touching a
                //    ref_cnt==0 cached block consumes a free block, same as a
                //    fresh alloc). Mirrors the Python evictable-block accounting.
                let mut fresh_needed = 0usize;
                let mut evictable_reuse = 0usize;
                for i in 0..needed {
                    match reuse_at(i) {
                        Some(bid) if self.ref_cnt[bid] == 0 => evictable_reuse += 1,
                        Some(_) => {}
                        None => fresh_needed += 1,
                    }
                }
                if fresh_needed + evictable_reuse > self.num_free_blocks {
                    return None;
                }

                // 2. Touch ALL reused blocks FIRST so they leave the free queue
                //    before any fresh pop — otherwise `allocate_fresh_blocks`
                //    could reclaim a ref_cnt==0 (freed-but-still-hittable)
                //    block we intend to share, evicting its hash.
                for i in 0..needed {
                    if let Some(bid) = reuse_at(i) {
                        self.touch_block(bid);
                    }
                }

                // 3. Build the dense, ordinal block list (exactly one bid per
                //    slot) + register fresh FULL blocks so later (reordered)
                //    requests can hit them.
                let mut all_ids = Vec::with_capacity(needed);
                for i in 0..needed {
                    if let Some(bid) = reuse_at(i) {
                        all_ids.push(bid); // hash already registered by req1
                    } else {
                        let bid = self.allocate_fresh_blocks(1)?[0];
                        all_ids.push(bid);
                        if i < num_full_blocks && i < all_hashes.len() {
                            let h = all_hashes[i];
                            self.block_hash_to_id.insert(h, bid);
                            self.block_id_to_hash.insert(bid, h);
                        }
                    }
                }
                debug_assert_eq!(all_ids.len(), needed, "block table must be dense+ordinal");

                // 4. req_to_hashes = positional full-block hashes (hit + miss
                //    alike), so free/seal/re-admit accounting stays consistent.
                let full_hashes: Vec<u64> =
                    all_hashes.iter().take(num_full_blocks).copied().collect();
                self.req_to_hashes
                    .insert(request.request_id.clone(), full_hashes);
                self.allocations
                    .insert(request.request_id.clone(), (vec![all_ids.clone()], needed));
                return Some(vec![all_ids]);
            }
        }

        let additional = needed.saturating_sub(currently_held);
        if additional == 0 {
            // No new blocks needed — return existing allocation.
            return Some(
                self.allocations
                    .get(&request.request_id)
                    .map(|(blocks, _)| blocks.clone())
                    .unwrap_or_else(|| vec![Vec::new()]),
            );
        }

        // Pre-compute block hashes before mutably borrowing allocations.
        let all_hashes = if self.enable_caching {
            Some(self.hash_all_blocks(
                &request.all_token_ids,
                request.block_annotations.as_ref(),
                request.mm_data.as_ref(),
            ))
        } else {
            None
        };

        let new_block_ids = self.allocate_fresh_blocks(additional)?;

        // Merge with existing allocation.
        let entry = self
            .allocations
            .entry(request.request_id.clone())
            .or_insert_with(|| (vec![Vec::new()], 0));
        entry.0[0].extend(new_block_ids.iter().copied());
        entry.1 = needed;

        // When caching is enabled, record block hashes for full blocks.
        if let Some(all_hashes) = all_hashes {
            let all_block_ids = &entry.0[0];
            let mut hashes = self
                .req_to_hashes
                .remove(&request.request_id)
                .unwrap_or_default();

            let num_full_blocks = num_hashable_full_blocks;
            for i in hashes.len()..num_full_blocks {
                if i < all_hashes.len() && i < all_block_ids.len() {
                    let hash = all_hashes[i];
                    let bid = all_block_ids[i];
                    self.block_hash_to_id.insert(hash, bid);
                    self.block_id_to_hash.insert(bid, hash);
                    hashes.push(hash);
                }
            }

            if !hashes.is_empty() {
                self.req_to_hashes
                    .insert(request.request_id.clone(), hashes);
            }
        }

        // Return all block IDs for this request (single KV cache group).
        let result = entry.0.clone();
        Some(result)
    }

    fn free(&mut self, request_id: &str) {
        if let Some((block_ids_groups, _num_blocks)) = self.allocations.remove(request_id) {
            // Remove per-request hash tracking (hashes stay in block_hash_to_id).
            self.req_to_hashes.remove(request_id);
            self.hybrid_req_hashes.remove(request_id);

            // Hybrid path: return each group's (non-null) blocks to the shared
            // pool via the lowest-free-first heap. The null block is filtered.
            // Group-namespaced hash mappings persist (lazy eviction on realloc).
            if self.swa_groups.is_some() {
                for group in &block_ids_groups {
                    self.free_hybrid_blocks(group);
                }
                return;
            }

            // Free blocks in REVERSE order so tail (decode) blocks enter the
            // free queue first and are evicted first, while prefix blocks
            // survive longest. Matches Python's `reversed(req_blocks)`.
            //
            // Decrements ref_cnt and only adds to the free queue when
            // ref_cnt drops to 0 (shared blocks stay allocated).
            for group in &block_ids_groups {
                for &bid in group.iter().rev() {
                    debug_assert!(self.ref_cnt[bid] > 0, "double-free of block {bid}");
                    self.ref_cnt[bid] -= 1;
                    if self.ref_cnt[bid] == 0 {
                        self.num_free_blocks += 1;
                        self.free_queue.push_back(bid);
                    }
                }
            }
        }
    }

    fn seal(&mut self, request: &Request) {
        if !self.enable_caching {
            return;
        }

        // Hybrid: register group-namespaced hashes for every group's CURRENT
        // (in-window, non-null) full blocks. Out-of-window slots are NULL, so
        // they are never registered (their KV has been overwritten/freed).
        if let Some(groups) = self.swa_groups.clone() {
            let tables = match self.allocations.get(&request.request_id) {
                Some((t, _)) => t.clone(),
                None => return,
            };
            self.register_hybrid_hashes(request, &groups, &tables);
            return;
        }

        // Register hashes for any full blocks not yet tracked.
        // SealPadProcessor pads during generation so all_token_ids is
        // block-aligned by the time we get here. Just catch any hashes
        // that allocate_slots missed (additional==0 early return).
        let all_tokens = &request.all_token_ids;
        let annotations = request.block_annotations.as_ref();
        let all_hashes = self.hash_all_blocks(all_tokens, annotations, request.mm_data.as_ref());
        // ⛔ THE SAME RULE `allocate_slots` USES, THROUGH THE SAME METHOD, because this is a SECOND place
        // that registers hashes. Sealing a request whose keys spread past its token count would promise a
        // later request blocks covering a masked hole.
        let num_full_blocks = match request.kv_extent {
            Some(e) => e.hashable_full_blocks(
                scratchy_core_common::KvBlockTokens::new(self.block_size)
                    .expect("a KV block holds at least one token"),
                all_tokens.len(),
            ),
            None => all_tokens.len() / self.block_size,
        };

        let (block_ids_groups, _) = match self.allocations.get(&request.request_id) {
            Some(entry) => entry,
            None => return,
        };
        let block_ids = match block_ids_groups.first() {
            Some(ids) if !ids.is_empty() => ids,
            _ => return,
        };

        let hashes = self
            .req_to_hashes
            .entry(request.request_id.clone())
            .or_default();
        for i in hashes.len()..num_full_blocks {
            if i < all_hashes.len() && i < block_ids.len() {
                let hash = all_hashes[i];
                let bid = block_ids[i];
                self.block_hash_to_id.insert(hash, bid);
                self.block_id_to_hash.insert(bid, hash);
                hashes.push(hash);

                if self.spans_config.debug {
                    tracing::debug!(
                        "[SPANS] Sealed block {} (bid={}) for request {}",
                        i,
                        bid,
                        request.request_id,
                    );
                }
            }
        }
    }

    fn free_volatile(&mut self, request_id: &str) {
        if let Some((block_ids_groups, _num_blocks)) = self.allocations.remove(request_id) {
            self.req_to_hashes.remove(request_id);
            self.hybrid_req_hashes.remove(request_id);

            // Hybrid path: return to the shared pool (front-eviction not
            // applicable to the lowest-free-first heap; treat as a normal free).
            if self.swa_groups.is_some() {
                for group in &block_ids_groups {
                    self.free_hybrid_blocks(group);
                }
                return;
            }

            // Free blocks and push to the FRONT of the free queue so they
            // are evicted first (before non-volatile blocks).
            for group in &block_ids_groups {
                for &bid in group.iter().rev() {
                    debug_assert!(self.ref_cnt[bid] > 0, "double-free of block {bid}");
                    self.ref_cnt[bid] -= 1;
                    if self.ref_cnt[bid] == 0 {
                        self.num_free_blocks += 1;
                        self.free_queue.push_front(bid);
                    }
                }
            }
        }
    }

    fn get_blocks(&self, request_id: &str) -> Vec<Vec<usize>> {
        self.allocations
            .get(request_id)
            .map(|(blocks, _)| blocks.clone())
            .unwrap_or_else(|| vec![Vec::new()])
    }

    fn get_computed_blocks(&self, request: &Request) -> (u32, Vec<Vec<usize>>) {
        if !self.enable_caching {
            return (0, vec![Vec::new()]);
        }

        if self.swa_groups.is_some() {
            return self.get_computed_blocks_hybrid(request);
        }

        let all_tokens = &request.all_token_ids;
        let annotations = request.block_annotations.as_ref();
        let hashes = self.hash_all_blocks(all_tokens, annotations, request.mm_data.as_ref());
        let has_relocatable = annotations.is_some();

        if !has_relocatable {
            // No annotations: find longest contiguous prefix of cache hits.
            let mut matched_block_ids = Vec::new();
            let mut num_matched_tokens = 0u32;

            for hash in &hashes {
                if let Some(&block_id) = self.block_hash_to_id.get(hash) {
                    matched_block_ids.push(block_id);
                    num_matched_tokens += self.block_size as u32;
                } else {
                    break;
                }
            }

            tracing::info!(
                "[CACHE] req={} prefix: {}/{} blocks hit, {} computed tokens",
                request.request_id,
                matched_block_ids.len(),
                hashes.len(),
                num_matched_tokens,
            );
            return (num_matched_tokens, vec![matched_block_ids]);
        }

        // Span-aware path: relocatable HASHING (hash_block_with_parent chains
        // the parent within a span, resetting to NONE_HASH only at the span's
        // first block) makes a tool's prefix match regardless of what precedes
        // it in the sequence — that is the position-independence spans buy.
        //
        // Reuse is nonetheless restricted to the CONTIGUOUS prefix of hits
        // (break on first miss, same as the non-span path). Scattered post-gap
        // reuse shared a relocatable block at a DIFFERENT position (relocation,
        // served by rope-on-read). That is correct only for block-aligned spans;
        // for non-block-aligned spans (the `/v1/messages` tool spans, which are
        // NOT padded to a block) it feeds mis-positioned KV → garbage, and worse
        // it POISONS the cache — a later request that merely shares a 16-token
        // schema sub-block scatter-hits stale KV and garbles too. Contiguous
        // reuse (same position ⇒ rope-on-read is a no-op) is always safe and is
        // exactly what launch-claude needs (identical tools+system prefix reused
        // across launches). Relocation for non-aligned spans is a separate
        // rope-on-read fix; until then it stays off so the cache can't be
        // poisoned. Block-aligned callers (the `bench spans` reorder path) that
        // want post-gap sharing back must land that fix first.
        let mut matched_block_ids = Vec::new();
        for hash in &hashes {
            match self.block_hash_to_id.get(hash) {
                Some(&block_id) => matched_block_ids.push(block_id),
                None => break,
            }
        }
        let contiguous_tokens = (matched_block_ids.len() * self.block_size) as u32;

        tracing::info!(
            "[CACHE] req={} spans (contiguous-only): {}/{} blocks hit, {} computed tokens",
            request.request_id,
            matched_block_ids.len(),
            hashes.len(),
            contiguous_tokens,
        );

        (contiguous_tokens, vec![matched_block_ids])
    }

    fn new_step_starts(&mut self) {
        // Nothing to do for the simple tracker.
    }

    fn reset_prefix_cache(&mut self) -> bool {
        if !self.enable_caching {
            return true;
        }
        self.block_hash_to_id.clear();
        self.block_id_to_hash.clear();
        self.req_to_hashes.clear();
        self.hybrid_hash_to_id.clear();
        self.hybrid_id_to_key.clear();
        self.hybrid_req_hashes.clear();
        true
    }

    fn num_free_blocks(&self) -> usize {
        self.num_free_blocks
    }

    fn num_total_blocks(&self) -> usize {
        self.total_blocks
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn usage(&self) -> f64 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        1.0 - self.num_free_blocks as f64 / self.total_blocks as f64
    }

    fn num_cached_blocks(&self) -> usize {
        // Cached blocks = free blocks that have a hash mapping.
        if self.swa_groups.is_some() {
            return self
                .hybrid_id_to_key
                .keys()
                .filter(|&&bid| self.ref_cnt[bid] == 0)
                .count();
        }
        self.block_id_to_hash
            .keys()
            .filter(|&&bid| self.ref_cnt[bid] == 0)
            .count()
    }
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// The main scheduler implementation.
///
/// Ported from `vllm.v1.core.sched.scheduler.Scheduler`.
///
/// The scheduling algorithm works in two phases:
/// 1. Schedule RUNNING requests -- assign tokens and handle preemption when
///    blocks are exhausted.
/// 2. Schedule WAITING requests -- compute cached blocks, assign tokens, and
///    allocate new blocks.
///
/// The output is a [`SchedulerOutput`] that tells the model runner exactly
/// which requests to process and how many tokens each should get.
pub struct Scheduler {
    // -- Configuration --
    /// Maximum number of requests that can be in the running state.
    max_num_running_reqs: usize,
    /// Maximum number of tokens to schedule in a single step.
    max_num_scheduled_tokens: usize,
    /// Maximum context length supported by the model.
    max_model_len: usize,
    /// Whether chunked prefill is enabled.
    enable_chunked_prefill: bool,
    /// Prefills longer than this threshold are split across steps.
    long_prefill_token_threshold: usize,
    /// Number of speculative lookahead tokens (0 = no speculation).
    num_lookahead_tokens: usize,
    /// Whether async scheduling is enabled (pre-scheduling overlap).
    /// When true, `update_after_schedule` increments `num_output_placeholders`
    /// for non-prefill requests so the next `schedule()` call accounts for
    /// tokens that are in-flight on the GPU but not yet finalized.
    async_scheduling: bool,
    /// Whether pipeline parallelism is active. When true and
    /// `async_scheduling` is false, the scheduler populates `new_token_ids`
    /// in `CachedRequestData`.
    use_pp: bool,

    // -- Request state --
    /// All tracked requests: `req_id -> Request`.
    requests: HashMap<String, Request>,
    /// Queue of requests waiting to be scheduled.
    waiting: Box<dyn RequestQueue>,
    /// Requests currently in the running state.
    running: Vec<Request>,
    /// Index: `req_id -> index in running`. Kept in sync for O(1) lookup.
    running_req_idx: HashMap<String, usize>,
    /// Request IDs finished between the previous and current steps.
    finished_req_ids: HashSet<String>,
    /// Scheduling pause state.
    pause_state: PauseState,

    // -- KV cache --
    /// The KV cache manager (abstracted via trait).
    kv_cache: Box<dyn KVCacheManagerOps>,
}

impl Scheduler {
    /// Create a new scheduler with a `SchedulerConfig` and a KV cache
    /// manager.
    pub fn new(
        scheduler_config: &SchedulerConfig,
        max_model_len: usize,
        kv_cache: Box<dyn KVCacheManagerOps>,
    ) -> Self {
        let policy = match &scheduler_config.policy {
            SchedulerPolicy::Fcfs => SchedulingPolicy::Fcfs,
            SchedulerPolicy::Priority => SchedulingPolicy::Priority,
        };

        let max_num_scheduled_tokens = scheduler_config
            .max_num_scheduled_tokens
            .unwrap_or(scheduler_config.max_num_batched_tokens);

        let async_scheduling = scheduler_config.async_scheduling.unwrap_or(false);
        let use_pp = scheduler_config.use_pp;

        Self {
            max_num_running_reqs: scheduler_config.max_num_seqs,
            max_num_scheduled_tokens,
            max_model_len,
            enable_chunked_prefill: scheduler_config.enable_chunked_prefill,
            long_prefill_token_threshold: scheduler_config.long_prefill_token_threshold,
            num_lookahead_tokens: scheduler_config.num_lookahead_tokens,
            async_scheduling,
            use_pp,

            requests: HashMap::new(),
            waiting: create_request_queue(policy),
            running: Vec::new(),
            running_req_idx: HashMap::new(),
            finished_req_ids: HashSet::new(),
            pause_state: PauseState::Unpaused,

            kv_cache,
        }
    }

    /// Convenience constructor using a [`SimpleBlockTracker`].
    ///
    /// Useful for testing.
    pub fn with_simple_blocks(
        scheduler_config: &SchedulerConfig,
        max_model_len: usize,
        num_gpu_blocks: usize,
        block_size: usize,
    ) -> Self {
        let kv_cache = Box::new(SimpleBlockTracker::new(num_gpu_blocks, block_size));
        Self::new(scheduler_config, max_model_len, kv_cache)
    }

    /// KV cache usage as a fraction in `[0.0, 1.0]`.
    pub fn kv_cache_usage(&self) -> f64 {
        self.kv_cache.usage()
    }

    /// Total number of GPU KV cache blocks.
    pub fn num_total_blocks(&self) -> usize {
        self.kv_cache.num_total_blocks()
    }

    /// Number of GPU KV cache blocks currently in use.
    pub fn num_used_blocks(&self) -> usize {
        self.kv_cache.num_total_blocks() - self.kv_cache.num_free_blocks()
    }

    /// Number of blocks retained in the prefix cache.
    pub fn num_cached_blocks(&self) -> usize {
        self.kv_cache.num_cached_blocks()
    }

    // -- Internal helpers --

    /// Push a request onto the running queue and update the O(1) index.
    fn running_push(&mut self, request: Request) {
        let idx = self.running.len();
        self.running_req_idx.insert(request.request_id.clone(), idx);
        self.running.push(request);
    }

    /// Remove the request at `pos` from the running queue and repair the index.
    ///
    /// After `Vec::remove(pos)`, every element at index > pos shifts down by
    /// one.  We fix those up in O(n) — acceptable because removals are rare
    /// (only on finish / abort by request_id).
    ///
    /// For tail removal (preemption), prefer `running_pop_tail()` which is O(1).
    fn running_remove(&mut self, pos: usize) -> Request {
        let req = self.running.remove(pos);
        self.running_req_idx.remove(&req.request_id);
        // Decrement indices for all elements that shifted.
        for idx in self.running_req_idx.values_mut() {
            if *idx > pos {
                *idx -= 1;
            }
        }
        req
    }

    /// Remove the last request from the running queue. O(1).
    ///
    /// Used for FCFS preemption which always evicts the tail.
    fn running_pop_tail(&mut self) -> Request {
        let req = self.running.pop().expect("running queue is non-empty");
        self.running_req_idx.remove(&req.request_id);
        req
    }

    /// Preempt a request: free its KV cache blocks, mark it as preempted,
    /// and move it back to the waiting queue.
    fn preempt_request(&mut self, request: &mut Request) {
        assert_eq!(
            request.status,
            RequestStatus::Running,
            "Only running requests can be preempted"
        );
        self.kv_cache.free(&request.request_id);
        request.status = RequestStatus::Preempted;
        request.num_computed_tokens = 0;
        request.spec_token_ids.clear();
        request.num_preemptions += 1;
        // Async scheduling: reset placeholder count so the request is
        // re-scheduled from scratch after resuming.
        request.num_output_placeholders = 0;

        // Sync to canonical requests map.
        if let Some(canonical) = self.requests.get_mut(&request.request_id) {
            canonical.status = RequestStatus::Preempted;
            canonical.num_computed_tokens = 0;
            canonical.spec_token_ids.clear();
            canonical.num_preemptions = request.num_preemptions;
            canonical.num_output_placeholders = 0;
        }
    }

    /// Build `CachedRequestData` for running + resumed requests.
    fn make_cached_request_data(
        &self,
        running_reqs: &[usize],
        resumed_reqs: &[usize],
        num_scheduled_tokens: &HashMap<String, usize>,
        req_to_new_blocks: &HashMap<String, Vec<Vec<usize>>>,
    ) -> CachedRequestData {
        let mut req_ids = Vec::new();
        let mut resumed_req_ids = HashSet::new();
        let mut new_token_ids = Vec::new();
        let mut new_block_ids = Vec::new();
        let mut num_computed_tokens_vec = Vec::new();
        let mut num_output_tokens_vec = Vec::new();

        // When PP is active and async scheduling is off, the scheduler sends
        // sampled tokens back because there's no direct communication between
        // the first-stage worker and the last-stage worker.
        // Matches Python: vllm/v1/core/sched/scheduler.py _make_cached_request_data
        let send_pp_tokens = self.use_pp && !self.async_scheduling;

        for (idx, &slot) in running_reqs.iter().chain(resumed_reqs.iter()).enumerate() {
            let is_resumed = idx >= running_reqs.len();
            let req = &self.running[slot];
            let req_id = req.request_id.clone();

            if is_resumed {
                resumed_req_ids.insert(req_id.clone());
            }

            if send_pp_tokens {
                let num_tokens = num_scheduled_tokens.get(&req_id).copied().unwrap_or(0);
                let start = req.num_computed_tokens as usize;
                let end = (start + num_tokens).min(req.all_token_ids.len());
                let token_ids = req.all_token_ids[start..end].to_vec();
                new_token_ids.push(token_ids);
            }

            // Block IDs: convert to the output format. Look up before moving req_id.
            let blocks = req_to_new_blocks.get(&req_id).cloned();
            new_block_ids.push(blocks);

            req_ids.push(req_id);

            num_computed_tokens_vec.push(req.num_computed_tokens);
            num_output_tokens_vec
                .push(req.num_output_tokens() as u32 + req.num_output_placeholders);
        }

        CachedRequestData {
            req_ids,
            resumed_req_ids,
            new_token_ids,
            new_block_ids,
            num_computed_tokens: num_computed_tokens_vec,
            num_output_tokens: num_output_tokens_vec,
        }
    }

    /// Update computed token counts after scheduling and clear finished IDs.
    fn update_after_schedule(&mut self, output: &SchedulerOutput) {
        for (req_id, &num_tokens) in &output.num_scheduled_tokens {
            if let Some(request) = self.requests.get_mut(req_id) {
                request.num_computed_tokens += num_tokens as u32;
                request.is_prefill_chunk = (request.num_computed_tokens as usize)
                    < request.num_tokens() + request.num_output_placeholders as usize;

                // Async scheduling: increment placeholders for non-prefill
                // decode requests. Each scheduled decode step will produce
                // 1 token (+ num_spec_tokens draft tokens) whose values are
                // unknown until `update_from_output` runs.
                if self.async_scheduling && !request.is_prefill_chunk {
                    let cur_num_spec_tokens = output
                        .scheduled_spec_decode_tokens
                        .get(req_id)
                        .map(|v| v.len() as u32)
                        .unwrap_or(0);
                    request.num_output_placeholders += 1 + cur_num_spec_tokens;
                }
            }
        }

        // Sync the running list with the authoritative requests map.
        for running_req in &mut self.running {
            if let Some(canonical) = self.requests.get(&running_req.request_id) {
                running_req.num_computed_tokens = canonical.num_computed_tokens;
                running_req.is_prefill_chunk = canonical.is_prefill_chunk;
                running_req.num_output_placeholders = canonical.num_output_placeholders;
            }
        }

        // finished_req_ids already moved into the output via mem::take.
    }

    /// Try to finish a single request. Returns the `(req_id, client_index)`
    /// if the request was found and not already finished.
    fn finish_single_request(
        &mut self,
        request_id: &str,
        status: RequestStatus,
    ) -> Option<(String, u32)> {
        // Check if the request exists.
        let request = self.requests.get(request_id)?;
        if request.status.is_finished() {
            return None;
        }
        let client_index = request.client_index;
        let req_id = request.request_id.clone();

        // Remove from running queue.
        if let Some(&pos) = self.running_req_idx.get(request_id) {
            let mut request = self.running_remove(pos);
            // 🦭 Seal: pad + hash the final partial block so it's cacheable.
            if request.seal {
                self.kv_cache.seal(&request);
            }
            // For sealed requests, always use normal free (back of LRU queue)
            // so the blocks persist for future cache hits. Volatile eviction
            // (front-of-queue) happens at read time, not finish time.
            if request.volatile && !request.seal {
                self.kv_cache.free_volatile(&request.request_id);
            } else {
                self.kv_cache.free(&request.request_id);
            }
            request.status = status;
            self.requests.insert(request.request_id.clone(), request);
        } else {
            // Remove from waiting queue.
            self.waiting.remove_request(request_id);
            if let Some(r) = self.requests.get_mut(request_id) {
                r.status = status;
            }
        }

        self.finished_req_ids.insert(req_id.clone());
        Some((req_id, client_index))
    }

    // -- Request accessors (used by EngineCore for stop criteria) --

    /// Get a reference to a request by ID.
    pub fn get_request(&self, request_id: &str) -> Option<&Request> {
        self.requests.get(request_id)
    }

    /// Get a mutable reference to a request by ID.
    pub fn get_request_mut(&mut self, request_id: &str) -> Option<&mut Request> {
        self.requests.get_mut(request_id)
    }

    /// Set speculative draft token IDs on a request, updating both the
    /// canonical `requests` map and the `running` list.
    pub fn set_spec_token_ids(&mut self, request_id: &str, spec_token_ids: Vec<u32>) {
        if let Some(request) = self.requests.get_mut(request_id) {
            request.spec_token_ids = spec_token_ids.clone();
        }
        if let Some(&idx) = self.running_req_idx.get(request_id) {
            self.running[idx].spec_token_ids = spec_token_ids;
        }
    }

    /// Rewind `num_computed_tokens` for rejected speculative decode drafts.
    ///
    /// When spec decode drafts are rejected, the scheduler already advanced
    /// `num_computed_tokens` by the full scheduled count (including all drafts).
    /// This method decrements it by `num_rejected` so the KV cache position is
    /// correct for the next step.
    ///
    /// Matches Python's `scheduler.update_from_output()`:
    ///   `request.num_computed_tokens -= num_rejected`
    ///   `request.num_output_placeholders -= num_rejected`
    pub fn rewind_num_computed_tokens(&mut self, request_id: &str, num_rejected: usize) {
        if let Some(request) = self.requests.get_mut(request_id) {
            request.num_computed_tokens = request
                .num_computed_tokens
                .saturating_sub(num_rejected as u32);
            request.num_output_placeholders = request
                .num_output_placeholders
                .saturating_sub(num_rejected as u32);
        }
        if let Some(&idx) = self.running_req_idx.get(request_id) {
            let running_req = &mut self.running[idx];
            running_req.num_computed_tokens = running_req
                .num_computed_tokens
                .saturating_sub(num_rejected as u32);
            running_req.num_output_placeholders = running_req
                .num_output_placeholders
                .saturating_sub(num_rejected as u32);
        }
    }

    /// Append output token IDs to a request, updating both the canonical
    /// `requests` map and the `running` list.
    pub fn append_output_tokens(&mut self, request_id: &str, token_ids: &[u32]) {
        if let Some(request) = self.requests.get_mut(request_id) {
            request.append_output_token_ids(token_ids);
            // Async scheduling: decrement placeholders now that actual
            // tokens have arrived from the GPU.
            request.num_output_placeholders = request
                .num_output_placeholders
                .saturating_sub(token_ids.len() as u32);
        }
        // Also update the running list copy via O(1) index lookup.
        if let Some(&idx) = self.running_req_idx.get(request_id) {
            let running_req = &mut self.running[idx];
            running_req.append_output_token_ids(token_ids);
            running_req.num_output_placeholders = running_req
                .num_output_placeholders
                .saturating_sub(token_ids.len() as u32);
        }
    }

    /// ⭐ RECORD WHAT THE WORKER SAID ITS KV LOOKS LIKE for one request — the slots to allocate blocks
    /// for and the prefix that may be hashed into the cache. Both are read by
    /// `KVCacheManager::allocate_slots` on the NEXT step.
    ///
    /// Written into BOTH the canonical map and the `running` copy, for the same reason
    /// [`append_output_tokens`](Self::append_output_tokens) is: the two are separate `Request` values and
    /// `allocate_slots` is called with whichever one the scheduling path happens to hold. A field updated
    /// in one of them is a value that is right or wrong depending on the caller.
    /// THE POOL-WIDE REACH the worker last reported — the deepest slot any live request will occupy, and
    /// therefore the slot a batched step's shared write lands at or before.
    ///
    /// ⛔ IT SIZES THE REQUESTS `set_kv_extent` CANNOT. That one is keyed by request id, so a request being
    /// ADMITTED has no entry and is sized by its token count alone; if it then joins a batch whose write slot
    /// is already deep, its write page is past every block it was given. `None` (and so inert) for every
    /// backend whose keys sit at their token positions.
    pub fn set_kv_pool_reach(&mut self, reach: scratchy_core_common::KvSlotSpan) {
        // ⛔ FORWARDED, NOT STORED. The allocator is what reads it, and a copy on the scheduler would be a
        // second home for one number — which is the shape every defect in this area has had.
        self.kv_cache.set_kv_pool_reach(reach);
    }

    pub fn set_kv_extent(&mut self, request_id: &str, extent: scratchy_core_common::KvExtent) {
        if let Some(request) = self.requests.get_mut(request_id) {
            request.kv_extent = Some(extent);
        }
        if let Some(&idx) = self.running_req_idx.get(request_id) {
            self.running[idx].kv_extent = Some(extent);
        }
    }
}

impl SchedulerInterface for Scheduler {
    fn schedule(&mut self) -> SchedulerOutput {
        // The scheduling algorithm, ported from Python's Scheduler.schedule():
        //
        // 1. Schedule RUNNING requests first: assign tokens, handle preemption
        //    when blocks run out.
        // 2. Schedule WAITING requests next: compute prefix cache hits,
        //    assign tokens, allocate blocks.
        // 3. Build and return SchedulerOutput.

        // ⭐⭐⭐ THE STEP'S REACH, COMPUTED BEFORE ANYTHING IS ALLOCATED.
        //
        // A batched step appends every row at ONE KV slot, as deep as the DEEPEST live request. So a request
        // admitted later in this very function deepens the write slot of every request allocated earlier in it,
        // whose allocation was sized by a worker report made before that request existed. Both queues are
        // visible here and nothing has been allocated yet, so the maximum is exact for running requests and an
        // upper bound for waiting ones — a prompt is fixed at admission, so a waiting request cannot end up
        // deeper than its prompt.
        //
        // Over-reaching costs pages the pool was already sized for; under-reaching costs a page nobody owns.
        //
        // ⛔⛔⛔ IT IS **POST-APPEND**, AND THE FIRST VERSION WAS NOT — WHICH THE CARD CAUGHT AND FOUR KANI
        // PROOFS DID NOT. A pre-append max looks right and is not: the shared write slot ends at
        // `deepest_row_end + THAT ROW'S append`, while each request was allocated `its own end + ITS OWN
        // append`. A deep row taking a full 96-token prefill chunk beside a shallow row appending one decode
        // token leaves the shallow row exactly one page short. MEASURED on the 2b gate with the pre-append
        // version: the hole arm still fired on `hits`-off, `short`-off and `longhits`-reversed (2-3 pages
        // each) — the caching-OFF runs, where full re-prefills make those appends large, and the permuted run,
        // where a different row is deepest. So a row's depth here INCLUDES the append it is about to make.
        //
        // The append bound mirrors the loop's own cap chain below (`num_new_tokens_raw`, the long-prefill
        // threshold, `max_num_scheduled_tokens` as the ceiling on `token_budget`, and `max_model_len`), so it is
        // an upper bound on what any row can append — and every term is a property of that request alone, with
        // no dependence on the allocation order. Any excess is pages the declaration already bought.
        //
        // ⛔ `num_tokens_with_spec`, NOT `num_tokens`: a row's draft tokens are slots it appends this step, so
        // they move the shared write slot for everyone else. Equal when no draft is attached.
        let append_ub = |r: &Request| -> usize {
            let raw = r
                .num_tokens_with_spec()
                .saturating_add(r.num_output_placeholders as usize)
                .saturating_sub(r.num_computed_tokens as usize);
            let capped = if self.long_prefill_token_threshold > 0 {
                raw.min(self.long_prefill_token_threshold)
            } else {
                raw
            };
            capped.min(self.max_num_scheduled_tokens).min(
                self.max_model_len
                    .saturating_sub(1 + r.num_computed_tokens as usize),
            )
        };
        let step_reach = scratchy_core_common::KvSlotSpan::new(
            self.running
                .iter()
                .chain(self.waiting.iter())
                .map(|r| {
                    // Where the row's keys end BEFORE this step: its reported span if the worker has run it,
                    // its token count otherwise — then the slots it is about to append.
                    //
                    // ⛔⛔⛔ THE SPAN IS AGED BY THE STEPS IN FLIGHT, AND WITHOUT THAT THIS WHOLE REACH IS
                    // ONE SLOT SHALLOW. A report reaches the scheduler from the last FINALIZED step, and the
                    // async loop schedules step `n+1` while step `n` is still on the card — so a reach built
                    // from the raw report is the reach of the step BEFORE the one being scheduled, and its
                    // page count is one short at every page boundary. The rescue arm in `blocks_this_step`
                    // reads this same number, so nothing else caught it.
                    let end = r.kv_extent.map_or(r.num_tokens_with_spec(), |e| {
                        (e.span_now(r.kv_inflight_slots()).get() as usize)
                            .max(r.num_tokens_with_spec())
                    });
                    end.saturating_add(append_ub(r))
                })
                .max()
                .unwrap_or(0)
                .try_into()
                .unwrap_or(u32::MAX),
        );
        self.kv_cache.set_step_reach(step_reach);

        let mut scheduled_new_reqs: Vec<Request> = Vec::new();
        // Zero-copy: running/resumed requests are recorded by their index into
        // `self.running` and read by reference when building CachedRequestData,
        // rather than deep-cloning the (growing) Request every step. Only new
        // requests, whose full data is sent to the worker once per lifetime,
        // are cloned (into NewRequestData below).
        let mut scheduled_resumed_reqs: Vec<usize> = Vec::new();
        let mut scheduled_running_reqs: Vec<usize> = Vec::new();
        let mut preempted_reqs: Vec<Request> = Vec::new();

        let mut req_to_new_blocks: HashMap<String, Vec<Vec<usize>>> = HashMap::new();
        // Spans Phase 2: per new request, the logical block indices that are
        // reused cache hits (shared via §a) → worker skips rewriting their KV.
        let mut req_to_reused_blocks: HashMap<String, Vec<usize>> = HashMap::new();
        let mut num_scheduled_tokens: HashMap<String, usize> = HashMap::new();
        let mut token_budget = self.max_num_scheduled_tokens;
        let mut scheduled_spec_decode_tokens: HashMap<String, Vec<u32>> = HashMap::new();

        if self.pause_state == PauseState::PausedAll {
            token_budget = 0;
        }

        self.kv_cache.new_step_starts();

        // ---------------------------------------------------------------
        // Phase 1: Schedule RUNNING requests
        // ---------------------------------------------------------------
        let mut req_index = 0;
        while req_index < self.running.len() && token_budget > 0 {
            let request = &self.running[req_index];

            // Async scheduling: skip requests that will certainly hit
            // max_tokens once the in-flight step completes. Without this
            // guard the scheduler would schedule one extra decode step
            // because `update_from_output` (which checks the finish
            // condition) hasn't run yet for the previous step.
            //
            // The formula is: (num_computed_tokens + 1) - (num_output_placeholders - 1)
            //   = num_computed_tokens + 2 - num_output_placeholders
            // Since placeholders are included in num_computed_tokens, we
            // subtract (placeholders - 1) to count only the guaranteed
            // minimum tokens (all drafts rejected).
            if request.num_output_placeholders > 0 {
                let effective_computed = request
                    .num_computed_tokens
                    .saturating_add(2)
                    .saturating_sub(request.num_output_placeholders);
                let max_total = request.num_prompt_tokens + request.max_tokens;
                if effective_computed >= max_total {
                    req_index += 1;
                    continue;
                }
            }

            let request = &self.running[req_index];

            // How many tokens does this request need computed?
            let num_new_tokens_raw = request
                .num_tokens_with_spec()
                .saturating_add(request.num_output_placeholders as usize)
                .saturating_sub(request.num_computed_tokens as usize);

            let mut num_new_tokens = num_new_tokens_raw;

            // Apply long-prefill threshold.
            if self.long_prefill_token_threshold > 0
                && num_new_tokens > self.long_prefill_token_threshold
            {
                num_new_tokens = self.long_prefill_token_threshold;
            }

            // Respect token budget.
            num_new_tokens = num_new_tokens.min(token_budget);

            // Ensure we don't exceed max model length.
            let max_remaining = self
                .max_model_len
                .saturating_sub(1 + request.num_computed_tokens as usize);
            num_new_tokens = num_new_tokens.min(max_remaining);

            if num_new_tokens == 0 {
                req_index += 1;
                continue;
            }

            // Zero-copy: pass a reference to the running request directly.
            // `kv_cache` and `running` are disjoint fields of `self`, so the
            // borrow checker permits `&mut self.kv_cache` (the method receiver)
            // alongside `&self.running[req_index]` — no Request clone needed.
            let new_blocks = self.kv_cache.allocate_slots(
                &self.running[req_index],
                num_new_tokens,
                self.num_lookahead_tokens,
                &[], // running/decode: blocks already held, no new cache reuse
            );

            if let Some(blocks) = new_blocks {
                // Successfully allocated. Zero-copy: record the running index;
                // the request is read by reference in make_cached_request_data.
                scheduled_running_reqs.push(req_index);
                let request_id = self.running[req_index].request_id.clone();

                // Handle speculative decode tokens.
                if !self.running[req_index].spec_token_ids.is_empty() {
                    let num_scheduled_spec = num_new_tokens
                        .saturating_add(self.running[req_index].num_computed_tokens as usize)
                        .saturating_sub(self.running[req_index].num_tokens());
                    if num_scheduled_spec > 0 {
                        let spec_ids = &self.running[req_index].spec_token_ids;
                        let truncated: Vec<u32> =
                            spec_ids.iter().take(num_scheduled_spec).copied().collect();
                        scheduled_spec_decode_tokens.insert(request_id.clone(), truncated);
                    }
                    // Clear spec tokens for next step (both running list and canonical map).
                    self.running[req_index].spec_token_ids.clear();
                    if let Some(canonical) = self.requests.get_mut(&request_id) {
                        canonical.spec_token_ids.clear();
                    }
                }

                req_to_new_blocks.insert(request_id.clone(), blocks);
                num_scheduled_tokens.insert(request_id, num_new_tokens);
                token_budget -= num_new_tokens;
                req_index += 1;
            } else {
                // Allocation failed -- preempt the last running request.
                // (FCFS: preempt from the back; Priority would pick the
                // lowest-priority request, but for simplicity we always
                // preempt from the back.)
                if self.running.len() <= 1 {
                    // Can't preempt — the only running request is this one,
                    // so nothing else will ever free pool blocks. Retrying
                    // next step would return this same empty schedule forever:
                    // a silent livelock the no-progress watchdog only breaks
                    // after 60s (measured, real launch-claude traffic). Fail
                    // the request NOW with a real reason instead.
                    let rid = self.running[req_index].request_id.clone();
                    warn!(
                        request_id = %rid,
                        computed = self.running[req_index].num_computed_tokens,
                        total = self.running[req_index].num_tokens(),
                        "KV pool exhausted mid-request with nothing to preempt \
                         — aborting the request (would otherwise livelock)"
                    );
                    self.finish_requests(&[&rid], RequestStatus::FinishedAborted);
                    break;
                }

                // Preempt the last request (lowest priority in FCFS order).
                // Both branches pop the tail, so use running_pop_tail() — O(1).
                let preempt_idx = self.running.len() - 1;
                if preempt_idx == req_index {
                    // The request we're trying to schedule is the last one;
                    // preempt it.
                    let mut preempted = self.running_pop_tail();
                    self.preempt_request(&mut preempted);

                    // Remove from scheduled lists if it was already scheduled.
                    let pid = preempted.request_id.clone();
                    if let Some(tokens) = num_scheduled_tokens.remove(&pid) {
                        token_budget += tokens;
                    }
                    req_to_new_blocks.remove(&pid);
                    scheduled_spec_decode_tokens.remove(&pid);
                    scheduled_running_reqs.retain(|&idx| idx != preempt_idx);

                    // Clone the request for the waiting queue; move into preempted list.
                    self.waiting.prepend_request(preempted.clone());
                    preempted_reqs.push(preempted);
                    break;
                } else {
                    let mut preempted = self.running_pop_tail();
                    self.preempt_request(&mut preempted);

                    // Restore budget if this request was scheduled.
                    let pid = preempted.request_id.clone();
                    if let Some(tokens) = num_scheduled_tokens.remove(&pid) {
                        token_budget += tokens;
                    }
                    req_to_new_blocks.remove(&pid);
                    scheduled_spec_decode_tokens.remove(&pid);
                    scheduled_running_reqs.retain(|&idx| idx != preempt_idx);

                    self.waiting.prepend_request(preempted.clone());
                    preempted_reqs.push(preempted);
                    // Retry the current request.
                    continue;
                }
            }
        }

        // ---------------------------------------------------------------
        // Phase 2: Schedule WAITING requests
        // ---------------------------------------------------------------
        if preempted_reqs.is_empty() && self.pause_state == PauseState::Unpaused {
            while !self.waiting.is_empty() && token_budget > 0 {
                if self.running.len() >= self.max_num_running_reqs {
                    break;
                }

                // Pop from the waiting queue directly to avoid peek+clone.
                let mut request = match self.waiting.pop_request() {
                    Some(r) => r,
                    None => break,
                };
                let request_id = request.request_id.clone();

                // Get computed blocks from prefix cache. `cached_blocks[0]` is
                // the per-logical-block hit list (usize::MAX for misses) —
                // threaded into allocate_slots so span blocks are SHARED.
                let (num_cached_tokens, cached_blocks) =
                    self.kv_cache.get_computed_blocks(&request);

                // How many tokens need to be scheduled.
                let total_tokens = request.num_tokens();

                // When the prefix cache covers the entire prompt, backing up
                // by one block ensures the model has real input tokens to
                // process. Without this, num_new_tokens would be 0 and the
                // request would be stuck in WAITING forever. Reprocessing the
                // last block is cheap and establishes the decode invariant
                // (num_tokens > num_computed after the first output token is
                // generated).
                let num_computed_tokens =
                    if num_cached_tokens as usize >= total_tokens && num_cached_tokens > 0 {
                        let bs = self.kv_cache.block_size();
                        (((num_cached_tokens as usize) / bs).saturating_sub(1) * bs) as u32
                    } else {
                        num_cached_tokens
                    };

                let num_new_tokens_raw = total_tokens.saturating_sub(num_computed_tokens as usize);

                // Defense-in-depth against the never-schedulable busy-spin: a
                // brand-new WAITING request whose prompt alone requires more KV
                // blocks than exist in the entire pool can NEVER be placed, even
                // with everything free. The old code's allocate_slots → None →
                // prepend+break path re-queued it every tick, re-hashing the
                // whole prompt forever at 100% CPU until the watchdog killed the
                // engine. Match vLLM: finish such a request as FinishedIgnored
                // (FinishReason::Length) and surface it via finished_req_ids,
                // rather than re-queuing. Only applies to fresh requests
                // (nothing computed yet) — never to requests that COULD fit and
                // are merely waiting on transient capacity (those keep the
                // normal preempt/retry path below).
                let num_total_blocks = self.kv_cache.num_total_blocks();
                let block_size = self.kv_cache.block_size().max(1);
                let required_blocks = num_new_tokens_raw.div_ceil(block_size);
                if num_computed_tokens == 0 && required_blocks > num_total_blocks {
                    warn!(
                        request_id = %request_id,
                        required_blocks,
                        num_total_blocks,
                        "request needs more KV blocks than total capacity; \
                         finishing as ignored (unschedulable)"
                    );
                    request.status = RequestStatus::FinishedIgnored;
                    if let Some(r) = self.requests.get_mut(&request_id) {
                        r.status = RequestStatus::FinishedIgnored;
                    }
                    self.finished_req_ids.insert(request_id.clone());
                    // Do NOT re-queue; move on to the next waiting request.
                    continue;
                }

                let mut num_new_tokens = num_new_tokens_raw;

                // Apply long-prefill threshold.
                if self.long_prefill_token_threshold > 0
                    && num_new_tokens > self.long_prefill_token_threshold
                {
                    num_new_tokens = self.long_prefill_token_threshold;
                }

                // If chunked prefill is disabled, skip if the request
                // doesn't fit in the remaining budget.
                if !self.enable_chunked_prefill && num_new_tokens > token_budget {
                    // Put the request back.
                    self.waiting.prepend_request(request);
                    break;
                }

                num_new_tokens = num_new_tokens.min(token_budget);
                if num_new_tokens == 0 {
                    // Put the request back.
                    self.waiting.prepend_request(request);
                    break;
                }

                // Allocate slots for the effective lookahead. For new
                // requests that haven't been computed yet, we use 0
                // lookahead tokens (matches Python: only running requests
                // get lookahead).
                let effective_lookahead = if request.num_computed_tokens == 0 {
                    0
                } else {
                    self.num_lookahead_tokens
                };

                // Set num_computed_tokens for proper block allocation.
                let orig_computed = request.num_computed_tokens;
                request.num_computed_tokens = num_computed_tokens;

                let new_blocks = self.kv_cache.allocate_slots(
                    &request,
                    num_new_tokens,
                    effective_lookahead,
                    &cached_blocks,
                );

                match new_blocks {
                    Some(blocks) => {
                        // Spans Phase 2: record which logical blocks are reused
                        // cache hits (matched != usize::MAX) so the worker can
                        // skip rewriting their KV. Only for span requests; the
                        // worker filters to the compute range (below-prefix
                        // hits aren't rewritten anyway).
                        if request.block_annotations.is_some() {
                            let hits = &cached_blocks[0];
                            let reused: Vec<usize> =
                                (0..hits.len()).filter(|&i| hits[i] != usize::MAX).collect();
                            if !reused.is_empty() {
                                req_to_reused_blocks.insert(request_id.clone(), reused);
                            }
                        }
                        // Mutate request to running state.
                        let was_waiting = request.status == RequestStatus::Waiting;
                        let was_preempted = request.status == RequestStatus::Preempted;

                        request.status = RequestStatus::Running;
                        request.num_computed_tokens = num_computed_tokens;
                        if request.num_cached_tokens < 0 {
                            request.num_cached_tokens = num_computed_tokens as i32;
                        }

                        // New requests need their full data copied once into
                        // NewRequestData (sent to the worker, per lifetime).
                        if was_waiting {
                            scheduled_new_reqs.push(request.clone());
                        } else if !was_preempted {
                            warn!("Unexpected request status for {}", request.request_id);
                        }

                        // Move to running list (no extra clone).
                        self.running_push(request);

                        // Zero-copy: resumed requests are recorded by their
                        // index in `self.running` (just pushed above) and read
                        // by reference in make_cached_request_data.
                        if was_preempted {
                            scheduled_resumed_reqs.push(self.running.len() - 1);
                        }

                        req_to_new_blocks.insert(request_id.clone(), blocks);
                        num_scheduled_tokens.insert(request_id.clone(), num_new_tokens);
                        token_budget -= num_new_tokens;

                        // Update the requests map.
                        if let Some(r) = self.requests.get_mut(&request_id) {
                            r.status = RequestStatus::Running;
                            r.num_computed_tokens = num_computed_tokens;
                            if r.num_cached_tokens < 0 {
                                r.num_cached_tokens = num_computed_tokens as i32;
                            }
                        }
                    }
                    None => {
                        // Cannot allocate -- put request back and stop.
                        request.num_computed_tokens = orig_computed;
                        self.waiting.prepend_request(request);
                        break;
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // Phase 3: Build SchedulerOutput
        // ---------------------------------------------------------------
        let total_num_scheduled_tokens: usize = num_scheduled_tokens.values().sum();

        // Build NewRequestData for newly scheduled requests.
        // Use into_iter() to move fields out instead of cloning.
        let new_reqs_data: Vec<NewRequestData> = scheduled_new_reqs
            .into_iter()
            .map(|req| {
                let blocks = req_to_new_blocks
                    .get(&req.request_id)
                    .cloned()
                    .unwrap_or_else(|| vec![Vec::new()]);
                let reused = req_to_reused_blocks.get(&req.request_id).cloned();
                let mut nrd = NewRequestData::new(
                    req.request_id,
                    Some(req.prompt_token_ids),
                    blocks,
                    req.num_computed_tokens,
                    Some(req.sampling_params),
                    req.mm_data,
                    req.block_annotations,
                );
                nrd.reused_block_idxs = reused;
                nrd
            })
            .collect();

        // Build CachedRequestData.
        let cached_reqs_data = self.make_cached_request_data(
            &scheduled_running_reqs,
            &scheduled_resumed_reqs,
            &num_scheduled_tokens,
            &req_to_new_blocks,
        );

        let preempted_req_ids: HashSet<String> =
            preempted_reqs.into_iter().map(|r| r.request_id).collect();

        let output = SchedulerOutput {
            scheduled_new_reqs: new_reqs_data,
            scheduled_cached_reqs: cached_reqs_data,
            num_scheduled_tokens,
            total_num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            scheduled_encoder_inputs: HashMap::new(),
            num_common_prefix_blocks: Vec::new(),
            finished_req_ids: std::mem::take(&mut self.finished_req_ids),
            free_encoder_mm_hashes: Vec::new(),
            preempted_req_ids: if preempted_req_ids.is_empty() {
                None
            } else {
                Some(preempted_req_ids)
            },
        };

        // Post-schedule updates.
        self.update_after_schedule(&output);

        output
    }

    fn add_request(&mut self, request: Request) {
        self.requests
            .insert(request.request_id.clone(), request.clone());
        self.waiting.add_request(request);
    }

    fn finish_requests(
        &mut self,
        request_ids: &[&str],
        finished_status: RequestStatus,
    ) -> Vec<(String, u32)> {
        let mut result = Vec::new();
        for &req_id in request_ids {
            if let Some(pair) = self.finish_single_request(req_id, finished_status) {
                result.push(pair);
            }
        }
        result
    }

    fn get_num_unfinished_requests(&self) -> usize {
        self.running.len() + self.waiting.len()
    }

    fn get_unfinished_request_ids(&self) -> Vec<String> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .map(|r| r.request_id.clone())
            .collect()
    }

    fn has_finished_requests(&self) -> bool {
        !self.finished_req_ids.is_empty()
    }

    fn pause_state(&self) -> PauseState {
        self.pause_state
    }

    fn set_pause_state(&mut self, state: PauseState) {
        self.pause_state = state;
    }

    fn reset_prefix_cache(&mut self) -> bool {
        if !self.running.is_empty() {
            return false;
        }
        self.kv_cache.reset_prefix_cache()
    }

    fn get_request_counts(&self) -> (usize, usize) {
        (self.running.len(), self.waiting.len())
    }

    fn kv_cache_usage(&self) -> f64 {
        self.kv_cache.usage()
    }

    fn num_total_blocks(&self) -> usize {
        self.kv_cache.num_total_blocks()
    }

    fn num_used_blocks(&self) -> usize {
        self.kv_cache.num_total_blocks() - self.kv_cache.num_free_blocks()
    }

    fn shutdown(&mut self) {
        // Free all running requests via drain to avoid intermediate Vec<String>.
        self.running_req_idx.clear();
        for req in self.running.drain(..) {
            self.kv_cache.free(&req.request_id);
        }

        // Drain waiting queue.
        self.waiting.drain_all();

        self.requests.clear();
        self.finished_req_ids.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scratchy_core_common::SamplingParams;
    use scratchy_core_config::SchedulerConfig;

    // Helper: create a default scheduler config for testing.
    fn test_scheduler_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            ..Default::default()
        }
    }

    // Helper: create a test request.
    fn make_request(id: &str, num_prompt_tokens: usize) -> Request {
        let prompt: Vec<u32> = (0..num_prompt_tokens as u32).collect();
        Request::new(
            id.into(),
            prompt,
            SamplingParams {
                max_tokens: Some(100),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        )
    }

    fn make_priority_request(
        id: &str,
        num_prompt_tokens: usize,
        priority: i32,
        arrival: f64,
    ) -> Request {
        let prompt: Vec<u32> = (0..num_prompt_tokens as u32).collect();
        Request::new(
            id.into(),
            prompt,
            SamplingParams {
                max_tokens: Some(100),
                ..Default::default()
            },
            arrival,
            0,
            priority,
            None,
        )
    }

    // ----- Basic scheduling tests -----

    #[test]
    fn test_empty_schedule() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        let output = sched.schedule();
        assert_eq!(output.total_num_scheduled_tokens, 0);
        assert!(output.scheduled_new_reqs.is_empty());
        assert_eq!(output.scheduled_cached_reqs.num_reqs(), 0);
    }

    #[test]
    fn test_add_and_schedule_single_request() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        let req = make_request("r1", 10);
        sched.add_request(req);

        assert_eq!(sched.get_num_unfinished_requests(), 1);
        assert!(sched.has_unfinished_requests());
        assert_eq!(sched.get_request_counts(), (0, 1));

        let output = sched.schedule();
        assert_eq!(output.scheduled_new_reqs.len(), 1);
        assert_eq!(output.scheduled_new_reqs[0].req_id, "r1");
        assert_eq!(*output.num_scheduled_tokens.get("r1").unwrap(), 10);
        assert_eq!(output.total_num_scheduled_tokens, 10);
        assert_eq!(sched.get_request_counts(), (1, 0));
    }

    #[test]
    fn test_schedule_multiple_requests() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.add_request(make_request("r2", 20));
        sched.add_request(make_request("r3", 30));

        let output = sched.schedule();
        assert_eq!(output.scheduled_new_reqs.len(), 3);
        assert_eq!(output.total_num_scheduled_tokens, 60);
    }

    #[test]
    fn test_max_num_seqs_limit() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 1024,
            max_num_seqs: 2,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.add_request(make_request("r2", 10));
        sched.add_request(make_request("r3", 10));

        let output = sched.schedule();
        // Only 2 requests should be scheduled due to max_num_seqs=2.
        assert_eq!(output.scheduled_new_reqs.len(), 2);
        assert_eq!(sched.get_request_counts(), (2, 1));
    }

    #[test]
    fn test_token_budget_limit() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 25,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.add_request(make_request("r2", 10));
        sched.add_request(make_request("r3", 10));

        let output = sched.schedule();
        // Total budget is 25. r1 (10) + r2 (10) = 20. r3 needs 10 but only
        // 5 remain. With chunked prefill, r3 gets 5.
        assert_eq!(output.total_num_scheduled_tokens, 25);
        assert_eq!(output.scheduled_new_reqs.len(), 3);
    }

    #[test]
    fn test_chunked_prefill_disabled() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 25,
            max_num_seqs: 10,
            enable_chunked_prefill: false,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.add_request(make_request("r2", 10));
        sched.add_request(make_request("r3", 10));

        let output = sched.schedule();
        // With chunked prefill disabled, r3 (10 tokens) doesn't fit in
        // the remaining budget (5), so only r1 and r2 are scheduled.
        assert_eq!(output.total_num_scheduled_tokens, 20);
        assert_eq!(output.scheduled_new_reqs.len(), 2);
    }

    // ----- Unschedulable (never-fits) guard -----

    #[test]
    fn test_unschedulable_request_finishes_not_requeued() {
        // Defense-in-depth for the "scr launch claude" meltdown: a brand-new
        // WAITING request whose prompt needs more KV blocks than exist in the
        // entire pool can never be placed. The scheduler must finish it (so the
        // engine reports it and the request leaves the queue) instead of
        // re-queuing and busy-spinning forever.
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 8192,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            ..Default::default()
        };
        // 4 blocks of size 16 = 64-token total KV capacity.
        let mut sched = Scheduler::with_simple_blocks(&cfg, 100_000, 4, 16);

        // 200-token prompt needs ceil(200/16) = 13 blocks > 4 total. Never fits.
        sched.add_request(make_request("toolong", 200));
        assert_eq!(sched.get_num_unfinished_requests(), 1);

        let output = sched.schedule();

        // It is finished (reported as ignored), NOT scheduled, NOT re-queued.
        assert!(
            output.finished_req_ids.contains("toolong"),
            "unschedulable request must be surfaced in finished_req_ids"
        );
        assert_eq!(output.scheduled_new_reqs.len(), 0);
        assert_eq!(output.total_num_scheduled_tokens, 0);
        assert_eq!(
            sched.get_num_unfinished_requests(),
            0,
            "request must leave the queue, not busy-spin"
        );
        // Status reflects vLLM's FinishedIgnored (Length finish reason).
        let req = sched.get_request("toolong").expect("request still tracked");
        assert_eq!(req.status, RequestStatus::FinishedIgnored);

        // A second schedule does not re-emit / re-queue it.
        let output2 = sched.schedule();
        assert!(!output2.finished_req_ids.contains("toolong"));
        assert_eq!(sched.get_num_unfinished_requests(), 0);
    }

    #[test]
    fn test_schedulable_request_not_falsely_ignored() {
        // A request that fits in total capacity must take the normal path and
        // NOT be falsely finished as ignored.
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 8192,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            ..Default::default()
        };
        // 8 blocks of size 16 = 128-token capacity.
        let mut sched = Scheduler::with_simple_blocks(&cfg, 100_000, 8, 16);

        // 100-token prompt needs ceil(100/16) = 7 blocks <= 8 total. Fits.
        sched.add_request(make_request("ok", 100));
        let output = sched.schedule();

        assert!(!output.finished_req_ids.contains("ok"));
        assert_eq!(output.scheduled_new_reqs.len(), 1);
        let req = sched.get_request("ok").expect("request tracked");
        assert_eq!(req.status, RequestStatus::Running);
    }

    // ----- Request lifecycle tests -----

    #[test]
    fn test_request_lifecycle() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Add request.
        sched.add_request(make_request("r1", 10));
        assert_eq!(sched.get_num_unfinished_requests(), 1);

        // Schedule it.
        let _output = sched.schedule();
        assert_eq!(sched.get_request_counts(), (1, 0));

        // Finish it.
        let finished = sched.finish_requests(&["r1"], RequestStatus::FinishedStopped);
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].0, "r1");
        assert_eq!(sched.get_num_unfinished_requests(), 0);
        assert!(sched.has_finished_requests());

        // The next schedule call should report the finished request.
        let output = sched.schedule();
        assert!(output.finished_req_ids.contains("r1"));
    }

    #[test]
    fn test_finish_nonexistent_request() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        let finished = sched.finish_requests(&["nonexistent"], RequestStatus::FinishedAborted);
        assert!(finished.is_empty());
    }

    #[test]
    fn test_finish_waiting_request() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));

        // Finish before scheduling.
        let finished = sched.finish_requests(&["r1"], RequestStatus::FinishedAborted);
        assert_eq!(finished.len(), 1);
        assert_eq!(sched.get_num_unfinished_requests(), 0);
    }

    // ----- Preemption tests -----

    #[test]
    fn test_preemption_when_blocks_exhausted() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        // Only 4 blocks of size 16 = 64 tokens total.
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 4, 16);

        // Add requests that together need more than 4 blocks.
        sched.add_request(make_request("r1", 16)); // 1 block
        sched.add_request(make_request("r2", 16)); // 1 block
        sched.add_request(make_request("r3", 16)); // 1 block
        sched.add_request(make_request("r4", 16)); // 1 block
        sched.add_request(make_request("r5", 16)); // would need 5th block

        let output = sched.schedule();

        // Not all 5 can be scheduled. At most 4 blocks available.
        assert!(output.total_num_scheduled_tokens <= 64);
        let total_scheduled =
            output.scheduled_new_reqs.len() + output.scheduled_cached_reqs.num_reqs();
        // At most 4 requests can be scheduled.
        assert!(total_scheduled <= 4);
    }

    #[test]
    fn test_running_request_preemption() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        // Very limited blocks: only 2 blocks of size 16 = 32 tokens.
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 2, 16);

        // First step: schedule r1 (16 tokens, 1 block).
        sched.add_request(make_request("r1", 16));
        let output1 = sched.schedule();
        assert_eq!(output1.scheduled_new_reqs.len(), 1);
        assert_eq!(output1.scheduled_new_reqs[0].req_id, "r1");

        // Simulate r1 generating an output token.
        sched.append_output_tokens("r1", &[99]);

        // Second step: r1 is running (needs 1 token), schedule it.
        let output2 = sched.schedule();
        // r1 should be in cached_reqs (already scheduled before).
        assert!(output2.num_scheduled_tokens.contains_key("r1"));
    }

    // ----- Pause state tests -----

    #[test]
    fn test_pause_all() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.set_pause_state(PauseState::PausedAll);

        let output = sched.schedule();
        assert_eq!(output.total_num_scheduled_tokens, 0);
        assert!(output.scheduled_new_reqs.is_empty());
    }

    #[test]
    fn test_pause_new() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Schedule r1 normally first.
        sched.add_request(make_request("r1", 10));
        let _output1 = sched.schedule();
        assert_eq!(sched.get_request_counts(), (1, 0));

        // Simulate r1 producing a token.
        sched.append_output_tokens("r1", &[99]);

        // Now pause new requests and add r2.
        sched.add_request(make_request("r2", 10));
        sched.set_pause_state(PauseState::PausedNew);

        let output2 = sched.schedule();
        // r1 (running) should still be scheduled, but r2 (waiting) should not.
        assert!(output2.num_scheduled_tokens.contains_key("r1"));
        assert!(!output2.num_scheduled_tokens.contains_key("r2"));
    }

    // ----- Shutdown test -----

    #[test]
    fn test_shutdown() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.add_request(make_request("r2", 20));
        let _output = sched.schedule();

        sched.shutdown();
        assert_eq!(sched.get_num_unfinished_requests(), 0);
        assert!(!sched.has_unfinished_requests());
    }

    // ----- Reset prefix cache test -----

    #[test]
    fn test_reset_prefix_cache_with_running_requests() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        let _output = sched.schedule();

        // Should fail because there are running requests.
        assert!(!sched.reset_prefix_cache());
    }

    #[test]
    fn test_reset_prefix_cache_without_running_requests() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // No running requests -- should succeed.
        assert!(sched.reset_prefix_cache());
    }

    // ----- Long prefill threshold test -----

    #[test]
    fn test_long_prefill_threshold() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 1024,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 50,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Request with 200 prompt tokens.
        sched.add_request(make_request("r1", 200));

        let output = sched.schedule();
        // Should schedule at most 50 tokens due to the threshold.
        assert_eq!(*output.num_scheduled_tokens.get("r1").unwrap(), 50);
    }

    // ----- Multiple scheduling steps test -----

    #[test]
    fn test_multi_step_scheduling() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 20,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Request with 50 prompt tokens -- needs multiple steps.
        sched.add_request(make_request("r1", 50));

        // Step 1: schedule up to 20 tokens.
        let output1 = sched.schedule();
        assert_eq!(*output1.num_scheduled_tokens.get("r1").unwrap(), 20);

        // Step 2: schedule next chunk. The request is now running.
        let output2 = sched.schedule();
        assert_eq!(*output2.num_scheduled_tokens.get("r1").unwrap(), 20);

        // Step 3: schedule remaining 10 tokens.
        let output3 = sched.schedule();
        assert_eq!(*output3.num_scheduled_tokens.get("r1").unwrap(), 10);
    }

    // ----- has_requests / has_finished tests -----

    #[test]
    fn test_has_requests_with_finished() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        let _output = sched.schedule();

        // Finish r1.
        sched.finish_requests(&["r1"], RequestStatus::FinishedStopped);

        // No unfinished, but has finished.
        assert!(!sched.has_unfinished_requests());
        assert!(sched.has_finished_requests());
        assert!(sched.has_requests());

        // After a schedule step, finished IDs are flushed.
        let _output = sched.schedule();
        assert!(!sched.has_requests());
    }

    // ----- Priority scheduling tests -----

    #[test]
    fn test_priority_scheduling_order() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 2,
            enable_chunked_prefill: true,
            policy: SchedulerPolicy::Priority,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Add a low-priority request first, then a high-priority one.
        sched.add_request(make_priority_request("r_low", 10, 10, 1.0));
        sched.add_request(make_priority_request("r_high", 10, -1, 2.0));

        let output = sched.schedule();
        assert_eq!(output.scheduled_new_reqs.len(), 2);

        // Both should be scheduled since max_num_seqs=2,
        // but r_high should come first in the new_reqs list because the
        // priority queue pops it first.
        let scheduled_ids: Vec<&str> = output
            .scheduled_new_reqs
            .iter()
            .map(|r| r.req_id.as_str())
            .collect();
        assert_eq!(scheduled_ids[0], "r_high");
        assert_eq!(scheduled_ids[1], "r_low");
    }

    #[test]
    fn test_priority_scheduling_with_limited_seqs() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 1, // Only 1 seq at a time.
            enable_chunked_prefill: true,
            policy: SchedulerPolicy::Priority,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // The high-priority request should be scheduled first.
        sched.add_request(make_priority_request("r_low", 10, 10, 1.0));
        sched.add_request(make_priority_request("r_high", 10, -1, 2.0));

        let output = sched.schedule();
        assert_eq!(output.scheduled_new_reqs.len(), 1);
        assert_eq!(output.scheduled_new_reqs[0].req_id, "r_high");
    }

    // -- Request accessor tests --

    #[test]
    fn test_get_request() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 5));
        assert!(sched.get_request("r1").is_some());
        assert_eq!(sched.get_request("r1").unwrap().request_id, "r1");
        assert!(sched.get_request("nonexistent").is_none());
    }

    #[test]
    fn test_get_request_mut() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 5));
        if let Some(req) = sched.get_request_mut("r1") {
            req.max_tokens = 42;
        }
        assert_eq!(sched.get_request("r1").unwrap().max_tokens, 42);
    }

    #[test]
    fn test_append_output_tokens() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 5));

        // Schedule to move from waiting to running.
        let _ = sched.schedule();

        // Append output tokens.
        sched.append_output_tokens("r1", &[100, 101]);

        let req = sched.get_request("r1").unwrap();
        assert_eq!(req.output_token_ids, vec![100, 101]);
        assert_eq!(req.num_output_tokens(), 2);
        // all_token_ids should be prompt + output.
        assert_eq!(req.all_token_ids.len(), 7); // 5 prompt + 2 output
    }

    #[test]
    fn test_append_output_tokens_nonexistent() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Appending to a nonexistent request should not panic.
        sched.append_output_tokens("nonexistent", &[1, 2, 3]);
    }

    // ----- KV cache usage tests -----

    #[test]
    fn test_simple_block_tracker_usage_empty() {
        let tracker = SimpleBlockTracker::new(100, 16);
        assert!((tracker.usage() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_simple_block_tracker_usage_zero_blocks() {
        let tracker = SimpleBlockTracker::new(0, 16);
        assert!((tracker.usage() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_simple_block_tracker_usage_after_alloc() {
        let mut tracker = SimpleBlockTracker::new(100, 16);
        // Allocate a request needing 2 blocks (32 tokens / 16 block_size).
        let req = make_request("r1", 32);
        tracker.allocate_slots(&req, 32, 0, &[]);
        // 2 out of 100 blocks used = 0.02
        assert!((tracker.usage() - 0.02).abs() < 1e-9);
    }

    #[test]
    fn test_simple_block_tracker_usage_after_free() {
        let mut tracker = SimpleBlockTracker::new(100, 16);
        let req = make_request("r1", 32);
        tracker.allocate_slots(&req, 32, 0, &[]);
        tracker.free("r1");
        assert!((tracker.usage() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scheduler_kv_cache_usage() {
        let cfg = test_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Initially empty.
        assert!((sched.kv_cache_usage() - 0.0).abs() < f64::EPSILON);

        // Add and schedule a request to trigger block allocation.
        let req = make_request("r1", 32);
        sched.add_request(req);
        sched.schedule();

        // After scheduling, some blocks should be allocated.
        assert!(sched.kv_cache_usage() > 0.0);
    }

    // ----- Async scheduling tests -----

    fn async_scheduler_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            async_scheduling: Some(true),
            ..Default::default()
        }
    }

    /// Helper: create a request with a specific max_tokens.
    fn make_request_with_max(id: &str, num_prompt_tokens: usize, max_tokens: u32) -> Request {
        let prompt: Vec<u32> = (0..num_prompt_tokens as u32).collect();
        Request::new(
            id.into(),
            prompt,
            SamplingParams {
                max_tokens: Some(max_tokens),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        )
    }

    #[test]
    fn test_async_placeholder_increment_on_decode() {
        let cfg = async_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Add request, schedule prefill (non-chunked).
        sched.add_request(make_request("r1", 10));
        let output1 = sched.schedule();

        // After a completed (non-chunked) prefill, is_prefill_chunk=false
        // so placeholders are incremented to 1 (the prefill step will
        // produce 1 output token whose value is unknown).
        assert_eq!(output1.scheduled_new_reqs.len(), 1);
        let req = sched.get_request("r1").unwrap();
        assert_eq!(req.num_output_placeholders, 1);

        // Simulate prefill producing a first output token.
        sched.append_output_tokens("r1", &[99]);
        assert_eq!(sched.get_request("r1").unwrap().num_output_placeholders, 0);

        // Now schedule decode step 2.
        let output2 = sched.schedule();
        assert!(output2.num_scheduled_tokens.contains_key("r1"));

        // After schedule, placeholder incremented again for the decode.
        let req = sched.get_request("r1").unwrap();
        assert_eq!(
            req.num_output_placeholders, 1,
            "Decode request should have 1 placeholder after async schedule"
        );
    }

    #[test]
    fn test_async_placeholder_decrement_on_output() {
        let cfg = async_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule(); // prefill

        // First decode token from prefill.
        sched.append_output_tokens("r1", &[99]);

        // Schedule decode (sets placeholder = 1).
        sched.schedule();
        assert_eq!(sched.get_request("r1").unwrap().num_output_placeholders, 1);

        // Simulate GPU returning 1 token → decrement placeholder.
        sched.append_output_tokens("r1", &[100]);
        assert_eq!(sched.get_request("r1").unwrap().num_output_placeholders, 0);
    }

    #[test]
    fn test_async_back_to_back_schedule_no_double_schedule() {
        // Core test: two consecutive schedule() calls without
        // update_from_output in between must not double-schedule the
        // same position.
        let cfg = async_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));

        // Prefill.
        let output1 = sched.schedule();
        assert_eq!(*output1.num_scheduled_tokens.get("r1").unwrap(), 10);

        // Simulate prefill producing first token.
        sched.append_output_tokens("r1", &[99]);

        // First decode schedule → r1 needs 1 token.
        let output2 = sched.schedule();
        assert_eq!(
            *output2.num_scheduled_tokens.get("r1").unwrap(),
            1,
            "First decode step should schedule 1 token"
        );
        // After schedule, placeholder = 1.

        // Second schedule() WITHOUT update_from_output — simulates
        // pre-scheduling while GPU is still running.
        let output3 = sched.schedule();
        assert_eq!(
            *output3.num_scheduled_tokens.get("r1").unwrap(),
            1,
            "Second decode should schedule 1 token (placeholder prevents overlap)"
        );

        // Placeholder should now be 2 (one for each in-flight step).
        let req = sched.get_request("r1").unwrap();
        assert_eq!(req.num_output_placeholders, 2);
    }

    #[test]
    fn test_async_max_tokens_guard_prevents_over_scheduling() {
        let cfg = async_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Request with only 2 max_tokens.
        sched.add_request(make_request_with_max("r1", 10, 2));
        sched.schedule(); // prefill (10 tokens)

        // Simulate generating 1st output token.
        sched.append_output_tokens("r1", &[99]);

        // Decode step 1 → schedules 1 token, placeholder = 1.
        let output2 = sched.schedule();
        assert_eq!(*output2.num_scheduled_tokens.get("r1").unwrap(), 1);
        assert_eq!(sched.get_request("r1").unwrap().num_output_placeholders, 1);

        // Now pre-schedule step 2 (without finalization).
        // num_computed = 12 (10 prompt + 1 output + 1 placeholder),
        // max_total = 10 + 2 = 12.
        // Guard: 12 + 2 - 1 = 13 >= 12 → skip!
        let output3 = sched.schedule();
        assert!(
            !output3.num_scheduled_tokens.contains_key("r1"),
            "Max-tokens guard should prevent scheduling request about to finish"
        );
    }

    #[test]
    fn test_async_preemption_resets_placeholders() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            async_scheduling: Some(true),
            ..Default::default()
        };
        // 3 blocks of size 8 = 24 tokens of KV capacity.
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 3, 8);

        // Schedule r1 (8 tokens = 1 block) and r2 (8 tokens = 1 block).
        sched.add_request(make_request("r1", 8));
        sched.add_request(make_request("r2", 8));
        sched.schedule(); // prefill both

        // Both should have placeholder = 1 after completed prefill.
        assert_eq!(sched.get_request("r1").unwrap().num_output_placeholders, 1);
        assert_eq!(sched.get_request("r2").unwrap().num_output_placeholders, 1);

        // Simulate output tokens for both.
        sched.append_output_tokens("r1", &[99]);
        sched.append_output_tokens("r2", &[99]);
        assert_eq!(sched.get_request("r1").unwrap().num_output_placeholders, 0);
        assert_eq!(sched.get_request("r2").unwrap().num_output_placeholders, 0);

        // Generate several more tokens for r1 to consume more blocks.
        // r1 now has 8 prompt + 1 output = 9 tokens → 2 blocks.
        // r2 has 8 prompt + 1 output = 9 tokens → 2 blocks.
        // Total: 4 blocks needed but only 3 available.

        // Schedule decode step: this should trigger preemption because
        // r1 (2 blocks) + r2 (2 blocks) > 3 blocks.
        let output = sched.schedule();

        // One of the two should be preempted. Check which.
        // (FCFS preempts from the back, so r2 should be preempted.)
        if sched.get_request("r2").unwrap().status == RequestStatus::Preempted {
            assert_eq!(
                sched.get_request("r2").unwrap().num_output_placeholders,
                0,
                "Preemption must reset placeholders"
            );
        } else if sched.get_request("r1").unwrap().status == RequestStatus::Preempted {
            assert_eq!(
                sched.get_request("r1").unwrap().num_output_placeholders,
                0,
                "Preemption must reset placeholders"
            );
        }

        // Only one should be scheduled.
        assert!(
            output.num_scheduled_tokens.len() <= 2,
            "At most 2 requests should be scheduled with limited blocks"
        );
    }

    #[test]
    fn test_async_placeholder_saturating_decrement() {
        // Ensure decrement never underflows.
        let cfg = async_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule(); // prefill

        // Placeholders = 0. Appending tokens should not underflow.
        sched.append_output_tokens("r1", &[99, 100, 101]);
        assert_eq!(sched.get_request("r1").unwrap().num_output_placeholders, 0);
    }

    #[test]
    fn test_sync_mode_no_placeholders() {
        // Verify that without async_scheduling, placeholders are never set.
        let cfg = test_scheduler_config(); // async_scheduling = None (false)
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule(); // prefill

        sched.append_output_tokens("r1", &[99]);

        sched.schedule(); // decode
        assert_eq!(
            sched.get_request("r1").unwrap().num_output_placeholders,
            0,
            "Sync mode should never set placeholders"
        );
    }

    // -----------------------------------------------------------------------
    // Prefix caching tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefix_cache_basic() {
        // Two requests with the same 32-token prompt should share cached blocks.
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(100, block_size);

        // Make a 32-token prompt (2 full blocks).
        let prompt: Vec<u32> = (0..32).collect();
        let mut r1 = Request::new(
            "r1".into(),
            prompt.clone(),
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );

        // First request: no cache hits.
        let (cached, _) = tracker.get_computed_blocks(&r1);
        assert_eq!(cached, 0);

        // Allocate for r1.
        let blocks = tracker.allocate_slots(&r1, 32, 0, &[]).unwrap();
        assert_eq!(blocks[0].len(), 2);
        let block0 = blocks[0][0];
        let block1 = blocks[0][1];

        // Simulate completion: r1 fully computed and freed.
        r1.num_computed_tokens = 32;
        tracker.free("r1");

        // Now both blocks should be in the cache.
        assert_eq!(tracker.num_cached_blocks(), 2);

        // Second request with the same prompt.
        let r2 = Request::new(
            "r2".into(),
            prompt.clone(),
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            2.0,
            0,
            0,
            None,
        );

        // Should get 32 cached tokens.
        let (cached2, cached_blocks2) = tracker.get_computed_blocks(&r2);
        assert_eq!(cached2, 32);
        assert_eq!(cached_blocks2[0].len(), 2);
        assert_eq!(cached_blocks2[0][0], block0);
        assert_eq!(cached_blocks2[0][1], block1);
    }

    #[test]
    fn test_prefix_cache_partial_match() {
        // Request with longer prompt should match the shared prefix.
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(100, block_size);

        // First request: 32 tokens.
        let prompt32: Vec<u32> = (0..32).collect();
        let mut r1 = Request::new(
            "r1".into(),
            prompt32.clone(),
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        tracker.allocate_slots(&r1, 32, 0, &[]).unwrap();
        r1.num_computed_tokens = 32;
        tracker.free("r1");

        // Second request: 48 tokens, first 32 overlap.
        let prompt48: Vec<u32> = (0..48).collect();
        let r2 = Request::new(
            "r2".into(),
            prompt48,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            2.0,
            0,
            0,
            None,
        );
        let (cached, _) = tracker.get_computed_blocks(&r2);
        assert_eq!(cached, 32, "first 2 blocks should match");
    }

    #[test]
    fn test_prefix_cache_no_match_different_prompt() {
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(100, block_size);

        let prompt_a: Vec<u32> = (0..32).collect();
        let mut r1 = Request::new(
            "r1".into(),
            prompt_a,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        tracker.allocate_slots(&r1, 32, 0, &[]).unwrap();
        r1.num_computed_tokens = 32;
        tracker.free("r1");

        // Different prompt — no match.
        let prompt_b: Vec<u32> = (100..132).collect();
        let r2 = Request::new(
            "r2".into(),
            prompt_b,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            2.0,
            0,
            0,
            None,
        );
        let (cached, _) = tracker.get_computed_blocks(&r2);
        assert_eq!(cached, 0);
    }

    #[test]
    fn test_prefix_cache_eviction_under_pressure() {
        // With limited blocks, cached blocks get evicted FIFO.
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(4, block_size);

        // Fill all 4 blocks with r1 (64 tokens).
        let prompt64: Vec<u32> = (0..64).collect();
        let mut r1 = Request::new(
            "r1".into(),
            prompt64,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        tracker.allocate_slots(&r1, 64, 0, &[]).unwrap();
        r1.num_computed_tokens = 64;
        tracker.free("r1");
        assert_eq!(tracker.num_cached_blocks(), 4);

        // New request needs 2 blocks — must evict 2 cached blocks.
        let prompt_new: Vec<u32> = (200..232).collect();
        let r2 = Request::new(
            "r2".into(),
            prompt_new,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            2.0,
            0,
            0,
            None,
        );
        let blocks = tracker.allocate_slots(&r2, 32, 0, &[]);
        assert!(blocks.is_some(), "should evict cached blocks to make room");
        assert_eq!(
            tracker.num_cached_blocks(),
            2,
            "2 of 4 cached blocks should remain"
        );
    }

    #[test]
    fn test_prefix_cache_reset() {
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(100, block_size);

        let prompt: Vec<u32> = (0..32).collect();
        let mut r1 = Request::new(
            "r1".into(),
            prompt.clone(),
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        tracker.allocate_slots(&r1, 32, 0, &[]).unwrap();
        r1.num_computed_tokens = 32;
        tracker.free("r1");
        assert_eq!(tracker.num_cached_blocks(), 2);

        tracker.reset_prefix_cache();
        assert_eq!(tracker.num_cached_blocks(), 0);

        // After reset, no cache hits.
        let r2 = Request::new(
            "r2".into(),
            prompt,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            2.0,
            0,
            0,
            None,
        );
        let (cached, _) = tracker.get_computed_blocks(&r2);
        assert_eq!(cached, 0);
    }

    #[test]
    fn test_prefix_cache_allocate_reuses_cached_blocks() {
        // When allocating for a request with cached prefix, the cached block IDs
        // should appear in the allocation.
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(100, block_size);

        let prompt: Vec<u32> = (0..32).collect();
        let mut r1 = Request::new(
            "r1".into(),
            prompt.clone(),
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        let r1_blocks = tracker.allocate_slots(&r1, 32, 0, &[]).unwrap();
        let cached_block0 = r1_blocks[0][0];
        let cached_block1 = r1_blocks[0][1];
        r1.num_computed_tokens = 32;
        tracker.free("r1");

        // r2 has same prompt + 8 extra tokens = 40 tokens total.
        let prompt40: Vec<u32> = (0..40).collect();
        let mut r2 = Request::new(
            "r2".into(),
            prompt40,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            2.0,
            0,
            0,
            None,
        );
        // Simulate what the scheduler does: get_computed_blocks, then set
        // num_computed_tokens, then allocate.
        let (cached, cached_blocks) = tracker.get_computed_blocks(&r2);
        assert_eq!(cached, 32);
        r2.num_computed_tokens = cached;

        let r2_blocks = tracker.allocate_slots(&r2, 8, 0, &cached_blocks).unwrap();
        // Should reuse the 2 cached blocks + 1 new block.
        assert_eq!(r2_blocks[0].len(), 3);
        assert_eq!(r2_blocks[0][0], cached_block0);
        assert_eq!(r2_blocks[0][1], cached_block1);
    }

    #[test]
    fn test_block_id_recycling() {
        // Verify that block IDs are recycled and stay within bounds.
        // With 4 blocks, serve many sequential requests — IDs must stay in 0..4.
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::new(4, block_size);

        for round in 0..20 {
            let prompt: Vec<u32> = (0..16).collect();
            let id = format!("r{round}");
            let r = Request::new(
                id.clone(),
                prompt,
                SamplingParams {
                    max_tokens: Some(1),
                    ..Default::default()
                },
                round as f64,
                0,
                0,
                None,
            );
            let blocks = tracker
                .allocate_slots(&r, 16, 0, &[])
                .expect("should allocate");
            // Block IDs must be within pool bounds.
            for &bid in &blocks[0] {
                assert!(bid < 4, "block ID {bid} out of bounds (round {round})");
            }
            tracker.free(&id);
        }
    }

    // Build a minimal request for the SWA allocator tests.
    fn swa_test_request(id: &str, block_size: usize) -> Request {
        Request::new(
            id.into(),
            (0..block_size as u32).collect(),
            SamplingParams {
                max_tokens: Some(1),
                ..Default::default()
            },
            0.0,
            0,
            0,
            None,
        )
    }

    // Build a hybrid request with a fresh prompt of `n` tokens.
    fn hybrid_req(id: &str, n: usize) -> Request {
        Request::new(
            id.into(),
            (0..n as u32).collect(),
            SamplingParams {
                max_tokens: Some(4),
                ..Default::default()
            },
            0.0,
            0,
            0,
            None,
        )
    }

    // gemma4-shaped tracker WITH caching: 1 full (bs 32) + 1 sliding (W=64, bs 16).
    fn caching_hybrid(pool: usize, window: usize) -> SimpleBlockTracker {
        let mut t = SimpleBlockTracker::with_caching(pool, 16);
        t.enable_hybrid(pool, vec![(false, 0, 32), (true, window, 16)]);
        t
    }

    #[test]
    fn test_hybrid_caching_full_prefix_reuse() {
        // r1 prefills 128 tokens, seals, frees. r2 with the same 128-token
        // prompt + a bit more must reuse the full-group prefix AND the sliding
        // group's in-window suffix, reporting num_computed near the full prompt.
        let mut t = caching_hybrid(512, 64);
        let mut r1 = hybrid_req("r1", 128);
        r1.num_computed_tokens = 0;
        t.allocate_slots(&r1, 128, 0, &[]).unwrap();
        r1.num_computed_tokens = 128;
        t.seal(&r1);
        // keep r1 alive so its in-window sliding blocks aren't recycled.

        let r2 = hybrid_req("r2", 128);
        let (p, tables) = t.get_computed_blocks(&r2);
        // Full group: 128/32 = 4 blocks. Window 64 → P can be up to 128.
        assert!(p > 0, "expected a cache hit, got P=0");
        assert_eq!(p, 128, "full 128-token prefix reusable (W=64 covered)");
        // Full group table fully populated, sliding group null-pads [0,P-W).
        assert_eq!(tables[0].len(), 4, "full group 4 blocks");
        let null_head = tables[1].iter().filter(|&&b| b == 0).count();
        assert_eq!(
            null_head,
            (128 - 64) / 16,
            "sliding out-of-window head nulled"
        );
        assert!(tables[1].iter().any(|&b| b != 0), "in-window suffix cached");
    }

    #[test]
    fn test_hybrid_caching_suffix_not_cached_caps_p() {
        // If the sliding group's in-window suffix is NOT cached (e.g. those
        // blocks fell out of window at seal and were never registered), P must
        // be capped so we never claim computed tokens whose KV is gone.
        let mut t = caching_hybrid(512, 64);
        // Manually register ONLY the full group's first 2 blocks (64 tokens),
        // leaving the sliding group with no hits → joint P must be 0.
        let r1 = hybrid_req("r1", 128);
        let full_hashes = t.hash_all_blocks_bs(&r1.all_token_ids, None, None, 32);
        // Pretend two full blocks are cached at fake ids 5,6 but NO sliding.
        t.ref_cnt[5] = 0;
        t.ref_cnt[6] = 0;
        t.hybrid_hash_to_id.insert((0, full_hashes[0]), 5);
        t.hybrid_hash_to_id.insert((0, full_hashes[1]), 6);
        t.hybrid_id_to_key.insert(5, (0, full_hashes[0]));
        t.hybrid_id_to_key.insert(6, (0, full_hashes[1]));

        let r2 = hybrid_req("r2", 128);
        let (p, _tables) = t.get_computed_blocks(&r2);
        assert_eq!(
            p, 0,
            "sliding in-window suffix not cached ⇒ P capped to 0 (no wrong reuse)"
        );
    }

    #[test]
    fn test_hybrid_caching_refcount_on_shared_hits() {
        // A reused cached block must be ref-counted so freeing one request does
        // not free a block another live request shares.
        let mut t = caching_hybrid(512, 1024); // big window: all in-window
        let mut r1 = hybrid_req("r1", 64);
        r1.num_computed_tokens = 0;
        t.allocate_slots(&r1, 64, 0, &[]).unwrap();
        r1.num_computed_tokens = 64;
        t.seal(&r1);

        let r2 = hybrid_req("r2", 64);
        let (p, matched) = t.get_computed_blocks(&r2);
        assert!(p > 0, "r2 should hit r1's cache");
        let shared = matched[0][0];
        let rc_before = t.ref_cnt[shared];
        let mut r2m = hybrid_req("r2", 64);
        r2m.num_computed_tokens = p;
        t.allocate_slots(&r2m, 64 - p as usize, 0, &matched)
            .unwrap();
        assert!(
            t.ref_cnt[shared] > rc_before,
            "shared cached block ref-count must rise on reuse"
        );
        // Freeing r2 must NOT drop the shared block to 0 (r1 still holds it).
        t.free("r2");
        assert!(
            t.ref_cnt[shared] > 0,
            "shared block still held by r1 after r2 freed (no premature free)"
        );
        // No double-free panic on freeing r1 either.
        t.free("r1");
    }

    #[test]
    fn test_hybrid_sliding_block_id_stays_bounded() {
        // THE KEYSTONE invariant. A sliding-only sequence decoded far past the
        // window must keep the MAX allocated block ID near the window — because
        // scratchy's lockstep `grow_to_cover` residency is keyed on the max
        // block ID. vLLM's free-to-back-of-queue would let the max climb into
        // the pool and commit it resident (the prior OOM). Lowest-free-first
        // recycling of the sliding group's freed low IDs is what prevents that.
        let bs = 16usize;
        let window = 64usize; // 4 blocks
        let win_blocks = window.div_ceil(bs); // 4
        let pool = 4096usize; // a big pool: FIFO would climb toward 4096
        let mut tracker = SimpleBlockTracker::new(pool, bs);
        tracker.enable_hybrid(pool, vec![(true, window, bs)]); // one sliding group

        let mut r = swa_test_request("r1", bs);
        let mut max_seen = 0usize;
        for tok in 1..=4000usize {
            r.num_computed_tokens = (tok - 1) as u32; // computed BEFORE this token
            let groups = tracker.allocate_slots(&r, 1, 0, &[]).expect("hybrid alloc");
            assert_eq!(groups.len(), 1, "one block table per group");
            // Live (non-null) blocks stay ~window — out-of-window freed each step.
            let live = groups[0].iter().filter(|&&b| b != 0).count();
            assert!(
                live <= win_blocks + 2,
                "live sliding blocks {live} exceed window+slack at tok {tok}",
            );
            max_seen = max_seen.max(tracker.max_allocated_block_id());
        }
        assert!(
            max_seen <= win_blocks + 4,
            "max allocated block id {max_seen} climbed past window+slack — \
             lockstep residency would commit the pool (the OOM this fix prevents)"
        );

        tracker.free("r1");
        assert_eq!(
            tracker.num_free_blocks(),
            pool - 1,
            "all data blocks freed (null block 0 held out), no leak/double-free"
        );
    }

    #[test]
    fn test_hybrid_groups_null_pad_and_distinctness() {
        // gemma4-shaped: 1 full (bs 32) + 2 sliding (window 64, bs 16) over ONE
        // shared pool. Per-group tables, null-padding of skipped head slots,
        // distinct in-window physical blocks (no aliasing), full never frees.
        let pool = 256usize;
        let mut t = SimpleBlockTracker::new(pool, 16);
        t.enable_hybrid(pool, vec![(false, 0, 32), (true, 64, 16), (true, 64, 16)]);

        let mut r = swa_test_request("r1", 16);
        r.num_computed_tokens = 0;
        let g0 = t.allocate_slots(&r, 320, 0, &[]).expect("prefill");
        assert_eq!(g0.len(), 3, "one table per group");
        assert_eq!(g0[0].len(), 320usize.div_ceil(32), "full bs=32 → 10 blocks");
        assert_eq!(
            g0[1].len(),
            320usize.div_ceil(16),
            "sliding bs=16 → 20 blocks"
        );
        for tbl in &g0 {
            assert!(
                tbl.iter().all(|&b| b != 0),
                "no null during first alloc (computed=0)"
            );
        }

        // Decode: now computed=320 → sliding head falls out of the 64-tok window.
        r.num_computed_tokens = 320;
        let g1 = t.allocate_slots(&r, 1, 0, &[]).expect("decode");
        // Full group never frees → still all-distinct, no null.
        assert!(
            g1[0].iter().all(|&b| b != 0),
            "full group keeps every block"
        );
        let full_distinct: std::collections::HashSet<_> = g1[0].iter().copied().collect();
        assert_eq!(
            full_distinct.len(),
            g1[0].len(),
            "full group never reuses a block"
        );
        // Sliding groups: head null-padded, in-window blocks distinct.
        for sg in &g1[1..] {
            assert!(
                sg.contains(&0),
                "sliding head null-padded after the window slides"
            );
            let live: Vec<_> = sg.iter().copied().filter(|&b| b != 0).collect();
            let distinct: std::collections::HashSet<_> = live.iter().copied().collect();
            assert_eq!(
                distinct.len(),
                live.len(),
                "in-window blocks distinct (no aliasing)"
            );
        }

        t.free("r1");
        assert_eq!(
            t.num_free_blocks(),
            pool - 1,
            "all freed, null block 0 held out"
        );
    }

    #[test]
    fn test_hybrid_capacity_atomic_across_groups() {
        // The shared pool must cover the SUM of all groups' demand; sized one
        // block short, allocation fails (None) rather than over-committing.
        let full = 320usize.div_ceil(32); // 10
        let sld = 320usize.div_ceil(16); // 20
        let demand = full + 2 * sld; // 50
        let groups = vec![(false, 0usize, 32usize), (true, 64, 16), (true, 64, 16)];

        // Pool = demand + 1 (the held null block) fits exactly.
        let mut t = SimpleBlockTracker::new(demand + 1, 16);
        t.enable_hybrid(demand + 1, groups.clone());
        let mut r = swa_test_request("r1", 16);
        r.num_computed_tokens = 0;
        assert!(
            t.allocate_slots(&r, 320, 0, &[]).is_some(),
            "exact fit (incl null block)"
        );
        assert_eq!(t.num_free_blocks(), 0, "exact fit consumes the pool");

        // One block short → reject (atomic cross-group check, no partial alloc).
        let mut t2 = SimpleBlockTracker::new(demand, 16);
        t2.enable_hybrid(demand, groups);
        let mut r2 = swa_test_request("r2", 16);
        r2.num_computed_tokens = 0;
        assert!(
            t2.allocate_slots(&r2, 320, 0, &[]).is_none(),
            "under-sized pool rejects rather than over-committing"
        );
        assert_eq!(
            t2.num_free_blocks(),
            demand - 1,
            "rejected alloc pulled zero blocks"
        );
    }

    #[test]
    fn test_prefix_cache_full_prompt_not_stuck() {
        // When the prefix cache covers the entire prompt, the scheduler
        // must still schedule at least one block of tokens so the request
        // can start decoding. Regression test for the bench-latency hang.
        let block_size = 16;
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        let kv: Box<dyn KVCacheManagerOps> =
            Box::new(SimpleBlockTracker::with_caching(100, block_size));
        let mut sched = Scheduler::new(&cfg, 8192, kv);

        // 32-token prompt = 2 full blocks with block_size=16.
        let prompt: Vec<u32> = (0..32).collect();

        // --- Iteration 1: no cache, should schedule 32 tokens ---
        let r1 = Request::new(
            "r1".into(),
            prompt.clone(),
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        sched.add_request(r1);
        let output1 = sched.schedule();
        let sched_tokens_1 = *output1.num_scheduled_tokens.get("r1").unwrap();
        assert_eq!(
            sched_tokens_1, 32,
            "first request should schedule all 32 prompt tokens"
        );

        // Simulate completion: update computed tokens, finish, free.
        sched.finish_requests(&["r1"], RequestStatus::FinishedStopped);
        // Consume the finished ID from the scheduler.
        let _ = sched.schedule();

        // --- Iteration 2: same prompt, fully cached ---
        let r2 = Request::new(
            "r2".into(),
            prompt.clone(),
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            2.0,
            0,
            0,
            None,
        );
        sched.add_request(r2);
        let output2 = sched.schedule();

        // Must schedule >0 tokens (the last block gets reprocessed).
        let sched_tokens_2 = *output2
            .num_scheduled_tokens
            .get("r2")
            .expect("r2 should be scheduled");
        assert!(
            sched_tokens_2 > 0,
            "fully-cached request must still schedule tokens, got 0"
        );
        // Specifically: we back up by one block, so 16 tokens get reprocessed.
        assert_eq!(
            sched_tokens_2, 16,
            "should reprocess the last block ({block_size} tokens)"
        );
    }

    #[test]
    fn test_block_id_recycling_with_caching() {
        // Same as above but with caching — evicted cached blocks must also
        // produce IDs within bounds.
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(4, block_size);

        for round in 0..20 {
            // Each request uses a different prompt so no cache hits.
            let base = (round * 16) as u32 + 1000;
            let prompt: Vec<u32> = (base..base + 16).collect();
            let id = format!("r{round}");
            let mut r = Request::new(
                id.clone(),
                prompt,
                SamplingParams {
                    max_tokens: Some(1),
                    ..Default::default()
                },
                round as f64,
                0,
                0,
                None,
            );
            let blocks = tracker
                .allocate_slots(&r, 16, 0, &[])
                .expect("should allocate");
            for &bid in &blocks[0] {
                assert!(bid < 4, "block ID {bid} out of bounds (round {round})");
            }
            r.num_computed_tokens = 16;
            tracker.free(&id);
        }
    }

    // -----------------------------------------------------------------------
    // Speculative decoding tests
    // -----------------------------------------------------------------------

    fn spec_decode_scheduler_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            num_lookahead_tokens: 5,
            ..Default::default()
        }
    }

    #[test]
    fn test_spec_decode_tokens_scheduled() {
        // Verify that spec_token_ids are included in the scheduling output.
        let cfg = spec_decode_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule(); // prefill
        sched.append_output_tokens("r1", &[99]);

        // Set spec tokens on the request.
        sched.set_spec_token_ids("r1", vec![100, 101, 102, 103, 104]);

        let output = sched.schedule(); // decode with spec tokens

        // Should schedule 1 (real token) + 5 (spec tokens) = 6 new tokens.
        let num_scheduled = output.num_scheduled_tokens.get("r1").copied().unwrap_or(0);
        assert_eq!(num_scheduled, 6, "Should schedule 1 real + 5 spec tokens");

        // Spec tokens should appear in scheduled_spec_decode_tokens.
        let spec_tokens = output.scheduled_spec_decode_tokens.get("r1").unwrap();
        assert_eq!(spec_tokens, &vec![100, 101, 102, 103, 104]);
    }

    #[test]
    fn test_spec_decode_tokens_cleared_after_schedule() {
        // After scheduling, spec_token_ids should be cleared.
        let cfg = spec_decode_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule(); // prefill
        sched.append_output_tokens("r1", &[99]);

        sched.set_spec_token_ids("r1", vec![100, 101, 102]);

        sched.schedule(); // consumes spec tokens

        // Spec tokens should be cleared.
        let req = sched.get_request("r1").unwrap();
        assert!(
            req.spec_token_ids.is_empty(),
            "spec_token_ids should be cleared after schedule"
        );
    }

    #[test]
    fn test_spec_decode_tokens_truncated_by_budget() {
        // When token budget is limited, spec tokens should be truncated.
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 4, // Very tight budget
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            num_lookahead_tokens: 5,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 2));
        sched.schedule(); // prefill (2 tokens)
        sched.append_output_tokens("r1", &[99]);

        sched.set_spec_token_ids("r1", vec![100, 101, 102, 103, 104]); // 5 draft tokens

        let output = sched.schedule(); // budget = 4, need 1+5=6, should truncate

        let num_scheduled = output.num_scheduled_tokens.get("r1").copied().unwrap_or(0);
        assert!(
            num_scheduled <= 4,
            "num_scheduled_tokens ({num_scheduled}) should not exceed budget (4)"
        );

        // Spec tokens in output should be truncated accordingly.
        if let Some(spec_tokens) = output.scheduled_spec_decode_tokens.get("r1") {
            assert!(
                spec_tokens.len() < 5,
                "Spec tokens should be truncated when budget is tight"
            );
        }
    }

    #[test]
    fn test_spec_decode_lookahead_allocates_blocks() {
        // Verify that num_lookahead_tokens causes extra block allocation.
        // With lookahead=5, a decode step needs more blocks than without.
        let cfg_no_spec = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 4,
            num_lookahead_tokens: 0,
            ..Default::default()
        };
        let cfg_spec = spec_decode_scheduler_config(); // lookahead=5

        // Use very limited blocks to see the difference.
        let mut sched_no = Scheduler::with_simple_blocks(&cfg_no_spec, 8192, 2, 16);
        let mut sched_sp = Scheduler::with_simple_blocks(&cfg_spec, 8192, 2, 16);

        // Add a request that fills nearly all blocks during prefill.
        sched_no.add_request(make_request("r1", 15));
        sched_sp.add_request(make_request("r1", 15));

        let out_no = sched_no.schedule(); // prefill
        let out_sp = sched_sp.schedule(); // prefill

        assert!(out_no.num_scheduled_tokens.contains_key("r1"));
        assert!(out_sp.num_scheduled_tokens.contains_key("r1"));

        // Append one token to move to decode.
        sched_no.append_output_tokens("r1", &[99]);
        sched_sp.append_output_tokens("r1", &[99]);

        // Decode step: without lookahead, 1 token is easy.
        // With lookahead=5, we need 1+5=6 slots which might be tight.
        let out_no = sched_no.schedule();
        let out_sp = sched_sp.schedule();

        // Both should schedule (we have enough blocks), but this confirms
        // lookahead is wired into the allocation path.
        assert!(
            out_no.num_scheduled_tokens.contains_key("r1"),
            "No-spec should schedule"
        );
        assert!(
            out_sp.num_scheduled_tokens.contains_key("r1"),
            "Spec should schedule"
        );
    }

    #[test]
    fn test_spec_decode_mixed_batch() {
        // Some requests have spec tokens, some don't.
        let cfg = spec_decode_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.add_request(make_request("r2", 10));

        sched.schedule(); // prefill both
        sched.append_output_tokens("r1", &[99]);
        sched.append_output_tokens("r2", &[99]);

        // Only r1 has spec tokens.
        sched.set_spec_token_ids("r1", vec![100, 101, 102]);

        let output = sched.schedule();

        // r1 should have 4 tokens (1 real + 3 spec).
        let n1 = output.num_scheduled_tokens.get("r1").copied().unwrap_or(0);
        assert_eq!(n1, 4, "r1 should schedule 1+3 tokens");

        // r2 should have 1 token (normal decode).
        let n2 = output.num_scheduled_tokens.get("r2").copied().unwrap_or(0);
        assert_eq!(n2, 1, "r2 should schedule 1 token");

        // Only r1 should have spec decode tokens in output.
        assert!(output.scheduled_spec_decode_tokens.contains_key("r1"));
        assert!(!output.scheduled_spec_decode_tokens.contains_key("r2"));
    }

    #[test]
    fn test_spec_decode_num_tokens_with_spec() {
        // Verify num_tokens_with_spec() matches scheduler's expectation.
        let mut req = make_request("r1", 10);
        assert_eq!(req.num_tokens_with_spec(), 10);

        req.spec_token_ids = vec![100, 101, 102];
        assert_eq!(req.num_tokens_with_spec(), 13);

        req.spec_token_ids.clear();
        assert_eq!(req.num_tokens_with_spec(), 10);
    }

    #[test]
    fn test_spec_decode_rewind_on_rejection() {
        // Verify that rewind_num_computed_tokens correctly decrements
        // num_computed_tokens when spec decode drafts are rejected.
        // This matches Python's scheduler.update_from_output().
        let cfg = spec_decode_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule(); // prefill: num_computed_tokens = 10
        sched.append_output_tokens("r1", &[99]);

        // Set 5 spec tokens and schedule them.
        sched.set_spec_token_ids("r1", vec![100, 101, 102, 103, 104]);
        let output = sched.schedule(); // num_computed_tokens += 6 (1 real + 5 spec)

        let num_scheduled = output.num_scheduled_tokens.get("r1").copied().unwrap_or(0);
        assert_eq!(num_scheduled, 6);

        // After schedule: num_computed_tokens = 10 + 6 = 16.
        let req = sched.get_request("r1").unwrap();
        assert_eq!(req.num_computed_tokens, 16);

        // Simulate partial rejection: only 2 of 5 drafts accepted (3 rejected).
        // The model would return 3 tokens (2 accepted + 1 recovered/bonus).
        sched.rewind_num_computed_tokens("r1", 3);

        let req = sched.get_request("r1").unwrap();
        assert_eq!(
            req.num_computed_tokens, 13,
            "num_computed_tokens should be 16 - 3 = 13"
        );
    }

    #[test]
    fn test_spec_decode_rewind_all_rejected() {
        // All drafts rejected: rewind by full draft count.
        let cfg = spec_decode_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule(); // prefill: num_computed = 10
        sched.append_output_tokens("r1", &[99]);

        sched.set_spec_token_ids("r1", vec![100, 101, 102]);
        sched.schedule(); // num_computed = 10 + 4 = 14

        // All 3 drafts rejected: model returns 1 token (recovered).
        // num_accepted = 0, num_rejected = 3.
        sched.rewind_num_computed_tokens("r1", 3);

        let req = sched.get_request("r1").unwrap();
        assert_eq!(
            req.num_computed_tokens, 11,
            "num_computed_tokens should be 14 - 3 = 11"
        );
    }

    #[test]
    fn test_spec_decode_rewind_none_rejected() {
        // All drafts accepted: no rewind needed.
        let cfg = spec_decode_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 10));
        sched.schedule();
        sched.append_output_tokens("r1", &[99]);

        sched.set_spec_token_ids("r1", vec![100, 101]);
        sched.schedule(); // num_computed = 10 + 3 = 13

        // All 2 drafts accepted: model returns 3 tokens (2 accepted + bonus).
        // num_accepted = 2, num_rejected = 0. No rewind.
        sched.rewind_num_computed_tokens("r1", 0);

        let req = sched.get_request("r1").unwrap();
        assert_eq!(req.num_computed_tokens, 13, "no rewind when all accepted");
    }

    // -----------------------------------------------------------------------
    // Pipeline parallelism tests
    // -----------------------------------------------------------------------

    fn pp_scheduler_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            async_scheduling: Some(false),
            use_pp: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_pp_new_token_ids_populated_on_decode() {
        // When use_pp=true and async_scheduling=false, the scheduler must
        // populate new_token_ids in CachedRequestData so non-last PP stages
        // can embed the correct token.
        let cfg = pp_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        // Add request with 10 prompt tokens.
        sched.add_request(make_request("r1", 10));

        // Step 1: Prefill. No cached reqs yet.
        let output1 = sched.schedule();
        assert_eq!(output1.scheduled_new_reqs.len(), 1);
        assert!(output1.scheduled_cached_reqs.req_ids.is_empty());

        // Simulate prefill producing token 42.
        sched.append_output_tokens("r1", &[42]);

        // Step 2: Decode. Request is now cached.
        let output2 = sched.schedule();
        assert_eq!(output2.scheduled_cached_reqs.req_ids.len(), 1);
        assert_eq!(output2.scheduled_cached_reqs.req_ids[0], "r1");

        // new_token_ids should contain the token the worker needs to embed.
        assert_eq!(output2.scheduled_cached_reqs.new_token_ids.len(), 1);
        assert!(
            !output2.scheduled_cached_reqs.new_token_ids[0].is_empty(),
            "PP sync scheduling must populate new_token_ids for cached requests"
        );
        // The token should be 42 (the one we appended).
        assert!(
            output2.scheduled_cached_reqs.new_token_ids[0].contains(&42),
            "new_token_ids should contain the sampled token"
        );
    }

    #[test]
    fn test_pp_new_token_ids_multiple_decode_steps() {
        // Verify new_token_ids is correct across multiple decode steps.
        let cfg = pp_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 5));

        // Prefill.
        sched.schedule();
        sched.append_output_tokens("r1", &[100]);

        // Decode step 1: should see token 100.
        let out1 = sched.schedule();
        assert_eq!(out1.scheduled_cached_reqs.new_token_ids.len(), 1);
        assert!(out1.scheduled_cached_reqs.new_token_ids[0].contains(&100));

        // Simulate decode producing token 200.
        sched.append_output_tokens("r1", &[200]);

        // Decode step 2: should see token 200.
        let out2 = sched.schedule();
        assert_eq!(out2.scheduled_cached_reqs.new_token_ids.len(), 1);
        assert!(out2.scheduled_cached_reqs.new_token_ids[0].contains(&200));

        // Simulate decode producing token 300.
        sched.append_output_tokens("r1", &[300]);

        // Decode step 3: should see token 300.
        let out3 = sched.schedule();
        assert_eq!(out3.scheduled_cached_reqs.new_token_ids.len(), 1);
        assert!(out3.scheduled_cached_reqs.new_token_ids[0].contains(&300));
    }

    #[test]
    fn test_pp_new_token_ids_multiple_requests() {
        // Verify new_token_ids works with multiple concurrent requests.
        let cfg = pp_scheduler_config();
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 5));
        sched.add_request(make_request("r2", 5));

        // Prefill both.
        sched.schedule();
        sched.append_output_tokens("r1", &[10]);
        sched.append_output_tokens("r2", &[20]);

        // Decode: both should have their correct new_token_ids.
        let out = sched.schedule();
        assert_eq!(out.scheduled_cached_reqs.req_ids.len(), 2);
        assert_eq!(out.scheduled_cached_reqs.new_token_ids.len(), 2);

        // Find which index is which request.
        for (i, req_id) in out.scheduled_cached_reqs.req_ids.iter().enumerate() {
            let tokens = &out.scheduled_cached_reqs.new_token_ids[i];
            match req_id.as_str() {
                "r1" => assert!(tokens.contains(&10), "r1 should have token 10"),
                "r2" => assert!(tokens.contains(&20), "r2 should have token 20"),
                _ => panic!("unexpected req_id"),
            }
        }
    }

    #[test]
    fn test_pp_no_new_token_ids_without_pp() {
        // Without use_pp, new_token_ids should be empty.
        let cfg = test_scheduler_config(); // use_pp = false
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 5));
        sched.schedule();
        sched.append_output_tokens("r1", &[42]);

        let out = sched.schedule();
        assert!(
            out.scheduled_cached_reqs.new_token_ids.is_empty(),
            "Without PP, new_token_ids should be empty"
        );
    }

    #[test]
    fn test_pp_no_new_token_ids_with_async_scheduling() {
        // With use_pp + async_scheduling, new_token_ids should be empty
        // (tokens go via GPU broadcast instead).
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 256,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            async_scheduling: Some(true),
            use_pp: true,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 100, 16);

        sched.add_request(make_request("r1", 5));
        sched.schedule();
        sched.append_output_tokens("r1", &[42]);

        let out = sched.schedule();
        assert!(
            out.scheduled_cached_reqs.new_token_ids.is_empty(),
            "PP + async scheduling should NOT populate new_token_ids (uses GPU broadcast)"
        );
    }

    #[test]
    fn test_decode_blocks_cached_after_preemption() {
        // After preemption, decode blocks should be recoverable from the cache.
        let block_size = 16;
        let mut tracker = SimpleBlockTracker::with_caching(20, block_size);

        // 32-token prompt + 32 decode tokens = 64 tokens = 4 full blocks.
        let prompt: Vec<u32> = (0..32).collect();
        let mut r1 = Request::new(
            "r1".into(),
            prompt,
            SamplingParams {
                max_tokens: Some(100),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );

        // Allocate prompt blocks (2 blocks for 32 tokens).
        let blocks = tracker.allocate_slots(&r1, 32, 0, &[]).unwrap();
        assert_eq!(blocks[0].len(), 2);
        r1.num_computed_tokens = 32;

        // Simulate 32 decode tokens, one at a time, allocating as needed.
        for i in 0..32u32 {
            r1.append_output_token_ids(&[100 + i]);
            r1.num_computed_tokens += 1;
            let _ = tracker.allocate_slots(&r1, 1, 0, &[]).unwrap();
        }

        // We have 4 full blocks + 1 partial (the last allocate_slots added a 5th
        // block for the partial tail). Only the 4 full blocks get hashed.
        let alloc = tracker.get_blocks("r1");
        assert_eq!(alloc[0].len(), 5);
        let original_block_ids: Vec<usize> = alloc[0][..4].to_vec();

        // Preempt: free blocks, reset computed tokens (but all_token_ids survives).
        tracker.free("r1");
        r1.num_computed_tokens = 0;
        r1.status = RequestStatus::Preempted;

        // 4 full blocks should be in the cache; the partial block goes to free list.
        assert_eq!(tracker.num_cached_blocks(), 4);

        // Re-admission: get_computed_blocks should find all 4 cached blocks.
        let (cached_tokens, cached_blocks) = tracker.get_computed_blocks(&r1);
        assert_eq!(cached_tokens, 64);
        assert_eq!(cached_blocks[0].len(), 4);
        assert_eq!(cached_blocks[0], original_block_ids);
    }

    // ----- running_pop_tail / running_req_idx consistency tests -----

    /// FCFS preemption always evicts the tail request.
    /// Setup: 2 blocks. r1+r2 fill them. Append 1 decode token each.
    /// Step 2: r1's decode needs a second block (ceil(17/16)=2), no free blocks
    /// → preempt r2 (tail). r1 gets r2's block. Only r2 is preempted.
    #[test]
    fn test_preemption_evicts_tail_fcfs() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 512,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 2, 16);

        sched.add_request(make_request("r1", 16));
        sched.add_request(make_request("r2", 16));
        let out1 = sched.schedule();
        assert_eq!(out1.scheduled_new_reqs.len(), 2);

        sched.append_output_tokens("r1", &[1]);
        sched.append_output_tokens("r2", &[2]);

        // r3 is a bystander sitting in waiting — should never be touched.
        sched.add_request(make_request("r3", 16));
        let out2 = sched.schedule();

        // r2 (tail) must be preempted; r1 and r3 must not be.
        let preempted = out2
            .preempted_req_ids
            .as_ref()
            .expect("preemption expected");
        assert!(
            preempted.contains("r2"),
            "expected r2 (tail) preempted, got {preempted:?}"
        );
        assert!(!preempted.contains("r1"), "r1 should not be preempted");
        assert!(!preempted.contains("r3"), "r3 was never running");

        let (running, _) = sched.get_request_counts();
        assert_eq!(running, 1, "only r1 should be running");
    }

    /// After tail preemption the running_req_idx for the remaining requests
    /// must still be correct (no stale indices from a skipped O(n) sweep).
    #[test]
    fn test_running_req_idx_consistent_after_tail_preemption() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 512,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 2, 16);

        sched.add_request(make_request("r1", 16));
        sched.add_request(make_request("r2", 16));
        let _out1 = sched.schedule();

        sched.append_output_tokens("r1", &[10]);
        sched.append_output_tokens("r2", &[20]);

        sched.add_request(make_request("r3", 16));
        let _out2 = sched.schedule(); // r2 preempted (tail)

        // Finish r1. If running_req_idx has a stale/wrong entry this panics
        // inside running_remove().
        sched.finish_requests(&["r1"], RequestStatus::FinishedStopped);

        // r2 (waiting/preempted) + r3 (waiting) remain.
        assert_eq!(sched.get_num_unfinished_requests(), 2);
    }

    // ----- block free ordering test -----

    /// When a request's blocks are freed, tail blocks enter the free queue
    /// first (evicted soonest), preserving prefix blocks longest.
    /// Use an exact-fit pool (4 blocks, block_size=4) so after freeing the
    /// queue contains only the freed blocks in the expected order.
    #[test]
    fn test_block_free_reverse_order() {
        // 4 blocks of size 4 — exactly fits a 16-token request.
        let mut tracker = SimpleBlockTracker::new(4, 4);
        let req = make_request("r1", 16);

        let alloc = tracker
            .allocate_slots(&req, 16, 0, &[])
            .expect("should allocate");
        let block_ids = alloc[0].clone(); // [0, 1, 2, 3]
        assert_eq!(block_ids.len(), 4);
        assert_eq!(tracker.num_free_blocks(), 0);

        tracker.free("r1");
        assert_eq!(tracker.num_free_blocks(), 4);

        // Allocate all 4 back: they should come out tail-first (block_ids[3]
        // first) because free() pushes in reverse order.
        let fresh = tracker.allocate_fresh_blocks(4).expect("allocate 4");
        assert_eq!(
            fresh[0], block_ids[3],
            "tail block (index 3) should be evicted first, got block {}",
            fresh[0]
        );
        assert_eq!(fresh[3], block_ids[0], "head block (index 0) evicted last");
    }

    /// Preempted requests are prepended to the waiting queue so they are
    /// re-admitted before newly arrived requests.
    ///
    /// Setup: 2 blocks. r1+r2 fill them. Step 2 decode grows r1 beyond 1
    /// block → r2 (tail) is preempted. r_new arrived while r1/r2 were
    /// running, so waiting = [r2(preempted), r_new].
    /// After finishing r1, 2 blocks are freed. r2 needs 2 blocks
    /// (ceil(17/16)) and gets them both; r_new gets nothing. Confirms r2
    /// (preempted) is ahead of r_new in the waiting queue.
    #[test]
    fn test_preempted_request_returns_to_front_of_waiting() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 512,
            max_num_seqs: 10,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        // 2 blocks of 16 tokens each.
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 2, 16);

        sched.add_request(make_request("r1", 16));
        sched.add_request(make_request("r2", 16));
        // schedule() internally calls update_after_schedule → num_computed = 16.
        let out1 = sched.schedule();
        assert_eq!(out1.scheduled_new_reqs.len(), 2);

        // Append 1 decode token: all_token_ids grows to 17, so next step
        // needs ceil(17/16)=2 blocks → triggers block allocation → preemption.
        sched.append_output_tokens("r1", &[1]);
        sched.append_output_tokens("r2", &[2]);

        // r_new arrives while r1/r2 are running.
        sched.add_request(make_request("r_new", 16));

        // Step 2: r1 decode needs a 2nd block; 0 free → preempt r2 (tail).
        // r1 gets r2's freed block.
        let out2 = sched.schedule();
        let preempted2 = out2
            .preempted_req_ids
            .as_ref()
            .expect("preemption expected");
        assert!(
            preempted2.contains("r2"),
            "r2 should be preempted: {preempted2:?}"
        );
        assert!(!preempted2.contains("r1"));

        // waiting = [r2 (preempted, at front), r_new]. 0 free blocks (r1 holds both).
        // Finish r1 to release its 2 blocks.
        sched.finish_requests(&["r1"], RequestStatus::FinishedStopped);

        // Step 3: Phase 2 tries r2 first (it was prepended). r2 needs
        // ceil(17/16)=2 blocks — gets both. r_new gets nothing.
        let out3 = sched.schedule();
        let resumed: Vec<&str> = out3
            .scheduled_cached_reqs
            .resumed_req_ids
            .iter()
            .map(|s| s.as_str())
            .collect();
        assert!(
            resumed.contains(&"r2"),
            "preempted r2 should be re-admitted: {resumed:?}"
        );
        assert!(
            !out3.num_scheduled_tokens.contains_key("r_new"),
            "r_new should still be waiting (r2 consumed all free blocks)"
        );
    }

    // -----------------------------------------------------------------------
    // Span-aware hashing tests
    // -----------------------------------------------------------------------

    fn spans_config_enabled() -> SpansConfig {
        SpansConfig { debug: false }
    }

    #[test]
    fn test_span_hash_all_blocks_chains_parents() {
        // hash_all_blocks chains parent hashes, so the same block content
        // at different positions produces different hashes.
        let tracker = SimpleBlockTracker::new(16, 4);
        let tokens: Vec<u32> = (0..12).collect(); // 3 full blocks of size 4
        let hashes = tracker.hash_all_blocks(&tokens, None, None);
        assert_eq!(hashes.len(), 3);
        // Block 0 has parent NONE_HASH, so its hash includes NONE_HASH + content.
        // Block 1 chains from block 0's hash, block 2 from block 1's.
        // All three should be distinct.
        assert_ne!(hashes[0], hashes[1]);
        assert_ne!(hashes[1], hashes[2]);
        assert_ne!(hashes[0], hashes[2]);
        // Same tokens, same order → same hashes.
        let hashes2 = tracker.hash_all_blocks(&tokens, None, None);
        assert_eq!(hashes, hashes2);
    }

    #[test]
    fn test_span_relocatable_resets_parent_hash() {
        let cfg = spans_config_enabled();
        let tracker = SimpleBlockTracker::with_spans_config(16, 4, cfg);

        // Block 1 is its OWN single-block relocatable span (first_token = 4, the
        // block's own first token, so it is the span's FIRST block). Under
        // chain-within-span hashing the span's first block resets its parent to
        // NONE_HASH → it hashes purely by content. Two different prefixes
        // followed by the same span-first block therefore produce the same hash.
        let seq_a: Vec<u32> = vec![0, 1, 2, 3, 100, 101, 102, 103];
        let seq_b: Vec<u32> = vec![9, 8, 7, 6, 100, 101, 102, 103];

        let mut ann = std::collections::BTreeMap::new();
        ann.insert(1, BlockKind::Relocatable { first_token: 4 });

        let hashes_a = tracker.hash_all_blocks(&seq_a, Some(&ann), None);
        let hashes_b = tracker.hash_all_blocks(&seq_b, Some(&ann), None);

        // First blocks differ (different tokens).
        assert_ne!(hashes_a[0], hashes_b[0]);
        // Second blocks (span-first, relocatable) are identical — same content,
        // parent reset to NONE_HASH regardless of the differing prefix.
        assert_eq!(hashes_a[1], hashes_b[1]);
    }

    #[test]
    fn test_span_prefixed_includes_context() {
        let cfg = spans_config_enabled();
        let tracker = SimpleBlockTracker::with_spans_config(16, 4, cfg);

        // Block 1 annotated as Prefixed should differ when preceded
        // by different tokens.
        let seq_a: Vec<u32> = vec![0, 1, 2, 3, 100, 101, 102, 103];
        let seq_b: Vec<u32> = vec![9, 8, 7, 6, 100, 101, 102, 103];

        let mut ann = std::collections::BTreeMap::new();
        ann.insert(1, BlockKind::Prefixed { first_token: 0 });

        let hashes_a = tracker.hash_all_blocks(&seq_a, Some(&ann), None);
        let hashes_b = tracker.hash_all_blocks(&seq_b, Some(&ann), None);

        // First blocks differ.
        assert_ne!(hashes_a[0], hashes_b[0]);
        // Second blocks (prefixed) should differ because preceding tokens differ.
        assert_ne!(hashes_a[1], hashes_b[1]);
    }

    #[test]
    fn test_span_relocatable_cache_reuse() {
        // Simulate: preload doc_a, then query [doc_a, doc_b, query].
        // doc_a's block should be a cache hit for the second request.
        let cfg = spans_config_enabled();
        let mut tracker = SimpleBlockTracker::with_spans_config(64, 4, cfg);

        let doc_a: Vec<u32> = vec![100, 101, 102, 103]; // relocatable block
        let doc_b: Vec<u32> = vec![200, 201, 202, 203]; // relocatable block
        let query: Vec<u32> = vec![300, 301, 302, 303]; // prefixed block

        // Request 1: [doc_a, query]
        let mut seq1 = Vec::new();
        seq1.extend_from_slice(&doc_a);
        seq1.extend_from_slice(&query);
        let mut req1 = Request::new(
            "r1".into(),
            seq1,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        // block 0 = relocatable (doc_a), block 1 = prefixed (query)
        let mut ann1 = std::collections::BTreeMap::new();
        ann1.insert(0, BlockKind::Relocatable { first_token: 0 });
        ann1.insert(1, BlockKind::Prefixed { first_token: 0 });
        req1.block_annotations = Some(ann1);

        let blocks = tracker.allocate_slots(&req1, req1.all_token_ids.len(), 0, &[]);
        assert!(blocks.is_some());

        // Free request 1 — blocks stay in cache.
        tracker.free("r1");

        // Request 2: [doc_a, doc_b, query]
        let mut seq2 = Vec::new();
        seq2.extend_from_slice(&doc_a);
        seq2.extend_from_slice(&doc_b);
        seq2.extend_from_slice(&query);
        let mut req2 = Request::new(
            "r2".into(),
            seq2,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        // block 0 = relocatable (doc_a), block 1 = relocatable (doc_b), block 2 = prefixed (query)
        let mut ann2 = std::collections::BTreeMap::new();
        ann2.insert(0, BlockKind::Relocatable { first_token: 0 });
        ann2.insert(1, BlockKind::Relocatable { first_token: 0 });
        ann2.insert(2, BlockKind::Prefixed { first_token: 0 });
        req2.block_annotations = Some(ann2);

        // doc_a's block should be a cache hit (relocatable, same content).
        let (num_computed, cached_ids) = tracker.get_computed_blocks(&req2);
        assert_eq!(num_computed, 4); // one full block of size 4
        assert!(!cached_ids[0].is_empty());
    }

    #[test]
    fn test_span_reuse_is_contiguous_only_across_a_gap() {
        // Relocatable span reuse is CONTIGUOUS-ONLY: if an earlier block MISSES,
        // reuse stops there — later blocks are recomputed even when their content
        // is cached. Scattered post-gap reuse is deliberately disabled: for the
        // non-block-aligned `/v1/messages` tool spans it fed mis-positioned KV
        // and POISONED the cache (see `get_computed_blocks`). launch-claude
        // reuses the identical leading tools+system prefix, which IS contiguous,
        // so contiguous-only is exactly the safe subset it needs. (Re-enabling
        // post-gap relocation is gated behind the non-aligned rope-on-read fix.)
        let cfg = spans_config_enabled();
        let mut tracker = SimpleBlockTracker::with_spans_config(64, 4, cfg);

        let doc_a: Vec<u32> = vec![100, 101, 102, 103];
        let doc_b: Vec<u32> = vec![200, 201, 202, 203];
        let doc_new: Vec<u32> = vec![900, 901, 902, 903];

        // req1: [doc_a, doc_b], both relocatable → populate cache, then free
        // (blocks stay hittable at ref_cnt 0 until evicted).
        let mut seq1 = doc_a.clone();
        seq1.extend_from_slice(&doc_b);
        let mut req1 = Request::new(
            "r1".into(),
            seq1,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        // doc_a and doc_b are each their OWN single-block relocatable span, so
        // first_token = each block's own start (0, 4).
        let mut ann1 = std::collections::BTreeMap::new();
        ann1.insert(0, BlockKind::Relocatable { first_token: 0 });
        ann1.insert(1, BlockKind::Relocatable { first_token: 4 });
        req1.block_annotations = Some(ann1);
        let r1 = tracker.allocate_slots(&req1, 8, 0, &[]).unwrap();
        let bid_a = r1[0][0];
        let bid_b = r1[0][1];
        tracker.free("r1");

        // req2: [doc_new, doc_a, doc_b] — block 0 MISSES (new content). doc_a and
        // doc_b are cached and content-hash the same (each a span-first block),
        // so they WOULD hit — but they sit past the miss, so contiguous-only
        // reuse skips them entirely.
        let mut seq2 = doc_new.clone();
        seq2.extend_from_slice(&doc_a);
        seq2.extend_from_slice(&doc_b);
        let mut req2 = Request::new(
            "r2".into(),
            seq2,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        let mut ann2 = std::collections::BTreeMap::new();
        ann2.insert(0, BlockKind::Relocatable { first_token: 0 });
        ann2.insert(1, BlockKind::Relocatable { first_token: 4 });
        ann2.insert(2, BlockKind::Relocatable { first_token: 8 });
        req2.block_annotations = Some(ann2);

        // The first block misses → reuse stops there. No scattered post-gap
        // hits: matched is the (empty) contiguous prefix, NOT [MISS, bid_a,
        // bid_b]. doc_a/doc_b are recomputed even though they are cached.
        let (num_computed, matched) = tracker.get_computed_blocks(&req2);
        assert_eq!(
            num_computed, 0,
            "first block misses → contiguous reuse is 0"
        );
        assert!(
            matched[0].is_empty(),
            "no scattered post-gap reuse — reuse stops at the gap"
        );

        // Allocating with the empty match table recomputes all three blocks into
        // a dense, ordinal block table (no mis-positioned relocatable KV bound).
        // (bid_a/bid_b were freed by req1; whether their physical slots get
        // recycled here is allocator-incidental, so we don't assert on them.)
        let _ = (bid_a, bid_b);
        req2.num_computed_tokens = num_computed;
        let r2 = tracker.allocate_slots(&req2, 12, 0, &matched).unwrap();
        assert_eq!(r2[0].len(), 3, "block table is dense (one bid per slot)");
    }

    #[test]
    fn test_no_annotations_still_chains_parents() {
        // Without annotations, parent hashing is used (matching Python vLLM).
        // Same block content at different positions should produce different hashes.
        let tracker = SimpleBlockTracker::with_spans_config(
            16,
            4,
            SpansConfig::default(), // disabled
        );
        // [A, B] and [B, A] — block content is the same but order differs.
        let seq1: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let seq2: Vec<u32> = vec![4, 5, 6, 7, 0, 1, 2, 3];
        let hashes1 = tracker.hash_all_blocks(&seq1, None, None);
        let hashes2 = tracker.hash_all_blocks(&seq2, None, None);
        // Block 0 differs (different content).
        assert_ne!(hashes1[0], hashes2[0]);
        // Block 1 also differs (same content but different parent).
        assert_ne!(hashes1[1], hashes2[1]);
    }

    #[test]
    fn test_seal_caches_partial_block() {
        // Simulates the full sealed-request lifecycle step by step, as it
        // happens during real generation with SealPadProcessor:
        //
        // 1. Prefill: 6-token prompt → 2 blocks allocated, block 0 hash registered
        // 2. Decode step 1: token 100 → 7 tokens (no new block needed)
        // 3. Decode step 2: token 2 (EOS) → 8 tokens (block-aligned)
        //    Engine would stop here; SealPadProcessor not needed in this case.
        //    But allocate_slots returns early (additional=0), block 1 hash NOT registered.
        // 4. seal() → registers block 1 hash
        // 5. Outer request with same 8 tokens → full cache hit on both blocks

        let cfg = spans_config_enabled();
        let mut tracker = SimpleBlockTracker::with_spans_config(64, 4, cfg);

        let prompt: Vec<u32> = vec![10, 20, 30, 40, 50, 60]; // 6 tokens
        let mut req = Request::new(
            "inner".into(),
            prompt,
            SamplingParams {
                max_tokens: Some(100),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        req.seal = true;

        // Step 1: Prefill — allocate 2 blocks for 6 tokens.
        let blocks = tracker.allocate_slots(&req, 6, 0, &[]);
        assert!(blocks.is_some());
        let block_ids = blocks.unwrap();
        assert_eq!(block_ids[0].len(), 2, "6 tokens need 2 blocks of size 4");

        // Block 0 (full: [10,20,30,40]) should have its hash registered.
        // Block 1 (partial: [50,60]) should NOT have its hash registered yet.
        let hashes_so_far = tracker.req_to_hashes.get("inner");
        assert_eq!(
            hashes_so_far.map(|h| h.len()),
            Some(1),
            "only 1 full block hash should be registered after prefill"
        );

        // Step 2: Decode — generate token 100. Total = 7 tokens.
        req.append_output_token_ids(&[100]);
        req.num_computed_tokens = 6; // prefill computed 6
        let blocks2 = tracker.allocate_slots(&req, 7, 0, &[]);
        assert!(blocks2.is_some());
        // Still 2 blocks (ceil(7/4) = 2), no new allocation.
        assert_eq!(
            tracker.req_to_hashes.get("inner").map(|h| h.len()),
            Some(1),
            "still only 1 hash — block 1 is still partial"
        );

        // Step 3: Decode — generate EOS token 2. Total = 8 tokens (block-aligned).
        req.append_output_token_ids(&[2]);
        req.num_computed_tokens = 7;
        let blocks3 = tracker.allocate_slots(&req, 8, 0, &[]);
        assert!(blocks3.is_some());
        // Still 2 blocks. additional=0, early return — block 1 hash NOT registered.
        assert_eq!(
            tracker.req_to_hashes.get("inner").map(|h| h.len()),
            Some(1),
            "allocate_slots with additional=0 does NOT register new hashes"
        );

        // Step 4: seal() — this is where block 1's hash gets registered.
        tracker.seal(&req);
        assert_eq!(
            tracker.req_to_hashes.get("inner").map(|h| h.len()),
            Some(2),
            "seal() must register hash for block 1"
        );

        // Free the request (blocks go to LRU, hashes persist).
        tracker.free("inner");

        // Step 5: Outer request with same 8 tokens → should hit both blocks.
        let outer_tokens: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 100, 2];
        let outer = Request::new(
            "outer".into(),
            outer_tokens,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );

        let (num_computed, cached_blocks) = tracker.get_computed_blocks(&outer);
        assert_eq!(num_computed, 8, "all 8 tokens should be cached");
        assert_eq!(
            cached_blocks[0].len(),
            2,
            "both blocks should be cache hits"
        );
    }

    #[test]
    fn test_volatile_evicts_before_normal() {
        // Volatile blocks go to the front of the free queue (evicted first).
        // Normal blocks go to the back.
        let cfg = spans_config_enabled();
        let mut tracker = SimpleBlockTracker::with_spans_config(4, 4, cfg);

        // Fill all 4 blocks with 2 requests.
        let req_normal = Request::new(
            "normal".into(),
            vec![1, 2, 3, 4],
            SamplingParams {
                max_tokens: Some(1),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        let req_volatile = Request::new(
            "volatile".into(),
            vec![5, 6, 7, 8],
            SamplingParams {
                max_tokens: Some(1),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );

        tracker.allocate_slots(&req_normal, 4, 0, &[]);
        tracker.allocate_slots(&req_volatile, 4, 0, &[]);
        assert_eq!(tracker.num_free_blocks(), 2); // 4 total - 2 used

        // Free both: normal to back, volatile to front.
        tracker.free("normal");
        tracker.free_volatile("volatile");

        // The next allocation should reuse the volatile block first
        // (it's at the front of the free queue).
        let req3 = Request::new(
            "r3".into(),
            vec![9, 10, 11, 12],
            SamplingParams {
                max_tokens: Some(1),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        let blocks = tracker.allocate_slots(&req3, 4, 0, &[]).unwrap();
        // The volatile request's block (block 1) should be allocated first.
        assert_eq!(blocks[0][0], 1);
    }

    #[test]
    fn test_volatile_blocks_still_hittable_before_eviction() {
        // After free_volatile, the cache mapping (hash→block_id) still exists.
        // A request arriving before eviction should get a cache hit.
        let cfg = spans_config_enabled();
        let mut tracker = SimpleBlockTracker::with_spans_config(64, 4, cfg);

        let tokens: Vec<u32> = vec![10, 20, 30, 40];
        let req1 = Request::new(
            "r1".into(),
            tokens.clone(),
            SamplingParams {
                max_tokens: Some(1),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        tracker.allocate_slots(&req1, 4, 0, &[]);
        tracker.free_volatile("r1");

        // Same tokens — should still hit the cached block.
        let req2 = Request::new(
            "r2".into(),
            tokens,
            SamplingParams {
                max_tokens: Some(1),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        let (num_computed, cached_ids) = tracker.get_computed_blocks(&req2);
        assert_eq!(num_computed, 4);
        assert!(!cached_ids[0].is_empty());
    }

    #[test]
    fn test_seal_plus_volatile_inner_generate_pattern() {
        // Full lifecycle test mimicking nested RAG: inner sealed generate,
        // then outer query using same tokens. Step-by-step through scheduler.
        //
        // Inner: 8 prompt + 3 decode + EOS + 3 pads = 16 tokens (4 blocks)
        // Outer: same 16 tokens as prefix, then more → all 4 inner blocks cached

        let cfg = spans_config_enabled();
        let mut tracker = SimpleBlockTracker::with_spans_config(64, 4, cfg);

        // Inner: 8-token prompt = 2 full blocks.
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut inner = Request::new(
            "inner".into(),
            prompt,
            SamplingParams {
                max_tokens: Some(20),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        inner.seal = true;
        inner.volatile = true;

        // Step 1: Prefill 8 tokens → 2 blocks.
        tracker.allocate_slots(&inner, 8, 0, &[]);
        inner.num_computed_tokens = 8;
        assert_eq!(tracker.req_to_hashes.get("inner").map(|h| h.len()), Some(2));

        // Steps 2-4: Decode tokens 100, 101, 102.
        for &tok in &[100u32, 101, 102] {
            inner.append_output_token_ids(&[tok]);
            inner.num_computed_tokens += 1;
            tracker.allocate_slots(&inner, inner.all_token_ids.len(), 0, &[]);
        }
        // 11 tokens → 3 blocks. Block 2 allocated at token 9 (need ceil(9/4)=3).
        // Block 2 hash NOT registered (partial).
        assert_eq!(inner.all_token_ids.len(), 11);

        // Step 5: EOS token (2). Total = 12, block-aligned.
        // Engine defers stop because request is sealed.
        inner.append_output_token_ids(&[2]);
        inner.num_computed_tokens += 1;
        tracker.allocate_slots(&inner, 12, 0, &[]);
        assert_eq!(inner.all_token_ids.len(), 12);

        // At this point, 12 tokens = 3 full blocks. But block 2 hash
        // was NOT registered by allocate_slots (additional=0 when we went
        // from 11→12 tokens, still needing 3 blocks).
        // Only blocks 0,1 have hashes from prefill, and block 2 got its
        // hash when it was first allocated at 9 tokens.
        // Actually: at 9 tokens, allocate_slots allocated 1 new block,
        // and registered hashes for num_full_blocks=9/4=2 blocks (0,1).
        // Block 2 is still partial (only 1 token).
        // At 10,11,12 tokens: additional=0, no hash registration.
        // So block 2's hash is NOT registered yet.

        // Step 6: seal() — registers ALL unregistered full block hashes.
        tracker.seal(&inner);
        let inner_hashes = tracker.req_to_hashes.get("inner").unwrap();
        assert_eq!(
            inner_hashes.len(),
            3,
            "seal must register all 3 block hashes"
        );

        // Sealed+volatile: seal wins, use normal free (back of LRU).
        // (finish_single_request in scheduler uses free() not free_volatile()
        // when request.seal is true.)
        tracker.free("inner");

        // Outer request with same 12 tokens as prefix (+ more for its own query).
        let mut outer_tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 100, 101, 102, 2];
        outer_tokens.extend_from_slice(&[200, 201, 202, 203]); // outer's own content
        let outer = Request::new(
            "outer".into(),
            outer_tokens,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );

        let (num_computed, cached_blocks) = tracker.get_computed_blocks(&outer);
        assert_eq!(num_computed, 12, "all 12 inner tokens should be cached");
        assert_eq!(
            cached_blocks[0].len(),
            3,
            "3 blocks from inner should be hits"
        );
    }

    #[test]
    fn test_seal_non_aligned_only_caches_full_blocks() {
        // If SealPadProcessor didn't pad (e.g. request stopped before EOS),
        // seal() only registers full blocks. The partial last block is lost.
        let cfg = spans_config_enabled();
        let mut tracker = SimpleBlockTracker::with_spans_config(64, 4, cfg);

        // 6 tokens = 1 full block + 2 partial.
        let prompt: Vec<u32> = vec![10, 20, 30, 40, 50, 60];
        let mut req = Request::new(
            "r1".into(),
            prompt,
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        req.seal = true;
        tracker.allocate_slots(&req, 6, 0, &[]);

        // seal() registers only the 1 full block. Partial block is not cached.
        tracker.seal(&req);
        let hashes = tracker.req_to_hashes.get("r1").unwrap();
        assert_eq!(hashes.len(), 1, "only full block registered");

        tracker.free("r1");

        // Outer request starting with same 4 tokens → hits block 0.
        let req2 = Request::new(
            "r2".into(),
            vec![10, 20, 30, 40, 50, 60, 70, 80],
            SamplingParams {
                max_tokens: Some(10),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        );
        let (num_computed, _) = tracker.get_computed_blocks(&req2);
        assert_eq!(num_computed, 4, "only block 0 (full) cached");
    }

    #[test]
    fn test_seal_hash_matches_across_requests() {
        // Verify that hash_all_blocks produces identical hashes for the same
        // token sequences, regardless of how those tokens arrived (prompt vs
        // prompt+output). This is the fundamental invariant for cache hits.
        let cfg = spans_config_enabled();
        let tracker = SimpleBlockTracker::with_spans_config(64, 4, cfg);

        // Inner's all_token_ids: prompt [1,2,3,4,5,6,7,8] + output [100,2,0,0]
        let inner_tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 100, 2, 0, 0];
        let inner_hashes = tracker.hash_all_blocks(&inner_tokens, None, None);

        // Outer's prompt contains the same 12 tokens as a prefix.
        let outer_tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 100, 2, 0, 0, 200, 201, 202, 203];
        let outer_hashes = tracker.hash_all_blocks(&outer_tokens, None, None);

        // First 3 blocks should have identical hashes.
        assert_eq!(inner_hashes.len(), 3);
        assert_eq!(outer_hashes.len(), 4);
        for i in 0..3 {
            assert_eq!(
                inner_hashes[i], outer_hashes[i],
                "block {i} hash must match between inner and outer"
            );
        }
    }
}
