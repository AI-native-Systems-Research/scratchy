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
use scratchy_core_common::{
    BlockKind, HybridKvStores, KvBackedBlocks, KvBacking, Request, RequestStatus,
};
use scratchy_core_config::{SchedulerConfig, SchedulerPolicy, SpansConfig};
use tracing::warn;

use super::interface::{PauseState, SchedulerInterface};
use super::output::{
    CachedKvOp, CachedRequestData, NewRequestData, RecurrentStateOps, SchedulerOutput,
    SharedMemoryOps, StateSnapshotOp, StepId,
};
use super::recurrent_state::{SnapshotBudget, StateSnapshots};
use super::request_queue::{RequestQueue, SchedulingPolicy, create_request_queue};
use super::shared_memory::SharedMemory;

/// ⭐ HOW FAR BEFORE A PROMPT'S END ITS RECURRENT-STATE SNAPSHOT SITS.
///
/// The next turn of a chat re-renders the generation prompt that ends this one. Qwen3.5/3.6 end a
/// prompt with `<|im_start|>assistant\n<think>\n` (5 tokens; 7 with thinking off), and the next
/// turn's history renders that same turn as `<|im_start|>assistant\n` + the reply — so it diverges
/// 2-4 tokens before this prompt's end (1 in a tool loop). A snapshot inside that tail is never
/// reused, and the turn that needed it falls back to the turn before. 8 covers every case measured
/// on the template, at the price of recomputing at most 8 tokens more per turn, and it needs no
/// knowledge of the template — so `/v1/completions` and `scr chat` get it too.
pub const GENERATION_PROMPT_ALLOWANCE: usize = 8;

/// The most snapshots one request's chunk takes in a step: the record the worker hands the
/// recurrent kernels carries two (`GdnSeqSlots::saves`), and a prompt has two save points.
pub const MAX_STATE_SAVES_PER_STEP: usize = 2;

/// Where a request with recurrent state resumes on admission — see
/// [`KVCacheManagerOps::plan_recurrent_prefix`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentPrefixPlan {
    /// Tokens the request skips: the deepest block-aligned position, within its KV hit and
    /// before its last token, that has a READY snapshot. 0 = cold.
    pub tokens: u32,
    /// The snapshot to restore when `tokens > 0`: `(slot, key)`.
    pub restore: Option<(usize, u64)>,
    /// Every ready snapshot slot on this request's prefix at or below `tokens`, `restore`
    /// included — touched on admission so a conversation's older snapshots age with its newest.
    /// An agent loop needs that: once the next user message arrives, the template re-renders the
    /// loop's turns, and the snapshot that turn can use is the OLDEST one of the loop. Touched,
    /// not pinned: only `restore` is read this step, so the others stay evictable.
    pub prefix_slots: Vec<usize>,
    /// Positions whose state this request's prefill snapshots as its scan passes them,
    /// ascending: `(position, key)`.
    pub save_points: Vec<(u32, u64)>,
}

// ---------------------------------------------------------------------------
// KVCacheManagerOps -- trait for KV cache interaction
// ---------------------------------------------------------------------------

/// Trait that abstracts the KV cache manager operations needed by the
/// scheduler. Implemented by [`SimpleBlockTracker`], the scheduler's KV block
/// tracker.
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

    /// Notify the KV cache manager that a new scheduling step is starting, and number it.
    fn new_step_starts(&mut self) -> StepId;

    /// THE WORKER'S POOL-WIDE KV REACH, for the allocator that sizes a request nothing has run yet.
    ///
    /// Defaulted to a no-op so a tracker that does not need it says nothing — the value only means anything
    /// where a batched step appends every row at one shared slot.
    fn set_kv_pool_reach(&mut self, _reach: scratchy_core_common::KvSlotSpan) {}

    /// HOW MANY BLOCKS OF EACH KV STORE THE WORKER HAS BACKED, reported after each step by a worker that
    /// declared [`scratchy_core_common::KvBacking::OnFirstUse`]. Defaulted to a no-op: only the hybrid
    /// path tells a cached free block from a never-used one by it.
    fn set_kv_backed(&mut self, _backed: scratchy_core_common::KvBackedBlocks) {}

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

    /// ⭐ WHERE A REQUEST WITH RECURRENT STATE MAY RESUME, given its KV hit.
    ///
    /// `None` = this cache tracks no recurrent state, and the KV hit stands. `Some` = it does:
    /// the request resumes at `plan.tokens` (possibly 0) and NEVER at a KV hit past it, because
    /// its recurrent state exists only where a snapshot does. Pure — nothing is committed until
    /// [`Self::admit_recurrent_prefix`], after the request's blocks were allocated.
    fn plan_recurrent_prefix(
        &self,
        _request: &Request,
        _kv_hit_tokens: u32,
    ) -> Option<RecurrentPrefixPlan> {
        None
    }

    /// Keep what a plan from [`Self::plan_recurrent_prefix`] restores from through this step,
    /// BEFORE the request's blocks are allocated: an allocation may take memory from the
    /// snapshots, and must not take the one the request is about to resume from.
    fn hold_recurrent_prefix(&mut self, _request_id: &str, _plan: &RecurrentPrefixPlan) {}

    /// Commit a plan from [`Self::plan_recurrent_prefix`] for a request that was just admitted.
    fn admit_recurrent_prefix(&mut self, _request_id: &str, _plan: &RecurrentPrefixPlan) {}

    /// `request_id`'s chunk this step runs positions `(from, to]`: reserve a snapshot slot for
    /// each of its save points the chunk passes, where a slot is free or evictable. Every save
    /// point through `to` is consumed, reserved or not. At most [`MAX_STATE_SAVES_PER_STEP`].
    fn reserve_state_saves(
        &mut self,
        _request_id: &str,
        _from: u32,
        _to: u32,
    ) -> Vec<StateSnapshotOp> {
        Vec::new()
    }

    /// The step that carried `saves` reported back: those snapshots now hold their state.
    fn commit_state_saves(&mut self, _saves: &[StateSnapshotOp]) {}

    /// Step `step` came back, and every step before it came back or failed: each block registered
    /// in the prefix cache while they were scheduled holds the KV it was registered for.
    fn commit_written(&mut self, step: StepId);

    /// ⛔ A FAILED STEP'S BLOCKS ARE NO PREFIX-CACHE HITS. Step `failed` never came back, and
    /// `requests` ran in it. A block is registered when the step that writes it is scheduled, so
    /// the steps behind can hit it, and no step that came back proves any block one of them
    /// registered since `failed` was scheduled: each stops being a hit, though its request has been
    /// freed since. So does every block a request that took one of them from the cache registered:
    /// each of its keys attended KV nobody wrote. Returns `requests` and those, each once: none may
    /// run on.
    fn forget_unwritten(&mut self, failed: StepId, requests: &[String]) -> Vec<String>;

    /// The memory this step moves between KV and the snapshots (see [`SharedMemoryOps`]),
    /// taken once its scheduling is done.
    fn take_shared_memory_ops(&mut self) -> SharedMemoryOps {
        SharedMemoryOps::default()
    }
}

// ---------------------------------------------------------------------------
// SimpleBlockTracker -- the scheduler's KV block tracker
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

    /// The step being scheduled, or the last one if none is: what a block
    /// registered now is journaled under.
    step: StepId,

    /// ⭐ THE PREFIX-CACHE REGISTRATIONS NO STEP THAT CAME BACK HAS PROVEN
    /// WRITTEN, per step, oldest first: per request, each block it registered
    /// while the step was scheduled. Kept apart from the per-request maps,
    /// which a request's `free` erases while its blocks stay mapped, so a
    /// request preempted or aborted with a step in flight is still found here
    /// ([`KVCacheManagerOps::forget_unwritten`]). A step's entries go when it
    /// or a later step comes back ([`KVCacheManagerOps::commit_written`]).
    journal: VecDeque<StepJournal>,

    /// The last step that came back: steps come back in the order they were
    /// scheduled, which `journal` is dropped by.
    committed: StepId,

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
    /// allocated block ID stays ~`full_blocks + Σ window_blocks`** plus the
    /// kept prompt windows (`hybrid_kept`) and, where every block is allocated
    /// up front, the cache ended requests leave (`Freed::Cached`). This is the
    /// load-bearing adaptation to scratchy's substrate: lockstep
    /// `grow_to_cover` residency is keyed on the max block ID, so an unbounded
    /// (FIFO, free-to-back) ID would commit the whole pool resident and
    /// re-create the OOM this project exists to fix. One per block space
    /// (`SwaGroup::space`), each numbering its own blocks; empty on the non-SWA
    /// path. Per-block state (`ref_cnt`, `hybrid_id_to_key`, `hybrid_kept`,
    /// `free_seq`) is indexed by `space_slot`.
    hybrid_spaces: Vec<BlockSpace>,

    /// Per `space_slot`: the sequence number of the block's latest free on the
    /// hybrid path. A free-list entry holding another one is stale.
    free_seq: Vec<u64>,

    /// The next `free_seq` to hand out.
    next_free_seq: u64,

    /// How the worker backs the hybrid path's blocks with memory — declared
    /// when the tracker is built ([`Self::with_kv_backing`]).
    kv_backing: KvBacking,

    /// Hybrid prefix cache: namespaced per group. `(group_id, block_hash) ->
    /// block_id`. A full group and a sliding group with identical token content
    /// MUST NOT collide (different KV semantics: full keeps everything, sliding
    /// stores unrotated relocatable spans), so the group id is part of the key.
    /// Only used on the hybrid path; empty otherwise.
    hybrid_hash_to_id: HashMap<(u32, u64), usize>,

    /// Reverse map for lazy eviction on the hybrid path: `space_slot ->
    /// (group_id, block_hash)`. A block belongs to exactly one group's table at
    /// a time, so a single reverse entry suffices.
    hybrid_id_to_key: HashMap<usize, (u32, u64)>,

    /// Hybrid per-request block hashes, per group: `request_id ->
    /// Vec_by_group<GroupHashes>`. Extended as the request's tokens fill
    /// blocks, so each block is hashed and registered once.
    hybrid_req_hashes: HashMap<String, Vec<GroupHashes>>,

    /// Hybrid + caching: per running request, the sliding blocks that slid
    /// out of the window but cover its PROMPT's last window — what a re-sent
    /// prompt needs (#328). Out of the block tables (the worker never reads
    /// them) yet still referenced, so the lowest-free-first heap can't hand
    /// them to the reply's next blocks and evict their hashes. Like any cached
    /// block they may grow the store; they are freed with the request, or all
    /// at once by `release_kept_windows` when an allocation would otherwise
    /// fail: keeping a cache entry must never cost a preemption.
    hybrid_kept: HashMap<String, Vec<usize>>,

    /// Per `space_slot`: how many `hybrid_kept` entries name the block. Each is
    /// also one of its `ref_cnt`.
    kept_refs: Vec<usize>,

    /// Recurrent-state snapshots, for a model whose layers carry a per-sequence scan state
    /// (Gated-DeltaNet hybrids). `Some` makes EVERY prefix hit resume at a snapshot — see
    /// [`KVCacheManagerOps::plan_recurrent_prefix`]. `None` for every attention-only model.
    recurrent: Option<RecurrentTracking>,
}

/// [`SimpleBlockTracker`]'s recurrent-state bookkeeping.
struct RecurrentTracking {
    snapshots: StateSnapshots,
    /// `request_id -> (position, key)` save points still ahead of the request, ascending.
    /// Committed at admission, dropped when the request's blocks are freed.
    save_points: HashMap<String, Vec<(u32, u64)>>,
    /// The budget KV and the snapshots share, and what each has mapped of it.
    memory: SharedMemory,
    /// Per block: the step it was last freed or touched, which is how old its cached contents
    /// are when the free queue is about to evict them.
    block_age: Vec<u64>,
}

impl RecurrentTracking {
    /// The mapped snapshot unit used least recently that no step reads or writes now, and when
    /// it was used.
    fn lru_unit(&self) -> Option<(usize, u64)> {
        (self.memory.mapped_units())
            .filter_map(|u| (self.snapshots.age_of(self.memory.slots_of(u))).map(|a| (u, a)))
            .min_by_key(|&(u, a)| (a, u))
    }

    /// Release snapshot unit `u` to KV: its snapshots are gone.
    fn release_unit(&mut self, u: usize) {
        self.snapshots.evict_all(self.memory.slots_of(u));
        self.memory.release_unit(u);
    }

    /// A request let `block` go at the current step.
    fn give_back(&mut self, block: usize) {
        let now = self.snapshots.now();
        self.memory.give_back(block, now);
        self.block_age[block] = now;
    }
}

/// One request's full-block hashes in one hybrid group, and how many of its
/// blocks have been registered in `hybrid_hash_to_id`.
#[derive(Clone, Debug, Default)]
struct GroupHashes {
    hashes: Vec<u64>,
    registered: usize,
}

/// One step's journal entry (`SimpleBlockTracker::journal`).
#[derive(Debug)]
struct StepJournal {
    step: StepId,
    /// Per request, each block it registered in the prefix cache while the
    /// step was scheduled.
    registered: HashMap<String, Vec<(Origin, Registered)>>,
}

/// How a request came by a block it registered in the prefix cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Origin {
    /// Its own step writes the block's KV.
    Wrote,
    /// It took the block from the cache: the block's hash already mapped to
    /// it, and its KV is its writer's — the hit block a fully cached prompt
    /// recomputes, too.
    Took,
}

/// A block as a request registered it in the prefix cache: its KV-cache
/// group, its `space_slot`, and the hash it is cached under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Registered {
    group: u32,
    slot: usize,
    hash: u64,
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
    /// The block space this group's IDs number: the store its KV lives in
    /// (`HybridKvStores`). Groups in one store share a space.
    space: usize,
}

/// Where a freed block of the hybrid path waits to be handed out again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Freed {
    /// No request is likely to reuse it (never used, or slid out of a running
    /// request's window): handed out lowest ID first, so a sliding group
    /// recycles its own out-of-window blocks and the store's highest block
    /// stays near the working set.
    Recycled,
    /// The cache a request leaves when it ends: handed out only once no backed
    /// recycled block is left (`pop_free`), least recently freed first, so it
    /// outlives other requests' traffic while the store has room for it (#330).
    Cached,
}

/// One block space of the hybrid path: the block IDs one KV store is addressed
/// with (`SwaGroup::space`).
#[derive(Debug, Default)]
struct BlockSpace {
    /// `Freed::Recycled` IDs, lowest first, each with the `free_seq` it was
    /// freed under. Stale entries (the block was touched for reuse, or freed
    /// again since) are skipped on pop.
    free_min: BinaryHeap<Reverse<(usize, u64)>>,
    /// `Freed::Cached` IDs, least recently freed first, each with its
    /// `free_seq`. Stale entries are skipped the same way.
    cached_lru: VecDeque<(usize, u64)>,
    /// Under [`KvBacking::OnFirstUse`], the IDs the worker last reported backed
    /// (`KvBackedBlocks`): a recycled block past them maps memory a cached one
    /// does not, so the cached one goes first.
    backed: usize,
    /// IDs in use (`ref_cnt > 0`), the null block excluded.
    used: usize,
    /// IDs in use that only kept prompt windows hold
    /// (`SimpleBlockTracker::kept_only`): free once `release_kept_windows` runs.
    kept_only: usize,
}

impl SwaGroup {
    /// The blocks this group must still hold for a reused prefix of `p`
    /// tokens: all of `[0, p)` for a full group, only the window `[p - W, p)`
    /// for a sliding one (keeping the partial block straddling the window
    /// edge). The one rule the cache lookup checks and `remove_skipped_blocks`
    /// keeps.
    fn prefix_blocks(&self, p: usize) -> std::ops::Range<usize> {
        let lo = if self.is_sliding {
            p.saturating_sub(self.window) / self.block_size
        } else {
            0
        };
        lo..p.div_ceil(self.block_size)
    }

    /// The global group — the (first) full group, else group 0. Its block size
    /// aligns every reusable prefix.
    fn global_index(groups: &[SwaGroup]) -> usize {
        groups.iter().position(|g| !g.is_sliding).unwrap_or(0)
    }

    /// The longest prefix of a `num_tokens`-token request the hybrid lookup
    /// may reuse: global-block aligned and strictly short of `num_tokens`
    /// (vLLM's `num_tokens - 1` cap). The P the lookup checks the sliding
    /// windows at is then the P the scheduler runs, because the scheduler's
    /// full-hit back-off never fires on this path.
    fn max_reusable_prefix(num_tokens: usize, global_bs: usize) -> usize {
        num_tokens.saturating_sub(1) / global_bs * global_bs
    }
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
            step: StepId::default(),
            journal: VecDeque::new(),
            committed: StepId::default(),
            spans_config: SpansConfig::from_env(),
            swa_groups: None,
            hybrid_spaces: Vec::new(),
            free_seq: Vec::new(),
            next_free_seq: 1,
            kv_backing: KvBacking::Allocated,
            hybrid_hash_to_id: HashMap::new(),
            hybrid_id_to_key: HashMap::new(),
            hybrid_req_hashes: HashMap::new(),
            hybrid_kept: HashMap::new(),
            kept_refs: Vec::new(),
            recurrent: None,
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
    ///
    /// `stores` says which groups' KV shares memory: under
    /// [`HybridKvStores::GlobalApart`] the global group numbers its blocks in a
    /// space of its own, so neither store is mapped for the other's blocks. The
    /// spaces share the one `num_blocks` budget; each reserves its own block 0
    /// as its null block.
    pub fn enable_hybrid(
        &mut self,
        num_blocks: usize,
        groups: Vec<(bool, usize, usize)>,
        stores: HybridKvStores,
    ) {
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
        let mut groups: Vec<SwaGroup> = groups
            .into_iter()
            .map(|(is_sliding, window, block_size)| SwaGroup {
                is_sliding,
                window,
                block_size,
                space: 0,
            })
            .collect();
        if stores == HybridKvStores::GlobalApart {
            let global = SwaGroup::global_index(&groups);
            groups[global].space = 1;
        }
        let num_spaces = groups.iter().map(|g| g.space + 1).max().unwrap_or(1);
        self.total_blocks = num_blocks;
        self.ref_cnt = vec![0; num_spaces * num_blocks];
        self.kept_refs = vec![0; num_spaces * num_blocks];
        // Every block starts free under `free_seq` 0.
        self.free_seq = vec![0; num_spaces * num_blocks];
        // Block 0 of each space = null (held out of the free set); data blocks
        // are 1..num_blocks.
        self.hybrid_spaces = (0..num_spaces)
            .map(|_| BlockSpace {
                free_min: (1..num_blocks).map(|id| Reverse((id, 0))).collect(),
                ..Default::default()
            })
            .collect();
        self.num_free_blocks = num_blocks - 1;
        self.free_queue.clear();
        self.swa_groups = Some(groups);
        // NB: prefix caching is HONORED on the hybrid path (do NOT force off).
        // `with_caching` sets `enable_caching`; the hybrid get_computed_blocks /
        // allocate_slots compute a per-group, joint-P-correct cached prefix
        // (full group fully cached + each sliding group's in-window suffix
        // cached). When caching is off this is a no-op (the alloc tail is all
        // fresh), preserving the prior force-off behavior.
    }

    /// Highest currently-allocated (`ref_cnt > 0`) block ID in `space`, or 0
    /// if none. The hybrid residency invariant: after any number of decode
    /// steps of a sliding-only sequence this must stay near the window, so
    /// lockstep `grow_to_cover` (keyed on the max block ID) never commits the
    /// pool.
    pub fn max_allocated_block_id(&self, space: usize) -> usize {
        (1..self.total_blocks)
            .rev()
            .find(|&i| self.ref_cnt[self.space_slot(space, i)] > 0)
            .unwrap_or(0)
    }

    /// Free blocks in block space `space`. Every space holds the pool's
    /// `total_blocks` (block 0 its null block): the worker sizes each store to
    /// it, and one block of every store fits the KV budget, so a block of one
    /// store never counts against another's.
    fn space_free(&self, space: usize) -> usize {
        self.total_blocks - 1 - self.hybrid_spaces[space].used
    }

    /// Whether every reference to block `slot` is a kept prompt window's, so
    /// releasing the kept windows frees it.
    fn kept_only(&self, slot: usize) -> bool {
        self.kept_refs[slot] > 0 && self.kept_refs[slot] == self.ref_cnt[slot]
    }

    /// Apply `change` to the reference counts of block `slot` of block space
    /// `space`, keeping the space's `kept_only` tally in step.
    fn tallied(&mut self, space: usize, slot: usize, change: impl FnOnce(&mut Self)) {
        let was = usize::from(self.kept_only(slot));
        change(self);
        let is = usize::from(self.kept_only(slot));
        let tally = &mut self.hybrid_spaces[space].kept_only;
        *tally = *tally + is - was;
    }

    /// Where block `id` of block space `space` keeps its per-block state
    /// (`ref_cnt`, `hybrid_id_to_key`).
    fn space_slot(&self, space: usize, id: usize) -> usize {
        space * self.total_blocks + id
    }

    /// The `(space, block_id)` a `space_slot` names.
    fn slot_block(&self, slot: usize) -> (usize, usize) {
        (slot / self.total_blocks, slot % self.total_blocks)
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

    /// DECLARE HOW THE WORKER BACKS THE HYBRID PATH'S BLOCKS WITH MEMORY. Under
    /// [`KvBacking::OnFirstUse`] a never-used block past what the worker last
    /// reported backed ([`KVCacheManagerOps::set_kv_backed`]) is handed out
    /// only once no cached one is left, so the prefix cache never maps memory
    /// of its own; nothing is backed until the first report.
    ///
    /// ⛔ DECLARED, NOT INFERRED from whether a report has arrived, which is
    /// false until a step has run (see [`scratchy_core_common::KvAddressing`]).
    pub fn with_kv_backing(mut self, backing: KvBacking) -> Self {
        self.kv_backing = backing;
        self
    }

    /// ⭐⭐⭐ DECLARE THAT THE MODEL CARRIES RECURRENT STATE, AND HOW ITS SNAPSHOTS SHARE THE KV
    /// BUDGET — Gated-DeltaNet and Mamba-2 hybrids (Qwen3.5 / Qwen3.6, Nemotron-H).
    ///
    /// From then on a prefix hit resumes only where a snapshot of that state exists, and a
    /// request's prefill snapshots its state at the positions worth resuming from as its scan
    /// passes them. KV and the snapshots draw on one budget — every block of the pool, as with
    /// caching off — mapped as each needs it and handed between them ([`SharedMemory`]): nothing
    /// is mapped, so no block is free, until a step maps a granule of them.
    ///
    /// ⛔ ONLY WITH PREFIX CACHING ON, AND NOT ON THE SLIDING-WINDOW PATH. Without caching there is
    /// nothing to resume; the SWA path has its own hit logic, and no arch mixes the two yet.
    pub fn with_state_snapshots(mut self, budget: SnapshotBudget) -> Self {
        assert!(
            self.enable_caching && self.swa_groups.is_none(),
            "recurrent-state snapshots need prefix caching on the uniform KV path"
        );
        assert!(
            budget.max_slots > 0,
            "recurrent-state snapshots need at least one slot"
        );
        self.free_queue.clear();
        self.num_free_blocks = 0;
        self.recurrent = Some(RecurrentTracking {
            snapshots: StateSnapshots::new(budget.max_slots),
            save_points: HashMap::new(),
            memory: SharedMemory::new(budget, self.total_blocks),
            block_age: vec![0; self.total_blocks],
        });
        self
    }

    /// Drop the recurrent-state bookkeeping of a request whose blocks are being freed.
    fn forget_recurrent(&mut self, request_id: &str) {
        if let Some(rec) = self.recurrent.as_mut() {
            rec.save_points.remove(request_id);
            rec.snapshots.drop_pending_of(request_id);
        }
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

    /// Build, for each block in `blocks`, the list of image hashes whose
    /// placeholder range overlaps that block's token range (indexed from
    /// `blocks.start`). Returns `None` for text-only requests so the hash is
    /// byte-identical to the pre-MM behavior.
    fn build_mm_extras(
        mm: Option<&MultimodalData>,
        blocks: std::ops::Range<usize>,
        block_size: usize,
    ) -> Option<Vec<Vec<u64>>> {
        let mm = mm?;
        if mm.images.is_empty() || blocks.is_empty() {
            return None;
        }
        let mut per_block = vec![Vec::<u64>::new(); blocks.len()];
        for (i, ph) in mm.image_placeholders.iter().enumerate() {
            let img = match mm.images.get(i) {
                Some(img) => img,
                None => continue,
            };
            let start_blk = (ph.offset / block_size).max(blocks.start);
            let end_blk = (ph.offset + ph.length).div_ceil(block_size).min(blocks.end);
            if start_blk >= end_blk {
                continue;
            }
            let img_hash = Self::hash_image(img);
            for slot in &mut per_block[start_blk - blocks.start..end_blk - blocks.start] {
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
        let mut hashes = Vec::new();
        self.extend_block_hashes(&mut hashes, all_tokens, annotations, mm, block_size);
        hashes
    }

    /// Append to `hashes` — the hashes of `all_tokens`' first `hashes.len()`
    /// full blocks — the hashes of every full block after them, chained from
    /// the last. A request's tokens only ever grow, so a request hashes each
    /// block once.
    fn extend_block_hashes(
        &self,
        hashes: &mut Vec<u64>,
        all_tokens: &[u32],
        annotations: Option<&scratchy_core_common::BlockAnnotations>,
        mm: Option<&MultimodalData>,
        block_size: usize,
    ) {
        let num_full_blocks = all_tokens.len() / block_size;
        let first = hashes.len();
        if first >= num_full_blocks {
            return;
        }
        let mm_extras = Self::build_mm_extras(mm, first..num_full_blocks, block_size);
        hashes.reserve(num_full_blocks - first);
        let mut parent_hash = hashes.last().copied().unwrap_or(Self::NONE_HASH);
        for i in first..num_full_blocks {
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
            let extras: &[u64] = (mm_extras.as_ref())
                .map(|v| v[i - first].as_slice())
                .unwrap_or(&[]);
            let hash =
                self.hash_block_with_parent(parent_hash, block_tokens, tokens_before, kind, extras);
            parent_hash = hash;
            hashes.push(hash);
        }
    }

    /// Try to allocate `count` fresh block IDs from the free queue.
    ///
    /// Pops blocks from the front of `free_queue`, skipping stale entries
    /// (blocks with ref_cnt > 0, already reclaimed via `touch()`). When a
    /// popped block has a hash mapping, the mapping is removed (lazy eviction
    /// matching Python's `_maybe_evict_cached_block`).
    ///
    /// Sharing its budget with recurrent-state snapshots, the pool first maps what the
    /// allocation needs ([`Self::make_room_for_kv`]), and before each pop that would evict a
    /// cached block grows instead while there is room ([`Self::grow_before_evicting`]).
    fn allocate_fresh_blocks(&mut self, count: usize) -> Option<Vec<usize>> {
        if self.available_blocks() < count {
            return None;
        }
        self.make_room_for_kv(count);

        let mut ids = Vec::with_capacity(count);
        while ids.len() < count {
            self.grow_before_evicting();
            let block_id = self.free_queue.pop_front()?;

            // Skip stale entries: blocks reclaimed via touch() have ref_cnt > 0, and a granule
            // released to the snapshots leaves its blocks' entries behind.
            if self.ref_cnt[block_id] > 0 || !self.block_mapped(block_id) {
                continue;
            }

            // Set ref_cnt to 1 (allocated).
            self.ref_cnt[block_id] = 1;
            self.num_free_blocks -= 1;
            self.evict_cached(block_id);
            if let Some(rec) = self.recurrent.as_mut() {
                rec.memory.take(block_id, rec.snapshots.now());
            }
            ids.push(block_id);
        }
        Some(ids)
    }

    /// Lazy eviction: remove the hash mapping of `block_id` if it was cached. Matches Python's
    /// `_maybe_evict_cached_block`. Only the mapping that points at THIS block: a later request
    /// that computed the same hash into a block of its own moved the mapping there, and evicting
    /// the older copy must not take it along.
    ///
    /// ⭐ A SNAPSHOT GOES WITH THE BLOCK IT ENDS AT. Its key is that block's hash, and a hit
    /// reaches it only through the cached KV of its whole prefix: with the block gone it can never
    /// be hit again, and its slot — and in the end its unit's memory — is free for what can.
    fn evict_cached(&mut self, block_id: usize) {
        if let Some(hash) = self.block_id_to_hash.remove(&block_id)
            && self.block_hash_to_id.get(&hash) == Some(&block_id)
        {
            self.block_hash_to_id.remove(&hash);
            if let Some(rec) = self.recurrent.as_mut() {
                rec.snapshots.evict_key(hash);
            }
        }
    }

    /// Journal that request `id` registered `blocks` in the prefix cache, under the step being
    /// scheduled (`journal`).
    fn record(&mut self, id: &str, blocks: impl IntoIterator<Item = (Origin, Registered)>) {
        let mut blocks = blocks.into_iter().peekable();
        if blocks.peek().is_none() {
            return;
        }
        if self.journal.back().is_none_or(|j| j.step != self.step) {
            let step = self.step;
            (self.journal).push_back(StepJournal {
                step,
                registered: HashMap::new(),
            });
        }
        let journal = self.journal.back_mut().expect("pushed above");
        match journal.registered.get_mut(id) {
            Some(registered) => registered.extend(blocks),
            None => {
                journal.registered.insert(id.to_owned(), blocks.collect());
            }
        }
    }

    /// Stop `block` being a prefix-cache hit, if the cache still maps it as it was registered: a
    /// block recycled since is another request's now, and so is a mapping another moved.
    fn uncache(&mut self, block: Registered) {
        if self.swa_groups.is_some() {
            if self.hybrid_id_to_key.get(&block.slot) == Some(&(block.group, block.hash)) {
                self.evict_hybrid_cached(block.slot);
            }
        } else if self.block_id_to_hash.get(&block.slot) == Some(&block.hash) {
            self.evict_cached(block.slot);
        }
    }

    /// Whether `block_id`'s memory is mapped: always, unless KV shares its budget with the
    /// snapshots and the block's granule is not.
    fn block_mapped(&self, block_id: usize) -> bool {
        (self.recurrent.as_ref()).is_none_or(|rec| rec.memory.block_mapped(block_id))
    }

    /// ⭐ THE BLOCKS AN ALLOCATION CAN HAVE: the free ones, and — sharing the budget with the
    /// snapshots — those of the granules the headroom, and the memory of every snapshot unit not
    /// read or written this step, would map. Exactly what [`Self::make_room_for_kv`] reaches, so
    /// an allocation that passes this check cannot fail halfway.
    fn available_blocks(&self) -> usize {
        let Some(rec) = self.recurrent.as_ref() else {
            return self.num_free_blocks;
        };
        let mem = &rec.memory;
        let units = (mem.mapped_units())
            .filter(|&u| rec.snapshots.age_of(mem.slots_of(u)).is_some())
            .count() as u64;
        let mut bytes = mem.headroom() + units * mem.unit_bytes();
        let mut blocks = self.num_free_blocks;
        for g in mem.unmapped_granules() {
            match bytes.checked_sub(mem.granule_bytes(g)) {
                Some(left) => {
                    bytes = left;
                    blocks += mem.blocks_of(g).len();
                }
                None => break,
            }
        }
        blocks
    }

    /// ⭐ LIVE KV FIRST: map granules, lowest first, until `count` blocks are free — out of the
    /// headroom, then out of the snapshots, least recently used unit first. A request that needs
    /// a block outranks a snapshot nobody is resuming from this step. The caller checked
    /// [`Self::available_blocks`].
    fn make_room_for_kv(&mut self, count: usize) {
        while self.num_free_blocks < count {
            let Some(rec) = self.recurrent.as_mut() else {
                return;
            };
            let g = (rec.memory.unmapped_granules().next()).expect("checked by available_blocks");
            while rec.memory.headroom() < rec.memory.granule_bytes(g) {
                let (u, _) = rec.lru_unit().expect("checked by available_blocks");
                rec.release_unit(u);
            }
            self.map_granule(g);
        }
    }

    /// ⭐ GROW BEFORE EVICTING — the next pop would evict a cached block: map a granule instead
    /// while the budget has room for one, or while a snapshot unit is older than that block (one
    /// LRU over both kinds of cached state). A cached block is evicted only when it is the oldest
    /// thing there is to give up.
    fn grow_before_evicting(&mut self) {
        let front = loop {
            match self.free_queue.front() {
                Some(&b) if self.ref_cnt[b] > 0 || !self.block_mapped(b) => {
                    self.free_queue.pop_front();
                }
                front => break front.copied(),
            }
        };
        let cached = front.is_some_and(|b| self.block_id_to_hash.contains_key(&b));
        let Some(rec) = self.recurrent.as_mut() else {
            return;
        };
        if front.is_some() && !cached {
            return;
        }
        let Some(g) = rec.memory.unmapped_granules().next() else {
            return;
        };
        let need = rec.memory.granule_bytes(g);
        if rec.memory.headroom() < need {
            let Some(front) = front else {
                return;
            };
            let age = rec.block_age[front];
            // Only if the units older than the block free enough: releasing one for nothing
            // would throw a snapshot away and evict the block all the same.
            let mut older: Vec<(usize, u64)> = (rec.memory.mapped_units())
                .filter_map(|u| rec.snapshots.age_of(rec.memory.slots_of(u)).map(|a| (u, a)))
                .filter(|&(_, a)| a < age)
                .collect();
            older.sort_by_key(|&(u, a)| (a, u));
            let reach = older.len() as u64 * rec.memory.unit_bytes() + rec.memory.headroom();
            if reach < need {
                return;
            }
            for (u, _) in older {
                if rec.memory.headroom() >= need {
                    break;
                }
                rec.release_unit(u);
            }
        }
        self.map_granule(g);
    }

    /// Map granule `g`: its blocks are free, at the front of the queue so they go before any
    /// cached one.
    fn map_granule(&mut self, g: usize) {
        let Some(rec) = self.recurrent.as_mut() else {
            return;
        };
        rec.memory.map_granule(g);
        let blocks = rec.memory.blocks_of(g);
        self.num_free_blocks += blocks.len();
        for b in blocks.rev() {
            self.free_queue.push_front(b);
        }
    }

    /// Release granule `g` to the snapshots: no request holds a block of it, and its cached
    /// blocks are evicted. Their queue entries are skipped as stale.
    fn release_granule(&mut self, g: usize) {
        let Some(rec) = self.recurrent.as_mut() else {
            return;
        };
        let blocks = rec.memory.blocks_of(g);
        rec.memory.release_granule(g);
        self.num_free_blocks -= blocks.len();
        for b in blocks {
            debug_assert_eq!(
                self.ref_cnt[b], 0,
                "block {b} of released granule {g} is held"
            );
            self.evict_cached(b);
        }
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
            if let Some(rec) = self.recurrent.as_mut() {
                rec.memory.take(block_id, rec.snapshots.now());
            }
        }
        self.ref_cnt[block_id] += 1;
    }

    /// The IDs of block space `space` backed by memory: every one under
    /// [`KvBacking::Allocated`], those the worker last reported under
    /// [`KvBacking::OnFirstUse`].
    fn backed(&self, space: usize) -> usize {
        match self.kv_backing {
            KvBacking::Allocated => self.total_blocks,
            KvBacking::OnFirstUse => self.hybrid_spaces[space].backed,
        }
    }

    /// The next free block of block space `space` to hand out: the lowest
    /// recycled one while it is backed, then the least recently freed cached
    /// one, and only once none is left a recycled one the worker must back
    /// first. Skips STALE entries: a block touched for cache reuse
    /// (`touch_hybrid_block`) or freed again since keeps its old entry. Mirrors
    /// `allocate_fresh_blocks`'s lazy free-queue skip.
    fn pop_free(&mut self, space: usize) -> Option<usize> {
        let backed = self.backed(space);
        loop {
            let st = &mut self.hybrid_spaces[space];
            let recycle = matches!(st.free_min.peek(), Some(&Reverse((id, _))) if id < backed);
            let entry = if recycle || st.cached_lru.is_empty() {
                st.free_min.pop()?.0
            } else {
                st.cached_lru.pop_front()?
            };
            if self.live_free_entry(space, entry) {
                return Some(entry.0);
            }
        }
    }

    /// Pop `count` free block IDs of block space `space` (hybrid path), in
    /// `pop_free` order. Returns `None` if the space can't satisfy the request
    /// (callers capacity-check first; this is the defensive guard).
    fn alloc_hybrid_fresh(&mut self, space: usize, count: usize) -> Option<Vec<usize>> {
        if self.space_free(space) < count {
            return None;
        }
        let mut ids = Vec::with_capacity(count);
        while ids.len() < count {
            let id = self.pop_free(space)?;
            let slot = self.space_slot(space, id);
            self.ref_cnt[slot] = 1;
            self.hybrid_spaces[space].used += 1;
            self.evict_hybrid_cached(slot);
            ids.push(id);
        }
        Some(ids)
    }

    /// Lazy eviction on the hybrid path: if block `slot` is still hash-mapped,
    /// drop both directions so it can't be handed out as a stale cache hit
    /// (mirrors [`Self::evict_cached`] / Python `_maybe_evict_cached_block`).
    /// The key's mapping goes only if it still points at THIS block: a later
    /// request that computed the same key into a block of its own moved it.
    fn evict_hybrid_cached(&mut self, slot: usize) {
        let (_, id) = self.slot_block(slot);
        if let Some(key) = self.hybrid_id_to_key.remove(&slot)
            && self.hybrid_hash_to_id.get(&key) == Some(&id)
        {
            self.hybrid_hash_to_id.remove(&key);
        }
    }

    /// Return `blocks` of block space `space` to that space (hybrid path),
    /// skipping the null block. A block freed as [`Freed::Recycled`], or one
    /// with no cache entry, goes back into the space's lowest-free-first heap so
    /// a sliding group reuses its own low IDs immediately — the bound on the max
    /// block ID. One freed as [`Freed::Cached`] queues behind every earlier one,
    /// a request's last block first as vLLM frees them, so its prefix is
    /// evicted last.
    fn free_hybrid_blocks(&mut self, space: usize, blocks: &[usize], freed: Freed) {
        for &b in blocks.iter().rev() {
            if b == Self::NULL_BLOCK {
                continue;
            }
            let slot = self.space_slot(space, b);
            self.tallied(space, slot, |t| {
                debug_assert!(t.ref_cnt[slot] > 0, "double-free of block {b}");
                t.ref_cnt[slot] -= 1;
                if t.ref_cnt[slot] == 0 {
                    let seq = t.next_free_seq;
                    t.next_free_seq += 1;
                    t.free_seq[slot] = seq;
                    let cached = freed == Freed::Cached && t.hybrid_id_to_key.contains_key(&slot);
                    let space_state = &mut t.hybrid_spaces[space];
                    space_state.used -= 1;
                    if cached {
                        space_state.cached_lru.push_back((b, seq));
                    } else {
                        space_state.free_min.push(Reverse((b, seq)));
                    }
                }
            });
        }
        self.drop_stale_free_entries(space);
    }

    /// Drop block space `space`'s stale free-list entries once either list
    /// holds more than twice the space's blocks, so blocks touched and freed
    /// again and again never grow them without bound.
    fn drop_stale_free_entries(&mut self, space: usize) {
        let limit = 2 * self.total_blocks;
        if self.hybrid_spaces[space].cached_lru.len() > limit {
            let mut lru = std::mem::take(&mut self.hybrid_spaces[space].cached_lru);
            lru.retain(|&entry| self.live_free_entry(space, entry));
            self.hybrid_spaces[space].cached_lru = lru;
        }
        if self.hybrid_spaces[space].free_min.len() > limit {
            let mut heap = std::mem::take(&mut self.hybrid_spaces[space].free_min);
            heap.retain(|&Reverse(entry)| self.live_free_entry(space, entry));
            self.hybrid_spaces[space].free_min = heap;
        }
    }

    /// Whether free-list entry `(id, seq)` of block space `space` still names
    /// a free block's latest free.
    fn live_free_entry(&self, space: usize, (id, seq): (usize, u64)) -> bool {
        let slot = self.space_slot(space, id);
        self.ref_cnt[slot] == 0 && self.free_seq[slot] == seq
    }

    /// vLLM `SlidingWindowManager.remove_skipped_blocks`: free the blocks that
    /// have fallen out of group `g`'s window, overwriting their table slots with
    /// the null block. `total_computed` = `request.num_computed_tokens` (tokens
    /// already cached BEFORE this step's new tokens). Idempotent — the reverse
    /// scan stops at the first null, so re-calling with a larger skip only frees
    /// the newly-exposed lower-index blocks. No-op for full groups.
    ///
    /// With prefix caching on, the skipped blocks a re-sent prompt needs
    /// (`prefix_blocks` at the prompt's `max_reusable_prefix`) move to
    /// `hybrid_kept` instead of the free heap, where the lowest-free-first
    /// order would hand them to the reply's own next blocks and evict their
    /// hashes (#328).
    fn remove_skipped_blocks(
        &mut self,
        request: &Request,
        g: usize,
        grp: SwaGroup,
        global_bs: usize,
    ) {
        if !grp.is_sliding {
            return; // full attention never frees
        }
        // skipped_tokens = max(0, num_computed - window + 1)
        let total_computed = request.num_computed_tokens as usize;
        let num_skipped_tokens = (total_computed + 1).saturating_sub(grp.window);
        if num_skipped_tokens == 0 {
            return;
        }
        let keep = if self.enable_caching {
            grp.prefix_blocks(SwaGroup::max_reusable_prefix(
                request.num_prompt_tokens as usize,
                global_bs,
            ))
        } else {
            0..0
        };
        let mut removed: Vec<usize> = Vec::new();
        let mut kept: Vec<usize> = Vec::new();
        if let Some((tables, _)) = self.allocations.get_mut(&request.request_id)
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
                if keep.contains(&i) {
                    kept.push(blocks[i]);
                } else {
                    removed.push(blocks[i]);
                }
                blocks[i] = Self::NULL_BLOCK;
            }
        }
        self.free_hybrid_blocks(grp.space, &removed, Freed::Recycled);
        if !kept.is_empty() {
            let slots: Vec<usize> = (kept.into_iter())
                .map(|b| self.space_slot(grp.space, b))
                .collect();
            for &slot in &slots {
                self.tallied(grp.space, slot, |t| t.kept_refs[slot] += 1);
            }
            self.hybrid_kept
                .entry(request.request_id.clone())
                .or_default()
                .extend(slots);
        }
    }

    /// Free every request's `hybrid_kept` blocks. Their hashes stay mapped
    /// until the heap hands them out again.
    fn release_kept_windows(&mut self) {
        let kept: Vec<usize> = self.hybrid_kept.drain().flat_map(|(_, b)| b).collect();
        self.free_kept(kept);
    }

    /// Free kept blocks, given as `space_slot`s, as the cache they are.
    fn free_kept(&mut self, slots: Vec<usize>) {
        for slot in slots {
            let (space, id) = self.slot_block(slot);
            self.tallied(space, slot, |t| t.kept_refs[slot] -= 1);
            self.free_hybrid_blocks(space, &[id], Freed::Cached);
        }
    }

    /// The capacity check of both hybrid allocation paths: may an allocation
    /// take `demand[s]` free blocks of each block space `s`, `reuses` being the
    /// `space_slot`s of the cached blocks it takes over? Each space fits on its
    /// own. When the allocation would otherwise fail, the kept windows are
    /// released first if that lets it fit. A reused block that only kept
    /// windows hold is a revival once they are released, so it then counts
    /// against its space too.
    fn make_hybrid_room(&mut self, demand: &[usize], reuses: &[usize]) -> bool {
        let spaces = 0..demand.len();
        if !spaces.clone().any(|s| demand[s] > self.space_free(s)) {
            return true;
        }
        // Per space: the reuses a release turns into revivals.
        let mut revived = vec![0usize; demand.len()];
        for &slot in reuses.iter().filter(|&&slot| self.kept_only(slot)) {
            revived[self.slot_block(slot).0] += 1;
        }
        if spaces
            .clone()
            .any(|s| demand[s] + revived[s] > self.space_free(s) + self.hybrid_spaces[s].kept_only)
        {
            return false;
        }
        self.release_kept_windows();
        true
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
        let global_bs = groups[SwaGroup::global_index(&groups)].block_size;
        for (g, grp) in groups.iter().enumerate() {
            self.remove_skipped_blocks(request, g, *grp, global_bs);
        }

        // 2. New tail blocks needed per group + ONE atomic cross-group check.
        let mut new_per_group = vec![0usize; num_groups];
        let mut new_per_space = vec![0usize; self.hybrid_spaces.len()];
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
                new_per_space[grp.space] += new_per_group[g];
            }
        }
        if !self.make_hybrid_room(&new_per_space, &[]) {
            return None; // single atomic capacity check (vLLM)
        }

        // 3. Allocate + append at the tail (lowest-free-first).
        for (g, &n) in new_per_group.iter().enumerate() {
            if n > 0 {
                let fresh = self.alloc_hybrid_fresh(groups[g].space, n)?;
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
        self.register_hybrid_hashes(request);
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

        // The `space_slot`s of the plan's reused (non-null, non-miss) blocks.
        let reuses: Vec<usize> = needed_per_group
            .iter()
            .enumerate()
            .flat_map(|(g, &need)| {
                let plan = matched.get(g).map(|v| v.as_slice()).unwrap_or(&[]);
                let space = groups[g].space;
                (plan.iter().take(need).copied())
                    .filter(|&b| b != usize::MAX && b != Self::NULL_BLOCK)
                    .map(move |b| (space, b))
            })
            .map(|(space, b)| self.space_slot(space, b))
            .collect();
        let demand = self.hybrid_reuse_demand(groups, &needed_per_group, matched);
        if !self.make_hybrid_room(&demand, &reuses) {
            return None;
        }

        // Touch all reused blocks FIRST so they leave the free pool before any
        // fresh pop can recycle (and evict) them.
        for &slot in &reuses {
            let (space, id) = self.slot_block(slot);
            self.touch_hybrid_block(space, id);
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
                        let fresh = self.alloc_hybrid_fresh(groups[g].space, 1)?;
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
        self.register_hybrid_hashes(request);

        Some(tables)
    }

    /// Free blocks a reuse plan consumes, per block space: every slot that is
    /// NOT a reuse of a live (ref>0) block takes one (a fresh alloc OR a touch
    /// of a ref==0 cached block). NULL slots are free (the shared null block).
    fn hybrid_reuse_demand(
        &self,
        groups: &[SwaGroup],
        needed_per_group: &[usize],
        matched: &[Vec<usize>],
    ) -> Vec<usize> {
        let mut demand = vec![0usize; self.hybrid_spaces.len()];
        for (g, need) in needed_per_group.iter().enumerate() {
            let plan = matched.get(g).map(|v| v.as_slice()).unwrap_or(&[]);
            let space = groups[g].space;
            for i in 0..*need {
                match plan.get(i).copied() {
                    Some(b) if b == Self::NULL_BLOCK => {}
                    Some(b) if b != usize::MAX => {
                        if self.ref_cnt[self.space_slot(space, b)] == 0 {
                            demand[space] += 1; // reviving a cached (free) block
                        }
                    }
                    _ => demand[space] += 1, // miss → fresh
                }
            }
        }
        demand
    }

    /// Touch block `block_id` of block space `space` (ref-count), accounting for
    /// free-pool membership. The block leaves the lowest-free-first heap lazily
    /// (stale heap entries are skipped on pop since ref_cnt > 0).
    fn touch_hybrid_block(&mut self, space: usize, block_id: usize) {
        let slot = self.space_slot(space, block_id);
        self.tallied(space, slot, |t| {
            if t.ref_cnt[slot] == 0 {
                t.hybrid_spaces[space].used += 1;
            }
            t.ref_cnt[slot] += 1;
        });
    }

    /// Register group-namespaced full-block hashes for the request's current
    /// tables. Each block is hashed once, when the request's tokens fill it,
    /// and registered once, when it is also allocated, as the uniform path's
    /// `seal` does: full groups register every full block; sliding groups only
    /// blocks still in-window (NULL slots skipped). The tables are read in
    /// place, so a step that fills no block costs a few lookups.
    fn register_hybrid_hashes(&mut self, request: &Request) {
        if !self.enable_caching {
            return;
        }
        let id = &request.request_id;
        let (Some(groups), Some((tables, _))) =
            (self.swa_groups.as_deref(), self.allocations.get(id))
        else {
            return;
        };
        let all_tokens = &request.all_token_ids;
        let annotations = request.block_annotations.as_ref();
        let mm = request.mm_data.as_ref();
        let mut req_hashes = (self.hybrid_req_hashes.get_mut(id))
            .map(std::mem::take)
            .unwrap_or_default();
        req_hashes.resize_with(groups.len(), GroupHashes::default);
        // `(key, block, space_slot, origin)` of each block to register: one the key already maps
        // to was taken from the cache.
        let mut new_keys = Vec::new();
        for (g, grp) in groups.iter().enumerate() {
            let group = &mut req_hashes[g];
            self.extend_block_hashes(
                &mut group.hashes,
                all_tokens,
                annotations,
                mm,
                grp.block_size,
            );
            let table = &tables[g];
            let ready = table.len().min(group.hashes.len());
            let new = group.registered.min(ready)..ready;
            for (&bid, &hash) in table[new.clone()].iter().zip(&group.hashes[new]) {
                if bid == Self::NULL_BLOCK {
                    continue; // out-of-window: never register (KV is gone)
                }
                let key = (g as u32, hash);
                let origin = match self.hybrid_hash_to_id.get(&key) == Some(&bid) {
                    true => Origin::Took,
                    false => Origin::Wrote,
                };
                new_keys.push((key, bid, self.space_slot(grp.space, bid), origin));
            }
            group.registered = group.registered.max(ready);
        }
        let mut journaled = Vec::with_capacity(new_keys.len());
        for (key, bid, slot, origin) in new_keys {
            self.hybrid_hash_to_id.insert(key, bid);
            self.hybrid_id_to_key.insert(slot, key);
            let (group, hash) = key;
            journaled.push((origin, Registered { group, slot, hash }));
        }
        self.record(id, journaled);
        match self.hybrid_req_hashes.get_mut(id) {
            Some(kept) => *kept = req_hashes,
            None => {
                self.hybrid_req_hashes.insert(id.clone(), req_hashes);
            }
        }
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

        // The global group's contiguous hit prefix block-aligns the candidate P.
        let global_g = SwaGroup::global_index(&groups);
        let global_bs = groups[global_g].block_size;
        let global_hits = &group_hits[global_g];
        let global_contig = global_hits.iter().take_while(|h| h.is_some()).count();

        // Group g is admissible at P iff every block it needs for P is cached.
        let admissible = |g: usize, p: usize| -> bool {
            groups[g]
                .prefix_blocks(p)
                .all(|i| group_hits[g].get(i).is_some_and(|h| h.is_some()))
        };

        // Candidate P walks DOWN from the global contiguous prefix, block by
        // (global) block, until every group is admissible. Max admissible wins.
        let max_p = SwaGroup::max_reusable_prefix(all_tokens.len(), global_bs);
        let mut chosen_p = 0usize;
        for blocks in (0..=global_contig).rev() {
            let p = blocks * global_bs;
            if p > max_p {
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
            let lo_blk = grp.prefix_blocks(chosen_p).start;
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

    fn set_kv_backed(&mut self, backed: KvBackedBlocks) {
        let Some(groups) = &self.swa_groups else {
            return;
        };
        let global = groups[SwaGroup::global_index(groups)].space;
        debug_assert_eq!(
            matches!(backed, KvBackedBlocks::GlobalApart { .. }),
            self.hybrid_spaces.len() == 2,
            "a {backed:?} report for {} block spaces",
            self.hybrid_spaces.len()
        );
        for (space, st) in self.hybrid_spaces.iter_mut().enumerate() {
            let blocks = match backed {
                KvBackedBlocks::Shared(blocks) => blocks,
                KvBackedBlocks::GlobalApart { global: blocks, .. } if space == global => blocks,
                KvBackedBlocks::GlobalApart { pool, .. } => pool,
            };
            // A report one step old under async scheduling: what is backed only grows.
            st.backed = st.backed.max(blocks);
        }
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
        // ⭐ EVERY POSITION THIS STEP COMPUTES IS A PROMPT POSITION — a prompt chunk, which writes its keys
        // AT those positions, inside its own token count. Only a step past the prompt can append at a shared
        // write slot beyond the span, so only it adds `writing` to the span (#694: a chunk charged both ways
        // held a whole token budget of blocks past its prompt for life).
        let prompt_chunk = request.num_computed_tokens as usize + num_new_tokens
            <= request.num_prompt_tokens as usize;
        let needed = match request.kv_extent {
            // ⭐ …AND IT EXTENDS THE RUN THE WORKER REPORTED. The chunk writes `[computed, computed + n)` at
            // token positions only after a contiguous run that stops at or before `computed` — which every
            // report of a prefilling request is (equal when synchronous, one step behind under async). A report
            // whose token-addressed run already reaches PAST `computed` describes keys the scheduler is about to
            // recompute, i.e. a history the two sides disagree about, and keeps the conservative arithmetic.
            Some(e)
                if prompt_chunk
                    && e.cacheable_tokens().get() as usize
                        <= request.num_computed_tokens as usize =>
            {
                e.blocks_for_prompt_chunk(block, total_tokens, reach, request.kv_inflight_slots())
            }
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
                if fresh_needed + evictable_reuse > self.available_blocks() {
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
                let mut journaled = Vec::new();
                for i in 0..needed {
                    if let Some(bid) = reuse_at(i) {
                        all_ids.push(bid); // hash already registered by req1
                        if let Some(&hash) = all_hashes.get(i) {
                            let block = Registered {
                                group: 0,
                                slot: bid,
                                hash,
                            };
                            journaled.push((Origin::Took, block));
                        }
                    } else {
                        let bid = self.allocate_fresh_blocks(1)?[0];
                        all_ids.push(bid);
                        if i < num_full_blocks && i < all_hashes.len() {
                            let h = all_hashes[i];
                            self.block_hash_to_id.insert(h, bid);
                            self.block_id_to_hash.insert(bid, h);
                            let block = Registered {
                                group: 0,
                                slot: bid,
                                hash: h,
                            };
                            journaled.push((Origin::Wrote, block));
                        }
                    }
                }
                debug_assert_eq!(all_ids.len(), needed, "block table must be dense+ordinal");
                self.record(&request.request_id, journaled);

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
        let mut journaled = Vec::new();
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
                    let block = Registered {
                        group: 0,
                        slot: bid,
                        hash,
                    };
                    journaled.push((Origin::Wrote, block));
                }
            }

            if !hashes.is_empty() {
                self.req_to_hashes
                    .insert(request.request_id.clone(), hashes);
            }
        }

        // Return all block IDs for this request (single KV cache group).
        let result = entry.0.clone();
        self.record(&request.request_id, journaled);
        Some(result)
    }

    fn free(&mut self, request_id: &str) {
        self.forget_recurrent(request_id);
        if let Some((block_ids_groups, _num_blocks)) = self.allocations.remove(request_id) {
            // Remove per-request hash tracking (hashes stay in block_hash_to_id).
            self.req_to_hashes.remove(request_id);
            self.hybrid_req_hashes.remove(request_id);

            // Hybrid path: return each group's (non-null) blocks to its block
            // space as the request's cache. The null block is filtered.
            // Group-namespaced hash mappings persist (lazy eviction on realloc).
            if let Some(groups) = &self.swa_groups {
                let spaces: Vec<usize> = groups.iter().map(|g| g.space).collect();
                for (space, group) in spaces.into_iter().zip(&block_ids_groups) {
                    self.free_hybrid_blocks(space, group, Freed::Cached);
                }
                let kept = self.hybrid_kept.remove(request_id).unwrap_or_default();
                self.free_kept(kept);
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
                        if let Some(rec) = self.recurrent.as_mut() {
                            rec.give_back(bid);
                        }
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
        if self.swa_groups.is_some() {
            self.register_hybrid_hashes(request);
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
        // An aborted request's last chunk may still be in flight: sealed blocks are journaled too.
        let mut journaled = Vec::new();
        for i in hashes.len()..num_full_blocks {
            if i < all_hashes.len() && i < block_ids.len() {
                let hash = all_hashes[i];
                let bid = block_ids[i];
                self.block_hash_to_id.insert(hash, bid);
                self.block_id_to_hash.insert(bid, hash);
                hashes.push(hash);
                let block = Registered {
                    group: 0,
                    slot: bid,
                    hash,
                };
                journaled.push((Origin::Wrote, block));

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
        self.record(&request.request_id, journaled);
    }

    fn free_volatile(&mut self, request_id: &str) {
        // The lowest-free-first heap has no front to free to.
        if self.swa_groups.is_some() {
            return self.free(request_id);
        }
        self.forget_recurrent(request_id);
        if let Some((block_ids_groups, _num_blocks)) = self.allocations.remove(request_id) {
            self.req_to_hashes.remove(request_id);
            self.hybrid_req_hashes.remove(request_id);

            // Free blocks and push to the FRONT of the free queue so they
            // are evicted first (before non-volatile blocks).
            for group in &block_ids_groups {
                for &bid in group.iter().rev() {
                    debug_assert!(self.ref_cnt[bid] > 0, "double-free of block {bid}");
                    self.ref_cnt[bid] -= 1;
                    if self.ref_cnt[bid] == 0 {
                        self.num_free_blocks += 1;
                        self.free_queue.push_front(bid);
                        if let Some(rec) = self.recurrent.as_mut() {
                            rec.give_back(bid);
                        }
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

    fn new_step_starts(&mut self) -> StepId {
        if let Some(rec) = self.recurrent.as_mut() {
            rec.snapshots.tick();
        }
        self.step = self.step.next();
        self.step
    }

    fn reset_prefix_cache(&mut self) -> bool {
        if !self.enable_caching {
            return true;
        }
        if let Some(rec) = self.recurrent.as_mut() {
            rec.snapshots.clear();
            rec.save_points.clear();
        }
        self.block_hash_to_id.clear();
        self.block_id_to_hash.clear();
        self.req_to_hashes.clear();
        self.journal.clear();
        self.hybrid_hash_to_id.clear();
        self.hybrid_id_to_key.clear();
        self.hybrid_req_hashes.clear();
        // With no cache entry left, every cached free block is a recycled one.
        for st in &mut self.hybrid_spaces {
            let cached = std::mem::take(&mut st.cached_lru);
            st.free_min.extend(cached.into_iter().map(Reverse));
        }
        true
    }

    /// On the hybrid path, the tightest block space's: it is the one that
    /// limits what fits. Blocks only kept prompt windows hold count as free,
    /// like any cached block: an allocation releases them before it fails.
    fn num_free_blocks(&self) -> usize {
        if self.swa_groups.is_some() {
            return (0..self.hybrid_spaces.len())
                .map(|s| self.space_free(s) + self.hybrid_spaces[s].kept_only)
                .min()
                .unwrap_or(0);
        }
        self.available_blocks()
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
        1.0 - KVCacheManagerOps::num_free_blocks(self) as f64 / self.total_blocks as f64
    }

    fn num_cached_blocks(&self) -> usize {
        // Cached blocks = free blocks that have a hash mapping.
        if self.swa_groups.is_some() {
            return self
                .hybrid_id_to_key
                .keys()
                .filter(|&&slot| self.ref_cnt[slot] == 0 || self.kept_only(slot))
                .count();
        }
        self.block_id_to_hash
            .keys()
            .filter(|&&bid| self.ref_cnt[bid] == 0)
            .count()
    }

    fn plan_recurrent_prefix(
        &self,
        request: &Request,
        kv_hit_tokens: u32,
    ) -> Option<RecurrentPrefixPlan> {
        let rec = self.recurrent.as_ref()?;
        let bs = self.block_size;
        // ⛔ Span annotations RELOCATE KV, which a scan state cannot follow, and a vision prompt
        // runs its encoder only on its first prefill step: neither resumes nor snapshots.
        if request.block_annotations.is_some() || request.mm_data.is_some() {
            return Some(RecurrentPrefixPlan {
                tokens: 0,
                restore: None,
                prefix_slots: Vec::new(),
                save_points: Vec::new(),
            });
        }
        // The strict-prefix chain: each block's hash covers every token before it, which is the
        // identity a snapshot has. (No annotations or images reach here, so it is also exactly the
        // chain `get_computed_blocks` matched.)
        let hashes = self.hash_all_blocks(&request.all_token_ids, None, None);
        let kv_blocks = (kv_hit_tokens as usize / bs).min(hashes.len());
        // At least one token must still run: its logits are the next token.
        let resumable = kv_blocks.min(request.num_tokens().saturating_sub(1) / bs);
        // ⛔ A RESTORE IS HELD THROUGH ITS REQUEST'S ALLOCATION (`hold_recurrent_prefix`), so a
        // request whose blocks and that snapshot's memory together exceed the budget could never
        // be admitted beside it: it resumes cold instead, over the same shared KV.
        let blocks = request.num_tokens().div_ceil(bs) as u64;
        let fits =
            blocks * rec.memory.block_bytes() + rec.memory.unit_bytes() <= rec.memory.limit();
        let hit = (1..=resumable).rev().filter(|_| fits).find_map(|b| {
            rec.snapshots
                .ready_slot(hashes[b - 1])
                .map(|slot| (b, slot))
        });
        let hit_blocks = hit.map_or(0, |(b, _)| b);
        let tokens = (hit_blocks * bs) as u32;
        let prefix_slots = hashes[..hit_blocks]
            .iter()
            .filter_map(|&h| rec.snapshots.ready_slot(h))
            .collect();

        let mut save_points = Vec::new();
        // A pooling request finishes after one step and a volatile one is one-shot by contract:
        // nothing resumes from either, so neither takes a snapshot slot.
        if !request.is_pooling && !request.volatile {
            // Three positions are worth a snapshot: the prompt's own end, short of the generation
            // prompt the next turn re-renders; for a request that starts cold, where its KV match
            // with an earlier prompt ends — a prefix two requests share with no state to show for
            // it, where the next one to share it resumes; and the client's prompt-cache
            // breakpoints (`Request::cache_breakpoints`), where it declares the prompt stable for
            // the requests after it, wherever those diverge from this one. A request that restored
            // a snapshot has state on its prefix; past it lies the conversation it continues,
            // whose next turn its own end serves. A resumed request is planned from its PROMPT,
            // not its output so far: the prompt end is the snapshot that it — and its next turn —
            // can use.
            let end = (request.num_prompt_tokens as usize)
                .saturating_sub(GENERATION_PROMPT_ALLOWANCE)
                / bs;
            let junction = (tokens == 0).then_some(kv_blocks.min(end));
            let breakpoints = (request.cache_breakpoints.iter())
                .map(|&t| t as usize / bs)
                .filter(|&b| b <= end);
            let mut points: Vec<usize> = junction
                .into_iter()
                .chain(breakpoints)
                .chain([end])
                .collect();
            points.sort_unstable();
            points.dedup();
            let mut from = tokens as usize;
            for b in points {
                let position = b * bs;
                if position > from && !rec.snapshots.contains(hashes[b - 1]) {
                    save_points.push((position as u32, hashes[b - 1]));
                    from = position;
                }
            }
        }
        tracing::info!(
            "[CACHE] req={} recurrent state: resume at {} (KV hit {}), snapshot at {:?} ({}/{} slots ready)",
            request.request_id,
            tokens,
            kv_hit_tokens,
            save_points.iter().map(|&(p, _)| p).collect::<Vec<_>>(),
            rec.snapshots.num_ready(),
            rec.snapshots.num_slots(),
        );
        Some(RecurrentPrefixPlan {
            tokens,
            restore: hit.map(|(b, slot)| (slot, hashes[b - 1])),
            prefix_slots,
            save_points,
        })
    }

    fn hold_recurrent_prefix(&mut self, request_id: &str, plan: &RecurrentPrefixPlan) {
        let Some(rec) = self.recurrent.as_mut() else {
            return;
        };
        // The restored slot is READ before this step's forward, so it is pinned for the step;
        // its ancestors only age with it.
        for &slot in &plan.prefix_slots {
            rec.snapshots.touch(slot);
        }
        if let Some((slot, _)) = plan.restore {
            rec.snapshots.pin(slot, request_id);
        }
    }

    fn admit_recurrent_prefix(&mut self, request_id: &str, plan: &RecurrentPrefixPlan) {
        let Some(rec) = self.recurrent.as_mut() else {
            return;
        };
        if plan.save_points.is_empty() {
            rec.save_points.remove(request_id);
        } else {
            rec.save_points
                .insert(request_id.to_owned(), plan.save_points.clone());
        }
    }

    fn reserve_state_saves(
        &mut self,
        request_id: &str,
        from: u32,
        to: u32,
    ) -> Vec<StateSnapshotOp> {
        let Some(rec) = self.recurrent.as_mut() else {
            return Vec::new();
        };
        let Some(points) = rec.save_points.get_mut(request_id) else {
            return Vec::new();
        };
        // Consumed whether or not a slot is free: the scan passes them this step.
        let passed = points.iter().take_while(|&&(p, _)| p <= to).count();
        let passed: Vec<(u32, u64)> = points.drain(..passed).collect();
        if points.is_empty() {
            rec.save_points.remove(request_id);
        }
        (passed.into_iter())
            .filter(|&(position, _)| position > from)
            .take(MAX_STATE_SAVES_PER_STEP)
            .filter_map(|(position, key)| {
                let slot = self.reserve_snapshot(key, position, request_id)?;
                Some(StateSnapshotOp {
                    req_id: request_id.to_owned(),
                    slot,
                    position,
                    key,
                })
            })
            .collect()
    }

    fn commit_state_saves(&mut self, saves: &[StateSnapshotOp]) {
        if let Some(rec) = self.recurrent.as_mut() {
            for op in saves {
                rec.snapshots.commit(op.slot, op.key, &op.req_id);
            }
        }
    }

    fn commit_written(&mut self, step: StepId) {
        debug_assert!(
            step >= self.committed,
            "{step:?} came back after {:?}: steps come back in order",
            self.committed
        );
        self.committed = step;
        while self.journal.front().is_some_and(|j| j.step <= step) {
            self.journal.pop_front();
        }
    }

    fn forget_unwritten(&mut self, failed: StepId, requests: &[String]) -> Vec<String> {
        let mut doomed = requests.to_vec();
        let mut newly = doomed.clone();
        let mut gone = HashSet::new();
        while !newly.is_empty() {
            // Their own writes since `failed` was scheduled: no step that came back proves one.
            let mut wrote = Vec::new();
            for journal in (self.journal.iter_mut()).filter(|j| j.step >= failed) {
                for id in &newly {
                    let blocks = journal.registered.remove(id).unwrap_or_default();
                    wrote.extend(blocks.into_iter().filter(|(o, _)| *o == Origin::Wrote));
                }
            }
            for (_, block) in wrote {
                self.uncache(block);
                gone.insert(block);
            }
            // A request that took one of them from the cache: every key it computed attended it.
            let took: HashSet<&String> = (self.journal.iter())
                .filter(|j| j.step >= failed)
                .flat_map(|j| j.registered.iter())
                .filter(|(_, blocks)| {
                    (blocks.iter()).any(|(o, b)| *o == Origin::Took && gone.contains(b))
                })
                .map(|(id, _)| id)
                .collect();
            newly = took.into_iter().cloned().collect();
            doomed.extend(newly.iter().cloned());
        }
        doomed
    }

    fn take_shared_memory_ops(&mut self) -> SharedMemoryOps {
        (self.recurrent.as_mut()).map_or_else(SharedMemoryOps::default, |rec| rec.memory.take_ops())
    }
}

impl SimpleBlockTracker {
    /// ⭐ A SLOT FOR `key`'s NEW SNAPSHOT, written by `writer`'s step: an empty one already
    /// mapped; else a unit mapped for it ([`Self::map_snapshot_unit`]); else the slot of the
    /// snapshot it evicts. `None` when `key` has one, or every snapshot is read or written this
    /// step and no memory can be had.
    fn reserve_snapshot(&mut self, key: u64, position: u32, writer: &str) -> Option<usize> {
        let rec = self.recurrent.as_ref()?;
        if rec.snapshots.contains(key) {
            return None;
        }
        let slot = match rec.snapshots.empty_slot(|s| rec.memory.slot_mapped(s)) {
            Some(empty) => empty,
            None => match self.map_snapshot_unit(writer) {
                Some(first) => first,
                None => self.recurrent.as_ref()?.snapshots.victim(writer)?.0,
            },
        };
        self.recurrent
            .as_mut()?
            .snapshots
            .place(slot, key, position, writer);
        Some(slot)
    }

    /// Map a snapshot unit for a save by `writer`, if the budget has it: out of the headroom, or
    /// out of KV granules no request holds — oldest first, and only those older than the
    /// snapshot the save would otherwise evict (one LRU over both kinds of cached state). Taken
    /// only if they free a whole unit: a granule released for nothing is cache thrown away.
    /// The unit's first slot.
    fn map_snapshot_unit(&mut self, writer: &str) -> Option<usize> {
        let rec = self.recurrent.as_ref()?;
        let u = rec.memory.lowest_unmapped_unit()?;
        let need = rec.memory.unit_bytes();
        if rec.memory.headroom() < need {
            let evicts = rec.snapshots.victim(writer).map(|(_, age)| age);
            let mut reach = rec.memory.headroom();
            let mut take = Vec::new();
            for (g, age) in rec.memory.free_granules() {
                if reach >= need || evicts.is_some_and(|evicts| age >= evicts) {
                    break;
                }
                reach += rec.memory.granule_bytes(g);
                take.push(g);
            }
            if reach < need {
                return None;
            }
            for g in take {
                self.release_granule(g);
            }
        }
        let rec = self.recurrent.as_mut()?;
        rec.memory.map_unit(u);
        Some(rec.memory.slots_of(u).start)
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
        // ⛔ ITS BLOCKS WERE JUST FREED, SO IT HAS NO KV EXTENT. The worker's report described keys this
        // request no longer holds; kept, it sizes the re-admission as if they were still there (#694).
        request.kv_extent = None;

        // Sync to canonical requests map.
        if let Some(canonical) = self.requests.get_mut(&request.request_id) {
            canonical.status = RequestStatus::Preempted;
            canonical.num_computed_tokens = 0;
            canonical.spec_token_ids.clear();
            canonical.num_preemptions = request.num_preemptions;
            canonical.num_output_placeholders = 0;
            canonical.kv_extent = None;
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

    /// Records a verify step's acceptance for `request_id`, feeding the miss
    /// policy: the step count and its accepted drafts. The engine stops
    /// drafting for a request whose drafts stop paying (its
    /// `spec_drafts_stopped`).
    pub fn record_spec_acceptance(&mut self, request_id: &str, accepted: usize) {
        if let Some(request) = self.requests.get_mut(request_id) {
            request.spec_verify_steps = request.spec_verify_steps.saturating_add(1);
            request.spec_accepted_drafts = request
                .spec_accepted_drafts
                .saturating_add(accepted as u32);
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
    /// [`SimpleBlockTracker`]'s `allocate_slots` on the NEXT step.
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

    /// HOW MANY BLOCKS OF EACH KV STORE THE WORKER HAS BACKED WITH MEMORY, as it last reported —
    /// forwarded to the allocator, which hands out the prefix cache's free blocks before a never-used
    /// block past them.
    pub fn set_kv_backed(&mut self, backed: KvBackedBlocks) {
        self.kv_cache.set_kv_backed(backed);
    }

    /// Step `step` came back, so what it wrote holds: its recurrent-state snapshots may be resumed
    /// from, and the prefix-cache blocks registered while it was scheduled stay hits whatever a
    /// later step does. Called from the step's output, never at scheduling time: a step that fails
    /// never gets here ([`Self::forget_unwritten`]).
    pub fn commit_step(&mut self, step: &SchedulerOutput) {
        self.kv_cache
            .commit_state_saves(&step.recurrent_state.saves);
        self.kv_cache.commit_written(step.step);
    }

    /// ⛔ THE STEP `failed` SCHEDULED NEVER CAME BACK, so the KV it was to write may not exist. Its
    /// blocks went into the prefix cache when it was scheduled, for the steps behind it to hit, and
    /// now stop being hits ([`KVCacheManagerOps::forget_unwritten`]). Returns the requests that
    /// cannot run on: the step's own, and any a step behind it admitted onto those blocks. The
    /// caller aborts them.
    pub fn forget_unwritten(&mut self, failed: &SchedulerOutput) -> Vec<String> {
        let requests: Vec<String> = failed.num_scheduled_tokens.keys().cloned().collect();
        self.kv_cache.forget_unwritten(failed.step, &requests)
    }

    ///
    /// ⛔ ONLY FOR A REQUEST THAT IS RUNNING. A worker reports every request it still holds, and it is told
    /// only about FINISHED ones — so a preempted request, whose blocks are freed, keeps being reported, and
    /// writing that into its record would size its re-admission by KV it no longer has (#694).
    pub fn set_kv_extent(&mut self, request_id: &str, extent: scratchy_core_common::KvExtent) {
        if let Some(request) = self.requests.get_mut(request_id) {
            if request.status != RequestStatus::Running {
                return;
            }
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
        let mut state_restores: Vec<StateSnapshotOp> = Vec::new();
        let mut cached_kv: Vec<CachedKvOp> = Vec::new();
        let mut token_budget = self.max_num_scheduled_tokens;
        let mut scheduled_spec_decode_tokens: HashMap<String, Vec<u32>> = HashMap::new();

        if self.pause_state == PauseState::PausedAll {
            token_budget = 0;
        }

        let step = self.kv_cache.new_step_starts();

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

                // The queue decides ORDER; the request admitted is the canonical record. The queue
                // holds a copy taken when the request was queued, and a request preempted with a
                // decode step still in flight (async scheduling) receives that step's token AFTER —
                // on the canonical record (`append_output_tokens`), never on this copy. Admitting the
                // copy schedules one token short: that step is an intermediate chunk that samples
                // nothing, then `num_tokens == num_computed` leaves the request unschedulable until
                // the server's no-progress watchdog aborts it.
                let mut request = match self.waiting.pop_request() {
                    Some(queued) => self
                        .requests
                        .get(&queued.request_id)
                        .cloned()
                        .expect("every queued request has a canonical record"),
                    None => break,
                };
                let request_id = request.request_id.clone();

                // Get computed blocks from prefix cache. `cached_blocks[0]` is
                // the per-logical-block hit list (usize::MAX for misses) —
                // threaded into allocate_slots so span blocks are SHARED.
                let (num_cached_tokens, cached_blocks) =
                    self.kv_cache.get_computed_blocks(&request);
                // ⭐ RECURRENT STATE: THE KV HIT STANDS ONLY WHERE A SNAPSHOT OF THE STATE DOES.
                // The plan resumes the request at the deepest snapshot within the hit (or at 0).
                // The matched blocks past that point stay shared: the request runs their positions
                // again only to rebuild its recurrent state, and the worker leaves their KV as the
                // cache holds it (`CachedKvOp`) — a cached block is never rewritten, so every
                // request that reads it reads the same KV.
                let kv_hit_tokens = num_cached_tokens;
                let recurrent = self
                    .kv_cache
                    .plan_recurrent_prefix(&request, num_cached_tokens);
                if let Some(plan) = &recurrent {
                    self.kv_cache.hold_recurrent_prefix(&request_id, plan);
                }
                let num_cached_tokens = recurrent
                    .as_ref()
                    .map_or(num_cached_tokens, |plan| plan.tokens);

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
                        if let Some(plan) = &recurrent {
                            self.kv_cache.admit_recurrent_prefix(&request_id, plan);
                            if let Some((slot, key)) = plan.restore {
                                state_restores.push(StateSnapshotOp {
                                    req_id: request_id.clone(),
                                    slot,
                                    position: plan.tokens,
                                    key,
                                });
                            }
                            if kv_hit_tokens > num_computed_tokens {
                                cached_kv.push(CachedKvOp {
                                    req_id: request_id.clone(),
                                    through: kv_hit_tokens,
                                });
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

        // Recurrent-state saves, reserved only now that this step's set of requests is final (a
        // request preempted above runs nothing). In `running` order, so which save gets the last
        // free slot does not depend on hash-map iteration. A row that verifies drafts passes its
        // save points without taking them: its scan runs past what it keeps.
        let mut state_saves: Vec<StateSnapshotOp> = Vec::new();
        for request in &self.running {
            if let Some(&n) = num_scheduled_tokens.get(&request.request_id) {
                let to = request.num_computed_tokens + n as u32;
                let from = match scheduled_spec_decode_tokens.contains_key(&request.request_id) {
                    true => to,
                    false => request.num_computed_tokens,
                };
                let saves = self
                    .kv_cache
                    .reserve_state_saves(&request.request_id, from, to);
                state_saves.extend(saves);
            }
        }

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
            draft_req_ids: HashSet::new(),
            primes_only: false,
            spec_pipeline: Default::default(),
            recurrent_state: RecurrentStateOps {
                restores: state_restores,
                cached_kv,
                saves: state_saves,
                memory: self.kv_cache.take_shared_memory_ops(),
            },
            step,
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
    use super::super::output::{KvGranule, SnapshotUnit};
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

    // ----- Multi-wave chunked prefill -----

    #[test]
    fn test_long_prompt_prefills_as_full_budget_continuation_waves() {
        // The scheduler-worker contract for a prompt longer than one budget:
        // every wave after the first is a CACHED-request continuation carrying
        // a full multi-token slice (n > 1) — never a stream of 1-token steps.
        // A backend that cannot run a mid-prompt continuation wave efficiently
        // is a backend defect, not a reason to shrink the wave.
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 2048,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            ..Default::default()
        };
        // 4096 blocks × 16 = 65k tokens of pool: the 5874-token prompt (368
        // blocks) must never come near the never-fits guard.
        let mut sched = Scheduler::with_simple_blocks(&cfg, 8192, 4096, 16);

        sched.add_request(make_request("r1", 5874));

        // Wave 1: admission, budget-capped.
        let out = sched.schedule();
        assert_eq!(out.scheduled_new_reqs.len(), 1);
        assert_eq!(*out.num_scheduled_tokens.get("r1").unwrap(), 2048);
        assert!(sched.get_request("r1").unwrap().is_prefill_chunk);

        // Wave 2: the MID-PROMPT continuation — must carry n = 2048, not 1.
        let out = sched.schedule();
        assert_eq!(out.scheduled_cached_reqs.num_reqs(), 1);
        assert_eq!(out.scheduled_cached_reqs.req_ids[0], "r1");
        assert_eq!(out.scheduled_cached_reqs.num_computed_tokens[0], 2048);
        assert_eq!(*out.num_scheduled_tokens.get("r1").unwrap(), 2048);
        assert!(sched.get_request("r1").unwrap().is_prefill_chunk);

        // Wave 3: the remainder finishes the prompt.
        let out = sched.schedule();
        assert_eq!(out.scheduled_cached_reqs.num_reqs(), 1);
        assert_eq!(out.scheduled_cached_reqs.num_computed_tokens[0], 4096);
        assert_eq!(*out.num_scheduled_tokens.get("r1").unwrap(), 1778);
        assert!(!sched.get_request("r1").unwrap().is_prefill_chunk);

        // The prompt-final wave samples one token (the engine appends it);
        // the next step is a 1-token decode from the full context.
        sched.append_output_tokens("r1", &[42]);
        let out = sched.schedule();
        assert_eq!(out.scheduled_cached_reqs.num_reqs(), 1);
        assert_eq!(out.scheduled_cached_reqs.num_computed_tokens[0], 5874);
        assert_eq!(*out.num_scheduled_tokens.get("r1").unwrap(), 1);
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
        t.enable_hybrid(
            pool,
            vec![(false, 0, 32), (true, window, 16)],
            HybridKvStores::Shared,
        );
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

        let r2 = hybrid_req("r2", 129);
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

    // `caching_hybrid`, the global group's KV in a store of its own.
    fn caching_hybrid_apart(pool: usize, window: usize) -> SimpleBlockTracker {
        let mut t = SimpleBlockTracker::with_caching(pool, 16);
        t.enable_hybrid(
            pool,
            vec![(false, 0, 32), (true, window, 16)],
            HybridKvStores::GlobalApart,
        );
        t
    }

    #[test]
    fn test_hybrid_global_apart_numbers_each_store_from_one() {
        // Each store numbers its blocks densely from 1, so neither is addressed with the other's.
        let mut t = caching_hybrid_apart(64, 1024);
        let tables = t
            .allocate_slots(&hybrid_req("r1", 128), 128, 0, &[])
            .unwrap();
        assert_eq!(tables[0], vec![1, 2, 3, 4]);
        assert_eq!(tables[1], (1..=8).collect::<Vec<_>>());
        assert_eq!(t.max_allocated_block_id(1), 4);
        assert_eq!(t.max_allocated_block_id(0), 8);
        t.free("r1");
        assert_eq!(t.num_free_blocks(), 63, "no leak/double-free");
    }

    #[test]
    fn test_hybrid_global_apart_gives_each_store_the_pool() {
        // Each store holds the pool's 63 blocks, and a block of one never counts against the
        // other: a 128-token request takes 4 global + 8 sliding blocks, so 7 fit (56 sliding,
        // 28 global) where one shared count of 63 fit 5. The 8th doesn't fit the sliding store.
        let mut t = caching_hybrid_apart(64, 1024);
        let request = |i: u32| {
            let mut r = hybrid_req(&format!("r{i}"), 128);
            r.prompt_token_ids
                .iter_mut()
                .for_each(|tok| *tok += i * 1000);
            r.all_token_ids = r.prompt_token_ids.clone();
            r
        };
        for i in 0..7u32 {
            assert!(t.allocate_slots(&request(i), 128, 0, &[]).is_some(), "r{i}");
        }
        assert_eq!(t.num_free_blocks(), 63 - 56, "the sliding store's");
        assert!(t.allocate_slots(&request(7), 128, 0, &[]).is_none());
        assert_eq!(t.num_free_blocks(), 7, "the failed allocation took nothing");
        for i in 0..7u32 {
            t.free(&format!("r{i}"));
        }
        assert_eq!(t.num_free_blocks(), 63, "no leak/double-free");
    }

    #[test]
    fn test_hybrid_global_apart_reuses_a_cached_prefix() {
        // The same prompt + a bit more reuses the global group's prefix and the sliding group's
        // in-window suffix, each from its own store, and every block comes back once.
        let mut t = caching_hybrid_apart(512, 64);
        let mut r1 = hybrid_req("r1", 128);
        t.allocate_slots(&r1, 128, 0, &[]).unwrap();
        r1.num_computed_tokens = 128;
        t.seal(&r1);
        let r1_tables = t.get_blocks("r1");

        let mut r2 = hybrid_req("r2", 129);
        let (p, matched) = t.get_computed_blocks(&r2);
        assert_eq!(p, 128);
        assert_eq!(matched[0], r1_tables[0], "the global group's blocks");
        assert_eq!(
            matched[1][(128 - 64) / 16..],
            r1_tables[1][(128 - 64) / 16..]
        );
        r2.num_computed_tokens = p;
        let tables = t.allocate_slots(&r2, 1, 0, &matched).unwrap();
        assert_eq!(tables[0][..4], r1_tables[0][..]);
        t.free("r1");
        t.free("r2");
        assert_eq!(t.num_free_blocks(), 511, "no leak/double-free");
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

    /// `caching_hybrid` after an earlier request used the whole pool, as on a server that has run
    /// for a while: the worker's KV pool covers every block, so keeping a window costs no memory.
    fn caching_hybrid_after_full_pool(pool: usize, window: usize) -> SimpleBlockTracker {
        let mut t = caching_hybrid(pool, window);
        // One full block (bs 32) and two sliding blocks (bs 16) per 32 tokens.
        let n = (pool - 1) / 3 * 32;
        let mut earlier = hybrid_req("earlier", n);
        earlier
            .prompt_token_ids
            .iter_mut()
            .for_each(|tok| *tok += 900_000);
        earlier.all_token_ids = earlier.prompt_token_ids.clone();
        t.allocate_slots(&earlier, n, 0, &[]).unwrap();
        t.free("earlier");
        t
    }

    /// Prefill `prompt_len` tokens, decode `reply_len` more one at a time (sealing each step, as
    /// the engine does), free, then look the same prompt up again. Returns the reusable P.
    fn hybrid_reuse_after_reply(
        mut t: SimpleBlockTracker,
        prompt_len: usize,
        reply_len: usize,
    ) -> u32 {
        let mut r1 = hybrid_req("r1", prompt_len);
        t.allocate_slots(&r1, prompt_len, 0, &[]).unwrap();
        r1.num_computed_tokens = prompt_len as u32;
        t.seal(&r1);
        for i in 0..reply_len {
            r1.append_output_token_ids(&[10_000 + i as u32]);
            t.allocate_slots(&r1, 1, 0, &[]).unwrap();
            r1.num_computed_tokens += 1;
            t.seal(&r1);
        }
        t.free("r1");
        t.get_computed_blocks(&hybrid_req("r2", prompt_len)).0
    }

    #[test]
    fn test_hybrid_prompt_window_survives_long_reply() {
        // #328: the sliding group's blocks covering the prompt's last window slide out of the
        // window during decode. Kept out of the lowest-free-first heap, they survive a reply of
        // any length — on a freshly started pool too, which has to grow to hold both the reply's
        // window and the kept one — so the same prompt sent again reuses everything but its last
        // partial block.
        for stores in [HybridKvStores::Shared, HybridKvStores::GlobalApart] {
            let reuse = |reply_len| {
                let mut t = SimpleBlockTracker::with_caching(512, 16);
                t.enable_hybrid(512, vec![(false, 0, 32), (true, 64, 16)], stores);
                hybrid_reuse_after_reply(t, 200, reply_len)
            };
            assert_eq!(reuse(8), 192, "{stores:?} short reply");
            assert_eq!(reuse(300), 192, "{stores:?} long reply");
            assert_eq!(reuse(1000), 192, "{stores:?} reply 15 windows long");
        }
    }

    #[test]
    fn test_hybrid_kept_prompt_window_is_reported_free() {
        // A kept window is a cache entry an allocation takes back before it fails, so the
        // reported KV usage counts it free, like every other cached block. Through a long reply
        // on Gemma 4's group layout, the tightest store holds more blocks with caching on than
        // off, yet usage reads the same.
        for stores in [HybridKvStores::Shared, HybridKvStores::GlobalApart] {
            let run = |caching: bool| {
                let mut t = if caching {
                    SimpleBlockTracker::with_caching(512, 16)
                } else {
                    SimpleBlockTracker::new(512, 16)
                };
                let mut groups = vec![(false, 0usize, 32usize)];
                groups.extend(std::iter::repeat_n((true, 64usize, 16usize), 5));
                t.enable_hybrid(512, groups, stores);
                let mut r1 = hybrid_req("r1", 200);
                t.allocate_slots(&r1, 200, 0, &[]).unwrap();
                r1.num_computed_tokens = 200;
                t.seal(&r1);
                let mut steps = Vec::new();
                for i in 0..300u32 {
                    r1.append_output_token_ids(&[10_000 + i]);
                    t.allocate_slots(&r1, 1, 0, &[]).unwrap();
                    r1.num_computed_tokens += 1;
                    t.seal(&r1);
                    let tightest = t.hybrid_spaces.iter().map(|s| s.used).max().unwrap();
                    steps.push((t.num_free_blocks(), tightest));
                }
                steps
            };
            let (cached, uncached) = (run(true), run(false));
            assert!(
                cached.iter().zip(&uncached).any(|(c, u)| c.1 > u.1),
                "{stores:?} test setup: caching keeps the prompt's window in the tightest store"
            );
            for (i, (c, u)) in cached.iter().zip(&uncached).enumerate() {
                assert_eq!(
                    c.0, u.0,
                    "{stores:?} free blocks reported at reply token {i}"
                );
            }
        }
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
        tracker.enable_hybrid(pool, vec![(true, window, bs)], HybridKvStores::Shared); // one sliding group

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
            max_seen = max_seen.max(tracker.max_allocated_block_id(0));
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
    fn test_hybrid_held_prompt_window_stays_bounded() {
        // With caching on, the prompt's last window is kept (#328) — that must cost at most one
        // extra window, not grow with the reply.
        let bs = 16usize;
        let window = 64usize;
        let win_blocks = window.div_ceil(bs);
        let pool = 4096usize;
        let mut tracker = SimpleBlockTracker::with_caching(pool, bs);
        tracker.enable_hybrid(pool, vec![(true, window, bs)], HybridKvStores::Shared);

        let mut r = hybrid_req("r1", 200);
        tracker.allocate_slots(&r, 200, 0, &[]).expect("prefill");
        r.num_computed_tokens = 200;
        let mut max_seen = 0usize;
        for i in 0..4000u32 {
            r.append_output_token_ids(&[10_000 + i]);
            tracker.allocate_slots(&r, 1, 0, &[]).expect("decode");
            r.num_computed_tokens += 1;
            // Kept blocks are out of the table, so count the pool, not the table.
            let used = tracker.hybrid_spaces[0].used;
            assert!(
                used <= 2 * win_blocks + 3,
                "{used} blocks in use exceed window + kept prompt window at tok {i}",
            );
            max_seen = max_seen.max(tracker.max_allocated_block_id(0));
        }
        assert!(
            max_seen <= 2 * win_blocks + 16,
            "max allocated block id {max_seen} climbed past two windows + prefill slack"
        );

        tracker.free("r1");
        assert_eq!(tracker.num_free_blocks(), pool - 1, "no leak/double-free");
    }

    #[test]
    fn test_hybrid_kept_prompt_window_yields_to_an_allocation_that_needs_it() {
        // The kept window is only a cache entry: an allocation that fits the pool only without it
        // must get it, instead of failing and making the scheduler preempt someone.
        let pool = 64usize;
        let mut t = caching_hybrid_after_full_pool(pool, 64);
        let mut r1 = hybrid_req("r1", 200);
        t.allocate_slots(&r1, 200, 0, &[]).unwrap();
        r1.num_computed_tokens = 200;
        for i in 0..300u32 {
            r1.append_output_token_ids(&[10_000 + i]);
            t.allocate_slots(&r1, 1, 0, &[]).unwrap();
            r1.num_computed_tokens += 1;
        }
        let kept = t.hybrid_kept.get("r1").map_or(0, Vec::len);
        assert!(kept > 0, "a 300-token reply slides the prompt's window out");

        let free = t.space_free(0);
        assert_eq!(
            t.num_free_blocks(),
            free + kept,
            "the kept window is reported free"
        );
        // The smallest fresh prompt that needs more than the free pool, but no more than the
        // free pool plus the kept window.
        let need = |n: usize| n.div_ceil(32) + n.div_ceil(16);
        let n = (1..).find(|&n| need(n) > free).unwrap();
        assert!(
            need(n) <= free + kept,
            "test setup: r2 must fit once the window is released"
        );
        let mut r2 = hybrid_req("r2", n);
        r2.prompt_token_ids
            .iter_mut()
            .for_each(|tok| *tok += 50_000);
        r2.all_token_ids = r2.prompt_token_ids.clone();
        assert!(
            t.allocate_slots(&r2, n, 0, &[]).is_some(),
            "kept blocks must be released before an allocation fails"
        );
        assert!(t.hybrid_kept.is_empty(), "every kept window released");

        t.free("r1");
        t.free("r2");
        assert_eq!(t.num_free_blocks(), pool - 1, "no leak/double-free");
    }

    /// r1 decodes a 300-token reply past a 200-token prompt, so its prompt window is kept. Then
    /// unrelated prompts fill the pool until `leave_free` blocks are left.
    fn kept_window_then_fill(leave_free: Option<usize>) -> (SimpleBlockTracker, Vec<&'static str>) {
        let mut t = caching_hybrid_after_full_pool(64, 64);
        let mut r1 = hybrid_req("r1", 200);
        t.allocate_slots(&r1, 200, 0, &[]).unwrap();
        r1.num_computed_tokens = 200;
        t.seal(&r1);
        for i in 0..300u32 {
            r1.append_output_token_ids(&[10_000 + i]);
            t.allocate_slots(&r1, 1, 0, &[]).unwrap();
            r1.num_computed_tokens += 1;
            t.seal(&r1);
        }
        assert!(t.hybrid_kept.get("r1").is_some_and(|k| !k.is_empty()));
        let mut fillers = Vec::new();
        if let Some(k) = leave_free {
            let need = |n: usize| n.div_ceil(32) + n.div_ceil(16);
            let target = t.space_free(0) - k;
            let (a, b) = (1..2000usize)
                .flat_map(|a| (1..200usize).map(move |b| (a, b)))
                .find(|&(a, b)| need(a) + need(b) == target)
                .unwrap();
            for (id, n, off) in [("f1", a, 50_000u32), ("f2", b, 80_000u32)] {
                let mut f = hybrid_req(id, n);
                f.prompt_token_ids.iter_mut().for_each(|tok| *tok += off);
                f.all_token_ids = f.prompt_token_ids.clone();
                t.allocate_slots(&f, n, 0, &[]).unwrap();
                fillers.push(id);
            }
        }
        assert!(
            t.hybrid_kept.contains_key("r1"),
            "test setup: the fill must not release r1's window"
        );
        (t, fillers)
    }

    #[test]
    fn test_hybrid_resend_that_cannot_fit_takes_nothing() {
        // r2 re-sends r1's prompt while r1 still keeps its window, with one block free. Its plan
        // shares r1's kept blocks; released, they would become revivals r2 needs free blocks for,
        // so r2 fits neither way. It must fail before taking a block, and leave r1's window kept.
        let (mut t, fillers) = kept_window_then_fill(Some(1));
        let mut r2 = hybrid_req("r2", 200);
        let (p, matched) = t.get_computed_blocks(&r2);
        assert_eq!(p, 192);
        r2.num_computed_tokens = p;
        assert!(
            t.allocate_slots(&r2, 200 - p as usize, 0, &matched)
                .is_none()
        );
        assert!(
            t.hybrid_kept.contains_key("r1"),
            "releasing r1's window can't make room, so it stays kept"
        );
        assert_eq!(t.space_free(0), 1, "the failed allocation took nothing");
        for id in ["r1", "r2"].into_iter().chain(fillers) {
            t.free(id);
        }
        assert_eq!(t.num_free_blocks(), 63, "no leak/double-free");
    }

    #[test]
    fn test_hybrid_two_requests_keep_the_same_window() {
        // r2 re-sends r1's prompt while r1 keeps its window, then decodes far enough to keep the
        // same blocks itself. Whatever the pool and whichever request ends first, every block
        // comes back exactly once.
        for leave_free in [None, Some(2)] {
            for owner_first in [true, false] {
                let (mut t, fillers) = kept_window_then_fill(leave_free);
                let mut r2 = hybrid_req("r2", 200);
                let (p, matched) = t.get_computed_blocks(&r2);
                r2.num_computed_tokens = p;
                t.allocate_slots(&r2, 200 - p as usize, 0, &matched)
                    .unwrap();
                r2.num_computed_tokens = 200;
                t.seal(&r2);
                for i in 0..300u32 {
                    r2.append_output_token_ids(&[20_000 + i]);
                    if t.allocate_slots(&r2, 1, 0, &[]).is_none() {
                        break;
                    }
                    r2.num_computed_tokens += 1;
                    t.seal(&r2);
                }
                if leave_free.is_none() {
                    let r1_kept = &t.hybrid_kept["r1"];
                    assert!(
                        t.hybrid_kept["r2"].iter().any(|b| r1_kept.contains(b)),
                        "test setup: both requests keep the same blocks"
                    );
                }
                let order = if owner_first {
                    ["r1", "r2"]
                } else {
                    ["r2", "r1"]
                };
                for id in order.into_iter().chain(fillers) {
                    t.free(id);
                }
                assert_eq!(t.num_free_blocks(), 63, "no leak/double-free");
            }
        }
    }

    /// A `hybrid_req` whose `n` tokens start at `first`.
    fn hybrid_req_at(id: &str, n: usize, first: u32) -> Request {
        let mut r = hybrid_req(id, n);
        r.prompt_token_ids = (first..first + n as u32).collect();
        r.all_token_ids = r.prompt_token_ids.clone();
        r
    }

    /// Look `r` up, allocate the rest of its prompt and seal it, as one prefill step does.
    fn hybrid_prefill(t: &mut SimpleBlockTracker, mut r: Request) {
        let (p, matched) = t.get_computed_blocks(&r);
        r.num_computed_tokens = p;
        let rest = r.num_prompt_tokens - p;
        t.allocate_slots(&r, rest as usize, 0, &matched).unwrap();
        r.num_computed_tokens = r.num_prompt_tokens;
        t.seal(&r);
    }

    #[test]
    fn test_hybrid_ended_requests_cache_outlives_other_requests_blocks() {
        // #330: r1 ends and r2 takes blocks from the same stores. Lowest ID first handed r2 r1's
        // freed blocks and erased its cache. Where every block is allocated up front, r2 now takes
        // never-used blocks and r1's prompt is reused. Where blocks are mapped as the pool reaches
        // them, r2 spends r1's cache before any block past what is mapped, so the cache never maps
        // memory of its own.
        for backing in [KvBacking::Allocated, KvBacking::OnFirstUse] {
            let mut t = caching_hybrid_apart(512, 64).with_kv_backing(backing);
            hybrid_prefill(&mut t, hybrid_req("r1", 200));
            let backed = (
                t.max_allocated_block_id(0) + 1,
                t.max_allocated_block_id(1) + 1,
            );
            assert_eq!(backed, (14, 8), "r1's 13 sliding and 7 global blocks");
            t.set_kv_backed(KvBackedBlocks::GlobalApart {
                pool: backed.0,
                global: backed.1,
            });
            t.free("r1");
            hybrid_prefill(&mut t, hybrid_req_at("r2", 64, 50_000));
            let reach = (t.max_allocated_block_id(0), t.max_allocated_block_id(1));
            let reused = t.get_computed_blocks(&hybrid_req("r3", 200)).0;
            match backing {
                KvBacking::Allocated => {
                    assert_eq!(reused, 192, "r1's prompt");
                    assert_eq!(reach, (16, 8), "r2 took never-used blocks");
                }
                KvBacking::OnFirstUse => {
                    assert!(
                        reach.0 < backed.0 && reach.1 < backed.1,
                        "{reach:?} maps more"
                    );
                    assert!(reused < 192, "r2 spent r1's cache");
                }
            }
        }
    }

    #[test]
    fn test_hybrid_cache_hit_moves_blocks_to_the_back_of_the_lru() {
        // r1 ends, then r2; r3 extends r1's prompt, reusing its blocks, and ends last. A filler
        // that takes every never-used block and 4 cached ones evicts the least recently freed:
        // r2's, not the blocks r1 freed first and r3 freed again.
        let mut t = caching_hybrid_apart(40, 64);
        hybrid_prefill(&mut t, hybrid_req("r1", 64));
        t.free("r1");
        hybrid_prefill(&mut t, hybrid_req_at("r2", 64, 50_000));
        t.free("r2");
        let r3 = hybrid_req("r3", 96);
        assert_eq!(t.get_computed_blocks(&r3).0, 64, "r3 extends r1's prompt");
        hybrid_prefill(&mut t, r3);
        t.free("r3");
        // Of the sliding store's 39 blocks, r2 and r3 left 4 + 6 cached: 29 were never used.
        hybrid_prefill(&mut t, hybrid_req_at("filler", (29 + 4) * 16, 900_000));
        assert_eq!(
            t.get_computed_blocks(&hybrid_req("r1", 64)).0,
            32,
            "r1's prompt"
        );
        assert_eq!(
            t.get_computed_blocks(&hybrid_req_at("r2", 64, 50_000)).0,
            0,
            "r2's prompt, evicted first"
        );
    }

    #[test]
    fn test_hybrid_free_lists_stay_bounded_under_repeated_hits() {
        // Each turn hits the last one's cache and ends, freeing those blocks again and leaving their
        // older free-list entries stale. Dropped once a list passes twice the store's blocks, they
        // never pile up; resetting the prefix cache recycles every cached block.
        let mut t = caching_hybrid_apart(64, 64);
        for turn in 0..500 {
            let id = format!("turn{turn}");
            hybrid_prefill(&mut t, hybrid_req(&id, 96));
            t.free(&id);
            for st in &t.hybrid_spaces {
                let lens = (st.cached_lru.len(), st.free_min.len());
                assert!(lens.0 <= 128 && lens.1 <= 128, "turn {turn}: {lens:?}");
            }
        }
        assert_eq!(t.get_computed_blocks(&hybrid_req("again", 96)).0, 64);
        t.reset_prefix_cache();
        assert!(t.hybrid_spaces.iter().all(|st| st.cached_lru.is_empty()));
        assert_eq!(t.num_free_blocks(), 63, "no leak/double-free");
    }

    #[test]
    fn test_hybrid_groups_null_pad_and_distinctness() {
        // gemma4-shaped: 1 full (bs 32) + 2 sliding (window 64, bs 16) over ONE
        // shared pool. Per-group tables, null-padding of skipped head slots,
        // distinct in-window physical blocks (no aliasing), full never frees.
        let pool = 256usize;
        let mut t = SimpleBlockTracker::new(pool, 16);
        t.enable_hybrid(
            pool,
            vec![(false, 0, 32), (true, 64, 16), (true, 64, 16)],
            HybridKvStores::Shared,
        );

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
        t.enable_hybrid(demand + 1, groups.clone(), HybridKvStores::Shared);
        let mut r = swa_test_request("r1", 16);
        r.num_computed_tokens = 0;
        assert!(
            t.allocate_slots(&r, 320, 0, &[]).is_some(),
            "exact fit (incl null block)"
        );
        assert_eq!(t.num_free_blocks(), 0, "exact fit consumes the pool");

        // One block short → reject (atomic cross-group check, no partial alloc).
        let mut t2 = SimpleBlockTracker::new(demand, 16);
        t2.enable_hybrid(demand, groups, HybridKvStores::Shared);
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

    /// A request PREEMPTED WITH A DECODE STEP IN FLIGHT must resume and finish. Under async
    /// scheduling the next step is scheduled before the previous one's tokens arrive, so a request
    /// preempted at step N+1 still receives step N's token — on the canonical record only. Admitting
    /// the waiting queue's stale copy scheduled one token short, sampled nothing, and left the request
    /// unschedulable; the server's 60 s no-progress watchdog then aborted it. gemma-4-26b-shaped: one
    /// full group (bs 32) + five sliding groups (window 1024, bs 16) over the 2634-block pool its
    /// server sized, 12 concurrent 512-token prompts × 128 tokens — which overflows near the end, so
    /// the tail request is preempted mid-decode. The worker is modelled the way `InputBatch` runs: a
    /// request samples when its chunk reaches the end of the history the WORKER holds. Both store
    /// layouts share the one budget, so both overflow.
    #[test]
    fn test_hybrid_preempted_mid_decode_resumes_under_async_scheduling() {
        for stores in [HybridKvStores::Shared, HybridKvStores::GlobalApart] {
            preempted_mid_decode_resumes(stores);
        }
    }

    fn preempted_mid_decode_resumes(stores: HybridKvStores) {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 2048,
            max_num_seqs: 16,
            enable_chunked_prefill: true,
            async_scheduling: Some(true),
            ..Default::default()
        };
        // Shared, 12 requests' 20 global + 200 sliding blocks outgrow 2,633. Apart, each store
        // holds the pool, so the sliding store alone must outgrow it: 12 × 160 fit at admission,
        // 12 × 200 don't by the end of the replies.
        let pool = match stores {
            HybridKvStores::Shared => 2634usize,
            HybridKvStores::GlobalApart => 2300,
        };
        let mut tracker = SimpleBlockTracker::new(pool, 16);
        let mut groups = vec![(false, 0usize, 32usize)];
        groups.extend(std::iter::repeat_n((true, 1024usize, 16usize), 5));
        tracker.enable_hybrid(pool, groups, stores);
        let mut sched = Scheduler::new(&cfg, 32768, Box::new(tracker));
        let (num_reqs, prompt_len, max_tokens) = (12usize, 512u32, 128u32);
        for i in 0..num_reqs {
            sched.add_request(Request::new(
                format!("r{i}"),
                (0..prompt_len).collect(),
                SamplingParams {
                    max_tokens: Some(max_tokens),
                    ..Default::default()
                },
                i as f64,
                0,
                0,
                None,
            ));
        }

        // The worker's history length per request: prompt ++ every token sampled, in flight or not.
        let mut worker_history: HashMap<String, usize> = HashMap::new();
        let mut in_flight: Vec<String> = Vec::new();
        let mut preempted_mid_decode = false;
        for _ in 0..10_000 {
            let out = sched.schedule();
            preempted_mid_decode |= out.preempted_req_ids.as_ref().is_some_and(|p| {
                p.iter().any(|id| {
                    sched
                        .get_request(id)
                        .is_some_and(|r| r.num_output_tokens() > 0)
                })
            });
            let mut sampled = Vec::new();
            for (id, &n) in &out.num_scheduled_tokens {
                let r = sched.get_request(id).unwrap();
                let start = r.num_computed_tokens as usize - n;
                let history = worker_history
                    .entry(id.clone())
                    .or_insert(r.num_prompt_tokens as usize);
                if start + n >= *history {
                    *history += 1;
                    sampled.push(id.clone());
                }
            }
            // One-step lag: the PREVIOUS step's tokens land after this step was scheduled.
            for id in std::mem::replace(&mut in_flight, sampled) {
                if sched.get_request(&id).unwrap().status.is_finished() {
                    continue;
                }
                sched.append_output_tokens(&id, &[7]);
                if sched.get_request(&id).unwrap().num_output_tokens() >= max_tokens as usize {
                    sched.finish_requests(&[id.as_str()], RequestStatus::FinishedLengthCapped);
                }
            }
            if !sched.has_unfinished_requests() {
                break;
            }
        }

        assert!(
            preempted_mid_decode,
            "{stores:?}: the pool must overflow mid-decode, or this test does not exercise the resume"
        );
        let unfinished = sched.get_unfinished_request_ids();
        assert!(
            unfinished.is_empty(),
            "{stores:?}: requests stuck after a mid-decode preemption: {unfinished:?}"
        );
        for i in 0..num_reqs {
            let r = sched.get_request(&format!("r{i}")).unwrap();
            assert_eq!(
                r.num_output_tokens(),
                max_tokens as usize,
                "{stores:?}: r{i} must emit all its tokens"
            );
        }
    }

    #[test]
    fn test_hybrid_global_apart_maps_each_store_for_its_own_blocks() {
        // #317's 8 users × 4096/512 cell. Sharing one space, the global group's blocks interleave
        // with the sliding groups', so its store is addressed far past the blocks it holds, and the
        // shared pool past the sliding groups'. Apart, the global store's highest block is exactly
        // the global group's peak: 8 requests of 4,608 tokens hold 144 blocks of 32 each.
        let shared = run_317_cell(HybridKvStores::Shared, false, 27_000, CELL_4096_512);
        let apart = run_317_cell(HybridKvStores::GlobalApart, false, 27_000, CELL_4096_512);
        assert_eq!(apart.global_reach, 8 * (4096 + 512usize).div_ceil(32));
        assert_eq!(
            (shared.global_reach, shared.pool_reach, apart.pool_reach),
            (3153, 4376, 3240)
        );
    }

    #[test]
    fn evicting_an_older_copy_of_a_block_keeps_the_newer_ones_mapping() {
        // Two requests compute the same prefix into blocks of their own; the second's
        // registration moves the mapping. Recycling the FIRST copy must not delete it.
        let mut t = SimpleBlockTracker::with_caching(4, 16);
        let b1 = t
            .allocate_slots(&make_request("r1", 32), 32, 0, &[])
            .unwrap()[0]
            .clone();
        let b2 = t
            .allocate_slots(&make_request("r2", 32), 32, 0, &[])
            .unwrap()[0]
            .clone();
        assert_ne!(b1, b2);
        let hit =
            |t: &SimpleBlockTracker| t.get_computed_blocks(&make_request("q", 32)).1[0].clone();
        assert_eq!(hit(&t), b2, "the latest registration stands");
        t.free("r1");
        t.free("r2");
        // r1's blocks were freed first, so they are recycled first.
        let params = SamplingParams {
            max_tokens: Some(100),
            ..Default::default()
        };
        let other = Request::new("r3".into(), (900..932).collect(), params, 1.0, 0, 0, None);
        let b3 = t.allocate_slots(&other, 32, 0, &[]).unwrap()[0].clone();
        assert_eq!(b3, b1.iter().rev().copied().collect::<Vec<_>>());
        assert_eq!(hit(&t), b2);
    }

    #[test]
    fn evicting_an_older_copy_of_a_hybrid_block_keeps_the_newer_ones_mapping() {
        // The same scenario on the hybrid (SWA) path: two requests fill the
        // same tokens into blocks of their own, the second registration moves
        // the mapping, and recycling the FIRST copy must not delete it.
        let mut t = caching_hybrid(64, 1024); // big window: nothing out-of-window
        let r1 = hybrid_req("r1", 32);
        let r2 = hybrid_req("r2", 32);
        t.allocate_slots(&r1, 32, 0, &[]).unwrap();
        t.allocate_slots(&r2, 32, 0, &[]).unwrap();
        // One token past the cached 32: the lookup never reuses a whole request.
        let hit = |t: &SimpleBlockTracker| {
            let (p, tables) = t.get_computed_blocks(&hybrid_req("q", 33));
            (p, tables[0].clone())
        };
        let (p0, t0) = hit(&t);
        assert!(p0 > 0, "expected a hit before any eviction");
        t.free("r1");
        t.free("r2");
        // r1 holds the lowest IDs; the lowest-free-first heap recycles them first.
        let params = SamplingParams {
            max_tokens: Some(100),
            ..Default::default()
        };
        let other = Request::new("r3".into(), (900..932).collect(), params, 1.0, 0, 0, None);
        t.allocate_slots(&other, 32, 0, &[]).unwrap();
        let (p1, t1) = hit(&t);
        assert!(
            p1 > 0,
            "the newer copy's mapping must survive recycling the older copy"
        );
        assert_eq!(t1, t0);
    }

    // ----- A failed step's blocks (#759) -----

    /// The prefix-cache hit, in tokens, of a new request whose prompt is `0..len`.
    fn hit_of(t: &SimpleBlockTracker, len: usize) -> u32 {
        t.get_computed_blocks(&make_request("probe", len)).0
    }

    fn ids(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|id| id.to_string()).collect()
    }

    fn sorted(mut ids: Vec<String>) -> Vec<String> {
        ids.sort();
        ids
    }

    /// A step's blocks are registered when it is scheduled, for the steps behind it to hit. Once
    /// it fails they are no hits, and those of the step before it, which came back, still are.
    #[test]
    fn a_failed_steps_blocks_are_no_prefix_cache_hits() {
        let mut t = SimpleBlockTracker::with_caching(16, 16);
        let mut r = make_request("r1", 64);
        let ran = t.new_step_starts();
        t.allocate_slots(&r, 32, 0, &[]).unwrap();
        t.commit_written(ran);
        r.num_computed_tokens = 32;
        let failed = t.new_step_starts();
        t.allocate_slots(&r, 32, 0, &[]).unwrap();
        assert_eq!(
            hit_of(&t, 64),
            64,
            "the second chunk is a hit once scheduled"
        );

        assert_eq!(t.forget_unwritten(failed, &ids(&["r1"])), ["r1"]);
        assert_eq!(hit_of(&t, 64), 32, "the first chunk came back");
        let next = t.new_step_starts();
        t.commit_written(next);
        assert!(
            t.journal.is_empty(),
            "a step that comes back drops what it proved"
        );
    }

    /// A request a step behind the failed one admitted onto its blocks attended KV nobody wrote:
    /// it fails with it, and the blocks it wrote are no hits either. A bystander's still are.
    #[test]
    fn a_request_admitted_onto_a_failed_steps_blocks_fails_with_it() {
        let mut t = SimpleBlockTracker::with_caching(32, 16);
        let failed = t.new_step_starts();
        t.allocate_slots(&make_request("r1", 64), 64, 0, &[])
            .unwrap();
        t.new_step_starts();
        let mut q = make_request("q", 96);
        let (hit, blocks) = t.get_computed_blocks(&q);
        assert_eq!(hit, 64, "q is admitted onto r1's blocks");
        q.num_computed_tokens = hit;
        let q_blocks = t.allocate_slots(&q, 32, 0, &blocks).unwrap()[0].clone();
        let params = SamplingParams {
            max_tokens: Some(100),
            ..Default::default()
        };
        let other = Request::new("b".into(), (900..932).collect(), params, 1.0, 0, 0, None);
        t.allocate_slots(&other, 32, 0, &[]).unwrap();
        assert!(t.block_id_to_hash.contains_key(&q_blocks[4]));

        let failed = sorted(t.forget_unwritten(failed, &ids(&["r1"])));
        assert_eq!(failed, ["q", "r1"]);
        assert_eq!(hit_of(&t, 64), 0);
        assert!(
            !t.block_id_to_hash.contains_key(&q_blocks[4]),
            "q's own block"
        );
        assert_eq!(t.get_computed_blocks(&other).0, 32, "the bystander's");
    }

    /// A request freed with its step in flight, preempted or aborted, leaves its blocks mapped:
    /// its step failing finds them all the same, and the request admitted onto them.
    #[test]
    fn a_request_freed_before_its_step_failed_is_forgotten_all_the_same() {
        let mut t = SimpleBlockTracker::with_caching(32, 16);
        let failed = t.new_step_starts();
        t.allocate_slots(&make_request("r1", 64), 64, 0, &[])
            .unwrap();
        t.new_step_starts();
        t.free("r1");
        let mut q = make_request("q", 96);
        let (hit, blocks) = t.get_computed_blocks(&q);
        assert_eq!(hit, 64, "r1's freed blocks are still hits");
        q.num_computed_tokens = hit;
        t.allocate_slots(&q, 32, 0, &blocks).unwrap();

        let failed = sorted(t.forget_unwritten(failed, &ids(&["r1"])));
        assert_eq!(failed, ["q", "r1"]);
        assert_eq!(hit_of(&t, 64), 0);
    }

    /// A failed request's blocks registered in a step behind its failed one go too: their KV was
    /// the failed step's to write.
    #[test]
    fn a_failed_requests_blocks_registered_behind_its_step_go_too() {
        let mut t = SimpleBlockTracker::with_caching(16, 16);
        let mut r = make_request("r1", 63);
        let ran = t.new_step_starts();
        t.allocate_slots(&r, 63, 0, &[]).unwrap();
        t.commit_written(ran);
        r.num_computed_tokens = 63;
        // It writes position 63, whose token the step before it has just sampled.
        let failed = t.new_step_starts();
        t.allocate_slots(&r, 1, 0, &[]).unwrap();
        r.num_computed_tokens = 64;
        r.append_output_token_ids(&[7]);
        // The step behind it fills block 3 in the request's tokens, so registers it.
        t.new_step_starts();
        t.allocate_slots(&r, 1, 0, &[]).unwrap();
        let params = SamplingParams {
            max_tokens: Some(100),
            ..Default::default()
        };
        let probe = Request::new("p".into(), r.all_token_ids.clone(), params, 1.0, 0, 0, None);
        assert_eq!(t.get_computed_blocks(&probe).0, 64);

        assert_eq!(t.forget_unwritten(failed, &ids(&["r1"])), ["r1"]);
        assert_eq!(t.get_computed_blocks(&probe).0, 48);
    }

    /// A prompt cached in full is admitted a block short, and its step recomputes that block in
    /// place. The block is its writer's: the step failing leaves it cached, and its writer runs on.
    #[test]
    fn a_failed_step_leaves_the_cached_block_it_recomputes() {
        let mut t = SimpleBlockTracker::with_caching(16, 16);
        let ran = t.new_step_starts();
        t.allocate_slots(&make_request("p", 64), 64, 0, &[])
            .unwrap();
        t.commit_written(ran);
        let failed = t.new_step_starts();
        let mut r = make_request("r1", 64);
        let (_, blocks) = t.get_computed_blocks(&r);
        r.num_computed_tokens = 48;
        t.allocate_slots(&r, 16, 0, &blocks).unwrap();

        assert_eq!(
            t.forget_unwritten(failed, &ids(&["r1"])),
            ["r1"],
            "p runs on"
        );
        assert_eq!(hit_of(&t, 64), 64);
    }

    /// The hybrid path's lookup, one token past a cached `0..128` prompt.
    fn hybrid_hit(t: &SimpleBlockTracker) -> u32 {
        t.get_computed_blocks(&hybrid_req("probe", 129)).0
    }

    #[test]
    fn a_failed_steps_blocks_are_no_hybrid_prefix_cache_hits() {
        for stores in BOTH_STORES {
            let mut t = SimpleBlockTracker::with_caching(64, 16);
            t.enable_hybrid(64, vec![(false, 0, 32), (true, 1024, 16)], stores);
            let mut r = hybrid_req("r1", 128);
            let ran = t.new_step_starts();
            t.allocate_slots(&r, 64, 0, &[]).unwrap();
            t.commit_written(ran);
            r.num_computed_tokens = 64;
            let failed = t.new_step_starts();
            t.allocate_slots(&r, 64, 0, &[]).unwrap();
            assert_eq!(hybrid_hit(&t), 128, "{stores:?}");

            assert_eq!(t.forget_unwritten(failed, &ids(&["r1"])), ["r1"]);
            assert_eq!(hybrid_hit(&t), 64, "{stores:?}");
        }
    }

    /// A gemma-4-shaped tracker with a window no prompt here leaves: one full group (bs 32) and
    /// one sliding group (bs 16).
    fn wide_window_hybrid(stores: HybridKvStores) -> SimpleBlockTracker {
        let mut t = SimpleBlockTracker::with_caching(64, 16);
        t.enable_hybrid(64, vec![(false, 0, 32), (true, 1024, 16)], stores);
        t
    }

    /// Whether request `id`'s block `i` of group `g` is still mapped in the prefix cache.
    fn hybrid_mapped(t: &SimpleBlockTracker, id: &str, g: usize, i: usize) -> bool {
        let space = t.swa_groups.as_ref().unwrap()[g].space;
        let slot = t.space_slot(space, t.get_blocks(id)[g][i]);
        t.hybrid_id_to_key.contains_key(&slot)
    }

    /// The hybrid path's blocks taken from the cache: a request admitted onto a failed step's
    /// blocks is found by them and fails with it, and the blocks it wrote are no hits either.
    #[test]
    fn a_request_admitted_onto_a_failed_steps_hybrid_blocks_fails_with_it() {
        for stores in BOTH_STORES {
            let mut t = wide_window_hybrid(stores);
            let failed = t.new_step_starts();
            t.allocate_slots(&hybrid_req("r1", 128), 128, 0, &[])
                .unwrap();
            t.new_step_starts();
            let mut q = hybrid_req("q", 192);
            let (hit, blocks) = t.get_computed_blocks(&q);
            assert_eq!(hit, 128, "{stores:?}: q is admitted onto r1's blocks");
            q.num_computed_tokens = hit;
            t.allocate_slots(&q, 64, 0, &blocks).unwrap();
            assert!(hybrid_mapped(&t, "q", 0, 4), "{stores:?}");

            let failed = sorted(t.forget_unwritten(failed, &ids(&["r1"])));
            assert_eq!(failed, ["q", "r1"], "{stores:?}");
            assert_eq!(hybrid_hit(&t), 0, "{stores:?}");
            assert!(!hybrid_mapped(&t, "q", 0, 4), "{stores:?}: q's own block");
        }
    }

    /// The hybrid path's blocks taken from the cache are their writer's: a failed request's step
    /// leaves them cached, and their writer runs on.
    #[test]
    fn a_failed_step_leaves_the_hybrid_blocks_it_took() {
        for stores in BOTH_STORES {
            let mut t = wide_window_hybrid(stores);
            let ran = t.new_step_starts();
            t.allocate_slots(&hybrid_req("p", 128), 128, 0, &[])
                .unwrap();
            t.commit_written(ran);
            let failed = t.new_step_starts();
            let mut r = hybrid_req("r1", 128);
            let (hit, blocks) = t.get_computed_blocks(&r);
            assert_eq!(hit, 96, "{stores:?}: never a whole request");
            r.num_computed_tokens = hit;
            t.allocate_slots(&r, 32, 0, &blocks).unwrap();

            assert_eq!(
                t.forget_unwritten(failed, &ids(&["r1"])),
                ["r1"],
                "p runs on"
            );
            assert!((0..3).all(|i| hybrid_mapped(&t, "p", 0, i)), "{stores:?}");
            assert!((0..6).all(|i| hybrid_mapped(&t, "p", 1, i)), "{stores:?}");
            assert_eq!(hybrid_hit(&t), 96, "{stores:?}");
        }
    }

    /// A block of the failed step can slide out of its request's window before the step fails,
    /// when the step behind it was scheduled past the window: out of the table, still mapped. It
    /// goes too, whether freed or kept as the prompt's last window.
    #[test]
    fn a_failed_steps_block_that_slid_out_of_the_window_is_no_hit() {
        let mut t = caching_hybrid(64, 64);
        let mut r = hybrid_req("r1", 256);
        let ran = t.new_step_starts();
        t.allocate_slots(&r, 128, 0, &[]).unwrap();
        t.commit_written(ran);
        r.num_computed_tokens = 128;
        let failed = t.new_step_starts();
        t.allocate_slots(&r, 128, 0, &[]).unwrap();
        // The sliding group's blocks of the failed step: tokens 128..256, blocks 8..16.
        let hashes = &t.hybrid_req_hashes["r1"][1].hashes;
        let written: Vec<((u32, u64), usize)> = (8..16)
            .map(|i| ((1, hashes[i]), t.get_blocks("r1")[1][i]))
            .collect();
        t.new_step_starts();
        r.num_computed_tokens = 256;
        r.append_output_token_ids(&[7]);
        t.allocate_slots(&r, 1, 0, &[]).unwrap();
        let tables = t.get_blocks("r1");
        assert!((tables[1][8..12].iter()).all(|&b| b == SimpleBlockTracker::NULL_BLOCK));
        let mapped = |t: &SimpleBlockTracker| {
            (written.iter())
                .filter(|(key, bid)| t.hybrid_hash_to_id.get(key) == Some(bid))
                .count()
        };
        assert!(mapped(&t) > 4, "some slid-out block is still mapped");

        t.forget_unwritten(failed, &ids(&["r1"]));
        assert_eq!(mapped(&t), 0);
    }

    /// `out`'s new request `id`'s prefix-cache hit, in tokens.
    fn admitted_with(out: &SchedulerOutput, id: &str) -> Option<u32> {
        (out.scheduled_new_reqs.iter())
            .find(|r| r.req_id == id)
            .map(|r| r.num_computed_tokens)
    }

    /// Through the scheduler, as the engine abandons a step: the request a step behind it admitted
    /// onto its blocks fails with it, and the next copy of the prompt computes it from scratch.
    #[test]
    fn a_failed_step_takes_the_requests_admitted_onto_its_blocks() {
        let kv: Box<dyn KVCacheManagerOps> = Box::new(SimpleBlockTracker::with_caching(100, 16));
        let mut sched = Scheduler::new(&test_scheduler_config(), 8192, kv);
        sched.add_request(make_request("r1", 64));
        let failed = sched.schedule();
        sched.add_request(make_request("r2", 80));
        let behind = sched.schedule();
        assert_eq!(admitted_with(&behind, "r2"), Some(64), "onto r1's blocks");

        assert_eq!(sorted(sched.forget_unwritten(&failed)), ["r1", "r2"]);
        sched.finish_requests(&["r1", "r2"], RequestStatus::FinishedAborted);
        sched.add_request(make_request("r3", 64));
        assert_eq!(admitted_with(&sched.schedule(), "r3"), Some(0));
    }

    /// Through the scheduler: a client abort frees the request while its step is in flight, and
    /// the step failing still forgets its blocks.
    #[test]
    fn a_request_aborted_while_its_step_was_in_flight_is_forgotten_when_it_fails() {
        let kv: Box<dyn KVCacheManagerOps> = Box::new(SimpleBlockTracker::with_caching(100, 16));
        let mut sched = Scheduler::new(&test_scheduler_config(), 8192, kv);
        sched.add_request(make_request("r1", 64));
        let failed = sched.schedule();
        sched.finish_requests(&["r1"], RequestStatus::FinishedAborted);
        sched.add_request(make_request("r2", 80));
        let behind = sched.schedule();
        assert_eq!(
            admitted_with(&behind, "r2"),
            Some(64),
            "onto r1's freed blocks"
        );

        assert_eq!(sorted(sched.forget_unwritten(&failed)), ["r1", "r2"]);
        sched.finish_requests(&["r2"], RequestStatus::FinishedAborted);
        sched.add_request(make_request("r3", 64));
        assert_eq!(admitted_with(&sched.schedule(), "r3"), Some(0));
    }

    /// A gemma-4-shaped scheduler (one full group, bs 32, + five sliding groups, window 1024,
    /// bs 16) over a `pool`-block pool.
    fn gemma4_scheduler(
        pool: usize,
        max_num_seqs: usize,
        stores: HybridKvStores,
        caching: bool,
        backing: KvBacking,
    ) -> Scheduler {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 2048,
            max_num_seqs,
            enable_chunked_prefill: true,
            ..Default::default()
        };
        let mut tracker = match caching {
            true => SimpleBlockTracker::with_caching(pool, 16),
            false => SimpleBlockTracker::new(pool, 16),
        };
        let mut groups = vec![(false, 0usize, 32usize)];
        groups.extend(std::iter::repeat_n((true, 1024usize, 16usize), 5));
        tracker.enable_hybrid(pool, groups, stores);
        Scheduler::new(&cfg, 32768, Box::new(tracker.with_kv_backing(backing)))
    }

    fn request_from(id: &str, first_token: u32, prompt_len: u32, max_tokens: u32) -> Request {
        Request::new(
            id.into(),
            (first_token..first_token + prompt_len).collect(),
            SamplingParams {
                max_tokens: Some(max_tokens),
                ..Default::default()
            },
            0.0,
            0,
            0,
            None,
        )
    }

    // ----- Recurrent-state (Gated-DeltaNet) snapshots -----

    /// `slots` snapshots that take none of the KV budget, KV mapped a block at a time: the
    /// snapshots and the blocks never compete, so a test sees each alone.
    fn apart(slots: usize) -> SnapshotBudget {
        SnapshotBudget {
            kv_block_bytes: 1,
            kv_granule_blocks: std::num::NonZeroUsize::MIN,
            slot_bytes: 0,
            unit_slots: std::num::NonZeroUsize::MIN,
            max_slots: slots,
        }
    }

    /// 8 blocks of 10 bytes — an 80-byte budget — in granules of 4, and up to 4 snapshot slots of
    /// 20 bytes, a unit each: a pool where KV and the snapshots compete.
    fn shared() -> SimpleBlockTracker {
        SimpleBlockTracker::with_caching(8, 16).with_state_snapshots(SnapshotBudget {
            kv_block_bytes: 10,
            kv_granule_blocks: std::num::NonZeroUsize::new(4).unwrap(),
            slot_bytes: 20,
            unit_slots: std::num::NonZeroUsize::MIN,
            max_slots: 4,
        })
    }

    /// `n` blocks of prompt unlike any other `seed`'s.
    fn prompt_blocks(id: &str, n: usize, seed: u32) -> Request {
        request_with(
            id,
            (0..(n * 16) as u32).map(|t| seed * 10_000 + t).collect(),
        )
    }

    fn alloc(t: &mut SimpleBlockTracker, r: &Request) -> Option<Vec<usize>> {
        (t.allocate_slots(r, r.num_tokens(), 0, &[])).map(|mut groups| groups.remove(0))
    }

    fn holds_snapshot(t: &SimpleBlockTracker, key: u64) -> bool {
        t.recurrent.as_ref().unwrap().snapshots.contains(key)
    }

    #[test]
    fn sharing_the_budget_kv_counts_every_block_and_maps_a_granule_when_a_step_needs_it() {
        let mut t = shared();
        assert_eq!(KVCacheManagerOps::num_total_blocks(&t), 8);
        assert_eq!(
            KVCacheManagerOps::num_free_blocks(&t),
            8,
            "the whole budget, as with caching off"
        );
        assert!(t.take_shared_memory_ops().is_empty(), "nothing mapped yet");
        assert_eq!(alloc(&mut t, &prompt_blocks("a", 1, 1)), Some(vec![0]));
        assert_eq!(t.take_shared_memory_ops().map_kv, vec![KvGranule(0)]);
        assert_eq!(KVCacheManagerOps::num_free_blocks(&t), 7);
    }

    #[test]
    fn a_snapshot_maps_from_the_headroom_then_from_kv_no_request_holds() {
        let mut t = shared();
        alloc(&mut t, &prompt_blocks("a", 8, 1)).unwrap();
        t.free("a");
        t.take_shared_memory_ops();
        t.new_step_starts();
        // The budget is all KV, cached and held by none, and there is no snapshot to evict.
        assert_eq!(t.reserve_snapshot(1, 16, "w"), Some(0));
        let ops = t.take_shared_memory_ops();
        assert_eq!(
            (ops.release_kv, ops.map_snapshots),
            (vec![KvGranule(0)], vec![SnapshotUnit(0)])
        );
        assert_eq!(
            KVCacheManagerOps::num_cached_blocks(&t),
            4,
            "granule 0's cached blocks went with it"
        );
        assert_eq!(
            KVCacheManagerOps::num_free_blocks(&t),
            4,
            "granule 1's: the 20 bytes left make no granule"
        );
        // The rest of granule 0's memory: a second unit, no KV released.
        assert_eq!(t.reserve_snapshot(2, 32, "w"), Some(1));
        assert_eq!(
            t.take_shared_memory_ops().map_snapshots,
            vec![SnapshotUnit(1)]
        );
    }

    #[test]
    fn a_save_evicts_a_snapshot_older_than_the_kv_it_would_take() {
        let mut t = shared();
        assert_eq!(t.reserve_snapshot(1, 16, "w"), Some(0));
        assert_eq!(t.reserve_snapshot(2, 32, "w"), Some(1));
        t.new_step_starts();
        alloc(&mut t, &prompt_blocks("a", 4, 1)).unwrap();
        t.free("a");
        t.take_shared_memory_ops();
        t.new_step_starts();
        // No headroom; a's granule, freed after both snapshots were saved, is the newer state.
        let slot = t.reserve_snapshot(3, 48, "w").unwrap();
        assert!(slot < 2, "a snapshot's slot");
        assert!(t.take_shared_memory_ops().is_empty(), "no memory moved");
        assert_eq!(
            KVCacheManagerOps::num_cached_blocks(&t),
            4,
            "a's blocks stay cached"
        );
    }

    #[test]
    fn live_kv_takes_the_snapshots_memory_but_not_a_unit_read_this_step() {
        let mut t = shared();
        for key in 0..4 {
            assert_eq!(t.reserve_snapshot(key, 16, "w"), Some(key as usize));
        }
        t.take_shared_memory_ops();
        t.new_step_starts();
        t.recurrent.as_mut().unwrap().snapshots.pin(0, "r");
        // 8 blocks would need slot 0's 20 bytes too: refused, and nothing is released for it.
        assert_eq!(alloc(&mut t, &prompt_blocks("big", 8, 1)), None);
        assert!(t.take_shared_memory_ops().is_empty());
        // 4 blocks: two units go, least recently used first, and never the one read this step.
        alloc(&mut t, &prompt_blocks("a", 4, 2)).unwrap();
        let ops = t.take_shared_memory_ops();
        assert_eq!(
            (ops.release_snapshots, ops.map_kv),
            (vec![SnapshotUnit(1), SnapshotUnit(2)], vec![KvGranule(0)])
        );
        assert!(holds_snapshot(&t, 0) && holds_snapshot(&t, 3));
    }

    #[test]
    fn a_snapshot_goes_with_the_block_it_ends_at() {
        let mut t = SimpleBlockTracker::with_caching(4, 16).with_state_snapshots(apart(2));
        let a = alloc(&mut t, &prompt_blocks("a", 4, 1)).unwrap();
        let key = t.block_id_to_hash[&a[2]];
        assert_eq!(t.reserve_snapshot(key, 48, "a"), Some(0));
        t.commit_state_saves(&[StateSnapshotOp {
            req_id: "a".into(),
            slot: 0,
            position: 48,
            key,
        }]);
        t.free("a");
        t.new_step_starts();
        // a's blocks wait in the free queue tail first: b's first block is a's last, which the
        // snapshot does not end at.
        let one_block = |id: &str, seed: u32| request_with(id, (seed..seed + 8).collect());
        assert_eq!(
            alloc(&mut t, &one_block("b", 20_000)).map(|b| b.len()),
            Some(1)
        );
        assert!(holds_snapshot(&t, key), "its block is still cached");
        alloc(&mut t, &one_block("c", 30_000)).unwrap();
        assert!(
            !holds_snapshot(&t, key),
            "its block was evicted, and it with it"
        );
    }

    #[test]
    fn a_cached_block_outlives_an_older_snapshot_and_gives_way_to_a_newer_one() {
        // The snapshot is older than a's cached blocks: c's granule comes out of the snapshot.
        let mut t = shared();
        assert_eq!(t.reserve_snapshot(9, 16, "w"), Some(0));
        t.new_step_starts();
        alloc(&mut t, &prompt_blocks("a", 4, 1)).unwrap();
        t.free("a");
        t.take_shared_memory_ops();
        t.new_step_starts();
        assert_eq!(
            alloc(&mut t, &prompt_blocks("c", 4, 2)),
            Some(vec![4, 5, 6, 7])
        );
        let ops = t.take_shared_memory_ops();
        assert_eq!(
            (ops.release_snapshots, ops.map_kv),
            (vec![SnapshotUnit(0)], vec![KvGranule(1)])
        );
        assert_eq!(
            KVCacheManagerOps::num_cached_blocks(&t),
            4,
            "a's blocks stay cached"
        );

        // The snapshot is newer: c evicts a's cached blocks and the snapshot stays.
        let mut t = shared();
        alloc(&mut t, &prompt_blocks("a", 4, 1)).unwrap();
        t.free("a");
        t.new_step_starts();
        assert_eq!(t.reserve_snapshot(9, 16, "w"), Some(0));
        t.take_shared_memory_ops();
        t.new_step_starts();
        assert_eq!(
            alloc(&mut t, &prompt_blocks("c", 4, 2)),
            Some(vec![3, 2, 1, 0]),
            "a's blocks, tail first"
        );
        assert!(t.take_shared_memory_ops().is_empty());
        assert_eq!(KVCacheManagerOps::num_cached_blocks(&t), 0);
        assert!(holds_snapshot(&t, 9));
    }

    /// A caching tracker over a uniform pool that tracks `slots` recurrent-state snapshots.
    fn recurrent_scheduler(slots: usize, budget: usize) -> Scheduler {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: budget,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            ..Default::default()
        };
        let kv: Box<dyn KVCacheManagerOps> =
            Box::new(SimpleBlockTracker::with_caching(1000, 16).with_state_snapshots(apart(slots)));
        Scheduler::new(&cfg, 8192, kv)
    }

    fn request_with(id: &str, prompt: Vec<u32>) -> Request {
        Request::new(
            id.into(),
            prompt,
            SamplingParams {
                max_tokens: Some(4),
                ..Default::default()
            },
            1.0,
            0,
            0,
            None,
        )
    }

    /// One scheduler step, with the engine's part: every request whose chunk reached the end of
    /// its history samples a token, and finishes at `max_tokens`. Returns the step's output.
    fn step_and_sample(sched: &mut Scheduler, max_tokens: u32) -> SchedulerOutput {
        let out = sched.schedule();
        for id in out.num_scheduled_tokens.keys() {
            let r = sched.get_request(id).unwrap();
            if r.num_computed_tokens as usize >= r.num_tokens() {
                sched.append_output_tokens(id, &[7]);
                if sched.get_request(id).unwrap().num_output_tokens() >= max_tokens as usize {
                    sched.finish_requests(&[id.as_str()], RequestStatus::FinishedLengthCapped);
                }
            }
        }
        out
    }

    /// #328, through a freshly started scheduler: a `prompt_len`-token prompt with a
    /// `reply`-token reply, then the same prompt again. Returns how much of it the resend reused,
    /// after checking that every sliding group still holds the window the first recomputed token
    /// reads.
    fn resend_after_long_reply(stores: HybridKvStores, prompt_len: u32, reply: u32) -> u32 {
        let mut sched = gemma4_scheduler(2634, 1, stores, true, KvBacking::Allocated);
        sched.add_request(request_from("r1", 0, prompt_len, reply));
        while sched.has_unfinished_requests() {
            step_and_sample(&mut sched, reply);
        }
        sched.add_request(request_from("r2", 0, prompt_len, reply));
        sched.schedule();
        let reused = sched.get_request("r2").unwrap().num_cached_tokens as u32;
        let tables = sched.kv_cache.get_blocks("r2");
        let sliding = SwaGroup {
            is_sliding: true,
            window: 1024,
            block_size: 16,
            space: 0,
        };
        for (g, table) in tables.iter().enumerate().skip(1) {
            for i in sliding.prefix_blocks(reused as usize) {
                assert_ne!(
                    table[i],
                    SimpleBlockTracker::NULL_BLOCK,
                    "sliding group {g} block {i} is in the window at P={reused} but null"
                );
            }
        }
        reused
    }

    const BOTH_STORES: [HybridKvStores; 2] = [HybridKvStores::Shared, HybridKvStores::GlobalApart];

    #[test]
    fn test_hybrid_same_prompt_after_long_reply_hits_prefix_cache() {
        // The issue's ~1.4k-token prompt: it reused 0 once the first reply passed ~150 tokens,
        // whatever the reply's length.
        for stores in BOTH_STORES {
            for reply in [300, 2048] {
                let reused = resend_after_long_reply(stores, 1418, reply);
                assert_eq!(reused, 1408, "{stores:?}, {reply}-token reply");
            }
        }
    }

    #[test]
    fn test_hybrid_exact_resend_reads_no_null_block() {
        // A prompt that is a whole number of global blocks: the lookup stops a global block short,
        // so the scheduler never backs P off below the window the lookup checked, and the first
        // recomputed token's window holds no null block.
        for stores in BOTH_STORES {
            assert_eq!(
                resend_after_long_reply(stores, 1408, 300),
                1376,
                "{stores:?}"
            );
        }
    }

    /// #330's traffic through a gemma-4 scheduler on a 27,000-block pool: r1 (1,418/300) with three
    /// 2,000-token prompts decoding long replies; once r1 ends, 256 more steps, then r1's prompt
    /// again. Under [`KvBacking::OnFirstUse`] the worker reports each step what it has mapped,
    /// growing like metal's stores (128 blocks at least, doubling). Returns what the resend reused.
    fn resend_while_others_decode(stores: HybridKvStores, backing: KvBacking) -> u32 {
        let pool = 27_000;
        let mut sched = gemma4_scheduler(pool, 256, stores, true, backing);
        sched.add_request(request_from("r1", 0, 1418, 300));
        for i in 1..=3 {
            sched.add_request(request_from(&format!("o{i}"), i * 100_000, 2000, 100_000));
        }
        // Blocks mapped per store, `[pool, global]`.
        let mut mapped = [0usize; 2];
        // One step and the engine's part. Whether r1 ended in it.
        let mut step = |sched: &mut Scheduler| {
            let out = sched.schedule();
            let mut r1_ended = false;
            let tables = (out.scheduled_new_reqs.iter())
                .map(|r| &r.block_ids)
                .chain(out.scheduled_cached_reqs.new_block_ids.iter().flatten());
            for table in tables {
                for (g, ids) in table.iter().enumerate() {
                    let store = usize::from(g == 0 && stores == HybridKvStores::GlobalApart);
                    let highest = ids.iter().copied().max().unwrap_or(0);
                    if highest >= mapped[store] {
                        mapped[store] = (highest + 1)
                            .next_multiple_of(128)
                            .max(2 * mapped[store])
                            .min(pool);
                    }
                }
            }
            sched.set_kv_backed(match stores {
                HybridKvStores::Shared => KvBackedBlocks::Shared(mapped[0]),
                HybridKvStores::GlobalApart => KvBackedBlocks::GlobalApart {
                    pool: mapped[0],
                    global: mapped[1],
                },
            });
            for id in out.num_scheduled_tokens.keys() {
                let r = sched.get_request(id).unwrap();
                if r.num_computed_tokens as usize >= r.num_tokens() {
                    sched.append_output_tokens(id, &[7]);
                    let max = sched.get_request(id).unwrap().sampling_params.max_tokens;
                    if Some(sched.get_request(id).unwrap().num_output_tokens() as u32) >= max {
                        sched.finish_requests(&[id.as_str()], RequestStatus::FinishedLengthCapped);
                        r1_ended |= id == "r1";
                    }
                }
            }
            r1_ended
        };
        while !step(&mut sched) {}
        for _ in 0..256 {
            step(&mut sched);
        }
        sched.add_request(request_from("r1 again", 0, 1418, 1));
        step(&mut sched);
        sched.get_request("r1 again").unwrap().num_cached_tokens as u32
    }

    #[test]
    fn test_hybrid_prompt_resent_while_others_decode_hits_prefix_cache() {
        // #330: lowest ID first handed an ended request's freed blocks to the next ones the others
        // asked for, so its prompt sent again reused nothing. With every block allocated up front
        // it now reuses all but its last partial block under either layout. With blocks mapped as
        // the pool reaches them, the cache only lives in what is mapped: on metal's layout for
        // Gemma 4 (`GlobalApart`), the TurboQuant store's last doubling holds the others' new
        // global blocks and the prompt is reused too. Shared, the others' growth needs the space
        // r1's cache holds, which is evicted before the pool maps more.
        for stores in BOTH_STORES {
            let reused = resend_while_others_decode(stores, KvBacking::Allocated);
            assert_eq!(reused, 1408, "{stores:?}");
        }
        let reused = resend_while_others_decode(HybridKvStores::GlobalApart, KvBacking::OnFirstUse);
        assert_eq!(reused, 1408, "mapped as reached");
    }

    /// One of #317's A/B cells: `prompts` distinct `prompt_len`-token prompts, `users` at a time,
    /// each replying `reply_len` tokens.
    #[derive(Clone, Copy)]
    struct Cell {
        users: u32,
        prompts: u32,
        prompt_len: u32,
        reply_len: u32,
    }

    const CELL_4096_512: Cell = Cell {
        users: 8,
        prompts: 12,
        prompt_len: 4096,
        reply_len: 512,
    };

    /// What one cell costs through the scheduler.
    struct CellRun {
        computed: usize,
        preemptions: usize,
        /// The highest block the global group's store is addressed with: group 0's.
        global_reach: usize,
        /// The highest block the shared pool is addressed with: every group's whose KV lives in it.
        pool_reach: usize,
    }

    /// `cell` on the gemma-4 layout over a `pool`-block pool, with the global group's KV where
    /// `stores` puts it.
    fn run_317_cell(stores: HybridKvStores, caching: bool, pool: usize, cell: Cell) -> CellRun {
        let mut sched = gemma4_scheduler(pool, 256, stores, caching, KvBacking::Allocated);
        let mut next = 0u32;
        let add = |sched: &mut Scheduler, next: &mut u32| {
            let id = format!("r{next}");
            let first = (*next + 1) * 100_000;
            sched.add_request(request_from(&id, first, cell.prompt_len, cell.reply_len));
            *next += 1;
        };
        while next < cell.users {
            add(&mut sched, &mut next);
        }
        let mut run = CellRun {
            computed: 0,
            preemptions: 0,
            global_reach: 0,
            pool_reach: 0,
        };
        while sched.has_unfinished_requests() {
            let running = sched.get_unfinished_request_ids().len();
            let out = step_and_sample(&mut sched, cell.reply_len);
            run.computed += out.num_scheduled_tokens.values().sum::<usize>();
            run.preemptions += out.preempted_req_ids.as_ref().map_or(0, |p| p.len());
            let tables = (out.scheduled_new_reqs.iter())
                .map(|r| &r.block_ids)
                .chain(out.scheduled_cached_reqs.new_block_ids.iter().flatten());
            for table in tables {
                for (g, ids) in table.iter().enumerate() {
                    let highest = ids.iter().copied().max().unwrap_or(0);
                    if g == 0 {
                        run.global_reach = run.global_reach.max(highest);
                    }
                    if g != 0 || stores == HybridKvStores::Shared {
                        run.pool_reach = run.pool_reach.max(highest);
                    }
                }
            }
            let finished = running - sched.get_unfinished_request_ids().len();
            for _ in 0..finished.min((cell.prompts - next) as usize) {
                add(&mut sched, &mut next);
            }
        }
        run
    }

    #[test]
    fn test_hybrid_kept_prompt_windows_never_cost_a_preemption() {
        // #317's 8 users × 4096/512 cell on the 32 GB box's 3,407-block pool. Keeping every running
        // request's prompt window with no way to release it preempted 75 times and computed 3.7×
        // the tokens; released on demand, it is #317's single preemption.
        let cell = CELL_4096_512;
        let ideal = (cell.prompts * (cell.prompt_len + cell.reply_len - 1)) as usize;
        for stores in BOTH_STORES {
            let run = run_317_cell(stores, true, 3407, cell);
            let preemptions = run.preemptions;
            assert!(
                preemptions <= 1,
                "{stores:?}: {preemptions} preemptions (#317: 1)"
            );
            let computed = run.computed;
            assert!(
                computed <= ideal,
                "{stores:?}: {computed} tokens for an ideal {ideal}"
            );
        }
    }

    #[test]
    fn test_hybrid_global_apart_serves_long_contexts_without_preempting() {
        // 4 users × 16,384/512 on the 32 GB box's pool. Counted against one shared pool, a
        // request's 528 global blocks and its sliding groups' window plus a prefill chunk overran
        // it: 94 preemptions, 2.36× the tokens recomputed. Each store holding the pool on its own,
        // all four run together.
        let cell = Cell {
            users: 4,
            prompts: 8,
            prompt_len: 16_384,
            reply_len: 512,
        };
        let run = run_317_cell(HybridKvStores::GlobalApart, true, 3406, cell);
        let ideal = (cell.prompts * (cell.prompt_len + cell.reply_len - 1)) as usize;
        assert_eq!(run.preemptions, 0);
        assert!(
            run.computed <= ideal,
            "{} for an ideal {ideal}",
            run.computed
        );
    }

    #[test]
    fn test_hybrid_kept_prompt_windows_cost_a_window_per_running_request() {
        // #317's 8 users × 4096/1280 cell on a roomy pool. Each running request keeps its prompt's
        // last window while its reply decodes in another, so the pool holding the sliding groups
        // grows by up to one window (5 groups × 65 blocks) per request running at once, and no
        // more. Apart, the kept windows live in the sliding groups' pool and never reach the
        // global store.
        let cell = Cell {
            prompts: 8,
            reply_len: 1280,
            ..CELL_4096_512
        };
        let window = 5 * (1024usize.div_ceil(16) + 1);
        for stores in BOTH_STORES {
            let kept = run_317_cell(stores, true, 27_000, cell);
            let uncached = run_317_cell(stores, false, 27_000, cell);
            let bound = uncached.pool_reach + cell.users as usize * window;
            assert!(
                kept.pool_reach > uncached.pool_reach,
                "{stores:?}: windows were kept"
            );
            assert!(
                kept.pool_reach <= bound,
                "{stores:?}: {} > {bound}",
                kept.pool_reach
            );
            if stores == HybridKvStores::GlobalApart {
                assert_eq!(kept.global_reach, uncached.global_reach);
            }
            assert_eq!(kept.preemptions, 0, "{stores:?}");
        }
    }

    #[test]
    fn test_block_hashes_extended_in_pieces_equal_a_full_rehash() {
        // Extending a request's hashes as its tokens grow gives, at every length, exactly what
        // hashing all its tokens gives: across relocatable and prefixed span blocks (annotated by
        // token offset at the config block size) and image placeholders, at both block sizes.
        use scratchy_core_common::multimodal::{ImageData, PlaceholderRange};
        let t = SimpleBlockTracker::with_caching(64, 16);
        let tokens: Vec<u32> = (0..400u32).map(|i| i * 7 % 251).collect();
        let mut ann = std::collections::BTreeMap::new();
        ann.insert(3, BlockKind::Relocatable { first_token: 48 });
        ann.insert(4, BlockKind::Relocatable { first_token: 48 });
        ann.insert(9, BlockKind::Prefixed { first_token: 0 });
        let image = |v: f32| ImageData {
            pixels: vec![v; 12],
            height: 2,
            width: 2,
        };
        let mm = MultimodalData {
            images: vec![image(0.5), image(0.25)],
            image_placeholders: vec![
                PlaceholderRange {
                    offset: 20,
                    length: 30,
                },
                PlaceholderRange {
                    offset: 250,
                    length: 70,
                },
            ],
        };
        for block_size in [16usize, 32] {
            let mut hashes = Vec::new();
            let mut len = 0;
            for step in [1usize, 7, 37, 5, 64, 13, 100, 3, 170] {
                len = (len + step).min(tokens.len());
                let seen = &tokens[..len];
                t.extend_block_hashes(&mut hashes, seen, Some(&ann), Some(&mm), block_size);
                let full = t.hash_all_blocks_bs(seen, Some(&ann), Some(&mm), block_size);
                assert_eq!(hashes, full, "block size {block_size}, {len} tokens");
            }
            assert_eq!(hashes.len(), tokens.len() / block_size);
        }
    }

    #[test]
    fn test_hybrid_decode_registers_every_block_at_its_full_hash() {
        // A request prefilled then decoded token by token (sealing each step, as the engine does)
        // ends holding exactly the hashes a full rehash of its tokens gives, each in-window block
        // registered under its own.
        for stores in BOTH_STORES {
            let mut t = SimpleBlockTracker::with_caching(512, 16);
            t.enable_hybrid(512, vec![(false, 0, 32), (true, 64, 16)], stores);
            let mut r = hybrid_req("r1", 200);
            t.allocate_slots(&r, 200, 0, &[]).unwrap();
            r.num_computed_tokens = 200;
            t.seal(&r);
            for i in 0..300u32 {
                r.append_output_token_ids(&[10_000 + i]);
                t.allocate_slots(&r, 1, 0, &[]).unwrap();
                r.num_computed_tokens += 1;
                t.seal(&r);
            }
            let groups = t.swa_groups.clone().unwrap();
            let tables = t.get_blocks("r1");
            for (g, grp) in groups.iter().enumerate() {
                let full = t.hash_all_blocks_bs(&r.all_token_ids, None, None, grp.block_size);
                assert_eq!(
                    t.hybrid_req_hashes["r1"][g].hashes, full,
                    "{stores:?} group {g}"
                );
                for (i, &bid) in tables[g].iter().enumerate().take(full.len()) {
                    if bid != SimpleBlockTracker::NULL_BLOCK {
                        let key = (g as u32, full[i]);
                        assert_eq!(t.hybrid_hash_to_id.get(&key), Some(&bid), "{stores:?}");
                    }
                }
            }
        }
    }

    /// One step as the engine runs it: schedule, and — the step having executed — commit the
    /// snapshots it saved, as `update_from_output` does.
    fn run_step(sched: &mut Scheduler) -> SchedulerOutput {
        let out = sched.schedule();
        sched.commit_step(&out);
        out
    }

    /// Run `id`'s prefill to completion through committed steps, then finish it.
    fn prefill_and_finish(sched: &mut Scheduler, id: &str) {
        let done = |sched: &Scheduler| {
            let r = sched.get_request(id).unwrap();
            r.num_computed_tokens >= r.num_tokens() as u32
        };
        for _ in 0..64 {
            if done(sched) {
                break;
            }
            run_step(sched);
        }
        assert!(done(sched), "{id} never finished its prefill");
        sched.finish_requests(&[id], RequestStatus::FinishedStopped);
        run_step(sched);
    }

    /// A request whose blocks need snapshot memory keeps the snapshot it resumes from: the restore
    /// is held before its blocks are allocated, so the allocation takes the other unit even though
    /// the restored one is older.
    #[test]
    fn recurrent_an_allocation_never_takes_the_snapshot_its_own_request_restores() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 4096,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            ..Default::default()
        };
        // 10 blocks of 10 bytes, mapped one at a time, and 30-byte snapshot slots.
        let budget = SnapshotBudget {
            kv_block_bytes: 10,
            kv_granule_blocks: std::num::NonZeroUsize::MIN,
            slot_bytes: 30,
            unit_slots: std::num::NonZeroUsize::MIN,
            max_slots: 3,
        };
        let kv = SimpleBlockTracker::with_caching(10, 16).with_state_snapshots(budget);
        let mut sched = Scheduler::new(&cfg, 8192, Box::new(kv));
        let saved = |sched: &mut Scheduler, id: &str, prompt: Vec<u32>| {
            sched.add_request(request_with(id, prompt));
            let out = run_step(sched);
            prefill_and_finish(sched, id);
            out.recurrent_state.saves[0].clone()
        };
        // Two 2-block prompts and their snapshots: 2 × (20 + 30) bytes, the whole budget.
        let turn1: Vec<u32> = (0..32).collect();
        let t1 = saved(&mut sched, "t1", turn1.clone());
        let other = saved(&mut sched, "o", (5000..5032).collect());

        // The next turn resumes at t1's snapshot and needs 3 blocks more than are free.
        let turn2: Vec<u32> = turn1.iter().copied().chain(1000..1048).collect();
        sched.add_request(request_with("t2", turn2));
        let out = run_step(&mut sched);
        assert_eq!(
            out.recurrent_state.restores,
            vec![StateSnapshotOp {
                req_id: "t2".into(),
                slot: t1.slot,
                position: 16,
                key: t1.key,
            }]
        );
        assert_eq!(
            out.recurrent_state.memory.release_snapshots,
            vec![SnapshotUnit(other.slot as u32)],
            "the other snapshot's memory, not the restored one's"
        );
    }

    /// A request's prompt-cache breakpoint is a snapshot of its own, beside its end: the request
    /// after it shares the prompt only through the breakpoint (Claude Code's permission classifier
    /// inserts each entry before a fixed tail) and resumes there.
    #[test]
    fn recurrent_a_cache_breakpoint_is_snapshotted_and_resumed_from() {
        let mut sched = recurrent_scheduler(4, 4096);
        let entries: Vec<u32> = (0..500).collect();
        let tail: Vec<u32> = (9000..9100).collect();
        let mut first = request_with("c1", entries.iter().chain(&tail).copied().collect());
        first.cache_breakpoints = vec![500];
        sched.add_request(first);
        let out = run_step(&mut sched);
        let saved: Vec<u32> = (out.recurrent_state.saves.iter())
            .map(|op| op.position)
            .collect();
        // floor(500 / 16) * 16 = 496, and the end: floor((600 - 8) / 16) * 16 = 592.
        assert_eq!(saved, vec![496, 592]);
        prefill_and_finish(&mut sched, "c1");

        // The next call: the same entries, a new one, then the tail again.
        let next: Vec<u32> = (entries.iter())
            .chain(&(5000..5040).collect::<Vec<u32>>())
            .chain(&tail)
            .copied()
            .collect();
        sched.add_request(request_with("c2", next));
        let out = run_step(&mut sched);
        assert_eq!(out.recurrent_state.restores.len(), 1);
        assert_eq!(
            out.recurrent_state.restores[0].position, 496,
            "at the breakpoint"
        );
    }

    #[test]
    fn recurrent_cold_prompt_snapshots_its_end_in_its_one_forward() {
        let mut sched = recurrent_scheduler(4, 4096);
        sched.add_request(make_request("r1", 600));
        let out = run_step(&mut sched);
        assert_eq!(
            out.num_scheduled_tokens["r1"], 600,
            "a snapshot does not split the prompt"
        );
        assert!(out.recurrent_state.restores.is_empty());
        assert_eq!(out.recurrent_state.saves.len(), 1);
        assert_eq!(out.recurrent_state.saves[0].req_id, "r1");
        // floor((600 - GENERATION_PROMPT_ALLOWANCE) / 16) * 16 = 592.
        assert_eq!(out.recurrent_state.saves[0].position, 592);
    }

    #[test]
    fn recurrent_hit_resumes_at_a_ready_snapshot_and_restores_it() {
        let mut sched = recurrent_scheduler(4, 4096);
        let turn1: Vec<u32> = (0..600).collect();
        sched.add_request(request_with("t1", turn1.clone()));
        let saved = sched.schedule();
        sched.commit_step(&saved);
        prefill_and_finish(&mut sched, "t1");

        // The next turn extends the prompt; its KV hit is t1's 37 full blocks.
        let turn2: Vec<u32> = turn1.iter().copied().chain(1000..1100).collect();
        sched.add_request(request_with("t2", turn2));
        let out = run_step(&mut sched);
        assert_eq!(
            out.recurrent_state.restores,
            vec![StateSnapshotOp {
                req_id: "t2".into(),
                slot: saved.recurrent_state.saves[0].slot,
                position: 592,
                key: saved.recurrent_state.saves[0].key,
            }]
        );
        assert_eq!(out.scheduled_new_reqs[0].num_computed_tokens, 592);
        assert_eq!(out.num_scheduled_tokens["t2"], 108);
        assert_eq!(sched.get_request("t2").unwrap().num_cached_tokens, 592);
        assert!(
            out.recurrent_state.cached_kv.is_empty(),
            "it resumes at the end of its KV hit"
        );
        let saved: Vec<u32> = (out.recurrent_state.saves.iter())
            .map(|op| op.position)
            .collect();
        assert_eq!(saved, vec![688], "its own end, past the restored state");
    }

    #[test]
    fn recurrent_kv_hit_without_a_snapshot_resumes_at_zero_over_the_cached_kv() {
        let mut sched = recurrent_scheduler(4, 4096);
        sched.add_request(make_request("r1", 200));
        prefill_and_finish(&mut sched, "r1");

        // r2 shares r1's first 6 blocks; r1 snapshotted only its own end (192).
        sched.add_request(request_with("r2", (0..96).chain(5000..5104).collect()));
        let out = run_step(&mut sched);
        assert_eq!(out.scheduled_new_reqs[0].num_computed_tokens, 0);
        assert_eq!(out.num_scheduled_tokens["r2"], 200);
        assert!(out.recurrent_state.restores.is_empty());
        assert_eq!(
            out.recurrent_state.cached_kv,
            vec![CachedKvOp {
                req_id: "r2".into(),
                through: 96
            }],
            "its first 96 positions run again for their state alone"
        );
        let saved: Vec<u32> = (out.recurrent_state.saves.iter())
            .map(|op| op.position)
            .collect();
        assert_eq!(saved, vec![96, 192], "the junction and its own end");
    }

    #[test]
    fn recurrent_snapshot_is_not_hit_before_its_step_reports_back() {
        let mut sched = recurrent_scheduler(4, 4096);
        sched.add_request(make_request("r1", 600));
        let first = sched.schedule(); // its saves are NOT committed: the step is still running
        assert_eq!(first.recurrent_state.saves.len(), 1);

        sched.add_request(make_request("r2", 600));
        let out = sched.schedule();
        assert!(
            out.recurrent_state.restores.is_empty(),
            "a pending snapshot is not hittable"
        );
        assert_eq!(
            out.num_scheduled_tokens["r2"], 600,
            "no second split for the same prefix"
        );
        assert!(
            out.recurrent_state.saves.is_empty(),
            "one prefix is snapshotted once"
        );

        // Once it reports back, the next request with that prefix resumes from it.
        sched.commit_step(&first);
        sched.add_request(make_request("r3", 600));
        let out = sched.schedule();
        assert_eq!(out.recurrent_state.restores.len(), 1);
        assert_eq!(out.recurrent_state.restores[0].req_id, "r3");
    }

    #[test]
    fn recurrent_snapshot_of_a_failed_step_never_becomes_hittable() {
        let mut sched = recurrent_scheduler(4, 4096);
        sched.add_request(make_request("r1", 600));
        let failed = sched.schedule();
        assert_eq!(failed.recurrent_state.saves.len(), 1);
        // The step failed: the engine aborts its requests and never commits.
        sched.finish_requests(&["r1"], RequestStatus::FinishedAborted);
        let _ = sched.schedule();

        sched.add_request(make_request("r2", 600));
        let retry = sched.schedule();
        assert!(retry.recurrent_state.restores.is_empty());
        assert_eq!(
            (retry.recurrent_state.saves.iter())
                .map(|op| (op.req_id.as_str(), op.position))
                .collect::<Vec<_>>(),
            vec![("r2", 592)],
            "r2 takes the snapshot r1 never wrote"
        );
        // r1's step ran after all (queued behind the failure) and reports back: it must not
        // make r2's reservation of the same prefix ready before r2's own step writes it.
        sched.commit_step(&failed);
        sched.add_request(make_request("r3", 600));
        assert!(sched.schedule().recurrent_state.restores.is_empty());
        sched.commit_step(&retry);
        sched.add_request(make_request("r4", 600));
        let out = sched.schedule();
        assert_eq!(out.recurrent_state.restores.len(), 1);
        assert_eq!(out.recurrent_state.restores[0].req_id, "r4");
    }

    #[test]
    fn recurrent_junction_snapshot_where_two_prompts_share_a_prefix() {
        let mut sched = recurrent_scheduler(4, 4096);
        let system: Vec<u32> = (0..512).collect();
        let with_user = |user: u32| -> Vec<u32> {
            system
                .iter()
                .copied()
                .chain((0..100).map(|t| user * 1000 + t))
                .collect()
        };
        sched.add_request(request_with("a", with_user(1)));
        prefill_and_finish(&mut sched, "a");

        // b shares a's 512-token system prompt but no state exists there: snapshot it, and
        // b's own end, in b's one forward.
        sched.add_request(request_with("b", with_user(2)));
        let out = run_step(&mut sched);
        assert!(out.recurrent_state.restores.is_empty());
        assert_eq!(out.num_scheduled_tokens["b"], 612);
        let saved: Vec<u32> = (out.recurrent_state.saves.iter())
            .map(|op| op.position)
            .collect();
        assert_eq!(saved, vec![512, 592]);
        prefill_and_finish(&mut sched, "b");

        // c resumes at the junction.
        sched.add_request(request_with("c", with_user(3)));
        let out = run_step(&mut sched);
        assert_eq!(out.recurrent_state.restores.len(), 1);
        assert_eq!(out.recurrent_state.restores[0].position, 512);
        assert_eq!(out.scheduled_new_reqs[0].num_computed_tokens, 512);
    }

    #[test]
    fn recurrent_long_prompt_snapshots_in_the_chunk_that_passes_its_save_point() {
        let mut sched = recurrent_scheduler(4, 256);
        sched.add_request(make_request("r1", 600));
        let chunks: Vec<(usize, Vec<u32>)> = (0..3)
            .map(|_| {
                let out = run_step(&mut sched);
                let saved = (out.recurrent_state.saves.iter())
                    .map(|op| op.position)
                    .collect();
                (out.num_scheduled_tokens["r1"], saved)
            })
            .collect();
        assert_eq!(chunks, vec![(256, vec![]), (256, vec![]), (88, vec![592])]);
    }

    #[test]
    fn recurrent_annotated_volatile_and_pooling_requests_take_no_snapshot() {
        let mut sched = recurrent_scheduler(4, 4096);
        let mut annotated = make_request("ann", 600);
        annotated.block_annotations = Some(Default::default());
        let mut volatile = make_request("vol", 600);
        volatile.volatile = true;
        let mut pooling = make_request("pool", 600);
        pooling.is_pooling = true;
        for r in [annotated, volatile, pooling] {
            sched.add_request(r);
        }
        let out = run_step(&mut sched);
        assert!(out.recurrent_state.restores.is_empty() && out.recurrent_state.saves.is_empty());
        assert!(out.num_scheduled_tokens.values().all(|&n| n == 600));
    }

    #[test]
    fn recurrent_annotated_requests_never_resume() {
        let mut sched = recurrent_scheduler(4, 4096);
        sched.add_request(make_request("r1", 600));
        prefill_and_finish(&mut sched, "r1");
        let mut annotated = make_request("ann", 600);
        annotated.block_annotations = Some(Default::default());
        sched.add_request(annotated);
        let out = run_step(&mut sched);
        assert!(out.recurrent_state.restores.is_empty());
        assert_eq!(out.scheduled_new_reqs[0].num_computed_tokens, 0);
    }

    #[test]
    fn recurrent_snapshots_are_evicted_least_recently_used_first() {
        let mut sched = recurrent_scheduler(1, 4096);
        sched.add_request(make_request("r1", 600));
        prefill_and_finish(&mut sched, "r1");
        // A different prompt takes the only slot.
        sched.add_request(request_with("other", (5000..5600).collect()));
        prefill_and_finish(&mut sched, "other");

        sched.add_request(make_request("r2", 600));
        let out = run_step(&mut sched);
        assert!(
            out.recurrent_state.restores.is_empty(),
            "r1's snapshot was evicted"
        );
        assert_eq!(out.scheduled_new_reqs[0].num_computed_tokens, 0);
    }

    #[test]
    fn recurrent_reset_prefix_cache_forgets_snapshots() {
        let mut sched = recurrent_scheduler(2, 4096);
        sched.add_request(make_request("r1", 600));
        prefill_and_finish(&mut sched, "r1");
        assert!(sched.reset_prefix_cache());
        sched.add_request(make_request("r2", 600));
        let out = run_step(&mut sched);
        assert!(out.recurrent_state.restores.is_empty());
        assert_eq!(
            out.recurrent_state.saves.len(),
            1,
            "the slot is free to snapshot again"
        );
    }

    #[test]
    fn recurrent_resume_below_the_kv_hit_shares_the_blocks_and_leaves_their_kv() {
        let mut sched = recurrent_scheduler(4, 4096);
        let turn1: Vec<u32> = (0..640).collect(); // 40 full blocks, snapshot at 624
        sched.add_request(request_with("t1", turn1.clone()));
        let first = run_step(&mut sched);
        let t1_blocks = first.scheduled_new_reqs[0].block_ids[0].clone();
        assert_eq!(t1_blocks.len(), 40);
        sched.finish_requests(&["t1"], RequestStatus::FinishedStopped);
        run_step(&mut sched);

        sched.add_request(request_with(
            "t2",
            turn1.iter().copied().chain(900..1000).collect(),
        ));
        let out = run_step(&mut sched);
        assert_eq!(out.recurrent_state.restores[0].position, 624);
        assert_eq!(out.scheduled_new_reqs[0].num_computed_tokens, 624);
        let t2_blocks = &out.scheduled_new_reqs[0].block_ids[0];
        assert_eq!(
            &t2_blocks[..40],
            &t1_blocks[..40],
            "the whole KV hit is shared"
        );
        assert!(!t1_blocks.contains(&t2_blocks[40]));
        assert_eq!(
            out.recurrent_state.cached_kv,
            vec![CachedKvOp {
                req_id: "t2".into(),
                through: 640
            }],
            "the block past the snapshot runs again, its KV left as cached"
        );
    }

    #[test]
    fn recurrent_multimodal_requests_neither_resume_nor_snapshot() {
        let mut sched = recurrent_scheduler(4, 4096);
        sched.add_request(make_request("r1", 600));
        prefill_and_finish(&mut sched, "r1");
        let mut mm = make_request("mm", 600);
        mm.mm_data = Some(Default::default());
        sched.add_request(mm);
        let out = run_step(&mut sched);
        assert!(out.recurrent_state.restores.is_empty() && out.recurrent_state.saves.is_empty());
        assert_eq!(out.scheduled_new_reqs[0].num_computed_tokens, 0);
        assert_eq!(out.num_scheduled_tokens["mm"], 600);
    }

    #[test]
    fn recurrent_junction_and_end_snapshots_in_one_prefill() {
        let mut sched = recurrent_scheduler(4, 4096);
        let system: Vec<u32> = (0..512).collect();
        let with_user = |user: u32| -> Vec<u32> {
            system
                .iter()
                .copied()
                .chain((0..400).map(|t| user * 1000 + t))
                .collect()
        };
        sched.add_request(request_with("a", with_user(1)));
        prefill_and_finish(&mut sched, "a");
        // b shares a's 512-token system prompt: a junction there, then its own end at 896.
        sched.add_request(request_with("b", with_user(2)));
        let out = run_step(&mut sched);
        let saved: Vec<u32> = (out.recurrent_state.saves.iter())
            .map(|op| op.position)
            .collect();
        assert_eq!(
            (out.num_scheduled_tokens["b"], saved),
            (912, vec![512, 896])
        );
        sched.finish_requests(&["b"], RequestStatus::FinishedStopped);
        run_step(&mut sched);

        let mut next = with_user(2);
        next.extend(5000..5050);
        sched.add_request(request_with("b2", next));
        let out = run_step(&mut sched);
        assert_eq!(out.recurrent_state.restores[0].position, 896);
    }

    #[test]
    fn recurrent_a_request_resumes_at_its_snapshot_after_preemption() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 4096,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            ..Default::default()
        };
        // r0 (1 block) + r1 (600 tokens, 38 blocks) fill the pool exactly.
        let kv: Box<dyn KVCacheManagerOps> =
            Box::new(SimpleBlockTracker::with_caching(39, 16).with_state_snapshots(apart(4)));
        let mut sched = Scheduler::new(&cfg, 8192, kv);
        let long = |id: &str, n: usize| {
            let mut r = make_request(id, n);
            r.max_tokens = 64;
            r.sampling_params.max_tokens = Some(64);
            r
        };
        sched.add_request(long("r0", 15));
        sched.add_request(long("r1", 600));
        let mut preempted = false;
        for _ in 0..32 {
            let out = run_step(&mut sched);
            if out
                .preempted_req_ids
                .as_ref()
                .is_some_and(|p| p.contains("r1"))
            {
                preempted = true;
                break;
            }
            for id in out.num_scheduled_tokens.keys() {
                let r = sched.get_request(id).unwrap();
                if r.num_computed_tokens as usize >= r.num_tokens() {
                    sched.append_output_tokens(id, &[7]);
                }
            }
        }
        assert!(
            preempted,
            "r0's decode must preempt r1 for this test to mean anything"
        );
        sched.finish_requests(&["r0"], RequestStatus::FinishedStopped);
        let out = run_step(&mut sched);
        assert_eq!(
            out.recurrent_state.restores,
            vec![StateSnapshotOp {
                req_id: "r1".into(),
                slot: out.recurrent_state.restores[0].slot,
                position: 592,
                key: out.recurrent_state.restores[0].key,
            }]
        );
        assert!(out.scheduled_cached_reqs.resumed_req_ids.contains("r1"));
        let i = out
            .scheduled_cached_reqs
            .req_ids
            .iter()
            .position(|r| r == "r1")
            .unwrap();
        assert_eq!(out.scheduled_cached_reqs.num_computed_tokens[i], 592);
        assert!(
            out.recurrent_state.saves.is_empty(),
            "its prompt end is already snapshotted"
        );
    }

    #[test]
    fn recurrent_a_request_that_cannot_be_admitted_restores_nothing_until_it_is() {
        let cfg = SchedulerConfig {
            max_num_batched_tokens: 4096,
            max_num_seqs: 4,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            ..Default::default()
        };
        let kv: Box<dyn KVCacheManagerOps> =
            Box::new(SimpleBlockTracker::with_caching(79, 16).with_state_snapshots(apart(4)));
        let mut sched = Scheduler::new(&cfg, 8192, kv);
        sched.add_request(make_request("r1", 600));
        prefill_and_finish(&mut sched, "r1");
        // A blocker holds 38 fresh blocks, leaving 3 free beside r1's 38 cached ones.
        sched.add_request(request_with("blocker", (7000..7600).collect()));
        for _ in 0..3 {
            run_step(&mut sched);
        }
        // r2 resumes at r1's snapshot but needs 7 fresh blocks: not admissible yet.
        let mut prompt: Vec<u32> = (0..600).collect();
        prompt.extend(8000..8100);
        sched.add_request(request_with("r2", prompt));
        let out = run_step(&mut sched);
        assert!(!out.num_scheduled_tokens.contains_key("r2"));
        assert!(out.recurrent_state.is_empty());
        sched.finish_requests(&["blocker"], RequestStatus::FinishedStopped);
        let out = run_step(&mut sched);
        assert_eq!(out.recurrent_state.restores.len(), 1);
        assert_eq!(out.recurrent_state.restores[0].req_id, "r2");
        assert_eq!(out.recurrent_state.restores[0].position, 592);
    }

    #[test]
    fn recurrent_older_snapshots_of_a_conversation_age_with_its_newest() {
        let mut sched = recurrent_scheduler(3, 4096);
        let p: Vec<u32> = (0..600).collect();
        let grow = |base: &[u32], from: u32, n: u32| -> Vec<u32> {
            base.iter().copied().chain(from..from + n).collect()
        };
        sched.add_request(request_with("a1", p.clone())); // snapshot at 592
        prefill_and_finish(&mut sched, "a1");
        let a2 = grow(&p, 10_000, 400); // resumes at 592, snapshot at 992
        sched.add_request(request_with("a2", a2.clone()));
        prefill_and_finish(&mut sched, "a2");
        sched.add_request(request_with("b", (20_000..20_600).collect()));
        prefill_and_finish(&mut sched, "b");
        // a3 resumes at 992 and touches 592 with it; too short to pass a save point of its own.
        sched.add_request(request_with("a3", grow(&a2, 30_000, 10)));
        let out = run_step(&mut sched);
        assert_eq!(out.recurrent_state.restores[0].position, 992);
        assert!(out.recurrent_state.saves.is_empty());
        sched.finish_requests(&["a3"], RequestStatus::FinishedStopped);
        run_step(&mut sched);
        // A fourth snapshot evicts b's — the least recently used — not the conversation's 592.
        sched.add_request(request_with("x", (40_000..40_600).collect()));
        prefill_and_finish(&mut sched, "x");
        sched.add_request(request_with("a4", grow(&p, 50_000, 100)));
        let out = run_step(&mut sched);
        assert_eq!(out.recurrent_state.restores[0].position, 592);
    }

    #[test]
    fn recurrent_a_conversation_that_fills_the_pool_keeps_taking_snapshots() {
        // Two slots, one conversation of growing turns: once its own snapshots fill the pool,
        // each turn restores the newest (pinned), touches the older (evictable), and still
        // snapshots its own end.
        let mut sched = recurrent_scheduler(2, 4096);
        let mut prompt: Vec<u32> = (0..600).collect();
        let mut previous_end = None;
        for turn in 0..5u32 {
            let id = format!("t{turn}");
            sched.add_request(request_with(&id, prompt.clone()));
            let out = run_step(&mut sched);
            if let Some(previous_end) = previous_end {
                assert_eq!(
                    out.recurrent_state.restores[0].position, previous_end,
                    "turn {turn} resumes at the previous turn's end"
                );
            }
            let end = (prompt.len() as u32 - 8) / 16 * 16;
            previous_end = Some(end);
            assert_eq!(
                out.recurrent_state
                    .saves
                    .iter()
                    .map(|op| op.position)
                    .collect::<Vec<_>>(),
                vec![end],
                "turn {turn} snapshots its own end"
            );
            prefill_and_finish(&mut sched, &id);
            // The next turn drops the 4-token generation prompt and adds 400 tokens.
            prompt.truncate(prompt.len() - 4);
            prompt.extend((0..400).map(|t| 100_000 * (turn + 1) + t));
        }
    }
}
