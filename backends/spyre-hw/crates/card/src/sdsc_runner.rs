// SPDX-License-Identifier: Apache-2.0
//! Execute an emitted **SuperDSC** bundle on the real IBM Spyre AIU — the `--target sendnn` silicon
//! run path, as the worker sees it.
//!
//! This is the worker-facing wrapper: f32 host buffers in, a submitted step's landing out, names
//! as `t{id}` (the SubtileIR tensor id). The work is [`crate::superdsc_exec`]'s, which drives the
//! card through the typed flex-rs surface in [`crate::runtime`].
//!
//! ⛔ THE SENGRAPH PATH IS GONE. The offline_decoder `GraphLoader`/`Predict` families (`SpyreSession`
//! / `GroupedSession` and their C++ shim) were the pre-SuperDSC bring-up path and had no live
//! caller: the worker routes every sendnn bundle to `SuperDsc`. Under `sendnn` the only bundle that
//! runs is the dxp-compiled SuperDSC bundle.

/// The worker's [`LaunchEnd`] machinery names which suffix a forward launches by this enum's
/// path through this module — re-exported, never restated.
pub use crate::superdsc_exec::SuffixLaunch;
use crate::superdsc_exec::{
    DeferredRead, Executor, ForwardLatch, KvDim, NewTokens, PoolPages, SegIdx, SeqPos, Vocab,
};
use anyhow::{Result, anyhow};
use scratchy_tensors::DType as SDType;
use tracing::debug;

use scratchy_target_spyre::manifest::bytes_to_f32;

use ktir_superdsc::collective::PerRank;
use scratchy_target_spyre::sen_convert::f32_to_f16_le;

/// ⭐ ONE STEP'S HOST INPUTS — every channel a launch writes into each card's shadow, borrowed for
/// the step:
/// * `tables` — the host kernels' tables (embedding rows, rotary tables, the mask stack), each in
///   the card's own fp16 ([`scratchy_target_spyre::forward_tape::CardTable`]), written by each card
///   straight into its own slot;
/// * `f32` — the constants, narrowed to SEN169 at the bind;
/// * `f16` — already-IEEE-f16 bytes, as-is;
/// * `i32` — the index tables, RAW LE words through `bind_input_raw`;
/// * `seeds` — the chained head's seeds, the one input NOT the same on every card: rank `r` binds
///   the words of ITS OWN vocabulary block ([`scratchy_target_spyre::forward_tape::ChainSeed::words`]).
pub struct StepBinds<'s, 't> {
    pub tables: &'s [(
        scratchy_target_spyre::bundle_code::PlaceId,
        Box<dyn scratchy_target_spyre::forward_tape::CardTable + 't>,
    )],
    pub f32: &'s [scratchy_target_spyre::wiring::Bind],
    pub f16: &'s [(scratchy_target_spyre::bundle_code::PlaceId, Vec<u8>)],
    pub i32: &'s [(scratchy_target_spyre::bundle_code::PlaceId, Vec<u8>)],
    pub seeds: &'s [(
        scratchy_target_spyre::bundle_code::PlaceId,
        scratchy_target_spyre::forward_tape::ChainSeed,
    )],
}

impl StepBinds<'static, 'static> {
    /// A step that binds nothing — its shadow already holds every input.
    pub const NONE: StepBinds<'static, 'static> = StepBinds {
        tables: &[],
        f32: &[],
        f16: &[],
        i32: &[],
        seeds: &[],
    };
}

/// ⭐ ONE CARD'S BINDS OF A STEP, INTO ITS OWN SHADOW — the one bind body every door runs
/// ([`SuperDscSession::bind_all`], card after card on the caller, before any doorbell). One body,
/// so a new channel or a new guard cannot land on one step shape and skip the other.
fn bind_rank<T: ktir_superdsc::collective::Tp>(
    e: &mut Executor<'_, '_, T>,
    b: &StepBinds<'_, '_>,
) -> Result<()> {
    // BIND-LOOP TIMING (SCRATCHY_SDSC_PHASE_TIME, the executor's phase-breakdown flag): the binds
    // run BEFORE the forward's doorbells, so the preamble_h2d/compute/read split cannot see them.
    let t_bind = std::time::Instant::now();
    // ⭐ A HOST KERNEL'S TABLE IS WRITTEN IN THE CARD'S OWN FORMAT, at its own slot — no f32 table,
    // no clone of it, no per-card convert of one.
    for (id, t) in b.tables {
        e.bind_input_table(*id, &**t)?;
    }
    // ⭐ ONE PASS, STRAIGHT INTO THE SHADOW: f32 → SEN169 at the activation's own slot.
    for (id, vals) in b.f32 {
        e.bind_input_f32(*id, vals)?;
    }
    for (id, bytes) in b.f16 {
        e.bind_input(*id, bytes.clone());
    }
    // ⛔ `bind_input_raw`, NOT `bind_input`. See `Executor::raw_ids`: the shadow write applies
    // `ieee_to_sen_bytes` to every ordinary bind, which reads each 4-byte index entry as two fp16
    // values and rewrites both — turning block 37 into a different, valid, arbitrary block.
    // ⛔ AND THE PAYLOAD ARRIVES PAIRED WITH ITS PLACEMENT. `place_raw` is the only constructor of
    // the value `bind_input_raw` takes, so the three ways a raw bind used to go wrong — a byte
    // count that is not whole int32 entries, a payload longer than its own placement, and a name
    // this bundle never placed (silently SKIPPED, and a skipped index reads as entry 0, a real
    // address) — are one refusal at one door.
    for (id, bytes) in b.i32 {
        let bind = e.place_raw(*id, bytes.clone()).ok_or_else(|| {
            anyhow!(
                "bind: raw tensor '{id}' ({} B) has no placement in this bundle, does not fit the \
                 one it has, or is not a whole number of int32 entries — a partial trailing entry \
                 is a truncated address, an over-long bind overwrites the next tensor, and an \
                 unplaced one would be skipped in silence (entry 0 is page 0's first block, a REAL \
                 address)",
                bytes.len()
            )
        })?;
        e.bind_input_raw(bind);
    }
    for (id, seed) in b.seeds {
        let bytes: Vec<u8> = seed
            .words(e.tp_identity())
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let n = bytes.len();
        let bind = e.place_raw(*id, bytes).ok_or_else(|| {
            anyhow!(
                "bind: the chain seed '{id}' ({n} B) has no placement in this bundle or does not \
                 fit the one it has — an unstaged stick gathers the wrong row"
            )
        })?;
        e.bind_input_raw(bind);
    }
    if std::env::var_os("SCRATCHY_SDSC_PHASE_TIME").is_some() {
        let elems: usize = b
            .tables
            .iter()
            .map(|(_, t)| scratchy_target_spyre::forward_tape::card_elems(t.layout()))
            .sum();
        eprintln!(
            "[sdsc-host] bind_loop={:.2} ms  rank {:?}  ({} tables, {elems} sen written; {} f32 \
             acts, {} f16 acts)",
            t_bind.elapsed().as_secs_f64() * 1e3,
            e.tp_identity().map(|r| r.get()),
            b.tables.len(),
            b.f32.len(),
            b.f16.len(),
        );
    }
    Ok(())
}

/// ⭐ ONE SUBMITTED STEP'S LANDING, PER CARD — rank `r`'s [`ForwardLatch`] for the forward that
/// step submitted on card `r`, in rank order. The lead's (rank 0's) is the step's own landing
/// when nobody reads a pick: the launch stream retires the forward in order, so its latch firing
/// IS the step having landed. Every peer's is consumed at a pick's collect
/// ([`SuperDscSession::finish_deferred_read_with`]) — each peer parks its OWN forward of the
/// step and takes its shadow's quiet point, never a whole-stream drain behind it.
pub struct StepLanding {
    /// In rank order: `forwards[r]` is rank `r`'s.
    pub forwards: Vec<ForwardLatch>,
}

impl StepLanding {
    /// The lead card's (rank 0's) forward latch — a readless step's queue-arm form: the
    /// landing the lead card's step queue takes when nobody reads this step's pick. The
    /// forwards are in rank order by construction (one per card the step spans), so rank 0
    /// is the front, never an index to agree on.
    pub fn lead(&self) -> &ForwardLatch {
        &self.forwards[0]
    }
}

/// ⭐ WHAT ONE SUBMITTED STEP OWES ITS CALLER — the typed landing (design §3.2: a step is
/// submitted, not run). The doorbells ring and the host returns; what the step owes rides this
/// value to whoever drains it, in submission order:
/// * `Nothing` — nobody reads this step's pick (a prompt's middle chunk, a discarded tail): the
///   step owes NOTHING. A VALUE, not the refusal the chunk loop used to raise on the chunk that
///   did not finish its prompt — the type states the case, so the refusal cannot exist.
/// * `Picks` — the step's pick is read (a prompt's final chunk, a decode step): the idx-stick
///   read is already deferred onto the ring exactly like a decode's, and the step owes that
///   pick, collected through the one collect door whenever its caller drains.
pub enum ForwardLanding {
    Nothing,
    Picks(DeferredRead),
}

/// The three model extents every constructor here forwards unchanged — one value because they are
/// always read from the same parse and passed together, and separately they pushed `new_inner` past the
/// argument limit. Typed wrappers (`Vocab`/`KvDim`/`PoolPages`) are minted from these downstream.
#[derive(Clone, Copy)]
pub struct SessionDims {
    pub vocab: usize,
    pub kv_dim: usize,
    pub num_blocks: usize,
}

/// LE f16 bytes → f32.
fn f16_le_to_f32(bytes: &[u8]) -> Vec<f32> {
    // `as_chunks` over `chunks_exact`: the chunk width is a CONSTANT, so this yields `&[u8; 2]` and the
    // element indexing below cannot be out of bounds by construction (clippy::chunks_exact_to_as_chunks).
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| half::f16::from_le_bytes(*c).to_f32())
        .collect()
}

/// A resident SUPERDSC session: the dxp-compiled bundle loaded from its AoT cache directory and run
/// through the flex runtime. Weights pack into resident device segments and the KV grows in a
/// resident pool; a submitted step ([`SuperDscSession::submit_bound`]) appends `n_new` NEW
/// positions at `[seq_pos..seq_pos+n_new)` and hands back its landing — what the card owes, not
/// the last new position's logits.
pub struct SuperDscSession<'a, 'c, T: ktir_superdsc::collective::Tp> {
    /// ⭐ ONE EXECUTOR PER CARD THE FORWARD SPANS, in rank order — one for a world of one, `W` for a
    /// sharded world's ranks, all driven by THIS session in one process. Every step is submitted on
    /// every card before any card is waited on (a rank's collective waits on its peers' sends, so a
    /// host that waited on one card first would stall the others' sends behind it), and every
    /// host-staged input is staged on every card (the forward is SPMD: each rank stages the same
    /// activations, its weights are its own shard). The token comes back from rank 0's suffix — the
    /// ranks' replicated values are bit-identical, the canonical reduction order's promise.
    ///
    /// Rank `r`'s executor BORROWS card `r`'s runtime for `'c`, and that runtime borrows its
    /// CARD for `'a`: the cards are opened once, by their owner, and the compiler checks that no
    /// session outlives its runtime (`'c`) and no runtime outlives its card (`'a`).
    ranks: PerRank<T, Executor<'a, 'c, T>>,
    /// Logits column count.
    pub vocab: usize,
    /// One resident K/V row width (= kv_heads·head_dim).
    pub kv_dim: usize,
    /// ⛔ THE BROADCAST PREFIX MASK'S EXTENT, from the bundle's own placement — see
    /// [`Executor::pmask_slots`]. NOT `cap`: `cap` is the resident POOL's slot count (what the KV probes
    /// read back at), and using it for the mask is what over-bound the mask by 4x on every prompt.
    /// `None` ⇒ the bundle placed no broadcast mask, and staging one must be REFUSED, not guessed.
    pub pmask_slots: Option<usize>,
}

/// One card's resident weights, `(source id, bytes, dtype, shape)` each. The bytes are by
/// provenance ([`scratchy_target_spyre::allocator::HostBytes`]): a weight the load viewed off its mmap stays a
/// `View` through the bind — the executor's staging walk reads the page cache directly and the
/// keepalive lives as long as the bound entry does.
pub type CardWeights = Vec<(
    usize,
    scratchy_target_spyre::allocator::HostBytes,
    SDType,
    Vec<usize>,
)>;

/// The resident weights of every card a session spans — one set per rank, rank `r`'s its own shard.
/// The OWNER session's only; a borrower binds none (`None`), so "one set per card" is a type, never a
/// count to check.
pub type RankWeights<T> = Option<PerRank<T, CardWeights>>;

/// What a session binds before prepare — the private bundle of [`SuperDscSession::new_inner`]'s
/// staged inputs.
struct Staged<'a, T: ktir_superdsc::collective::Tp> {
    /// The resident weights, moved in by the OWNER (the decode session) — one set per card, rank
    /// `r`'s its own shard — and empty for every borrower, which reads the owner's bytes through the
    /// alias.
    weights: RankWeights<T>,
    /// The wiring's staged-constant list — bound once before prepare (see [`SuperDscSession::new`]).
    load_consts: &'a [(u32, scratchy_target_spyre::wiring::LoadConst)],
    /// The KV pool's own tables — bound once, verbatim, by each session whose kernels read them.
    pool_statics: &'a [(u32, Vec<u8>)],
}

/// ⭐ EVERY RANK'S `f(r, its input)` AT ONCE — one scoped thread per rank, the results in rank
/// order. A sharded world's BUILD work (its staging walks, its per-card session builds) is per
/// rank and independent — each rank's thread touches only its own card and its own input — and
/// run serially the last rank starts a whole world minus one builds after the first (measured at
/// tp/8: staging 4.0 s as eight sequential walks, the dxp session 3.1 s as eight sequential
/// per-card builds; each is ~the single-card wall on its own). A world of one runs on the
/// caller's thread — the same law as [`SuperDscSession::each_parallel`].
///
/// Each rank's `input` is MOVED into its thread: the one way a unique resource (the allocator
/// whose buffers exactly one rank's staging takes) crosses the fan-out without a shared `&mut`.
/// Threads are spawned for EVERY rank first and joined in a second pass (the prefill ladder's
/// own shape); a panicked rank re-panics on the caller with its own payload, and the first error
/// in rank order is the one returned — the same rank the serial map would have stopped at.
/// NUMA pinning is the caller's law (the session builds pin to their card's CPUs; staging runs
/// where the loader ran, as it always has).
pub fn each_rank_parallel<T, I, X, E>(
    inputs: PerRank<T, I>,
    f: impl Fn(ktir_superdsc::collective::Rank<T>, I) -> Result<X, E> + Sync,
) -> Result<PerRank<T, X>, E>
where
    T: ktir_superdsc::collective::World,
    I: Send,
    X: Send,
    E: Send,
{
    use ktir_superdsc::collective::{Rank, ranks};
    let mut slots: Vec<Option<I>> = inputs.into_values().map(Some).collect();
    let mut take_input = |r: Rank<T>| {
        slots[r.get() as usize]
            .take()
            .expect("every rank's input is taken exactly once")
    };
    // A world of one runs on the caller's thread — nothing to fan out over.
    if !T::SHARDED {
        return PerRank::try_from_fn(|r| f(r, take_input(r)));
    }
    let f = &f;
    std::thread::scope(|scope| {
        // Spawn EVERY rank first, join in a second pass — a map that joins inside the spawn
        // loop would run each rank's build to completion before the next thread exists.
        let handles: Vec<_> = ranks::<T>()
            .map(|r| {
                let input = take_input(r);
                std::thread::Builder::new()
                    .name(format!("BuildR{}", r.get()))
                    .spawn_scoped(scope, move || f(r, input))
                    // A spawn refusal at boot is a resource-exhaustion event, not a load error
                    // to route around — the ladder fan-outs' own shape (`scope::spawn` panics).
                    .unwrap_or_else(|err| panic!("rank {}: spawn its build thread: {err}", r.get()))
            })
            .collect::<Vec<_>>();
        // `handles` is in rank order (built over `ranks`), so the consuming iterator joins in
        // rank order — the first error the serial map would have stopped at is the one returned.
        let mut joins = handles.into_iter();
        PerRank::try_from_fn(|_| {
            joins
                .next()
                .expect("one handle per rank")
                .join()
                .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
        })
    })
}

impl<'a, 'c, T: ktir_superdsc::collective::Tp> SuperDscSession<'a, 'c, T> {
    /// Load the bundle, bind the weights, and prepare (allocate + upload the resident program,
    /// weights and KV pool).
    ///
    /// ⭐ THE GRAPH'S OWN SOURCE FACTS ARE READ OFF `code`, NEVER ARGUED: the
    /// source count and the resident source ids are stamped on the bundle's
    /// layout at the bake, so a session built here walks exactly the source
    /// range of the graph it runs — a caller cannot hand it another graph's
    /// numbers. Only the spilled weight tail is an argument (a load-time copy
    /// decision the layout cannot make for the caller).
    ///
    /// `load_consts` is the wiring's OWN staged-constant list — the tables a
    /// spliced kernel declares as sources. They are bound ONCE here, before
    /// prepare, resident like a weight (each is a function of the graph
    /// alone); a session that binds none passes an empty slice. A BORROWING
    /// rung binds its OWN copy through
    /// [`new_borrowing_with_consts`](Self::new_borrowing_with_consts) — the
    /// tables sit in the per-session activation segment, which no alias
    /// covers.
    ///
    /// `pool_statics` are the KV pool's own int32 tables
    /// ([`scratchy_target_spyre::paged_step::card_statics`]) — staged VERBATIM once here by the session whose
    /// bundles read them (the one-request decode's).
    pub fn new(
        cards: &'c PerRank<T, crate::runtime::CardRuntime<'a, T>>,
        code: &'static scratchy_target_spyre::bundle_code::BundleCode<'static, T>,
        dims: SessionDims,
        weights: RankWeights<T>,
        load_consts: &[(u32, scratchy_target_spyre::wiring::LoadConst)],
        pool_statics: &[(u32, Vec<u8>)],
        spilled_tail: &[u32],
    ) -> Result<Self> {
        Self::new_inner(
            cards,
            code,
            dims,
            Staged {
                weights,
                load_consts,
                pool_statics,
            },
            &[],
            spilled_tail,
        )
    }

    /// [`new`](Self::new) for a session whose WEIGHTS come from another session by alias — every rung
    /// of the prefill width ladder. Binds nothing and stages nothing: each `borrowed` segment is
    /// declared dead-on-arrival before prepare, so prepare gives it a placeholder and skips its H2D.
    ///
    /// The caller MUST [`alias_seg_from`](Self::alias_seg_from) every listed segment immediately
    /// after this returns — a borrowed segment left un-aliased holds ONLY the placeholder, so a
    /// failed alias must be fatal for this session, never a warning that leaves it on a stub.
    pub fn new_borrowing(
        cards: &'c PerRank<T, crate::runtime::CardRuntime<'a, T>>,
        code: &'static scratchy_target_spyre::bundle_code::BundleCode<'static, T>,
        vocab: usize,
        kv_dim: usize,
        num_blocks: usize,
        borrowed: &[i64],
        spilled_tail: &[u32],
    ) -> Result<Self> {
        Self::new_borrowing_with_consts(
            cards,
            code,
            SessionDims {
                vocab,
                kv_dim,
                num_blocks,
            },
            borrowed,
            &[],
            &[],
            spilled_tail,
        )
    }

    /// [`new_borrowing`](Self::new_borrowing) for a session that ALSO binds its own staged
    /// constants — every BATCHED DECODE rung. The sampler's nine tables are SOURCES placed in
    /// the ACTIVATION segment, which is never aliased (each session's seg3 is its own region,
    /// and the tables' offsets are rung-local: the `[rows, PAD_V]` noise block packs ahead of
    /// them at THIS rung's row count), so a borrower that binds nothing reads ZEROS where its
    /// tables sit. Each rung stages its own copy — a few MB against the rung's programs, once
    /// at load — through the same load-time channel the owner uses.
    pub fn new_borrowing_with_consts(
        cards: &'c PerRank<T, crate::runtime::CardRuntime<'a, T>>,
        code: &'static scratchy_target_spyre::bundle_code::BundleCode<'static, T>,
        dims: SessionDims,
        borrowed: &[i64],
        load_consts: &[(u32, scratchy_target_spyre::wiring::LoadConst)],
        // ⛔ THE POOL'S OWN TABLES, FOR A SESSION WHOSE ATTENTION READS THEM. They are NOT covered by
        // `borrowed`: the pool statics are placed as an INTERMEDIATE colour segment
        // (`CardSeat::Static` -> `SegRole::Intermediate`), and the alias list is the RESIDENT segments
        // (weights, KV) alone — so a borrower that reads `kv_row_pool` and passes none here gathers
        // through an UNSTAGED buffer. That is a clean launch and garbage from the first token, which
        // is what the prompt trip did on the card the moment it began reading the pool. Same reason
        // `load_consts` is per-session here rather than aliased.
        pool_statics: &[(u32, Vec<u8>)],
        spilled_tail: &[u32],
    ) -> Result<Self> {
        let mut sess = Self::build_borrowing_with_consts(
            cards,
            code,
            dims,
            borrowed,
            load_consts,
            pool_statics,
            spilled_tail,
        )?;
        sess.prepare()?;
        Ok(sess)
    }

    /// [`new_borrowing_with_consts`](Self::new_borrowing_with_consts) WITHOUT the device half —
    /// the HOST build alone, for a ladder fan-out that calls [`prepare`](Self::prepare) later,
    /// once the owner session has allocated. The split changes nothing the borrowing law demands
    /// of the caller: the `alias_seg_from` adoption still runs after that prepare, on the thread
    /// that owns the owner, in ladder order.
    pub fn build_borrowing_with_consts(
        cards: &'c PerRank<T, crate::runtime::CardRuntime<'a, T>>,
        code: &'static scratchy_target_spyre::bundle_code::BundleCode<'static, T>,
        dims: SessionDims,
        borrowed: &[i64],
        load_consts: &[(u32, scratchy_target_spyre::wiring::LoadConst)],
        pool_statics: &[(u32, Vec<u8>)],
        spilled_tail: &[u32],
    ) -> Result<Self> {
        Self::build_inner(
            cards,
            code,
            dims,
            Staged {
                weights: None,
                load_consts,
                pool_statics,
            },
            borrowed,
            spilled_tail,
        )
    }

    /// Allocate every segment at FULL size but populate none — no weights bound, so staging no-ops.
    ///
    /// This exists for the batched-prefill session, whose 2.6 GB seg1 reservation is what PLACES the
    /// decode session's weight region: borrowing (a 128 B placeholder) instead moves decode's weights
    /// ~2.6 GB earlier in HBM and costs 1.8 tok/s, measured. The placement effect comes from the
    /// ALLOCATION, not from the bytes — and those bytes are dead, since the session is aliased onto
    /// decode's copy immediately afterwards. So reserving without populating keeps the placement and
    /// drops the bind + stage (6.97s of startup for granite-3.1-8b, measured).
    ///
    /// `load_consts` — the prefill wiring's own staged-constant list. The general mint puts the
    /// sampler's nine tables in EVERY prefill rung too, and they sit in the session's own seg3
    /// (never aliased), so the anchor stages its own copy exactly as a borrowing rung does.
    pub fn new_reserving(
        cards: &'c PerRank<T, crate::runtime::CardRuntime<'a, T>>,
        code: &'static scratchy_target_spyre::bundle_code::BundleCode<'static, T>,
        dims: SessionDims,
        load_consts: &[(u32, scratchy_target_spyre::wiring::LoadConst)],
        // The pool's own tables, for the same reason the borrowing constructor takes them.
        pool_statics: &[(u32, Vec<u8>)],
        spilled_tail: &[u32],
    ) -> Result<Self> {
        Self::new_inner(
            cards,
            code,
            dims,
            Staged {
                weights: None,
                load_consts,
                pool_statics,
            },
            &[],
            spilled_tail,
        )
    }

    /// Every constructor, composed: the HOST half then the DEVICE half. The ladder fan-outs are
    /// the one caller that splits them (see
    /// [`build_borrowing_with_consts`](Self::build_borrowing_with_consts) and
    /// [`prepare`](Self::prepare)); everyone else gets both halves in the order they always had.
    fn new_inner(
        cards: &'c PerRank<T, crate::runtime::CardRuntime<'a, T>>,
        code: &'static scratchy_target_spyre::bundle_code::BundleCode<'static, T>,
        dims: SessionDims,
        staged: Staged<'_, T>,
        borrowed: &[i64],
        spilled_tail: &[u32],
    ) -> Result<Self> {
        let mut sess = Self::build_inner(cards, code, dims, staged, borrowed, spilled_tail)?;
        sess.prepare()?;
        Ok(sess)
    }

    /// ⭐ THE HOST HALF of every constructor — `Executor::load`, the source and borrow
    /// declarations, the weight and constant binds, and the staging walk. It touches NO device
    /// state: every allocation, program upload, H2D and sync lives in [`prepare`](Self::prepare),
    /// so what this returns is a plan plus host bytes and nothing on the card. The ladder
    /// fan-outs build their rungs through this half WHILE the owner session builds — the
    /// owner-first laws order the ladder's DEVICE half, not this one.
    fn build_inner(
        cards: &'c PerRank<T, crate::runtime::CardRuntime<'a, T>>,
        code: &'static scratchy_target_spyre::bundle_code::BundleCode<'static, T>,
        dims: SessionDims,
        staged: Staged<'_, T>,
        borrowed: &[i64],
        // The spilled weight tail's tids — the ONE resident slice the layout cannot state: the
        // spill is a load-time copy decision (a weight whose slot also holds an m-dependent
        // intermediate color cannot be aliased, so `adopt_from_owner` copies it into this
        // session's own region), so the LOADER declares it resident here, exactly as it hands it
        // over. Leaving it out is a REFUSED LAUNCH, not a silent wrong answer — the completeness
        // check classifies the un-copied tail `SourceFiller::Nobody` and stops.
        spilled_tail: &[u32],
    ) -> Result<Self> {
        let SessionDims {
            vocab,
            kv_dim,
            num_blocks,
        } = dims;
        let Staged {
            weights,
            load_consts,
            pool_statics,
        } = staged;
        // The owner binds rank r's weight set on card r; a borrower binds none on any.
        let weights: PerRank<T, CardWeights> = match weights {
            Some(w) => w,
            None => PerRank::from_fn(|_| CardWeights::new()),
        };
        // ⭐ EVERY RANK'S BUILD AT ONCE — [`each_rank_parallel`]: a rank's HOST build (its
        // executor load, its binds, its stage) touches only its own card and its own weight set,
        // so the ranks build concurrently instead of the last card starting a whole world minus
        // one builds after the first (measured at tp/8: the dxp session alone was 3.1 s of
        // sequential per-card builds). The DEVICE half ([`prepare`](Self::prepare)) is a separate
        // wave the ladder laws order. The per-card alloc order (anchor → owner → rungs) is PER
        // CARD and each rank's thread keeps its own card's order; cross-card order was never
        // constrained — device memory is per card.
        let ranks: PerRank<T, Executor<'a, 'c, T>> =
            each_rank_parallel(weights, |r, card_weights| -> Result<Executor<'a, 'c, T>> {
                let card = cards.get(r);
                // This rank's build thread runs on its card's NUMA node before it touches the card
                // (its doorbell and its host writes stay on the card's socket) — the same law as
                // [`SuperDscSession::each_parallel`]'s submits. A world of one runs this closure on
                // the caller's thread, already where the world's shared work runs, and skips the pin.
                if <T as ktir_superdsc::collective::World>::SHARDED {
                    card.cpus()
                        .pin_current_thread()
                        .map_err(anyhow::Error::from)?;
                }
                let mut exec = Executor::load(
                    card,
                    code,
                    Vocab(vocab),
                    KvDim(kv_dim),
                    PoolPages(num_blocks as i64),
                )?;
                // ⭐ THE GRAPH'S OWN SOURCE FACTS — the count and the resident ids are stamped on the
                // layout this session's `code` carries, so the completeness walk covers exactly the
                // source range of the graph being run: prefix-KV caches and load-const tables excused
                // by the bundle's own list, the spilled tail by the caller's, and nothing else.
                exec.declare_sources(
                    code.layout.num_sources,
                    code.layout
                        .resident_source_tids
                        .iter()
                        .copied()
                        .chain(spilled_tail.iter().copied()),
                );
                // BORROWED segments must be declared BEFORE prepare — afterwards the full region is already
                // allocated and transferred, so the flag would be a silent no-op (the executor refuses).
                for &seg in borrowed {
                    let s = SegIdx::checked(seg)
                        .ok_or_else(|| anyhow!("mark_borrowed: segment {seg} out of range"))?;
                    exec.mark_borrowed(s)?;
                }

                // Bind every weight to its input `t{id}` as device-dtype bytes. Prepare packs them, so they
                // must be bound before prepare. Ownership MOVES in — no copy anywhere on this path: a
                // VIEW moves as its keepalive handle and the staging walk reads the mmap itself.
                let t_bind = std::time::Instant::now();
                let n_bound = card_weights.len();
                for (id, data, dt, _shape) in card_weights {
                    let bytes = match dt {
                        // Already the device dtype (the narrowing pass's product, or fp8 verbatim —
                        // do NOT dequant to f16): the bytes bind as they are, view or owned.
                        SDType::F16 | SDType::Fp8E4m3 => data,
                        other => scratchy_target_spyre::allocator::HostBytes::Owned(f32_to_f16_le(
                            &bytes_to_f32(data.as_slice(), other),
                        )),
                    };
                    exec.bind_weight(
                        scratchy_target_spyre::bundle_code::PlaceId::Act(id as u32),
                        bytes,
                    );
                }
                // ⭐ THE STAGED CONSTANTS ride the same law: bound once, before
                // prepare, into the same weight channel `stage_weights` converts and
                // uploads — a table the owner never binds reads as ZEROS on card, and
                // the kernel composes its answer from zeros (measured: the argmax
                // mint's first card run answered token 0 — EOS — on every decode
                // step, while the build's own numeric gate passed, because the gate's
                // emulator binds its own sources and never crosses this seat).
                exec.bind_load_consts(load_consts)?;
                // The pool's int32 tables ride the same verbatim channel as an i32 load constant: their LE
                // words ARE the device's.
                for (tid, bytes) in pool_statics {
                    exec.bind_const_fp32(
                        scratchy_target_spyre::bundle_code::PlaceId::Act(*tid),
                        bytes.clone(),
                    )?;
                }
                let n_staged = n_bound + load_consts.len() + pool_statics.len();
                debug!(
                    "[timing] bind weights loop took {:.2}s",
                    t_bind.elapsed().as_secs_f64()
                );

                // Convert + stage the bound weights into host staging NOW, on this thread, while the
                // background device prewarm is still bringing the card up — this takes the ~1s parallel
                // IEEE→SEN convert off the prepare critical path. Bytes reaching the device are identical;
                // pure scheduling.
                //
                // NOTHING BOUND is the same situation as BORROWED: with no weights this call only zero-fills
                // a host shadow of a segment that is about to be replaced by an alias. A BORROWER THAT
                // BINDS STAGED CONSTANTS still stages: `stage_weights` gives every borrowed segment its
                // 128 B placeholder, so the only shadow this writes is the ACTIVATION segment's — the one
                // holding this session's own load-const tables, which no alias can deliver (the tables are
                // per-session placements, and a rung's seg3 is its own region).
                if n_staged == 0 {
                    debug!(
                        "[timing] weight bind + stage SKIPPED (nothing bound — segments reserved, \
                     populated by alias)"
                    );
                } else {
                    let t_stage = std::time::Instant::now();
                    exec.stage_weights()?;
                    debug!(
                        "[timing] weight stage took {:.2}s",
                        t_stage.elapsed().as_secs_f64()
                    );
                }

                Ok(exec)
            })?;
        let exec = ranks.first();
        Ok(Self {
            vocab: exec.vocab.0,
            kv_dim: exec.kv_dim.0,
            pmask_slots: exec.pmask_slots(),
            ranks,
        })
    }

    /// ⭐ THE DEVICE HALF — every rank's [`Executor::prepare`]: the stream, the program and
    /// tensor-segment allocations, the sends and the sync. Idempotent, and composed into every
    /// `new*` constructor; the ladder fan-outs call it as a separate wave (see
    /// [`Self::prepare_in_rank_order`]) so a rung's host build hides inside the owner's while the
    /// rung's ALLOCATIONS still happen only after the owner's — the tp-window and weight-region
    /// placement laws are laws about THIS half alone.
    ///
    /// ⭐ A LONE SESSION'S RANKS PREPARE TOGETHER ([`Self::each_parallel`], the host half's own
    /// law): one thread per rank on its card's node. Run serially the half re-serializes what the
    /// host fan-out parallelized (measured at tp/2: the owner's two prepares back-to-back cost
    /// 0.34 s of boot).
    pub fn prepare(&mut self) -> Result<()> {
        self.each_parallel("Prepare", |e| e.prepare()).map(drop)
    }

    /// [`Self::prepare`] with the WAVE's rank schedule: ranks in rank order, on the caller's
    /// thread. A wave's concurrency is its SESSION count — one thread per rung, the width a
    /// one-card boot already rides (a k=1 wave is this exact shape and is the fastest measured) —
    /// so a session preparing among many walks its ranks in order instead of fanning them:
    /// fanned, tp/8's wave was 33 sessions × 8 ranks = 264 prepare threads on one host driver
    /// path, measured 3755 ms for ~a tenth of that in per-card device work; in rank order the
    /// thread count stays at the session count and the cards queue the same device work.
    pub fn prepare_in_rank_order(&mut self) -> Result<()> {
        self.each(|e| e.prepare()).map(drop)
    }

    /// ⭐ RANK 0 — the card whose readback a step returns. In a sharded world every value the host
    /// reads (the picked token, the logits row) is replicated across the ranks — the forward ends in
    /// the whole hidden state on every rank, so each rank's suffix computes the same row — so one
    /// rank answers for all.
    fn lead(&mut self) -> &mut Executor<'a, 'c, T> {
        self.ranks.first_mut()
    }

    /// Apply `f` on EVERY card, rank order, stopping at the first refusal (named by its rank in a
    /// sharded world).
    fn each<R>(
        &mut self,
        mut f: impl FnMut(&mut Executor<'a, 'c, T>) -> Result<R>,
    ) -> Result<Vec<R>> {
        use ktir_superdsc::collective::World;
        self.ranks
            .iter_mut()
            .map(|(r, e)| {
                f(e).map_err(|err| {
                    if <T as World>::SHARDED {
                        err.context(format!("rank {}", r.get()))
                    } else {
                        err
                    }
                })
            })
            .collect()
    }

    /// ⭐ [`Self::each`] ON EVERY CARD AT ONCE — one scoped thread per rank, the results in rank
    /// order. A sharded step's host work (its refill, its CB render, its doorbell) is per card and
    /// independent, and run serially the last card starts a whole `W - 1` submits after the first,
    /// stalling every collective that waits on it (measured at tp/8: ~2.2 ms a rank, the eighth
    /// card's doorbell 18 ms into the step). A world of one runs on the caller's thread.
    ///
    /// Each rank's thread runs on its card's NUMA node (`CardRuntime::cpus`) before it touches the
    /// card: its doorbell and its host writes stay on the card's socket.
    ///
    /// `name` prefixes the per-rank thread name, so the thread's phase (a step submit, a boot
    /// prepare) and its placement can be read per rank.
    fn each_parallel<R: Send>(
        &mut self,
        name: &str,
        f: impl Fn(&mut Executor<'a, 'c, T>) -> Result<R> + Sync,
    ) -> Result<Vec<R>> {
        use ktir_superdsc::collective::World;
        if !<T as World>::SHARDED {
            return self.each(f);
        }
        let f = &f;
        std::thread::scope(|scope| {
            let handles = self
                .ranks
                .iter_mut()
                .map(|(r, e)| {
                    // Named `<name>R<rank>`, so the thread's phase and placement can be read per
                    // rank.
                    std::thread::Builder::new()
                        .name(format!("{name}R{}", r.get()))
                        .spawn_scoped(scope, move || {
                            let pinned = e
                                .card_cpus()
                                .pin_current_thread()
                                .map_err(anyhow::Error::from);
                            pinned
                                .and_then(|()| f(e))
                                .map_err(|err| err.context(format!("rank {}", r.get())))
                        })
                        .map_err(|err| anyhow!("rank {}: spawn its submit thread: {err}", r.get()))
                })
                .collect::<Result<Vec<_>>>()?;
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
                .collect()
        })
    }

    /// ⭐ EVERY CARD'S BINDS OF A STEP, CARD AFTER CARD ON THE CALLER, BEFORE ANY DOORBELL — so no
    /// card rings for a step another card failed to bind (the first error returns before the
    /// submit section starts). Returns how long the binds took, the bind row of the step segments.
    fn bind_all(&mut self, b: &StepBinds<'_, '_>) -> Result<std::time::Duration> {
        let t = std::time::Instant::now();
        self.each(|e| bind_rank(e, b))?;
        Ok(t.elapsed())
    }

    /// ⭐⭐⭐⭐⭐ WHICH BODY THE NEXT LAUNCH AT `start` WILL RUN, and what it declares — see
    /// [`superdsc_exec::StepBody`]. The host asks this BEFORE it stages anything, because both the
    /// gather's index table and the prefix mask's block form are decisions about the selected body.
    ///
    /// ⛔ `start` IS THE SAME `seq_pos` THE HOST PASSES TO `submit_bound`, and the selection is derived from it
    /// alone — so the body staged for and the body launched are the same body by construction, not by two
    /// call sites agreeing.
    ///
    /// [`superdsc_exec::StepBody`]: crate::superdsc_exec::StepBody
    /// ⭐ WHAT THIS SESSION'S OWN BUNDLE PLACED ([`scratchy_target_spyre::wiring::BakeFacts::of`] its layout) — the
    /// constants a launch on it binds, its scalar registry index for index. Every rung's bundle
    /// places its own: a decode batch rung's registry carries the batched kernel's clamp, which the
    /// one-request decode's does not, so a launch bound through another bundle's facts read that
    /// slot unbound.
    pub fn bake_facts(&self) -> scratchy_target_spyre::wiring::BakeFacts {
        scratchy_target_spyre::wiring::BakeFacts::of(&self.ranks.first().code().layout)
    }

    pub fn step_body(&mut self, start: usize) -> Result<crate::superdsc_exec::StepBody> {
        // Every rank runs the same bundle at the same position, so every rank selects the same body.
        self.lead()
            .step_body(crate::superdsc_exec::SeqPos(start as i64))
    }

    /// The [`CausalKey`] this session's staged activation `id` was last bound under, if it was
    /// bound keyed. A forward whose key equals it has nothing to bind for `id`.
    ///
    /// [`CausalKey`]: ktir_superdsc::sdsc_abstract::CausalKey
    pub fn keyed_holds(
        &self,
        id: scratchy_target_spyre::bundle_code::PlaceId,
    ) -> Option<ktir_superdsc::sdsc_abstract::CausalKey> {
        self.ranks.first().keyed_holds(id)
    }

    /// ⭐ BIND AND SUBMIT ONE PIPELINED STEP — every card's binds ([`Self::bind_all`]), then
    /// every card's doorbell in one parallel section, waiting on nothing. §3.5's protected
    /// shape, unchanged: ONE bind pass, ONE parallel submit — problem 1's measured work, byte
    /// for byte. Returns the binds' time (the bind row of the step segments) and the step's
    /// landing: rank `r`'s [`ForwardLatch`] for the forward this step submitted on card `r`,
    /// armed on the forward's last submission, so the caller arms the lead card's step queue
    /// with it, probes what the card still owes, and settles every peer at a pick's collect.
    /// Every launch binds — a step whose shadow already holds every input binds the empty
    /// [`StepBinds::NONE`] — which is why there is no second, bindless submit door.
    pub fn submit_bound(
        &mut self,
        b: &StepBinds<'_, '_>,
        n_new: usize,
        seq_pos: usize,
        suffix: SuffixLaunch,
    ) -> Result<(std::time::Duration, StepLanding)> {
        let bind = self.bind_all(b)?;
        // ⭐ EVERY CARD'S DOORBELL BEFORE ANY WAIT — this door waits on nothing, so the ranks' steps
        // are all in flight together and their collectives meet on the cards.
        let t0 = std::time::Instant::now();
        let landed = self.each_parallel("Submit", |e| {
            let landed = e.submit_step(NewTokens(n_new as i64), SeqPos(seq_pos as i64), suffix)?;
            Ok((landed, t0.elapsed().as_secs_f64() * 1e3))
        })?;
        if std::env::var_os("SCRATCHY_SDSC_RB_TIME").is_some() {
            let at = landed.iter().map(|(_, ms)| *ms).collect::<Vec<_>>();
            eprintln!("[sdsc-submit] step {seq_pos}: rank submits done at {at:.2?} ms");
        }
        Ok((
            bind,
            StepLanding {
                forwards: landed.into_iter().map(|(l, _)| l).collect(),
            },
        ))
    }

    /// ⭐ THE LEAD CARD'S STEP QUEUE — the submission-order cadence over the launch stream's
    /// landings. The single submitting thread arms it with each step's landing right after the
    /// submit that carries it (a readless step's forward latch, a read-bearing step's read —
    /// design §3.8.2), and the drain pops the oldest. The depth and busy facts the step
    /// segments read are THE QUEUE'S OWN (`depth`, and the peek of the last-armed landing) —
    /// never a host count restated beside them.
    pub fn lead_steps(&mut self) -> &flex_rs::step::StepQueue<'a> {
        self.lead().steps()
    }

    /// ⭐ RETIRE THE LEAD'S LANDED STEPS THROUGH ONE — pop the lead card's queue oldest-first
    /// until (and including) `through`, the landing the collected step armed. The launch
    /// stream is in order, so a step whose read has landed has every older landing fired:
    /// each pop's park returns immediately, and what the queue keeps is exactly the steps
    /// still outstanding — the depth fact, returned to truth at every collect. Called by the
    /// one collect door, with the collected step's own read as `through`.
    pub fn drain_steps_through(
        &mut self,
        through: &std::sync::Arc<flex_rs::step::Landing>,
    ) -> Result<()> {
        loop {
            let popped = self
                .lead_steps()
                .drain_oldest()
                .map_err(|e| anyhow!("{e}"))?;
            if std::sync::Arc::ptr_eq(&popped, through) {
                return Ok(());
            }
        }
    }

    /// ⭐ #32 (A) — ENQUEUE THE STEP'S STICK READ, no drain: the D2H rides the
    /// compute pipeline behind the step's CBs, and the host keeps working until
    /// the loop's ONE fence (the runner spelling of
    /// `Executor::defer_read_tensor_raw`).
    pub fn defer_read_tensor_raw(
        &mut self,
        id: scratchy_target_spyre::bundle_code::PlaceId,
        cap_bytes: usize,
    ) -> Result<DeferredRead> {
        self.lead().defer_read_tensor_raw(id, cap_bytes)
    }

    /// ⭐ #32 (C) — THE RING TWIN, for a pick not consumed until a later drain:
    /// two outstanding shadow reads for the same stick race, so an in-flight
    /// step's read lands in its own ring slot (the runner spelling of
    /// `Executor::defer_read_tensor_ring`).
    pub fn defer_read_tensor_ring(
        &mut self,
        id: scratchy_target_spyre::bundle_code::PlaceId,
        cap_bytes: usize,
    ) -> Result<DeferredRead> {
        self.lead().defer_read_tensor_ring(id, cap_bytes)
    }

    /// ⭐ #32 (A) — THE LOOP'S ONE FENCE, ZERO-ALLOC: park on this read's own
    /// completion (and the shadow's quiet point), then hand the landed window
    /// to `f` BY BORROW — no copy, no fresh Vec per step (the runner spelling
    /// of `Executor::finish_deferred_read_with`). `step` is the landing the
    /// step that deferred this read handed back: every PEER settles on its own
    /// forward of that step here — the collect is where a sharded world
    /// retires the step on every card.
    pub fn finish_deferred_read_with<R>(
        &mut self,
        read: &DeferredRead,
        step: &StepLanding,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<R> {
        // Rank 0's read lands behind its step. Every other rank settles on its OWN forward of
        // that step — NEVER a whole-stream drain: each peer's stream already holds the NEXT step
        // the pipelined loop submitted, and draining it would wait that step out too, so the card
        // would idle through every submit (measured at tp/8: busy at 1% of submits, against
        // tp/1's 99%). The forwards are zipped rank-to-rank — both are in rank order by
        // construction, so no index can disagree.
        let out = self.lead().finish_deferred_read_with(read, f)?;
        for ((r, e), l) in self.ranks.iter_mut().zip(&step.forwards) {
            if r.get() != 0 {
                e.settle_peer_pick(l)?;
            }
        }
        Ok(out)
    }

    /// The same fence COMPOSED WITH A COPY — the one-shot owned contract over
    /// [`Self::finish_deferred_read_with`] (the runner spelling of
    /// `Executor::finish_deferred_read_raw`).
    pub fn finish_deferred_read_raw(
        &mut self,
        read: &DeferredRead,
        step: &StepLanding,
    ) -> Result<Vec<u8>> {
        self.finish_deferred_read_with(read, step, |bytes| bytes.to_vec())
    }

    /// Bind `id` as a function of `key` alone, narrowed to f16 here. The next step uploads it, and
    /// no later step re-uploads it until another key is bound (see `Executor::bind_input_keyed`).
    pub fn bind_keyed(
        &mut self,
        id: scratchy_target_spyre::bundle_code::PlaceId,
        key: ktir_superdsc::sdsc_abstract::CausalKey,
        vals: &[f32],
    ) -> Result<()> {
        let bytes = f32_to_f16_le(vals);
        self.each(|e| e.bind_input_keyed(id, key, bytes.clone()))
            .map(|_| ())
    }

    /// [`Self::bind_keyed`], uploaded NOW rather than by the next step — a session's load-time
    /// staging of what its first launch would bind (see `Executor::stage_keyed`). Returns the bytes
    /// sent.
    pub fn stage_keyed(
        &mut self,
        id: scratchy_target_spyre::bundle_code::PlaceId,
        key: ktir_superdsc::sdsc_abstract::CausalKey,
        vals: &[f32],
    ) -> Result<u64> {
        let bytes = f32_to_f16_le(vals);
        Ok(self
            .each(|e| e.stage_keyed(id, key, bytes.clone()))?
            .into_iter()
            .sum())
    }

    /// ⭐⭐⭐ THE ONCARD HAND-OFF — the sticks a stable decode seating leaves to the card's own
    /// ChainStage writes. Ends the refill's re-send of a previous `Host` launch's seed bind
    /// (see `Executor::hand_to_card`); a churned seating's re-bind re-inserts it.
    pub fn hand_to_card(&mut self, id: scratchy_target_spyre::bundle_code::PlaceId) {
        for e in self.ranks.values_mut() {
            e.hand_to_card(id);
        }
    }

    // ── DEBUG numeric bisection (SPYRE_SUPERDSC_SELFTEST). NOT a serving path. ──

    /// Bind an activation source (f32 → f16) before a prefix run.
    pub fn bind(
        &mut self,
        id: scratchy_target_spyre::bundle_code::PlaceId,
        vals: &[f32],
    ) -> Result<()> {
        let bytes = f32_to_f16_le(vals);
        for e in self.ranks.values_mut() {
            e.bind_input(id, bytes.clone());
        }
        Ok(())
    }

    /// Run ONLY the prefix program (after binding synthetic activations).
    pub fn run_prefix_only(&mut self) -> Result<()> {
        self.each(Executor::run_prefix_only).map(|_| ())
    }

    /// Run the FULL forward (prefix → body×iters → seam → suffix) after binding synthetic
    /// activations, leaving every tid resident. `Ok(false)` if the bundle is not the per-op rolled
    /// path.
    pub fn run_full(&mut self) -> Result<bool> {
        Ok(self.each(Executor::run_full)?.into_iter().all(|ok| ok))
    }

    /// Read back a tensor (up to `cap` elems) from its resident segment.
    pub fn read_tensor(
        &mut self,
        id: scratchy_target_spyre::bundle_code::PlaceId,
        cap: usize,
    ) -> Result<Vec<f32>> {
        Ok(f16_le_to_f32(&self.lead().read_tensor(id, cap)?))
    }

    /// ⭐ THE GENERIC RAW-READ VARIANT of [`Self::read_tensor`] — the tensor's
    /// own bytes, VERBATIM, no f16 decode. A step whose rung's result is not
    /// the f16 row reads the result here: the CALLER decodes at the dtype the
    /// wiring's `result_dtype` states (an index stick's raw LE int32 words),
    /// because only the wiring knows it.
    pub fn read_tensor_raw(
        &mut self,
        id: scratchy_target_spyre::bundle_code::PlaceId,
        cap_bytes: usize,
    ) -> Result<Vec<u8>> {
        self.lead().read_tensor_raw(id, cap_bytes)
    }

    /// Zero a whole tensor segment ON THE DEVICE, without a host mirror of it.
    ///
    /// The KV pool must start zeroed (decode computes q·k over a whole page before the validity mask
    /// applies, so uninitialised bytes go through the arithmetic as denormals/NaN). Doing that as
    /// read_seg → write_seg required a host buffer the size of the pool — 960 MB, and 3.84 GB at a
    /// 1024-slot page — to mirror bytes only the device ever reads.
    pub fn zero_seg(&mut self, seg: i64) -> Result<()> {
        let s =
            SegIdx::checked(seg).ok_or_else(|| anyhow!("zero_seg: segment {seg} out of range"))?;
        self.each(|e| e.zero_seg(s)).map(|_| ())
    }

    /// D2H the whole tensor segment as raw sen-fp16 bytes (the prefill→decode KV handoff: read the
    /// PREFILL session's seg2, then hand it to the decode session's alias door).
    pub fn read_seg(&mut self, seg: i64) -> Result<PerRank<T, Vec<u8>>> {
        let s =
            SegIdx::checked(seg).ok_or_else(|| anyhow!("read_seg: segment {seg} out of range"))?;
        // A segment image is a rank's own (its weight shard, its KV heads): one per rank.
        PerRank::try_from_fn(|r| self.ranks.get_mut(r).read_seg(s))
    }

    /// Overwrite the FIRST `images.get(r).len()` bytes of rank `r`'s tensor segment + H2D exactly that window — the
    /// spilled-weight-tail copy, whose source is one bundle's segment and whose destination is
    /// another's of a DIFFERENT extent. See [`Executor::write_seg_prefix`].
    pub fn write_seg_prefix(&mut self, seg: i64, images: &PerRank<T, Vec<u8>>) -> Result<()> {
        let s = SegIdx::checked(seg)
            .ok_or_else(|| anyhow!("write_seg_prefix: segment {seg} out of range"))?;
        // Rank r's image onto card r: a segment image is that rank's own.
        for (r, e) in self.ranks.iter_mut() {
            e.write_seg_prefix(s, images.get(r))?;
        }
        Ok(())
    }

    /// Point THIS session's segment at `owner`'s device region, so the two SHARE it and no copy is
    /// needed. Replaces only this session's descriptor — `owner` is untouched and its behaviour is
    /// bit-for-bit unchanged.
    ///
    /// Used to alias the batched-PREFILL session's seg2 (the resident KV) onto the DECODE session's,
    /// so prefill writes the prompt's KV directly where decode reads it. That deletes the per-prompt
    /// read_seg(2) → write_seg(2) round trip: 188.7 MB of DMA plus two 90 MiB host memcpys and a
    /// 90 MiB zeroed allocation, all serialized around a synchronize (~125–133 ms, prompt-length
    /// independent).
    ///
    /// MUST be called after BOTH sessions have been prepared — prepare zero-inits and H2Ds every
    /// segment, so aliasing earlier would let the owner's prepare wipe the now-shared region.
    /// ⭐ POINT EVERY CARD'S LANDING SLOTS AT ITS RECEIVE WINDOW, then check every rank's window
    /// lands at one address. Called on the OWNER session once it is prepared — its weight segment
    /// holds the window every borrower aliases. A world of one points nothing.
    ///
    /// # Errors
    /// A rank whose slots will not point, or windows that land apart.
    pub fn point_landing_slots(
        &mut self,
        cards: &PerRank<T, crate::runtime::CardRuntime<'_, T>>,
    ) -> Result<()> {
        use ktir_superdsc::collective::World;
        if !<T as World>::SHARDED {
            return Ok(());
        }
        self.each(|e| e.point_landing_slots())?;
        let landing = crate::runtime::refuse_if_windows_apart(cards).map_err(|e| anyhow!("{e}"))?;
        tracing::info!("[spyre-tp] every rank's receive window lands at 0x{landing:x}");
        Ok(())
    }

    pub fn alias_seg_from(&mut self, owner: &SuperDscSession<'a, '_, T>, seg: i64) -> Result<()> {
        let s =
            SegIdx::checked(seg).ok_or_else(|| anyhow!("alias_seg: segment {seg} out of range"))?;
        // Rank r aliases the owner's rank r: both sessions are one executor per rank of `T`.
        for (r, e) in self.ranks.iter_mut() {
            e.alias_seg_from(owner.ranks.get(r), s)?;
        }
        Ok(())
    }
}
