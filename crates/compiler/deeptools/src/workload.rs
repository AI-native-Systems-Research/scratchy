//! THE WORKLOAD POINT, AS CONSTANTS — which rung of the ladder this program is baked for.
//!
//! ⭐⭐ THE AXIS IS TWO-DIMENSIONAL AND BOTH HALVES ARE REAL. A bake is keyed by
//! `WorkloadPoint { num_tokens, sk_bucket }` (`macros/src/assignment.rs:95-98`): how many rows the
//! launch processes, and how far back into the KV cache it reads. A target that carried only the
//! first would bake one program for every cache depth.
//!
//! ⭐ THE ROW COUNTS ARE A DECLARED LADDER, NOT A RANGE. `PREFILL_RUNGS`
//! (`macros/src/codegen.rs:9975-9977`) is twenty-one values from 7 to 96, and the ceiling is
//! load-bearing elsewhere — a `const` assertion ties it to `PagedKvPool::PREFILL_CHUNK_SLOTS`
//! because the pool sizes its write slack from it. Decode rungs are keyed by `active_cap`, the
//! sweep extent (`codegen.rs:9411-9429`).
//!
//! ⛔⛔ AND THESE ARE NOT DECORATION. See [`Exploit`]: every constant here is READ by a branch that
//! changes what comes out. A constant that only reaches an attribute has been expressed and not
//! exploited, and three versions of this crate were thrown away for exactly that.

use crate::arch::Arch;
use crate::generated::DataType;
use crate::model::Model;

/// ONE RUNG OF THE LADDER, AS A TYPE.
///
/// ⛔ BOTH CONSTANTS ARE STATED, neither defaulted. A rung that forgot its `ACTIVE_CAP` and still
/// compiled would bake a program reading a KV span nobody chose.
pub trait Workload {
    /// `num_tokens` — how many rows this rung's launch processes.
    ///
    /// One of `PREFILL_RUNGS` for a prefill rung; the batch width `mq` for a decode one.
    const ROWS: u32;

    /// `sk_bucket` / `active_cap` — how far back into the KV cache this rung reads, RESOLVED.
    ///
    /// ⛔ THE SWEEP EXTENT, NOT THE BATCH WIDTH. `sk_bucket_rungs` is keyed by `active_cap`, and
    /// the two were confused once already (`sdsc_abstract.rs:4879` says so in as many words).
    ///
    /// ⛔⛔ AND RESOLVED, NOT THE SENTINEL. `ActiveCap` is a sentinel type: `FULL` is 0 ("sweep
    /// everything", meaning the bundle's whole cap) and `NONE` is `u32::MAX` ("sweep nothing").
    /// `ActiveCap::resolve(cap, stick)` is documented as THE ONLY place a rung becomes a tile
    /// extent, and passing `.get()` here instead handed this door a sentinel where an extent
    /// belongs — which the acceptance build reported as `rung (rows 1, active_cap 0)` and
    /// `active_cap 4294967295`.
    ///
    /// ⭐ ZERO IS LEGITIMATE and means the bundle sweeps NO resident prefix — a prefill chunk whose
    /// `start == 0` has none to attend. It is the resolved value of `NONE`, not a missing one.
    const ACTIVE_CAP: u32;

    /// ⭐⭐ THE RUNG'S OWN INVARIANTS, evaluated per rung at build time.
    const WELL_FORMED: () = {
        assert!(Self::ROWS > 0, "a rung that processes no rows");
        // ⛔ NO ASSERTION ON `ACTIVE_CAP` BEING NON-ZERO. An earlier version refused zero as "a rung
        // that reads no cache at all", which is a real and common bundle: a prefill chunk with
        // `start == 0` sweeps no resident prefix, and that is `ActiveCap::NONE` resolved.
    };
}

/// ⭐⭐ WHAT THE CONSTANTS ACTUALLY CHANGE.
///
/// ⛔⛔ EXPRESSING A CONSTANT ONLY PUTS A VALUE IN THE OUTPUT; EXPLOITING IT CHANGES WHAT THE
/// OUTPUT IS. Every flag below is read by a branch in the emitter that emits DIFFERENT OPS, not a
/// different bound on the same ops — an `affine.for` that runs once is still a region, still a
/// barrier, and still an induction variable every enclosed transfer is strided by.
///
/// ⛔ AND EACH ONE READS [`Workload::WELL_FORMED`] BEFORE COMPUTING. An unreferenced associated
/// const is never evaluated — a `Model` with `nqh=32, nkvh=5` once compiled clean for exactly that
/// reason — so the invariant is forced on any use of a derived flag rather than on a call someone
/// remembered to write.
pub struct Exploit<A: Arch, M: Model, W: Workload>(core::marker::PhantomData<(A, M, W)>);

impl<A: Arch, M: Model, W: Workload> Exploit<A, M, W> {
    /// HOW MANY ACTIVATION ELEMENTS FIT IN ONE STICK.
    ///
    /// ⛔⛔ NOT [`Arch::SLICES_PER_STICK`], WHICH IS EIGHT. A slice is a hardware sub-unit of a
    /// stick — `numSlicesPerStick` (`sysdef.cpp:229`) — so at 128 bytes a slice is sixteen bytes,
    /// not an element. Using it as an element count made a 2048-wide row look like 256 sticks
    /// instead of 32 and a 64-position cache span look like eight vectors instead of one; the
    /// two-emission test caught it by counting loops that should not have existed.
    ///
    /// ⭐ THE ACTIVATION STREAM IS fp16 whatever the weights are quantised to — the tape carries
    /// `Df::Fp16` on every node's dims — so this is the fp16 packing: 128 bytes / 2 = 64.
    pub const ACT_PER_STICK: u32 =
        DataType::Sen169Fp16.per_stick((A::BYTES_PER_STICK.get() * 8) as u32);

    /// ONE ROW — the decode case.
    ///
    /// ⭐ WHAT IT REMOVES: the row nest entirely. A single-row launch has no `m` loop, so every
    /// access inside it loses an induction variable and every address that was strided by it
    /// becomes a constant.
    pub const IS_DECODE: bool = {
        let () = W::WELL_FORMED;
        W::ROWS == 1
    };

    /// THE WHOLE ROW FITS IN THE SCRATCHPAD.
    ///
    /// ⭐ WHAT IT REMOVES: the tiling loop. Not "sets its bound to one" — REMOVES it, along with
    /// the induction variable every enclosed transfer is strided by.
    ///
    /// ⛔ CARRY THE VALUE. This compares BYTES against [`Arch::LX_CAPACITY`], hand-workable and
    /// asserted as a number in the tests. An earlier version of this idea compared a FLOP count
    /// (`M*K*N`) against a byte capacity and over-tiled by 64x while every ratio assertion passed.
    pub const FITS_LX: bool = {
        let () = W::WELL_FORMED;
        let () = M::WELL_FORMED;
        // One row of the hidden state, at two bytes an element, for as many rows as the rung runs.
        let working = (M::HIDDEN as u64) * (W::ROWS as u64) * 2;
        working <= A::LX_CAPACITY.0
    };

    /// THE ROW IS A WHOLE NUMBER OF STICKS.
    ///
    /// ⭐ WHAT IT REMOVES: the affine mask and every `element_wise_selection` that consumes it. A
    /// ragged tail needs a predicate per lane; an aligned one needs none, so the ops disappear
    /// rather than becoming a mask of all ones.
    pub const STICK_ALIGNED: bool = {
        let () = M::WELL_FORMED;
        M::HIDDEN % Self::ACT_PER_STICK == 0
    };

    /// HOW MANY BYTES THIS RUNG'S KV SPAN OCCUPIES, for one layer.
    ///
    /// ⛔⛔ CARRY THE VALUE, AND CARRY BOTH HALVES. `2 * ACTIVE_CAP * KV_WIDTH * 2`: two tensors
    /// (K and V), `ACTIVE_CAP` cache positions, [`Model::KV_WIDTH`] elements each, two bytes an
    /// element. The width is the KV one, NARROWER than the query stream by exactly
    /// [`Model::GQA`] — using `Q_WIDTH` here over-counts a grouped-query model's cache by that
    /// factor and tiles against a span that does not exist.
    pub const CACHE_BYTES: u64 = {
        let () = W::WELL_FORMED;
        let () = M::WELL_FORMED;
        2 * (W::ACTIVE_CAP as u64) * (M::KV_WIDTH as u64) * 2
    };

    /// THIS RUNG'S CACHE SPAN FITS THE SCRATCHPAD ALONGSIDE ITS ROWS.
    ///
    /// ⭐ WHAT IT CHANGES: whether the cache is staged ONCE before the attention walk or streamed
    /// from the HBM inside it. Those are different programs — one transfer against one per step —
    /// not one program with a different bound.
    ///
    /// ⛔ AND THE ROW IS COUNTED TOO. A cache that fits only because the activations were forgotten
    /// is the arena arithmetic that does not count the bytes it then binds against.
    pub const CACHE_FITS_LX: bool = {
        let row = (M::HIDDEN as u64) * (W::ROWS as u64) * 2;
        Self::CACHE_BYTES + row <= A::LX_CAPACITY.0
    };

    /// HOW MANY HARDWARE VECTORS THIS RUNG'S CACHE SPAN COVERS.
    pub const KV_VECTORS: u32 = {
        let () = W::WELL_FORMED;
        W::ACTIVE_CAP.div_ceil(Self::ACT_PER_STICK)
    };

    /// THERE IS NOTHING TO WALK.
    ///
    /// ⭐ WHAT IT REMOVES: the cache walk. At a span no wider than one vector there is nothing to
    /// step over, so the loop is not emitted — and with it goes the induction variable every cache
    /// address inside it was strided by.
    ///
    /// ⛔ `<= 1`, NOT `== 1`. Zero vectors is a bundle that sweeps no resident prefix at all
    /// (`ActiveCap::NONE`), and `== 1` left it emitting a walk of ZERO trips — a loop that runs
    /// never, which is the shape `audit_op_work` refuses elsewhere for exactly this reason.
    pub const NO_CACHE_WALK: bool = Self::KV_VECTORS <= 1;

    /// HOW MANY STICKS ONE ROW OF THE HIDDEN STATE OCCUPIES, rounded up.
    ///
    /// ⛔ ROUNDED UP, because a ragged row still occupies the stick it partly fills. Rounding down
    /// addresses one stick short of the row and reads the previous tensor's tail.
    pub const STICKS_PER_ROW: u32 = {
        let () = M::WELL_FORMED;
        M::HIDDEN.div_ceil(Self::ACT_PER_STICK)
    };
}
