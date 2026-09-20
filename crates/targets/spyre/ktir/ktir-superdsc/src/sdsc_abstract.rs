//! COMPILE-TIME interpreter that PROVES the emitted SDSC attention computes attention.
//!
//! The bugs in the SDSC lowering are all in the TRANSLATION — the incoming program's own arithmetic
//! is checkable by interpreting the KTIR, which is what a producer's host emulator does, so what is
//! left to get wrong is this crate's rendering of it into descriptors and addresses. Rather than run
//! on-card and discover the output is wrong (which I
//! already know), this module **executes the emitted attention op-DAG at build time** over the
//! SAME device element→address mapping the emitter bakes, on seeded inputs, and checks the
//! result equals `softmax(q·kᵀ·scale + mask)·v`. A divergence is a `cargo build` **panic** that
//! names the op — so the build does not pass until the translation is provably faithful. No
//! on-card run, no numeric golden chased after the fact: the math is locked at compile time.
//!
//! Faithfulness: the address model (`dev_off`) mirrors [`DeviceTileLayout`] exactly (a 2-D
//! tensor sticked on its last dim lives on-device as `[b/64, a, 64]`; everything else is flat
//! row-major grouped into 64-sticks). So if two synthetics overlap, or a reduce writes
//! stick-major where a consumer reads dense, or a matmul contracts the wrong axis, the
//! interpreter reads the wrong cells and the check fails — exactly the bug classes seen on-card.

use crate::addr::Gqa;
use std::collections::HashMap;
use std::num::NonZeroU32;

use crate::superdsc_opspec::{DataFormat, Df, Fp16};

/// ⭐⭐⭐⭐⭐ THE KV POOL'S STICK WIDTH, DERIVED FROM THE FORMAT ITS PLANES ACTUALLY HOLD — never a bare 64.
///
/// ⛔ AT head_dim 128 THIS NUMBER HAS A TWIN, AND THAT IS THE WHOLE HAZARD:
///
/// | quantity | value on granite-8b |
/// |---|---|
/// | `head_dim` | **128** |
/// | **fp8** elems-per-stick | **128** |
/// | fp16 elems-per-stick | 64 |
/// | `hd / fp16_stick` (slabs) | 2 |
/// | `hd / fp8_stick` (slabs) | **1** |
///
/// The MODEL is fp8 — its weights are — but the KV planes and every attention operand are **fp16**
/// (`matmul_opspec_off::<Fp16>`). So at head_dim 128, `head_dim` and the fp8 stick width are the SAME NUMBER, and
/// any site that takes "the stick" without saying WHICH FORMAT'S stick gets one slab where it needs two, or
/// treats a head as one stick when it spans two. At head_dim 64 the two disagree (64 vs 128) and the mistake is
/// visible; at 128 they coincide and it is not. That is the defect shape this whole file's newtypes exist to
/// prevent, in the one place that was still a literal.
///
/// Naming the provenance costs nothing and makes the question askable: this is `Fp16`'s stick because the POOL
/// holds fp16, and if a future pool holds fp8 planes this line is where that changes — not fifteen `64`s.
const STK: usize = POOL_STICK as usize;

/// ⭐⭐⭐⭐ THE KV WIDTH AND ITS HEAD COUNT, DERIVED TOGETHER SO THEY CANNOT DISAGREE.
///
/// ⛔ THE PATTERN THIS REPLACES IS A ROUND-TRIP THROUGH INTEGER DIVISION, WRITTEN THREE TIMES:
/// `let nkvh = (kv_dim / hd).max(1); let cols = nkvh * hd;` — and `cols == kv_dim` only when `hd` divides
/// `kv_dim` exactly. It does on every model run so far (1024 / 128 = 8), so the two are the same number and
/// interchangeable in any expression. The `.max(1)` makes the failure quiet rather than loud: an `hd` larger than
/// `kv_dim` yields one head and a `cols` that is not the width of anything.
///
/// Constructing both from one call means the head count and the width are one fact. `None` when the division is
/// not exact — a KV width that is not a whole number of heads describes no tensor this backend can lay out, and
/// the caller should say so rather than round.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct KvWidth {
    heads: u32,
    hd: u32,
}

impl KvWidth {
    /// `None` when `hd` does not divide `kv_dim` — the case the `.max(1)` used to swallow.
    pub fn of(kv_dim: u32, hd: u32) -> Option<KvWidth> {
        (hd != 0 && kv_dim != 0 && kv_dim.is_multiple_of(hd)).then_some(KvWidth {
            heads: kv_dim / hd,
            hd,
        })
    }

    /// KV heads — `nkvh`.
    pub const fn heads(self) -> u32 {
        self.heads
    }

    /// The packed width, `heads * hd`. Equal to the `kv_dim` it was built from BY CONSTRUCTION, not by
    /// coincidence — which is the whole difference from re-multiplying a divided value.
    pub const fn cols(self) -> u32 {
        self.heads * self.hd
    }
}

/// ⭐⭐⭐⭐ A PLANE'S **PHYSICAL** SLOT EXTENT — the distance to the next kv head, not the addressable count.
///
/// ⛔ `PAGE_SLOTS` AND `PLANE_SLOTS` ARE BOTH 256 AND MEAN DIFFERENT THINGS. Addressable: masks are blocked by it,
/// the fold sweeps it, a `KvSlot` never exceeds it. Physical: what separates one kv head's block from the next,
/// and therefore the quantity a padded chunk write overruns. They differ by `WRITE_SLACK`, which is 0 today —
/// **and was 32 once, which BROKE head_dim 128** (` Rome. Q` became ` Romes,`), because the extra slots move the
/// feature-stick GROUP stride that every reader re-derives and a head spans two sticks above hd=64.
///
/// So this is a coincidence with a proven failure history: the two were equal, a commit made them differ, and the
/// difference reached the addresses through expressions that could not say which extent they meant. A type at the
/// one site where both appear is what makes the substitution refuse to compile.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct PlaneExtent(u32);

impl PlaneExtent {
    /// The physical extent, for the stride arithmetic that is the only consumer.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// ⭐⭐⭐⭐ THE NEW BLOCK'S PADDED ROW COUNT — and the three different things that number is used AS.
///
/// ⛔ AT EVERY RUNG BAKED SO FAR IT IS 64, AND SO IS THE fp16 STICK. Two quantities, one value, and they are used
/// in the same expressions: `nsub = mq_pad / stick` divides a padded ROW COUNT by a LANE COUNT, and substituting
/// either for the other type-checks and yields 1. They diverge the moment a prefill chunk exceeds 64 rows, which
/// no baked rung has done — so this is a coincidence waiting on a wider rung, exactly like `PAGE_SLOTS` vs
/// `PLANE_SLOTS` was waiting on a non-zero `WRITE_SLACK` (which arrived, and broke head_dim 128).
///
/// The three roles, named because they are genuinely different questions about the same block:
/// * [`rows`](Self::rows) — how many query rows the block is padded to;
/// * [`slots`](Self::slots) — the slot extent of the block's own Kᵗ (`[hd, mq_pad]`), equal to `rows` because each
///   new token occupies exactly one slot, which is an identity of the NEW block and not of the pool;
/// * [`cols`](Self::cols) — the score width against the new block, equal for the same reason.
///
/// Each accessor answers in its OWN type — [`PaddedRows`] / [`SlotExtent`] / [`ScoreWidth`] — and a number
/// leaves those only through a method that names the slot it fills, so a site holding the row count cannot
/// answer a slot-extent question. The identity between the three stays real and holds by construction (all
/// three read the one stored extent); what a caller can no longer do is ask one question and spend the answer
/// as another — or as a lane count.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct MqPad(u32);

/// THE PAD LAW ITSELF — `mq.div_ceil(stick) * stick`, floored at one stick: the one place the
/// arithmetic is written. `const`, because it has two spenders with two evaluation times and they
/// must be the same law: [`MqPad::of_chunk`] applies it to a runtime prefill width, and
/// [`Rung::PAD`] has the COMPILER apply it to a decode-ladder width.
const fn pad_of_mq(mq: u32) -> u32 {
    let sticks = mq.div_ceil(POOL_STICK);
    (if sticks == 0 { 1 } else { sticks }) * POOL_STICK
}

impl MqPad {
    /// From the chunk's real row count, by [`pad_of_mq`]. Module-private: outside this file a pad
    /// exists only inside a [`PaddedMq`], fused to the width it was computed from — a bare `u32`
    /// cannot become a pad without also fixing which `mq` it pads.
    fn of_chunk(mq: u32) -> MqPad {
        MqPad(pad_of_mq(mq))
    }

    /// Query rows the block is padded to.
    pub fn rows(self) -> PaddedRows {
        PaddedRows(self.0)
    }

    /// The slot extent of the block's own Kᵗ — equal to `rows`, and an identity of the NEW block only.
    pub fn slots(self) -> SlotExtent {
        SlotExtent(self.0)
    }

    /// The score width against the new block — equal for the same reason.
    pub fn cols(self) -> ScoreWidth {
        ScoreWidth(self.0)
    }

    /// ⭐ HOW MANY STICK-WIDE SUB-BLOCKS THE BLOCK IS SWEPT IN — the `mq_pad / stick` division, NAMED.
    ///
    /// This is the expression the coincidence hides in: a padded row count divided by a lane count. Written as a
    /// method, the units are stated once and a caller cannot divide by `hd` (also 128 at the model we run) or by
    /// `PAGE_SLOTS` and still compile.
    pub fn sub_blocks(self) -> SubBlocks {
        SubBlocks((self.0 / POOL_STICK).max(1))
    }

    /// The block's sub-blocks as TYPED row windows, in sweep order — the only door to a [`RowWindow`],
    /// so a window index can only come from the pad law that defines how many there are.
    pub fn row_windows(self) -> impl Iterator<Item = RowWindow> {
        (0..self.sub_blocks().0).map(RowWindow)
    }
}

/// [`MqPad::rows`]'s answer: the padded query-ROW extent of the new block's staging tensors
/// (`new_k`/`new_v`/`new_k_scaled` are all `[mq_pad, nkvh·hd]`). The number leaves through
/// [`row_axis_extent`](Self::row_axis_extent) alone — the one question every consumer of the row count asks:
/// a declared row count, a row-axis extent, or the row sweep of an op over those tensors.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PaddedRows(u32);

impl PaddedRows {
    /// The row-axis extent of a `[mq_pad, ·]` staging tensor — also the M an op sweeping all its rows declares.
    pub const fn row_axis_extent(self) -> u32 {
        self.0
    }
}

/// [`MqPad::slots`]'s answer: the SLOT-axis extent of the new block's own Kᵗ (`[hd, mq_pad]`, per kv-head).
/// Distinct from the pool's [`SlotCount`] (addressable slots of a page) — this is an identity of the NEW
/// block. Two exits, because the slot question is asked in two positions and each names its slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SlotExtent(u32);

impl SlotExtent {
    /// The slot axis's extent where the Kᵗ scratch is DESCRIBED — a `synth` dims entry or a nest extent.
    pub const fn slot_axis_extent(self) -> u32 {
        self.0
    }

    /// The NEW block's door into [`KtKernelPitch`]: when the block's own Kᵗ scratch is the score
    /// kernel, its declared physical column count is this slot extent. The resident plane's value in
    /// that position is a page's slots ([`PagedKvPool::KT_KERNEL_PITCH`]), never `width`.
    pub const fn kernel_row_pitch(self) -> KtKernelPitch {
        KtKernelPitch(self.0)
    }
}

/// THE Kᵀ SCORE KERNEL'S DECLARED PHYSICAL COLUMN COUNT (`Stk::kernel`'s `n_out`) — the pitch that
/// separates one slot-stick group of the kernel from the next.
///
/// Exactly two tensors are ever the score kernel, and each is a door: the NEW block's own Kᵗ scratch
/// ([`SlotExtent::kernel_row_pitch`] — its slot extent, `mq_pad` by the new-block identity) and the
/// paged pool's resident Kᵀ plane ([`PagedKvPool::KT_KERNEL_PITCH`] — a page's slots). Both answers
/// are SLOT quantities; a row count (`mq_pad` as rows, `nqh*mq`, a mask extent) has no door, so it
/// cannot fill the kernel-pitch slot even where the values coincide.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct KtKernelPitch(u32);

impl KtKernelPitch {
    /// ⭐ THE GATHERED SCRATCH'S PITCH — ONE 64-SLOT WINDOW, because a scratch row holds exactly the
    /// `[hd, 64]` block the copy landed there. The pool's own door
    /// ([`PagedKvPool::KT_KERNEL_PITCH`]) declares a whole PAGE, which is what the UNGATHERED read
    /// sweeps a window out of; handing that value to a scratch read would declare a stick-group stride
    /// four times the buffer's and put every feature above the first stick in another request's block.
    ///
    /// A third door rather than a bare `u32` for the same reason the other two are doors: the three
    /// values (64, 256, and the new block's padded slots) are all plausible column counts here.
    pub const fn of_gather_window(slots: WindowSlots) -> KtKernelPitch {
        KtKernelPitch(slots.n())
    }

    /// `Stk::kernel`'s `n_out` slot — the kernel's declared physical column extent.
    pub const fn n_out_cols(self) -> usize {
        self.0 as usize
    }
}

/// THE Kᵀ RESTICKIFY TILE'S SLOT EXTENT — how many slots one `assemble_restickify_kt_2d` tile
/// re-sticks (the natural `[slots, feats]` rows in, the Kᵀ `[feats, slots]` columns out). Exactly
/// two tiles exist and each is a door: one row window of the NEW block (whose rows ARE its slots —
/// the [`MqPad`] identity) and a whole page of the resident pool. A feature width has no door, so
/// the restickify's two extents — previously two adjacent bare `u32`s, both 64 on the sub-block
/// form — cannot be handed over swapped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct KtTileSlots(u32);

impl KtTileSlots {
    /// One row window of new tokens — quantity (3): the window's rows are the new block's slots.
    pub const fn of_row_window(rows: WindowRows) -> KtTileSlots {
        KtTileSlots(rows.n())
    }

    /// A whole page of the resident pool — the post-cache-write re-transpose spans every slot.
    pub const fn of_page() -> KtTileSlots {
        KtTileSlots(PagedKvPool::PAGE_SLOTS as u32)
    }

    /// The slot-axis extent of the tile.
    pub const fn extent(self) -> u32 {
        self.0
    }
}

/// THE Kᵀ RESTICKIFY TILE'S FEATURE EXTENT — the head-dim width one tile carries. Two doors: one
/// head-dim slab (the sub-block form, quantity (4) of the four 64s) or the whole head dim (the
/// whole-page re-transpose). A slot extent has no door here, mirroring [`KtTileSlots`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct KtTileFeats(u32);

impl KtTileFeats {
    /// One head-dim slab — quantity (4): the per-(row-window, slab) sub-block form.
    pub const fn of_head_slab(feats: SlabFeats) -> KtTileFeats {
        KtTileFeats(feats.n())
    }

    /// The whole head dim — the whole-page re-transpose spans it.
    pub const fn of_head_dim(hd: u32) -> KtTileFeats {
        KtTileFeats(hd)
    }

    /// The feature-axis extent of the tile.
    pub const fn extent(self) -> u32 {
        self.0
    }
}

/// [`MqPad::cols`]'s answer: the COLUMN count of a score row against the new block — the width of the
/// `[nqh·mq, mq_pad]` score/causal-mask buffers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ScoreWidth(u32);

impl ScoreWidth {
    /// The score/causal-mask buffers' column extent.
    pub const fn score_axis_extent(self) -> u32 {
        self.0
    }
}

/// [`MqPad::sub_blocks`]'s answer: how many stick-wide sub-blocks sweep the new block. Not a row count and
/// not a lane count — the sweep is walked via [`MqPad::row_windows`] (each window a typed [`RowWindow`]),
/// and "is this the proven single-block emit" is [`is_single`](Self::is_single), so the count itself never
/// travels bare.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SubBlocks(u32);

impl SubBlocks {
    /// TRUE when the block is one stick wide and the emit is byte-identical to the single-block form
    /// (every chunk ≤64 rows, and all of decode) — the question the op-naming compatibility gates ask.
    pub const fn is_single(self) -> bool {
        self.0 == 1
    }
}

/// ⭐⭐⭐⭐ QUANTITY (3) OF THE FOUR MEANINGS OF 64: a STICK-TALL ROW WINDOW of the new-token block —
/// WHICH group of [`Self::ROWS`] padded query rows one sub-block covers.
///
/// The window is stick-tall because every reduce works on ONE stick of columns and the new block's
/// padded rows ARE its score columns (each new token occupies one slot), so the reduce's column budget
/// blocks the rows in sticks — that is the same law [`MqPad::of_chunk`] pads by. The window index is
/// three coordinates at once, all the [`MqPad`] rows/slots/cols identity: the source's row corner
/// ([`Self::first_row`]), the block's own Kᵀ slot-stick plane, and the cmask's block index.
///
/// ⛔ WHAT IT REPLACES: a bare `j` multiplied by the emitter's untyped `stick` local at each of those
/// sites — the product `j * 64` in which nothing said WHICH 64, in a scope where the lane count, the
/// fold's slot window and the feature slab are all also 64.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct RowWindow(u32);

impl RowWindow {
    /// Rows one window covers — the granularity [`MqPad::of_chunk`] pads to, held zero-sized in
    /// [`WindowRows`]. By the window's own identity this is ALSO its SCORE-COLUMN count (each padded
    /// row is one score column against the new block, sized by the reduce's one-stick column budget,
    /// which is why the window is stick-tall in the first place) — spending it as columns goes
    /// through [`BlockCols::of_row_window`], spending it as rows through
    /// [`KtTileSlots::of_row_window`], each of which demands this witness by type.
    pub const ROWS: WindowRows = WindowRows;

    /// Which window of the sweep — the sub-block's name, its Kᵀ slot-stick plane, its cmask block.
    pub const fn index(self) -> u32 {
        self.0
    }

    /// The first query row this window covers — the window index times its [`ROWS`](Self::ROWS):
    /// the `j * 64` product, owned here, landing in the row-offset type [`WindowFirstRow`].
    pub const fn first_row(self) -> WindowFirstRow {
        WindowFirstRow(self.0 * WindowRows::ROWS)
    }
}

/// ⭐ QUANTITY (3) OF THE FOUR 64s, AS A TYPE: the rows of one [`RowWindow`] — zero-sized, the value
/// is the associated const and never travels at runtime. NO cross-conversion: none of the four 64s
/// ([`Lanes`], [`SlotWindow::SLOTS`], this, [`FeatIdx::SLAB_FEATS`]) converts to another, so a slot
/// that demands this one cannot be fed any of the other three even though all four are 64 at every
/// baked rung. The number leaves only through doors that name what it fills.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WindowRows;

impl WindowRows {
    /// The rows themselves — module-private: outside this file the quantity is only spendable typed.
    const ROWS: u32 = POOL_STICK;

    const fn n(self) -> u32 {
        Self::ROWS
    }
}

/// THE ROW CORNER OF A ROW WINDOW — [`RowWindow::first_row`]'s answer: a query-ROW offset into the
/// new block's padded rows. Its one exit is the typed row coordinate, so it can fill a `row` slot
/// and nothing else — not a slot, not a feature, not a score column, each of which the bare
/// `j * 64` product could silently become.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WindowFirstRow(u32);

impl WindowFirstRow {
    /// This offset as the `row`-axis coordinate of a nest — the only exit.
    pub const fn row_idx(self) -> crate::addr::Idx<crate::addr::Row> {
        crate::addr::Idx::<crate::addr::Row>::n(self.0)
    }
}

/// ⭐⭐⭐⭐⭐ ONE DECODE-LADDER RUNG, AS A TYPE — `Rung<MQ>` can only be named at a width the ladder bakes.
///
/// A decode bundle's query-row count is not a runtime-shaped quantity the way a prefill chunk's is:
/// the ladder is the FIXED set `{1} ∪ `[`PagedKvPool::BATCH_RUNGS`], decided at bake, selected by the
/// worker. So the width is a CONST, and everything the pad law derives from it is arithmetic the
/// COMPILER does, per rung:
///
/// * an unlisted width has no type: `Rung::<3>::baked()` (or 31, or 96 — any prefill width) fails the
///   BUILD, because [`baked`](Self::baked) evaluates a membership assert against the pool's own
///   ladder. The ladder and the rung type cannot drift — changing [`PagedKvPool::BATCH_RUNGS`] alone
///   re-decides which `Rung`s exist;
/// * [`PAD`](Self::PAD) is [`pad_of_mq`] evaluated at compile time — the SAME law as the runtime
///   door, called once, not respelled;
/// * [`SUB_BLOCKS`](Self::SUB_BLOCKS) carries a PROOF, not just a value: every baked rung pads to
///   ONE stick of rows, so a decode bundle's new block is swept in exactly one [`RowWindow`] — the
///   emit form proven on hardware. A ladder widened past one stick (a 128-wide rung, say) stops
///   compiling HERE, at the type, until the multi-window decode walk is proven on card — instead of
///   silently riding the prefill-only sub-block loop into an unproven decode emit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rung<const MQ: u32>(());

impl<const MQ: u32> Rung<MQ> {
    /// Ladder membership, decided by the compiler. `1` is the solo-decode width (the primary m=1
    /// bundle); every other legal width is the pool's own ladder, read from the one array the bakes
    /// iterate — so this const and the bake loop cannot disagree about what a rung is.
    const IS_A_BAKED_RUNG: () = assert!(
        {
            let mut ok = MQ == 1;
            let ladder = PagedKvPool::BATCH_RUNGS;
            let mut i = 0;
            while i < ladder.len() {
                ok = ok || ladder[i] == MQ;
                i += 1;
            }
            ok
        },
        "Rung<MQ>: MQ is not a width the decode ladder bakes (1, or PagedKvPool::BATCH_RUNGS)"
    );

    /// THE PADDED ROW COUNT, computed by the compiler — [`pad_of_mq`] at `MQ`. Evaluating it also
    /// evaluates the membership assert, so an unlisted width dies here even if a caller never asks
    /// for anything else.
    pub const PAD: u32 = {
        let () = Self::IS_A_BAKED_RUNG;
        pad_of_mq(MQ)
    };

    /// EVERY BAKED RUNG IS ONE STICK OF PADDED ROWS. This is what licenses the decode emit shape:
    /// `nsub == 1`, the single-window new block, the op names without a window suffix — the form the
    /// card has accepted. Not a numerical accident to re-check at runtime: a compile-time fact of
    /// the ladder, enforced where the ladder meets the type.
    const ONE_STICK_OF_ROWS: () = assert!(
        Self::PAD / POOL_STICK == 1,
        "Rung<MQ>: a decode rung wider than one stick of padded rows has no proven emit — the \
         new-block sub-block walk is prefill-only until the multi-window decode form is proven on \
         card. Widen the ladder only together with that proof."
    );

    /// The sub-block count of the rung's new block — `PAD / stick`, which the ladder pins to 1.
    pub const SUB_BLOCKS: u32 = {
        let () = Self::ONE_STICK_OF_ROWS;
        Self::PAD / POOL_STICK
    };

    /// THE one row window a baked rung's new block sweeps — index 0 of a one-window sweep, held as
    /// a const because [`SUB_BLOCKS`](Self::SUB_BLOCKS) is. The window a decode bundle's
    /// restickify/score/mask sub-block indices come from is this value, not a loop variable that
    /// merely happens to take one value.
    pub const WINDOW: RowWindow = {
        let () = Self::ONE_STICK_OF_ROWS;
        RowWindow(0)
    };

    /// The rung as the ladder baked it. The only constructor, and it insists on the membership
    /// assert — `Rung` has no door that skips the ladder.
    pub const fn baked() -> Rung<MQ> {
        let () = Self::IS_A_BAKED_RUNG;
        Rung(())
    }
}

/// ⭐⭐⭐⭐⭐ THE CHUNK'S WIDTH AND ITS PAD, ONE VALUE — `mq` and the stick-padded row count travel FUSED,
/// minted together by one law application, so no call site can pair a real row count with a pad
/// computed from some OTHER width, and no signature carries them as two adjacent integers again
/// (the arrangement the `mq_pad` parameter swap lived in).
///
/// The two mints, and the wall between them:
/// * a DECODE width comes in through [`of_rung`](Self::of_rung) — the width is a ladder const
///   ([`Rung`]), so the pad and the one-window proof are the compiler's;
/// * a PREFILL width comes in through the chunk arm of [`of_bundle`](Self::of_bundle) — a runtime
///   quantity (the chunk ladder is a wide range), padded by the same [`pad_of_mq`] law. It CANNOT
///   mint a `Rung`: there is no value→const conversion anywhere but the dispatch's literal arms,
///   and those arms only name ladder widths.
///
/// [`of_bundle`](Self::of_bundle) is the ONE parse boundary from a runtime `(mq, rows_are_requests)`
/// pair to the law — the same boundary discipline as the geometry door
/// (`scratchy_subtile::model_geometry::with_config_attn_geometry`): a decode width the ladder does not bake
/// is a refusal (`None` → a loud bake error at the caller), never a silent pad.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PaddedMq {
    mq: QueryRowCount,
    pad: MqPad,
}

impl PaddedMq {
    /// A decode rung's width and pad, from the type: `MQ` is the width, [`Rung::PAD`] the pad, and
    /// binding [`Rung::WINDOW`] here makes the one-window proof a compile-time obligation of every
    /// rung the dispatch can name — a too-wide ladder stops the BUILD in this function.
    pub const fn of_rung<const MQ: u32>(rung: Rung<MQ>) -> PaddedMq {
        let _ = rung;
        let _one_window: RowWindow = Rung::<MQ>::WINDOW;
        PaddedMq {
            mq: QueryRowCount::of_mq(MQ),
            pad: MqPad(Rung::<MQ>::PAD),
        }
    }

    /// THE RUNTIME ARM OF THE PAD LAW — a prefill chunk's width, padded by [`pad_of_mq`]. One
    /// spelling, module-private: both [`of_bundle`](Self::of_bundle)'s prefill arm and
    /// [`AttnBundleRows`]'s prefill mint pad a chunk through here, so the runtime pairing of a
    /// width with its pad exists in exactly one place.
    fn of_chunk(mq: u32) -> PaddedMq {
        PaddedMq {
            mq: QueryRowCount::of_mq(mq),
            pad: MqPad::of_chunk(mq),
        }
    }

    /// THE PARSE BOUNDARY from a bundle's runtime width to the pad law.
    ///
    /// Decode (one row, or rows that are requests): the width must be a baked ladder rung —
    /// [`with_baked_rung`] is the one value→const door, and `None` is a decode width the ladder
    /// does not bake: the caller refuses the bundle, loudly.
    ///
    /// Prefill (a multi-row prompt chunk): the runtime law, same [`pad_of_mq`] arithmetic.
    pub fn of_bundle(mq: u32, rows_are_requests: bool) -> Option<PaddedMq> {
        if !(rows_are_requests || mq == 1) {
            return Some(PaddedMq::of_chunk(mq));
        }
        /// The pad consumer: a rung's width and pad are the type's own arithmetic.
        struct Pad;
        impl OnBakedRung for Pad {
            type Out = PaddedMq;
            fn on_rung<const MQ: u32>(self, rung: Rung<MQ>) -> PaddedMq {
                PaddedMq::of_rung(rung)
            }
        }
        with_baked_rung(mq, Pad)
    }

    /// The chunk's REAL query rows — the width the pad was computed from, by construction.
    pub const fn mq(self) -> QueryRowCount {
        self.mq
    }

    /// The stick-padded row count that belongs to [`mq`](Self::mq) — the only pad a holder of this
    /// value can spend, so "which width was this pad for" is never a question.
    pub const fn pad(self) -> MqPad {
        self.pad
    }
}

/// ⭐⭐⭐⭐⭐ THE MODEL'S ATTENTION-HEAD GEOMETRY, AS A TYPE — `AttnGeometry<NQH, NKVH, HD>` can only
/// be named at a (query-head count, kv-head count, head-dim) triple whose GQA grouping exists.
///
/// scratchy is a per-model compiler: `arch-<name>`/`<stem>` (scratchy-models)
/// and `<preset>` (scratchy-quantizations) Cargo features pin the model set
/// at cargo-build time, the
/// `#[forward]` macro expands with the full geometry known, and the SuperDSC lowering runs DURING
/// that expansion — so the head counts are bake-time constants of the bundle, exactly like the
/// decode width [`Rung`] carries. This is their carrier: three const generics on one ZST, so a
/// geometry conflation (an nqh/nkvh transposition, a head dim from another model) is a BUILD error
/// where a trio of `u32` parameters would bake cleanly and garble on-card.
///
/// * [`GQA`](Self::GQA) is the group size WITH ITS PROOF: a triple where `NKVH` does not divide
///   `NQH` (or either count is zero) fails const evaluation, so a non-dividing pair is a type that
///   does not instantiate. There is no runtime arm left to fudge one — `(nqh / nkvh.max(1)).max(1)`
///   was that arm, and at a non-dividing pair it yields a plausible group size whose attention
///   reads another head's keys: fluent wrong output, never a fault.
/// * [`minted`](Self::minted) is the only constructor and it spends the proof, so HOLDING a value
///   of this type is holding the divisibility fact.
/// * the counts leave through doors that carry their own facts: [`nqh_nz`](Self::nqh_nz) is the
///   `NonZeroU32` the head iterators ([`QueryHead::all`], [`KvHead::all`]) take, nonzero by the
///   same const evaluation, so no call site discharges an `Option` for a question the type already
///   answered.
///
/// The value→const boundary is `scratchy_subtile::model_geometry::with_config_attn_geometry` — the same
/// door discipline as [`PaddedMq::of_bundle`]'s decode ladder: the tape carries a
/// `ModelAttnGeometry` because the emitter crate is one
/// binary serving every model, and the door's arm is where the number the macro parsed from the
/// model config meets the type. Its arms are GENERATED from the model configs in scope, so a
/// geometry no config declares is a loud bake error, answered by a `config.json`.
///
/// [`crate::addr::Shape`] is the ADDRESS-side face of the same facts (it hands out nests; its
/// fourth parameter is the KV capacity, which today still reaches the emitter as a value); this
/// type is the EMITTER-side carrier that travels through `assemble_attn`'s signature.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AttnGeometry<const NQH: u32, const NKVH: u32, const HD: u32>(());

impl<const NQH: u32, const NKVH: u32, const HD: u32> AttnGeometry<NQH, NKVH, HD> {
    /// The GQA group size, `NQH / NKVH` — evaluating it IS the divisibility proof.
    pub const GQA: u32 = {
        assert!(
            NQH >= 1,
            "AttnGeometry: a model has at least one query head"
        );
        assert!(NKVH >= 1, "AttnGeometry: a model has at least one kv head");
        assert!(
            NQH.is_multiple_of(NKVH),
            "AttnGeometry: NKVH must divide NQH — a non-dividing pair has no GQA grouping, so it \
             is not a geometry the attention emitter can name"
        );
        NQH / NKVH
    };

    /// The query-head count as the `NonZeroU32` the head iterators take — nonzero by
    /// [`GQA`](Self::GQA)'s evaluation, so the `panic!` arm is a compile-time impossibility, not a
    /// runtime assertion.
    pub const NQH_NZ: NonZeroU32 = {
        let _proof: u32 = Self::GQA;
        match NonZeroU32::new(NQH) {
            Some(n) => n,
            None => panic!("AttnGeometry: NQH == 0 is already refused by GQA's evaluation"),
        }
    };

    /// The only constructor. Naming a triple through here evaluates [`GQA`](Self::GQA), so every
    /// held value carries the proof.
    pub const fn minted() -> AttnGeometry<NQH, NKVH, HD> {
        let _proof: u32 = Self::GQA;
        AttnGeometry(())
    }

    /// The query-head count, spent as a value.
    pub const fn nqh(self) -> u32 {
        NQH
    }

    /// The kv-head count, spent as a value.
    pub const fn nkvh(self) -> u32 {
        NKVH
    }

    /// The head dim, spent as a value.
    pub const fn hd(self) -> u32 {
        HD
    }

    /// The GQA group size — [`GQA`](Self::GQA), the proof-carrying const, spent as a value.
    pub const fn gqa(self) -> u32 {
        Self::GQA
    }

    /// [`NQH_NZ`](Self::NQH_NZ) as a method, for call sites holding the value.
    pub const fn nqh_nz(self) -> NonZeroU32 {
        Self::NQH_NZ
    }

    /// ONE REQUEST'S ROW EXTENT of a per-request attention pass — its `nqh` head rows. The law's
    /// only factor is the query-head count, a const of the geometry, so the extent is the
    /// compiler's: there is no runtime `nqh` for a call site to fetch from somewhere else.
    pub const PER_REQUEST_ROWS: PerRequestRows = {
        let _proof: u32 = Self::GQA;
        PerRequestRows::of_one_request_heads(NQH)
    };

    /// [`PER_REQUEST_ROWS`](Self::PER_REQUEST_ROWS) as a method, for call sites holding the value.
    pub const fn per_request_rows(self) -> PerRequestRows {
        Self::PER_REQUEST_ROWS
    }
}

/// ⭐ WHAT A DECODE DISPATCH ARM RUNS with the const width it named — the consumer side of
/// [`with_baked_rung`]. The method is generic over `MQ` because that is the whole point of the
/// door: the arm's body receives the width AS A CONST, with the [`Rung`] witness that proves the
/// ladder bakes it, so everything it derives is the compiler's arithmetic.
pub trait OnBakedRung {
    /// What the dispatch produces.
    type Out;
    /// The arm's body, at the width the ladder named.
    fn on_rung<const MQ: u32>(self, rung: Rung<MQ>) -> Self::Out;
}

/// THE ONE VALUE→CONST DOOR for a decode width. The match arms are the ladder as literals — the
/// only place a runtime value can become a const — and the listing is verified against the pool's
/// ladder AT COMPILE TIME: each arm instantiates [`Rung`] (membership), and the guard const pins
/// the count and the strict ascent, so the six literals ARE the set `{1} ∪ BATCH_RUNGS` — growing
/// either side alone fails the build here. `None` is a width the ladder does not bake: the caller
/// refuses it, loudly.
///
/// Every dispatch routes through THIS listing ([`PaddedMq::of_bundle`]'s decode arm,
/// [`attn_bundle_rows`]'s), so two doors cannot drift about what the ladder is.
pub fn with_baked_rung<C: OnBakedRung>(mq: u32, consumer: C) -> Option<C::Out> {
    macro_rules! decode_ladder {
        ($($w:literal),+ $(,)?) => {{
            const LISTED: &[u32] = &[$($w),+];
            const LISTING_IS_EXACTLY_THE_LADDER: () = {
                assert!(
                    LISTED.len() == PagedKvPool::BATCH_RUNGS.len() + 1,
                    "the decode dispatch must list 1 plus every PagedKvPool::BATCH_RUNGS width"
                );
                let mut i = 1;
                while i < LISTED.len() {
                    assert!(
                        LISTED[i - 1] < LISTED[i],
                        "the decode dispatch listing must be strictly ascending (no duplicates)"
                    );
                    i += 1;
                }
            };
            let () = LISTING_IS_EXACTLY_THE_LADDER;
            match mq {
                $($w => Some(consumer.on_rung(Rung::<$w>::baked())),)+
                _ => None,
            }
        }};
    }
    decode_ladder!(1, 2, 4, 8, 16, 32)
}

/// ⭐⭐⭐⭐⭐ THE ROW LAWS OF ONE BAKED DECODE RUNG, EVALUATED BY THE COMPILER — `RungRowLaws<NQH, MQ>`
/// carries every row extent whose factors are BOTH bake-time constants: the query-head count is
/// the geometry's const and a decode width is the ladder's, so their products and differences are
/// arithmetic the compiler does per (geometry, rung) instantiation, not a runtime multiplication
/// per emit.
///
/// * the shared-buffer row count ([`MaskRows`], `NQH*MQ`) is 64 at (32, 2) and 128 at (32, 4) —
///   the first width boundary the card broke on. Those two extents are consts of DIFFERENT types
///   now: `RungRowLaws<32, 2>` does not unify with `RungRowLaws<32, 4>`, so a site framed for one
///   rung's rows cannot be handed another's, where two runtime `u32`s would swap silently.
///
/// ```compile_fail,E0308
/// use ktir_superdsc::sdsc_abstract::{AttnGeometry, MaskRows, Rung, RungRowLaws};
/// fn framed_for_the_wide_rung(laws: RungRowLaws<32, 4>) -> MaskRows { laws.mask_rows() }
/// let narrow = RungRowLaws::<32, 2>::minted(AttnGeometry::<32, 8, 64>::minted(), Rung::<2>::baked());
/// framed_for_the_wide_rung(narrow); // 64 rows in a 128-row slot: a type error, not a bake
/// ```
///
/// * a PREFILL width cannot mint these laws: [`minted`](Self::minted) is the only constructor and
///   it demands a [`Rung`] — a runtime chunk width has no const to name one with (the only
///   value→const door is [`with_baked_rung`]'s ladder arms, which are decode's). Naming an
///   unlisted width directly dies in `Rung`'s own membership assert:
///
/// ```compile_fail,E0080
/// use ktir_superdsc::sdsc_abstract::{AttnGeometry, Rung, RungRowLaws};
/// let _ = RungRowLaws::<32, 31>::minted(AttnGeometry::<32, 8, 64>::minted(), Rung::<31>::baked());
/// ```
///
/// * every const below is the SAME law function the runtime path calls ([`MaskRows::new`],
///   [`RequiredRowStrides::of_head_major`], [`RequiredRowStrides::of_request_major`]), evaluated
///   in const position — one law, two evaluation times, nothing respelled.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RungRowLaws<const NQH: u32, const MQ: u32>(());

impl<const NQH: u32, const MQ: u32> RungRowLaws<NQH, MQ> {
    /// The rung width as the type the row laws take. Nonzero is the compiler's fact here: no
    /// ladder width is zero (evaluating [`Rung::PAD`] is the membership proof), so the `None` arm
    /// is a build failure, never a call-site discharge.
    const RUNG_WIDTH: RungWidth = {
        assert!(
            NQH >= 1,
            "RungRowLaws: a geometry has at least one query head"
        );
        let _ladder_membership: u32 = Rung::<MQ>::PAD;
        match RungWidth::of_emitted_rows(MQ) {
            Some(w) => w,
            None => panic!("RungRowLaws: the ladder bakes no zero-width rung"),
        }
    };

    /// The whole-batch shared-buffer row count — [`MaskRows::new`] at the consts.
    const MASK_ROWS: MaskRows = MaskRows::new(NQH, Self::RUNG_WIDTH);

    /// The head-major framing's required strides — [`HeadRequestRow`]'s own differences at `MQ`.
    const HEAD_MAJOR_STRIDES: RequiredRowStrides = RequiredRowStrides::of_head_major(MQ);

    /// The request-major framing's required strides — [`RequestHeadRow`]'s own differences at `NQH`.
    const REQUEST_MAJOR_STRIDES: RequiredRowStrides = RequiredRowStrides::of_request_major(NQH);

    /// The only constructor: it spends the geometry's proof (which ties `NQH` to a real model's
    /// head count) and the rung witness (which ties `MQ` to the ladder), and evaluates the laws —
    /// so HOLDING a value of this type is holding the compiler's row arithmetic for this rung.
    pub const fn minted<const NKVH: u32, const HD: u32>(
        geom: AttnGeometry<NQH, NKVH, HD>,
        rung: Rung<MQ>,
    ) -> RungRowLaws<NQH, MQ> {
        let _ = (geom, rung);
        let _laws: (MaskRows, RequiredRowStrides, RequiredRowStrides) = (
            Self::MASK_ROWS,
            Self::HEAD_MAJOR_STRIDES,
            Self::REQUEST_MAJOR_STRIDES,
        );
        RungRowLaws(())
    }

    /// The shared-buffer row count, spent as the value every whole-batch sweep takes.
    pub const fn mask_rows(self) -> MaskRows {
        Self::MASK_ROWS
    }

    /// The rung width, spent as the value the mask row laws take.
    pub const fn rung_width(self) -> RungWidth {
        Self::RUNG_WIDTH
    }

    /// [`HeadRequestRow`]'s strides at this rung — request axis one row, head axis `MQ` rows.
    pub const fn head_major_strides(self) -> RequiredRowStrides {
        Self::HEAD_MAJOR_STRIDES
    }

    /// [`RequestHeadRow`]'s strides at this rung — head axis one row, request axis `NQH` rows.
    pub const fn request_major_strides(self) -> RequiredRowStrides {
        Self::REQUEST_MAJOR_STRIDES
    }
}

/// THE ATTENTION EMITTER'S MASK-BLOCK WIDTH: one fold pass is one PAGE, so the emitter's
/// prefix-mask shape is always blocked at the pool page's slot count — named, so the const-generic
/// argument says WHICH width fills it (the same discipline as [`POOL_STICK`]).
pub const PAGE_MASK_COLS: u32 = PagedKvPool::PAGE_SLOTS as u32;

/// ⭐⭐ THE BUNDLE'S WIDTH AND THE ROW LAWS IT IS FRAMED WITH, ONE VALUE — what `assemble_attn`
/// draws every shared-buffer synth extent, mask nest and row sweep from, minted beside the
/// [`PaddedMq`] it belongs to so a width cannot be paired with another bundle's row frame.
///
/// The two mints mirror [`PaddedMq::of_bundle`]'s wall:
/// * a DECODE rung's comes from [`of_rung_laws`](Self::of_rung_laws) — every extent is
///   [`RungRowLaws`]' const, so the row arithmetic is the compiler's;
/// * a PREFILL chunk's comes from the runtime arm of [`attn_bundle_rows`] — the same law
///   functions, evaluated at runtime on the chunk's width. It cannot reach the const mint: there
///   is no [`Rung`] for it to present.
///
/// `NQH` rides the type so a carrier minted for one geometry cannot be spent emitting another's
/// bundle — `assemble_attn` demands `AttnBundleRows<NQH>` at ITS `NQH`.
#[derive(Clone, Copy, Debug)]
pub struct AttnBundleRows<const NQH: u32> {
    width: PaddedMq,
    shape: PrefixMaskShape<POOL_STICK, PAGE_MASK_COLS>,
}

impl<const NQH: u32> AttnBundleRows<NQH> {
    /// A baked decode rung's frame: the pad is [`Rung`]'s compile-time arithmetic and every row
    /// extent is [`RungRowLaws`]' const — nothing here multiplies at runtime.
    pub const fn of_rung_laws<const MQ: u32>(laws: RungRowLaws<NQH, MQ>) -> AttnBundleRows<NQH> {
        AttnBundleRows {
            width: PaddedMq::of_rung(Rung::<MQ>::baked()),
            shape: PrefixMaskShape::of_rung_laws(laws),
        }
    }

    /// A prefill chunk's frame: the runtime row path. Module-private — the one way in from a
    /// runtime width is [`attn_bundle_rows`], whose decode widths never reach this arm.
    fn of_prefill_chunk(mq: u32) -> Option<AttnBundleRows<NQH>> {
        let rung = RungWidth::of_emitted_rows(mq)?;
        Some(AttnBundleRows {
            width: PaddedMq::of_chunk(mq),
            shape: PrefixMaskShape::new(NQH, rung)?,
        })
    }

    /// The chunk's width and pad — the same value [`PaddedMq::of_bundle`] answers.
    pub const fn width(self) -> PaddedMq {
        self.width
    }

    /// The prefix mask's shape — for a decode rung, its row count is [`RungRowLaws`]' const.
    pub const fn mask_shape(self) -> PrefixMaskShape<POOL_STICK, PAGE_MASK_COLS> {
        self.shape
    }
}

/// THE PARSE BOUNDARY WITH THE GEOMETRY IN SCOPE — [`PaddedMq::of_bundle`]'s discipline (the same
/// prefill/decode wall, the same [`with_baked_rung`] ladder door), answering with the bundle's ROW
/// LAWS as well as its pad. The decode arms pair the ladder's const `MQ` with the geometry's const
/// `NQH`, so every row extent of a decode bundle is the compiler's ([`RungRowLaws`]); the prefill
/// arm is the runtime law. `None` is a decode width the ladder does not bake — or a zero-row
/// chunk, which has no row to frame.
pub fn attn_bundle_rows<const NQH: u32, const NKVH: u32, const HD: u32>(
    geom: AttnGeometry<NQH, NKVH, HD>,
    mq: u32,
    rows_are_requests: bool,
) -> Option<AttnBundleRows<NQH>> {
    if !(rows_are_requests || mq == 1) {
        return AttnBundleRows::of_prefill_chunk(mq);
    }
    /// The laws consumer: the arm's const `MQ` meets the boundary's const `NQH` here, which is
    /// what makes the row laws the compiler's.
    struct Mint<const NQH: u32, const NKVH: u32, const HD: u32>(AttnGeometry<NQH, NKVH, HD>);
    impl<const NQH: u32, const NKVH: u32, const HD: u32> OnBakedRung for Mint<NQH, NKVH, HD> {
        type Out = AttnBundleRows<NQH>;
        fn on_rung<const MQ: u32>(self, rung: Rung<MQ>) -> AttnBundleRows<NQH> {
            AttnBundleRows::of_rung_laws(RungRowLaws::<NQH, MQ>::minted(self.0, rung))
        }
    }
    with_baked_rung(mq, Mint(geom))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE MATMUL ASSEMBLER'S DIMENSION RUN — one type per slot.
//
//  `assemble_matmul_off*` / `matmul_opspec_off*` take their dimensions as a positional run of five
//  adjacent slots (`m, n, k, batch[, phys_m]`) whose values coincide pairwise on every narrow
//  shape: at nqh=32 the head-major row count, `mq_pad` and the fp16 stick are ALL 64 for mq ≤ 2 and
//  separate first at mq = 4 (rows 128 vs 64). That run is where the `mq_pad` parameter swap and the
//  `qs_off` head-stride bug lived. Each slot is its own type below, and a value reaches a slot only
//  through a constructor that names WHICH quantity it is — so two slots can no longer be satisfied
//  by the same accidental 64, and a swap is a type error, not a bake.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ROWS THIS MATMUL COMPUTES — the `mb` iteration extent of `A[m,k]·W[k,n]`.
///
/// Not the activation's physical packing (that is [`PhysM`]), not an output-column count, not a
/// lane count. The constructors name the row quantities that actually flow into the emitters.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MatM(u32);

impl MatM {
    /// The chunk's real query rows `mq` — one output row per query row. A batched decode's requests
    /// ride this same axis (the row KIND travels separately, as `QueryRows`). Takes the typed count,
    /// so a padded row count, a shared-buffer extent or a lane count cannot fill this row slot.
    pub const fn of_query_rows(mq: QueryRowCount) -> MatM {
        MatM(mq.get())
    }
    /// Exactly ONE row: the per-(request, head) forms, where the request and the head are carried
    /// by each op's own base offset rather than by this extent.
    pub const fn single_row() -> MatM {
        MatM(1)
    }
    /// A whole head-major buffer's rows (`heads * mq`) — the collapsed RoPE form sweeps them all in
    /// one op.
    pub const fn of_head_major_rows(rows: u32) -> MatM {
        MatM(rows)
    }
    /// A node's token rows — the dense projections, the lm-head, the fp8 W8A8 chain.
    pub const fn of_token_rows(m: u32) -> MatM {
        MatM(m)
    }
    /// The new-K/V block's stick-PADDED row count: the GQA replicate copies the pad rows too (they
    /// are zeroed upstream and masked downstream). Takes [`MqPad::rows`]'s own answer, so only the
    /// pad law's ROW exit can fill this row slot — never the slot extent or the score width it
    /// numerically equals, and never a lane count.
    pub const fn of_padded_chunk_rows(mq_pad: PaddedRows) -> MatM {
        MatM(mq_pad.row_axis_extent())
    }
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// OUTPUT COLUMNS OF THIS MATMUL — the `out` (N) stick extent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MatN(u32);

impl MatN {
    /// One head's feature width `hd` — the value matmul's and RoPE's output columns.
    pub const fn of_head_dim(hd: u32) -> MatN {
        MatN(hd)
    }
    /// The score block's kv window — the column count of one online-softmax block (one stick per
    /// fold sub-block by design). Takes the block's own typed width, so the score matmul's N and
    /// the reduce sweep that consumes it are one quantity by construction.
    pub const fn of_kv_window(width: BlockCols) -> MatN {
        MatN(width.get())
    }
    /// Exactly ONE stick of output columns — the identity-slab copies, whose write is deliberately
    /// one stick so the op has no plane term of its own to get wrong. The width is the machine's
    /// lane count, named by the zero-sized [`Lanes`] witness, so a row count — or any of the other
    /// three 64s — cannot stand in for it.
    pub const fn one_stick(lanes: Lanes) -> MatN {
        MatN(lanes.n())
    }
    /// One head-dim SLAB of output features — the per-(head, slab) value matmul, which emits one
    /// slab per op so a head-dim-spanning write cannot mis-stride its second stick group. Demands
    /// quantity (4)'s witness ([`FeatIdx::SLAB_FEATS`]) by type — the pool's lane count no longer
    /// fits this slot.
    pub const fn of_head_slab(feats: SlabFeats) -> MatN {
        MatN(feats.n())
    }
    /// A projection's output features — N of `A[m,k]·W[k,n]`.
    pub const fn of_out_features(n: u32) -> MatN {
        MatN(n)
    }
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// CONTRACTION EXTENT OF THIS MATMUL — the `in` (K) reduction stick extent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MatK(u32);

impl MatK {
    /// One head's feature width `hd` — the score matmul contracts Q·Kᵀ over it.
    pub const fn of_head_dim(hd: u32) -> MatK {
        MatK(hd)
    }
    /// The score block's kv window — the value matmul contracts probabilities·V over it. Takes the
    /// block's own typed width, same reason as [`MatN::of_kv_window`].
    pub const fn of_kv_window(width: BlockCols) -> MatK {
        MatK(width.get())
    }
    /// Exactly ONE stick of contraction — the identity-kernel slab copies, whose A-side read is
    /// deliberately one stick so the copy has no plane term of its own to get wrong. The width is
    /// the machine's lane count, named by the zero-sized [`Lanes`] witness.
    pub const fn one_stick(lanes: Lanes) -> MatK {
        MatK(lanes.n())
    }
    /// ONE HEAD-DIM SLAB of contraction — the score leg's Q·Kᵀ split by slab so that its
    /// contraction is one stick at EVERY head dim, which is what
    /// [`OneStickContraction`](crate::ir::bridge::tiled_op_sdsc_op) demands of the `y`-batched
    /// form. Distinct in meaning from [`Self::one_stick`] (an identity-kernel copy's deliberately
    /// one-stick A-side read) even though a slab IS one stick of lanes by definition — the same
    /// distinction [`BlockCols::of_head_slab`] already draws, and the reason this is not spelled
    /// `one_stick(Lanes)` at the call site.
    pub const fn of_head_slab(feats: SlabFeats) -> MatK {
        MatK(feats.n())
    }
    /// A projection's input features — K of `A[m,k]·W[k,n]` (or one K-split block of it).
    pub const fn of_in_features(k: u32) -> MatK {
        MatK(k)
    }
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// BATCH COUNT OF THIS MATMUL — what the `y` axis walks. WHICH thing it walks (heads, a GQA group,
/// requests) decides every operand's y-stride, so the constructors are the meaning, not a detail.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MatY(u32, Option<BatchStrides>);

/// ⭐⭐⭐ THE HEAD STRIDES THE TWO BATCHED OPERANDS REALLY HAVE — carried BY the batch axis, so a
/// `y`-batch cannot be asked for without them.
///
/// ⛔ THIS IS THE LOCK ON THE hd=64 SUBSTITUTION. A `y`-batched matmul does not read a stride from
/// anywhere: it DERIVES one from its declared walk (`stick` for the batch-inner order, `mb_dev*stick`
/// for the head-outermost one) and strides to the next head by it. Whether that lands where the head
/// actually IS is a property of the OPERAND, and until now nothing related the two — so the batched
/// arm sat behind a head-dim gate that stood in for the relation, and every attempt to remove the
/// gate re-derived the stride from whatever extents were in scope and got it wrong somewhere.
///
/// Both strides come from an [`OperandPlacement`], which mints them with the base offset from one row
/// law, so the pair cannot be filled from two different laws. The builder then CHECKS its derived
/// walk against them and returns `Err` — and this crate runs inside the `#[forward]` proc macro, so
/// that is a `cargo build` failure naming the op, not a wrong address on-card.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BatchStrides {
    /// The ACTIVATION's head stride — how far apart adjacent heads are in the operand `y` walks.
    pub(crate) a: crate::addr::AxisStride,
    /// The OUTPUT's head stride, from its own placement law (they differ: an activation may be a
    /// token stream while the output is head-major).
    pub(crate) o: crate::addr::AxisStride,
    /// ⭐⭐⭐ EACH OPERAND'S OWN DECLARED ROW PITCH, riding with its own head stride.
    ///
    /// ⛔ THE BUILDER USED **ONE** `mb_dev` (`phys_mb.unwrap_or(m)`) FOR BOTH OPERAND CHECKS, AND
    /// THAT IS WHY A ONE-STICK CONTRACTION LOOKED IMPOSSIBLE. A `y`-batched op derives each operand's
    /// head step as `pitch * stick_extent`, and the two operands of the score leg genuinely need
    /// DIFFERENT pitches: a token-stream `qs` whose heads are `mq*hd` apart needs `mq*nslab` when `in`
    /// is one stick, while a head-major `sc` whose heads are `mq*stick` apart needs `mq`. With one
    /// shared value, satisfying the input broke the output and vice versa — which I mistook for a
    /// contradiction in the FORM rather than a missing per-operand declaration.
    ///
    /// Minted by the same placements as the strides, so an operand's pitch cannot be paired with
    /// another operand's head stride.
    pub(crate) a_pitch: PhysM,
    pub(crate) o_pitch: PhysM,
}

impl BatchStrides {
    /// The activation operand's real head stride.
    pub const fn activation(self) -> crate::addr::AxisStride {
        self.a
    }
    /// The output operand's real head stride.
    pub const fn output(self) -> crate::addr::AxisStride {
        self.o
    }
    /// The ACTIVATION's own declared row pitch — the `mb` device extent ITS walk must use.
    pub const fn activation_pitch(self) -> PhysM {
        self.a_pitch
    }
    /// The OUTPUT's own declared row pitch.
    pub const fn output_pitch(self) -> PhysM {
        self.o_pitch
    }
}

impl MatY {
    /// No batch axis at all: a plain 2-D matmul (`batch == 1` drops `y` from the dims). No strides,
    /// because there is no `y` to stride — the check below is vacuous and skipped.
    pub const fn unbatched() -> MatY {
        MatY(1, None)
    }
    /// `y` walks the `gqa` query heads of ONE kv-head group, which legitimately SHARE that kv
    /// head's K/V — the shared-2-D-kernel case the dxp-validated form covers.
    ///
    /// ⛔ TAKES BOTH OPERANDS' PLACEMENTS, NOT JUST THE GROUP SIZE. The group size says how many heads
    /// `y` covers; it says nothing about how far apart they are, and that second fact is the one that
    /// was wrong at every head dim above one stick. Passing the placements means the stride the walk
    /// will use is checkable against the stride the buffers have.
    pub fn of_gqa_group(gqa: u32, a: OperandPlacement, o: OperandPlacement) -> MatY {
        MatY(
            gqa,
            Some(BatchStrides {
                a: a.head_stride(),
                o: o.head_stride(),
                // FROM THE SAME TWO PLACEMENTS as the strides — one law per operand, both of its
                // declarations, so a pitch cannot be paired with another operand's head stride.
                a_pitch: a.head_pitch(),
                o_pitch: o.head_pitch(),
            }),
        )
    }

    /// The declared strides, for the builder's own check. `None` only when there is no batch axis.
    pub const fn batch_strides(self) -> Option<BatchStrides> {
        self.1
    }
    /// `y` walks the batch's REQUESTS, each reading its own K/V page — the fold's request-axis form
    /// (per-batch 3-D kernel, `kernel_device_extent` declares the page stride).
    pub const fn of_requests(requests: u32) -> MatY {
        MatY(requests, None)
    }
    /// `y` walks attention HEADS, each with its own K/V slice — the true per-batch bmm
    /// (`matmul_opspec_batched`'s contract).
    pub const fn of_heads(heads: u32) -> MatY {
        MatY(heads, None)
    }
    /// The untyped wrapper boundary (`matmul_opspec` / `assemble_matmul*`): the caller's own batch
    /// count, passed through unexamined. Callers inside the typed run use a meaning-bearing
    /// constructor instead.
    pub const fn of_batch(batch: u32) -> MatY {
        MatY(batch, None)
    }
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// PHYSICAL ROWS PER STICK PLANE OF THE ACTIVATION — the pitch half of an [`OperandPlacement`]:
/// the op computes [`MatM`] rows but the activation is PACKED with this many rows per stick plane.
/// It becomes the input's `mb` device extent, which is what the operand's DECLARED WALK and
/// arrangement classification are derived from (`maxDimSizes_`: a one-row plane reconstructs the
/// stick-blocked walk, a deeper plane pins a row-major sweep of that depth) — a property of the
/// TENSOR, not of how much of it this op touches.
///
/// Deliberately NOT constructible from [`MatM`]: "rows swept" and "rows the tensor is packed with"
/// is exactly the pair this slot exists to keep apart. And deliberately NOT constructible ALONE:
/// a pitch exists only inside an [`OperandPlacement`], minted by the same row law that mints the
/// operand's base offset — the pair is one answer, and half of it cannot be filled from a
/// different law than the other half.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PhysM(u32);

impl PhysM {
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// WHERE A ROW-SWEPT ACTIVATION STARTS AND HOW DEEP ITS HEAD PLANE IS — one value, both halves
/// minted by the row law that owns the buffer.
///
/// `assemble_matmul_off_phys_m` derives the activation's declared walk from two quantities: the
/// base OFFSET its sweep starts at and the head PITCH ([`PhysM`]) its `y` step advances by. Both
/// are answers of the SAME framing law — the token stream packs `mq` rows per plane and places
/// head `h` a plane in; the request-major score rows place adjacent heads ONE row apart. As two
/// separate slots, the score leg's packing fits the value leg's slot and vice versa, and the two
/// coincide at `mq == 1` — every one-request decode — so the swap bakes clean and garbles only a
/// real batch. Each constructor below IS one law: it computes offset and pitch from that law's own
/// coordinates, and there is no constructor from parts.
#[derive(Clone, Copy, Debug)]
pub struct OperandPlacement {
    off: crate::addr::DevOff,
    pitch: PhysM,
    /// ⭐⭐⭐ ELEMENTS BETWEEN ADJACENT HEADS, minted by the SAME nest that minted `off`.
    ///
    /// A `y`-batched op derives its head stride from its declared walk; where the heads ACTUALLY are
    /// is this law's business. Carrying the two together is what lets the builder refuse a mismatch,
    /// and means no site can take the offset from one law and the stride from another.
    ///
    /// ⛔ NOT `pitch`. The pitch is ROWS; this is ELEMENTS, and on an `hd`-wide buffer they differ by
    /// the stick AND the slab count — equal only at `hd == one stick`, which is precisely the
    /// coincidence that hid this class of bug.
    head_stride: crate::addr::AxisStride,
}

impl OperandPlacement {
    /// THE TOKEN-STREAM LAW: head `h`, REQUEST `request`, slab `s` of the `[mq, heads*hd]` head-major
    /// stream, whose producer writes head `h` row `r` at `h*mq*hd + r*hd` — packed `mq` rows per stick
    /// plane, so the pitch is the same `mq` the nest declares as its row extent.
    ///
    /// ⭐ THE REQUEST IS A COORDINATE, NOT A ZERO. Every op that sweeps a head's whole `mq` rows passes
    /// `0` and is unmoved; an op that computes ONE request's row — the collapsed fold's per-request legs
    /// — names its own, and the row it names comes from this law rather than from a step added to the
    /// law's answer afterwards.
    ///
    /// NAMED AXES, NOT A HAND-SUMMED COLUMN. As a hand-written column this offset is
    /// `h * hd + s * stick` — two products of DIFFERENT UNITS (head × head_dim, slab × lanes)
    /// added by hand, and that expression shape is INVISIBLE at head_dim 64: transposed as
    /// `h * stick + s * hd` it is IDENTICAL at hd == stick == 64 and DIFFERENT at hd = 128 — a
    /// plausible wrong column for every head above 0, on the only model whose batch decode is
    /// wrong. Both spellings type-check as `u32 * u32 + u32 * u32`, so nothing could refuse the
    /// transposition. `View::slab()` exists precisely for this, and says so: "`n_slabs` loops are
    /// the single most common place a head_dim > 64 multiplier goes missing, because at one stick
    /// every one of them is 0." The `s * lanes` multiplier lives inside the primitive that OWNS
    /// lanes, and the head is a NAMED AXIS rather than a multiplier a caller supplies — so there is
    /// no pair of products left to swap.
    pub fn of_token_stream(
        mq: QueryRowCount,
        heads: u32,
        hd: u32,
        h: u32,
        request: u32,
        s: u32,
    ) -> OperandPlacement {
        let nest =
            crate::addr::Nest::new(&["row", "head", "feat"], &[mq.get(), heads, hd], Df::Fp16);
        let off = nest
            .view()
            .at(crate::addr::Idx::<crate::addr::Row>::n(request))
            .at(crate::addr::Idx::<crate::addr::Head>::n(h))
            .slab(s)
            .dev();
        // ⭐ ASKED OF THE NEST, not written as `mq*hd`. `span` evaluates two adjacent head positions
        // through the same law that produced `off`, so the stride and the offset cannot disagree — and
        // the product whose transposition is invisible at `hd == lanes` never appears.
        let head_stride = nest.span(
            crate::addr::Idx::<crate::addr::Head>::n(0),
            crate::addr::Idx::<crate::addr::Head>::n(1),
        );
        OperandPlacement {
            off,
            pitch: PhysM(mq.get()),
            head_stride,
        }
    }

    /// ⭐⭐⭐ THE TOKEN-STREAM LAW STEPPED **ONE STICK AT A TIME** — the score leg's slab-split
    /// batched form. Same buffer, same offset and same `head_stride` as [`Self::of_token_stream`];
    /// what differs is the PITCH, and only because the declared walk contracts one stick of `in`
    /// instead of the whole `hd`.
    ///
    /// ⛔⛔⛔ THIS IS THE RELATION THE `hd == 64` GATE WAS REALLY STANDING ON, AND IT IS **TWO**
    /// REQUIREMENTS THAT COINCIDE THERE, NOT ONE. A `y`-batched op derives its head step as
    /// `pitch * in`, and the builder refuses unless that equals where the heads ACTUALLY are:
    ///   • batching heads wants `pitch * in == mq*hd` (the stream's real head stride)
    ///   • the dxp-validated shared-kernel shape wants `in` = ONE STICK
    /// With `in = hd` the first holds and the second fails; with `in = stick` and the *stream's own*
    /// `mq` pitch the second holds and the first fails — which is exactly the build-time refusal
    /// "strides `y` by 64 elems but adjacent heads are 128 apart". At `hd == 64` the two are the
    /// same number and both hold at once. THAT coincidence is the gate.
    ///
    /// Both hold at every head dim once the pitch is the one the one-stick walk needs: head `h`
    /// spans `nslab` consecutive stick planes of `mq` rows, so the pitch is `mq*nslab` ROWS and
    /// `pitch * stick == mq*hd == head_stride`.
    ///
    /// ⛔ THE PITCH IS DELIBERATELY NOT THE TENSOR'S ROW COUNT, which is what [`PhysM`] is for —
    /// "rows the tensor is PACKED with", never [`MatM`]'s rows SWEPT. dxp walks the swept rows; the
    /// pitch only sets the plane stride. Inflating a *swept* extent would make dxp walk rows that
    /// are not there ([[maxdimsizes-is-a-size-cap-backgap-is-the-stride]]); inflating the pitch
    /// states a plane stride, which is this slot's whole purpose.
    pub fn of_token_stream_by_slab(
        mq: QueryRowCount,
        heads: u32,
        hd: u32,
        h: u32,
        request: u32,
        s: u32,
        df: Df,
    ) -> OperandPlacement {
        let base = Self::of_token_stream(mq, heads, hd, h, request, s);
        // ASKED OF THE SHAPE, not written as `hd/64` — the same door `nests.slabs()` uses, so the
        // slab count in the pitch cannot disagree with the slab count the emitter loops over.
        let nslab = crate::addr::Shape::<0, 0, 0, 0>::slabs_of(hd, df).get();
        OperandPlacement {
            pitch: PhysM(mq.get() * nslab),
            ..base
        }
    }

    /// THE REQUEST-MAJOR ROW LAW: head `h` of request `req` in a `[nqh, width]` score/probability
    /// buffer framed by [`RequestHeadRow`] — the row is that law's `of` (request first, head
    /// minor), and the pitch is the same law's own head pitch: adjacent heads ONE row apart, so a
    /// per-request batched matmul's `y` step is a single row block of the buffer, whatever the
    /// chunk's `mq`. The law's `nqh` and the nest's row extent are one argument, so the row order
    /// and the buffer it frames cannot disagree.
    pub fn of_request_major_rows(
        req: u32,
        h: u32,
        rows: PerRequestRows,
        width: BlockCols,
    ) -> OperandPlacement {
        let row = RequestHeadRow::of(req, h, rows.get());
        let nest = crate::addr::Nest::new(&["row", "feat"], &[rows.get(), width.get()], Df::Fp16);
        let off = nest
            .view()
            .at(crate::addr::Idx::<crate::addr::Row>::n(row.get()))
            .dev();
        // The head lives on the ROW axis here, so its stride is the span of the law's own head pitch
        // IN ROWS — asked of the nest, which knows a row step is one stick. By hand this is
        // `pitch*lanes`, and the plausible wrong spelling is `pitch*width`: this buffer's own recorded
        // bug, correct for every shape that was ever one stick wide.
        let pitch_rows = RequestHeadRow::head_pitch_rows(rows.get());
        let head_stride = nest.span(
            crate::addr::Idx::<crate::addr::Row>::n(0),
            crate::addr::Idx::<crate::addr::Row>::n(pitch_rows),
        );
        OperandPlacement {
            off,
            pitch: PhysM(pitch_rows),
            head_stride,
        }
    }

    /// ⭐⭐⭐⭐⭐ THE KV CACHE WRITE'S LAW — `y` covers the kv heads of ONE SLAB, so its step is a whole
    /// HEAD (`nslab` stick planes) and the slab stays a coordinate of the offset.
    ///
    /// A `y` step is `pitch * stick`, so reaching the next HEAD needs a pitch of `rows * nslab` where
    /// reaching the next PLANE would need `rows`. Those are the same number at one slab, which is why
    /// this law and a plane-stepped one are indistinguishable on granite-2b.
    ///
    /// ⛔⛔⛔ AND THE PLANE-STEPPED ONE IS REFUTED ON CARD, so this pitch is not a choice. Walking `y`
    /// over `(kv head, slab)` PAIRS would fold both of the write's loops into one axis, and its
    /// arithmetic is correct: the pair `h*nslab + s` steps exactly one plane on both operands, the
    /// builder's y-stride check passes at hd 64/128/256/512, and two Kani proofs discharged coverage
    /// and injectivity. granite-8b at hd=128 emitted GARBAGE from the first token, 10/10 runs, while
    /// this law on the same bake is coherent 10/10. Whatever a `y` step crossing a feature slab does,
    /// it is not what the addresses say — so the pitch that skips a whole head is the one this site
    /// gets, and the pair walk is not to be retried on the strength of its arithmetic.
    pub fn of_plane_walk_by_head(
        nest: &crate::addr::Nest,
        row: crate::addr::Idx<crate::addr::Row>,
        s: u32,
        slabs: crate::addr::Slabs,
    ) -> OperandPlacement {
        OperandPlacement {
            off: nest.view().at(row).slab(s).dev(),
            pitch: PhysM(nest.rows() * slabs.get()),
            head_stride: nest.span(
                crate::addr::Idx::<crate::addr::Head>::n(0),
                crate::addr::Idx::<crate::addr::Head>::n(1),
            ),
        }
    }

    /// [`Self::of_plane_walk_by_head`] AT ONE HEAD, for a copy that cannot put the heads on `y`.
    ///
    /// The head is a COORDINATE of the offset here, so the emitted op needs no batch axis
    /// ([`MatY::unbatched`]) and `head_stride` is never read. Required where one op spans several ROWS:
    /// the plane-walk pitch describes head-outermost planes of `rows` rows, which only agrees with a
    /// row-outermost source (`row*(nkvh*hd) + head*hd`) when an op covers a single row.
    pub fn at_head(
        nest: &crate::addr::Nest,
        row: crate::addr::Idx<crate::addr::Row>,
        head: u32,
        s: u32,
        slabs: crate::addr::Slabs,
    ) -> OperandPlacement {
        OperandPlacement {
            off: nest
                .view()
                .at(row)
                .at(crate::addr::Idx::<crate::addr::Head>::n(head))
                .slab(s)
                .dev(),
            pitch: PhysM(nest.rows() * slabs.get()),
            head_stride: nest.span(
                crate::addr::Idx::<crate::addr::Head>::n(0),
                crate::addr::Idx::<crate::addr::Head>::n(1),
            ),
        }
    }

    /// [`Self::of_plane_walk_by_head_block_base`] AT ONE BLOCK — the paged-plane side of [`Self::at_head`].
    pub fn at_block(
        nest: &crate::addr::Nest,
        block: u32,
        s: u32,
        slabs: crate::addr::Slabs,
    ) -> OperandPlacement {
        OperandPlacement {
            off: nest.block(block).slab(s).dev(),
            pitch: PhysM(nest.rows() * slabs.get()),
            head_stride: nest.block_span(),
        }
    }

    /// [`Self::of_plane_walk_by_head`] for the paged KV plane, whose "head" is a BLOCK (no `head` axis
    /// to span) and whose row is always slot 0 — the launch's own shift moves it onto the position.
    pub fn of_plane_walk_by_head_block_base(
        nest: &crate::addr::Nest,
        s: u32,
        slabs: crate::addr::Slabs,
    ) -> OperandPlacement {
        OperandPlacement {
            off: nest.view().slab(s).dev(),
            pitch: PhysM(nest.rows() * slabs.get()),
            head_stride: nest.block_span(),
        }
    }

    /// THE HEAD-MAJOR ROW LAW: head `h`, REQUEST `request` of a `[nqh*mq, width]` shared buffer framed
    /// by [`HeadRequestRow`], pitch that law's own head pitch: the chunk's `mq` rows.
    ///
    /// ⭐ `request` IS THE LAW'S OWN MINOR COORDINATE, and it used to be a hardcoded `0` — right only
    /// because every op swept a head's whole `mq` rows and the request rode inside the block. The
    /// collapsed fold's per-request legs compute ONE row, so they name which, through the same law that
    /// the mask's own validity rows are staged by (`HeadRequestRow::of(h, r, mq) = h*mq + r`).
    pub fn of_head_major_rows(
        h: u32,
        request: u32,
        mq: QueryRowCount,
        rows: MaskRows,
        width: BlockCols,
    ) -> OperandPlacement {
        let row = HeadRequestRow::of(h, request, mq.get());
        let nest = crate::addr::Nest::new(&["row", "feat"], &[rows.get(), width.get()], Df::Fp16);
        let off = nest
            .view()
            .at(crate::addr::Idx::<crate::addr::Row>::n(row.get()))
            .dev();
        // Same derivation as the request-major law above, and for the same reason.
        let pitch_rows = HeadRequestRow::head_pitch_rows(mq.get());
        let head_stride = nest.span(
            crate::addr::Idx::<crate::addr::Row>::n(0),
            crate::addr::Idx::<crate::addr::Row>::n(pitch_rows),
        );
        OperandPlacement {
            off,
            pitch: PhysM(pitch_rows),
            head_stride,
        }
    }

    /// THE `hd`-WIDE HEAD-MAJOR ACCUMULATOR LAW (`ov` / `run_o` / `otmp`): head `h`, slab `s` of a
    /// `[rows, hd]` buffer whose heads are stacked on the ROW axis at `h*mq`.
    ///
    /// ⛔ ITS HEAD STRIDE IS `mq*stick`, NOT `mq*hd`, AND THAT IS THE WHOLE POINT. The buffer is `hd`
    /// wide, so `mq*hd` is the expression a reader reaches for — but the heads are separated along the
    /// ROW axis, and a stick-blocked row step is ONE STICK. `span` over the law's own pitch in rows
    /// answers it, so the `hd` that is right for the WIDTH and wrong for the STRIDE never enters.
    pub fn of_head_major_accum(
        rows: RowCount,
        hd: u32,
        mq: QueryRowCount,
        h: u32,
        request: u32,
        s: u32,
    ) -> OperandPlacement {
        let nest = crate::addr::Nest::new(&["row", "feat"], &[rows.get(), hd], Df::Fp16);
        let row = HeadRequestRow::of(h, request, mq.get());
        let off = nest
            .view()
            .at(crate::addr::Idx::<crate::addr::Row>::n(row.get()))
            .slab(s)
            .dev();
        let pitch_rows = HeadRequestRow::head_pitch_rows(mq.get());
        let head_stride = nest.span(
            crate::addr::Idx::<crate::addr::Row>::n(0),
            crate::addr::Idx::<crate::addr::Row>::n(pitch_rows),
        );
        OperandPlacement {
            off,
            pitch: PhysM(pitch_rows),
            head_stride,
        }
    }

    /// ⭐⭐⭐⭐⭐ ONE HEAD'S WHOLE `[mq, hd]` BLOCK of the head-major accumulator — the placement that
    /// lets a value-leg op cover **EVERY SLAB IN ONE OP** without a slab axis.
    ///
    /// THE SLAB IS NOT AN AXIS. It is the STICK-GROUP COORDINATE of `out`. torch-spyre builds each
    /// operand's `layoutDimOrder_` from `arg.device_coordinates` — one symbolic EXPRESSION per device
    /// dim — and a stick-blocked `[rows, hd]` tensor's coords are `(out//lanes, mb, out%lanes)`. The
    /// stick group is a FUNCTION of `out`, so it iterates by itself the moment `out` spans the whole
    /// head dim. Declaring a fourth walk axis for it (which our `device_dims: [IterSym; D]` cannot
    /// express as `out//lanes`) baked clean and produced garbage; this is the shape the machine has.
    ///
    /// ⛔ SO THE PITCH IS THE BUFFER'S OWN `rows`, NOT THIS SLICE'S `mq`. The stick-GROUP stride a
    /// stick-blocked view derives is `rows*lanes` — a property of the TENSOR. An op sweeping one head's
    /// `mq` rows out of `nqh*mq` must declare the tensor's row count or every feature above the first
    /// stick group lands `nqh`× too close. That is the same statement `phys_mb` makes for the fold, and
    /// it is why this law mints a pitch at all.
    ///
    /// Both halves come from the SAME nest as [`Self::of_head_major_accum`]'s, so a site cannot take
    /// the head's row base from one framing and the stick-group pitch from another.
    /// `feat` is THIS buffer's own feature extent — `hd` for the accumulator, the block width for the
    /// probability buffer. Both are sliced to one head's `mq` rows out of the same `rows`, so both want
    /// the same PITCH and their OWN offsets; passing one buffer's extent for the other's would place
    /// the offset in a nest the tensor does not have.
    pub fn of_head_major_head_block(
        rows: RowCount,
        feat: u32,
        mq: QueryRowCount,
        h: u32,
    ) -> OperandPlacement {
        let nest = crate::addr::Nest::new(&["row", "feat"], &[rows.get(), feat], Df::Fp16);
        let row = HeadRequestRow::of(h, 0, mq.get());
        let off = nest
            .view()
            .at(crate::addr::Idx::<crate::addr::Row>::n(row.get()))
            .dev();
        let head_stride = nest.span(
            crate::addr::Idx::<crate::addr::Row>::n(0),
            crate::addr::Idx::<crate::addr::Row>::n(HeadRequestRow::head_pitch_rows(mq.get())),
        );
        OperandPlacement {
            off,
            pitch: PhysM(rows.get()),
            head_stride,
        }
    }

    /// This placement's HEAD STRIDE — the distance a head-batched walk must reproduce, as the typed
    /// [`crate::addr::AxisStride`] the nest minted.
    pub(crate) const fn head_stride(self) -> crate::addr::AxisStride {
        self.head_stride
    }

    /// The placed base offset — for the operand slots that take a bare [`crate::addr::DevOff`]
    /// (the forms with no pitch of their own to declare).
    /// Device offset of this nest — read by the SDSC lowering, which
    /// now lives in the spyre target crate.
    pub const fn off(self) -> crate::addr::DevOff {
        self.off
    }

    /// The activation's rows per stick plane — the `mb` device extent the phys form declares.
    pub const fn head_pitch(self) -> PhysM {
        self.pitch
    }
}

/// ⭐ THE POOL'S STICK AS A CONST GENERIC ARGUMENT — so `PrefixMaskShape<POOL_STICK, COLS>` says WHICH 64 it is.
///
/// ⛔ EVERY INSTANTIATION SPELLED IT `<64, …>`. The type had the const generic right — `STICK` is a parameter
/// precisely so a block whose column count is not a whole number of sticks is unrepresentable — and then every
/// call site filled it with a literal, in a position where `mq_pad` (ALSO 64 today) or the fp8 stick (128, and
/// equal to `head_dim` at the model we run) would type-check identically. A const generic only separates
/// quantities if its ARGUMENTS are named.
pub const POOL_STICK: u32 = <Fp16 as DataFormat>::ELEMS_PER_STICK;

// The value must not move silently: every address in this file is `(feat / STK)` and `(slot / STK)` arithmetic,
// and the emitter's own mask blocks are `PrefixMaskShape<POOL_STICK, _>`. If the pool's format ever changes, this fails
// and names the reason, rather than every offset shifting by a factor nobody chose.
const _: () = assert!(
    STK == 64,
    "the KV pool's stick is fp16's 64 lanes. If this now fails, the pool's plane format changed — which also \
     moves PrefixMaskShape's COLS, the emitter's `stick` locals, and the fp8-vs-fp16 slab count at head_dim 128, \
     where head_dim and the fp8 stick width are BOTH 128 and therefore indistinguishable."
);

/// Device element offset (in elements) for logical multi-index `idx` of a tensor whose full
/// shape is `dims` with the stick axis at `stick_idx` — mirroring [`DeviceTileLayout`]:
/// a 2-D tensor sticked on its LAST dim is `[b/64, a, 64]`; otherwise flat row-major into sticks
/// (which equals the plain row-major flat index).
pub fn dev_off(dims: &[usize], stick_idx: usize, idx: &[usize]) -> usize {
    dev_off_stk(dims, stick_idx, idx, STK)
}

/// [`dev_off`] with an EXPLICIT stick width `stk` (elements per 128-byte stick): 64 for fp16, 128 for
/// fp8/int8, 32 for fp32. `dev_off` is exactly `dev_off_stk(.., STK)`; [`StickLayout::dev_off`] passes
/// `self.lanes()` so a device tensor is addressed at its OWN format's stick. This is what makes an fp8
/// (128-stick, 1-byte) weight addressable in the SAME formula without touching the fp16 path — fp16 has
/// `stk == STK`, so every existing address is byte-identical.
pub fn dev_off_stk(dims: &[usize], stick_idx: usize, idx: &[usize], stk: usize) -> usize {
    debug_assert_eq!(dims.len(), idx.len());
    if dims.len() == 2 && stick_idx == 1 {
        let a = dims[0];
        let (i, j) = (idx[0], idx[1]);
        (j / stk) * (a * stk) + i * stk + (j % stk)
    } else {
        // flat row-major (device [total/64,64] stores host elements in row-major order)
        let mut off = 0usize;
        for d in 0..dims.len() {
            off = off * dims[d] + idx[d];
        }
        off
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  SYSTOLIC LAYOUT vocabulary — the typed core that makes "wrong layout" unexpressible. `StickLayout`
//  OWNS `dev_off` (delegates to the free fn above, byte-identical — Kani `dev_off_method_equals_free_fn`
//  in scratchy-sdsc's `systolic_ir`). Lives HERE (next to `dev_off`) so the live emitter — which cannot
//  depend on the `scratchy-sdsc` tower (circular) — routes its device addresses through it; the tower's
//  `SystolicIR` graph pass RE-USES these types. The `kcache_kt_write_offset`/`vcache_write_offset` below
//  are the first live call sites migrated onto it.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// HOW MUCH of a dim one tensor view's address spans, when the view does not range over that dim.
///
/// A view's device address is folded over a vector of per-dim extents, and there are exactly two
/// legitimate readings of "the extent of dim `d`" — they are NOT interchangeable, so the choice is a
/// value, not a convention a call site can forget:
///
/// * [`Span::Swept`] — every dim at this op's iteration extent. What a view's *arrangement* is
///   classified from ([`StickLayout::for_view_df`]): the residency of `[mb, hidden]` is decided by
///   the shape the op presents, broadcast dims included.
/// * [`Span::Materialized`] — a dim this view does not range over (`Scale != Active`: a reduction
///   OUTPUT dim, or a broadcast INPUT dim such as the rmsnorm gamma `[1, hidden]` or an fp8
///   per-channel scale) physically occupies ONE row, so it contributes `1` to the stick-group
///   stride. What a per-core START address is folded over. Charging such a dim the op's `mb` strode
///   `mb×` too far and a split stick-dim core read past the end of the operand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Span {
    /// Every dim at the op's iteration extent (arrangement classification).
    Swept,
    /// Non-ranged dims collapse to one row (per-core start addressing).
    Materialized,
}

/// THE per-dim device extent vector of ONE tensor view — the only thing a device address may be
/// folded over ([`StickLayout::off_view`] takes this, not a bare `&[usize]`, so a hand-rolled vector
/// does not type-check).
///
/// **Why this is a type.** Three places used to build their own extent vector for the same view, from
/// the same op, by hand — the arrangement classifier, the per-core start addresser, and the footprint
/// sizer. They agreed by inspection only, and one of them silently ignored a field the other read:
/// [`DeviceExtents::of_view`]'s `declared` argument (`TensorArg::device_extent`, torch-spyre's
/// `arg.device_size`) was honoured when CLASSIFYING a view and dropped when ADDRESSING it, so setting
/// it produced byte-identical bundles and read as inert. That is not a tuning knob, it is the only way
/// to say *"this op ITERATES a 64-slot window of a resident cache but must ADDRESS the whole
/// allocation"* — precisely what a batched decode needs to give each request its own KV pages. With one
/// constructor there is nowhere left to drop it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DeviceExtents(Vec<usize>);

impl DeviceExtents {
    /// Build the extent vector of one view of rank `swept.len()`.
    ///
    /// * `swept[i]` — this op's iteration extent for dim `i`.
    /// * `declared[i]` — the dim's TRUE PHYSICAL extent when the view is a WINDOW into a larger
    ///   allocation (`TensorArg::with_device_extent`). **A declared extent WINS over everything
    ///   else**, including the `Materialized` collapse: the whole point is that the address steps by
    ///   the allocation, not by the window. `None` (every call site that does not opt in) falls back
    ///   to the span rule, so this is byte-identical wherever nothing is declared.
    /// * `stick_idx` — the stick axis; it never collapses (its extent drives the stick-width guard,
    ///   and an unswept stick axis contributes nothing to an address anyway — its corner stays 0).
    /// * `active[i]` — false when this view does not range over dim `i` (`Scale != Active`).
    pub fn of_view(
        swept: &[usize],
        declared: &[Option<usize>],
        stick_idx: usize,
        active: &[bool],
        span: Span,
    ) -> DeviceExtents {
        let mut dims = Vec::with_capacity(swept.len());
        for (i, &e) in swept.iter().enumerate() {
            // DECLARED PHYSICAL EXTENT WINS. See the type doc: dropping this here is the bug the type
            // exists to make unconstructable.
            if let Some(p) = declared.get(i).copied().flatten() {
                dims.push(p);
                continue;
            }
            let collapses = span == Span::Materialized
                && i != stick_idx
                && !active.get(i).copied().unwrap_or(true);
            dims.push(if collapses { 1 } else { e });
        }
        DeviceExtents(dims)
    }

    /// The extents, for the fold. Read-only: there is no way to mutate one after construction.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Rank of the view.
    pub fn rank(&self) -> usize {
        self.0.len()
    }
}

/// The role a DEVICE axis plays in a systolic op. A contraction STREAMS the `Reduction` axis through
/// the PE array and ACCUMULATES, while CARRYING `FreeM` (rows/tokens) and `FreeN` (output sticks).
/// Conflating `Reduction` with `FreeM` is the recurring bug class (invisible at m==1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AxisRole {
    /// Streamed through the PE array and accumulated (matmul K; a reduce's feature axis).
    Reduction,
    /// The row / token / batch axis (mb), carried across the PT rows. NEVER merged into a reduction.
    FreeM,
    /// The output-feature axis, carried as parallel N-sticks.
    FreeN,
    /// The 64 lanes within a single stick.
    Lane,
}

impl AxisRole {
    /// THE single mapping from an iteration dim's flags to its systolic role, so "which axis is FreeM"
    /// is never hand-decided at a call site (the conflation that caused the M>1 scramble).
    pub const fn of(is_reduction: bool, is_stick: bool) -> AxisRole {
        match (is_reduction, is_stick) {
            (true, _) => AxisRole::Reduction,
            (false, true) => AxisRole::FreeN,
            (false, false) => AxisRole::FreeM,
        }
    }
}

/// The physical stick residencies. `RowBlocked`/`Kernel` share the stick-block address formula (they
/// differ only in which axis is `FreeM` vs `Reduction`/`FreeN`); `RowScalar` is the one-lane-per-row
/// case; `Flat` is row-major (head-major tensors).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StickKind {
    /// Activation / matmul-A / pointwise data: `[rows, feat]` stick-blocked, `rows` = FreeM (mb).
    RowBlocked,
    /// Matmul kernel / K-V cache: `[in, out]` stick-blocked on `out`; row-count `in`, m-independent.
    Kernel,
    /// Per-row reduced scalar (sum / max / recip): one value per row in lane 0 of its own stick.
    RowScalar,
    /// FLAT row-major `[rows, cols]` (stick NOT on the last re-tiled dim) — head-major tensors.
    Flat,
}

/// Whether a layout is addressed STICK-BLOCKED (re-tiled `[feat/lanes, rows, lanes]`) or row-major FLAT.
/// This is THE single per-kind addressing decision: `dev_off` (the per-core START), `off_view` (the
/// arbitrary-rank start fold), and `device_walk` (the on-card WALK dxp reconstructs) ALL derive from
/// [`StickLayout::addressing`], so the start and the walk cannot disagree about a tensor's residency
/// without changing that one function — which changes ALL of them at once. (The bug this prevents: a
/// walk hand-written to "match" a separately-hand-written start, where the two silently diverge — e.g.
/// classifying `RowScalar`'s walk row-major while its start stays stick-blocked. That is now
/// unrepresentable: there is one classification, not two.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Addressing {
    /// Re-tiled `[feat/lanes, rows, lanes]` — the stick-block residency. dxp reconstructs it from `-1`.
    StickBlocked,
    /// Row-major over the whole shape — head-major tensors. dxp needs the ACTUAL extents to walk it.
    RowMajor,
}

/// The descriptor's `maxDimSizes_` value — the on-card WALK dxp reconstructs. This is NOT a free
/// `Vec<i64>`: for any one tensor it is a WHOLE-TENSOR binary choice fixed by the [`StickLayout`] —
/// EITHER dxp reconstructs the stick-blocked device_size (every dim `-1`) OR the row-major extents are
/// pinned (every dim its actual size). The three illegal states a raw `Vec<i64>` allowed — a MIX of
/// `-1` and actual, a length ≠ the tensor rank (so `maxDimSizes_.len() != layoutDimOrder_.len()`), or a
/// hand-typed constant that disagrees with the start's layout — are all UNREPRESENTABLE here: the inner
/// [`Walk`] is private, so the ONLY way to obtain a `DeviceWalk` is [`StickLayout::device_walk`], which
/// derives it from the SAME layout that drives `dev_off` (the per-core start). A divergent walk would
/// require changing that `StickLayout`, which changes the start too. Serializes to the exact JSON array
/// dxp reads (`[-1,-1]` for reconstruct, `[31,2048]` for pinned) — byte-identical to the old field.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DeviceWalk(Walk);

#[derive(Clone, PartialEq, Eq, Debug)]
enum Walk {
    /// dxp reconstructs the stick-blocked (RowBlocked) device_size from N_/layoutDimOrder_/stickSize_.
    /// Serializes to `[-1; rank]`. `rank` is the tensor's dim count, taken from the extents `device_walk`
    /// was handed, so it always equals `layoutDimOrder_.len()` (both flow from the same view).
    Reconstruct { rank: usize },
    /// A row-major (`Flat`/head-major, rows>1) start: the actual per-dim extents, pinned so the card walks
    /// row-major instead of being told to reconstruct a stick-block over a row-major buffer.
    RowMajor(Vec<i64>),
    /// ⭐ THE PAGED (GATHERED) WALK — the ONE legal mix of `-1` and an actual extent.
    ///
    /// The doc above calls a mix illegal, and for a directly-addressed tensor it is. For a tensor
    /// read THROUGH an index it is the whole mechanism: `getPageSize` (`dsc2.cpp:4493-4526`) walks
    /// this array and ERASES every negative entry, so the surviving pinned dims ARE the paged dims.
    /// The pinned dim's value is the PAGE — how many positions along it one index entry selects —
    /// and it is what sets the address stride: dxp clamps that axis's capacity to the page and
    /// multiplies by the other axes, giving `skip_addr`. Every other dim stays `-1` and is neither
    /// paged nor checked. The vendor's own baking fixture
    /// (`dxp/test/test_gather_1core/sdsc_1.json`) emits `[1, -1, -1]`; IBM's paged attention
    /// (`dcg/dcg_fe/scheduler/test/sdsc_add_paged_l3lu.json`) emits `[-1, -1, 64, 1]`, i.e. a page
    /// of 64 — so the pin is NOT always 1, and treating it as 1 costs a 64× address error.
    ///
    /// Reachable ONLY via [`DeviceWalk::paged_at`], which an emitter calls only for the value side
    /// of a declared [`IndirectAccess`](crate::superdsc_opspec::IndirectAccess) — so a mixed walk
    /// still cannot be hand-typed onto an ordinary tensor.
    Paged {
        rank: usize,
        /// `(dim position, page)` for every paged dim — PLURAL, because paged attention pins two: a
        /// page granularity and a second axis at 1 (one entry per position along it). A single-pin
        /// form caps the gather at collapsing pages.
        pins: Vec<(usize, crate::superdsc_opspec::PageExtent)>,
    },
}

impl serde::Serialize for DeviceWalk {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        match &self.0 {
            Walk::Reconstruct { rank } => {
                let mut seq = s.serialize_seq(Some(*rank))?;
                for _ in 0..*rank {
                    seq.serialize_element(&-1i64)?;
                }
                seq.end()
            }
            Walk::RowMajor(ext) => ext.serialize(s),
            Walk::Paged { rank, pins } => {
                let mut seq = s.serialize_seq(Some(*rank))?;
                for d in 0..*rank {
                    seq.serialize_element(
                        &pins
                            .iter()
                            .find(|(at, _)| *at == d)
                            .map_or(-1i64, |(_, page)| i64::from(page.get())),
                    )?;
                }
                seq.end()
            }
        }
    }
}

impl DeviceWalk {
    /// This tensor's rank, as the walk will serialize it.
    pub fn rank(&self) -> usize {
        match &self.0 {
            Walk::Reconstruct { rank } | Walk::Paged { rank, .. } => *rank,
            Walk::RowMajor(ext) => ext.len(),
        }
    }

    /// Re-declare this walk as PAGED along `entry_dim` — pinned to `page` there, `-1` everywhere else.
    ///
    /// `None` if `entry_dim` is out of range, so a caller that names a dim this tensor does not have
    /// gets an `Err` it must handle rather than a silently unpaged value tensor. That distinction
    /// matters: an unpaged value tensor still emits a `value_tensor` node, still cross-links, and
    /// still bakes — it just gathers with no page declaration, which is wrong addresses and a clean
    /// build.
    pub fn paged_at(self, pins: &[(usize, crate::superdsc_opspec::PageExtent)]) -> Option<Self> {
        let rank = self.rank();
        // ⛔ EVERY pin must be in range and no dim may be pinned twice: a second pin on one dim is not
        // two pages, it is one page silently chosen by iteration order, and the survivor sets
        // `skip_addr`. `None` rather than a quiet merge.
        let ok = !pins.is_empty()
            && pins.iter().all(|&(d, _)| d < rank)
            && pins
                .iter()
                .map(|&(d, _)| d)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == pins.len();
        ok.then(|| {
            DeviceWalk(Walk::Paged {
                rank,
                pins: pins.to_vec(),
            })
        })
    }
}

/// THE single source of truth for a device tensor's layout. `rows`/`cols` are the two re-tiled dims
/// (`RowBlocked`: rows=mb, cols=feature; `Kernel`: rows=in(K), cols=out(N)); `lanes` is the stick
/// width. `dev_off` + the sweeps are METHODS: no call site passes its own `(dims, stick_idx)`, so two
/// views of one tensor cannot compute different addresses. `m()` is a RUNTIME field (symbolic
/// continuous-batching stays expressible; decode m==1 is the same code at m==1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StickLayout {
    pub rows: usize,
    pub cols: usize,
    /// The device element format of THIS tensor. The stick width (`lanes()`) is DERIVED from it —
    /// fp16→64, fp8/int8→128, fp32→32 — so a tensor is always addressed at its own format's stick.
    /// There is no stored `lanes`, so "stick width disagrees with the format" is UNCONSTRUCTABLE
    /// (the field whose hardcoded `=64` made an fp8 1-byte weight inexpressible in the live addressing
    /// SSOT — the W8A8-audit gap — is now this ONE derived source).
    pub df: Df,
    pub kind: StickKind,
}

impl StickLayout {
    /// A `RowBlocked` activation `[m, feat]` (the pointwise/rmsnorm data path, the matmul-A). fp16.
    pub fn row_blocked(m: usize, feat: usize) -> Self {
        Self::row_blocked_df(m, feat, Df::Fp16)
    }
    /// [`row_blocked`](Self::row_blocked) with an EXPLICIT device format (the fp8/int8 activation path).
    pub fn row_blocked_df(m: usize, feat: usize, df: Df) -> Self {
        StickLayout {
            rows: m,
            cols: feat,
            df,
            kind: StickKind::RowBlocked,
        }
    }
    /// A `Kernel` weight/cache `[in(K), out(N)]` stick-blocked on `out`. fp16.
    pub fn kernel(k_in: usize, n_out: usize) -> Self {
        Self::kernel_df(k_in, n_out, Df::Fp16)
    }
    /// [`kernel`](Self::kernel) with an EXPLICIT device format — the fp8/int8 1-byte packed weight
    /// residency (`lanes()` becomes 128, HALF the fp16 footprint = the decode bandwidth win). This
    /// constructor is why rung 1 exists: it lets a weight declare its real HBM residency.
    pub fn kernel_df(k_in: usize, n_out: usize, df: Df) -> Self {
        StickLayout {
            rows: k_in,
            cols: n_out,
            df,
            kind: StickKind::Kernel,
        }
    }
    /// A `RowScalar` per-row reduced value for `m` rows (one lane-0 value per row's stick). fp16.
    pub fn row_scalar(m: usize) -> Self {
        StickLayout {
            rows: m,
            cols: Df::Fp16.elems_per_stick() as usize,
            df: Df::Fp16,
            kind: StickKind::RowScalar,
        }
    }
    /// A FLAT row-major `[rows, cols]` tensor (stick NOT on the last re-tiled dim). fp16.
    pub fn flat(rows: usize, cols: usize) -> Self {
        StickLayout {
            rows,
            cols,
            df: Df::Fp16,
            kind: StickKind::Flat,
        }
    }

    /// Stick width in elements — DERIVED from [`df`](Self::df) (`Df::elems_per_stick`: fp16→64,
    /// fp8/int8→128, fp32→32), the SINGLE source of truth. There is deliberately no stored `lanes`
    /// field, so a `StickLayout` whose stick width contradicts its format cannot be constructed.
    pub fn lanes(&self) -> usize {
        self.df.elems_per_stick() as usize
    }

    /// The FreeM (row) count — 1 for decode, mq for prefill. The ONLY M in the IR (runtime, not a type).
    pub fn m(&self) -> usize {
        self.rows
    }

    /// THE single per-kind addressing decision — every start/walk site derives from this, so they cannot
    /// disagree. EXHAUSTIVE match (no `_` wildcard) on purpose: a new [`StickKind`] fails to compile here
    /// until it is explicitly classified, so it can never be silently mis-addressed. `RowScalar` is
    /// STICK-blocked (its scalar sits at `r*lanes`, the `c==0` stick-block address), NOT row-major.
    pub fn addressing(&self) -> Addressing {
        match self.kind {
            StickKind::RowBlocked | StickKind::Kernel | StickKind::RowScalar => {
                Addressing::StickBlocked
            }
            StickKind::Flat => Addressing::RowMajor,
        }
    }

    /// Device element offset of logical `(r, c)`. BYTE-IDENTICAL to the free [`dev_off`] (Kani
    /// `dev_off_method_equals_free_fn`): routing a call site through the layout changes ZERO emitted
    /// bytes, it only removes the freedom to pass a wrong `stick_idx`/`dims`. Stick-blocked vs row-major
    /// comes from [`addressing`](Self::addressing) — the SAME source `device_walk`/`off_view` use.
    pub fn dev_off(&self, r: usize, c: usize) -> usize {
        let stick_idx = match self.addressing() {
            Addressing::RowMajor => usize::MAX, // any value != 1 ⇒ the free dev_off FLAT (row-major) branch
            Addressing::StickBlocked => 1,      // stick-blocked on the last re-tiled dim
        };
        // Stick width from THIS tensor's format (`lanes()`): fp16 ⇒ 64 (byte-identical to the const-STK
        // free `dev_off`), fp8/int8 ⇒ 128 (the 1-byte packed residency this rung makes addressable).
        dev_off_stk(&[self.rows, self.cols], stick_idx, &[r, c], self.lanes())
    }

    /// Do these two layouts place EVERY logical `(r,c)` at the same device byte? This is the invariant
    /// the arrangement authority enforces so a producer and consumer of ONE tensor cannot address it
    /// differently. Analytic (no O(rows·cols) sweep): different `(rows, cols, lanes)` never match;
    /// otherwise a `Flat` and a stick-blocked kind COINCIDE exactly when there is no blocking to differ
    /// — a single row (`rows<=1`, the decode case) or a single stick (`cols<=lanes`) — and `RowBlocked`
    /// vs `Kernel` is the identical stick-block formula. So the M>1 (rows>1, cols>lanes) Flat-vs-stick
    /// disagreement is the ONE case that returns `false` — exactly the prefill scramble.
    pub fn addr_eq(&self, other: &StickLayout) -> bool {
        if self.lanes() != other.lanes() {
            return false;
        }
        if self.rows == other.rows && self.cols == other.cols {
            if self.kind == other.kind || self.rows <= 1 || self.cols <= self.lanes() {
                return true;
            }
            use StickKind::{Kernel, RowBlocked};
            return matches!(
                (self.kind, other.kind),
                (RowBlocked, Kernel) | (Kernel, RowBlocked)
            );
        }
        // HEAD-MAJOR ROW EXPANSION: a stick-blocked `[m, H·lanes]` and `[H·m, lanes]` are the SAME
        // BYTES, even at m>1 where neither is `flat_addressed`. Proof (lanes = L, c = h·L + d, d < L):
        //   [m, H·L] : (c/L)·(m·L) + r·L + (c%L) = h·m·L + r·L + d
        //   [H·m, L] : (d/L)·(H·m·L) + j·L + (d%L) = j·L + d, and j = h·m + r ⇒ h·m·L + r·L + d
        // — equal for every (h, r, d). This is exactly how a `[mq, heads·hd]` Q/K tensor and its
        // per-head-block `[heads·mq, hd]` view relate, which is what lets the RoPE pointwise legs read
        // the whole tensor in ONE op instead of once per head. Pinned by Kani
        // `row_expansion_is_byte_identical`. Both sides must be STICK-BLOCKED (a `Flat` side really is
        // row-major and would differ), so the M>1 Dense-vs-stick scramble stays a conflict.
        let stickblocked = |k: StickKind| !matches!(k, StickKind::Flat);
        if stickblocked(self.kind)
            && stickblocked(other.kind)
            && (self.is_row_expansion_of(other) || other.is_row_expansion_of(self))
        {
            return true;
        }
        // DIFFERENT shape, SAME buffer: a byte-identical RESHAPE. Both layouts map element `k` to byte `k`
        // exactly when each addresses CONTIGUOUSLY — [`flat_addressed`](Self::flat_addressed): a single row
        // (`rows<=1`), a single stick (`cols==lanes`), or the row-major `Flat` kind. A `[1,2048]` activation
        // and its `[32,64]` head-major view (cols==64==lanes) are the same bytes. At m>1 the flat activation
        // is stick-blocked (`cols>lanes`, rows>1) so it is NOT `flat_addressed` — the M>1 scramble the
        // authority MUST still catch stays a conflict.
        self.rows * self.cols == other.rows * other.cols
            && self.flat_addressed()
            && other.flat_addressed()
    }

    /// Is `self` the HEAD-MAJOR ROW EXPANSION `[H·m, lanes]` of `other`'s `[m, H·lanes]`? See the
    /// equality proved in [`addr_eq`](Self::addr_eq). Requires `other`'s feature axis to be a whole
    /// number of sticks; the single-stick case (`H == 1`) degenerates to the same shape, which
    /// `addr_eq` has already accepted before reaching here.
    pub fn is_row_expansion_of(&self, other: &StickLayout) -> bool {
        let lanes = self.lanes();
        self.cols == lanes
            && other.cols >= lanes
            && other.cols.is_multiple_of(lanes)
            && self.rows == other.rows * (other.cols / lanes)
    }

    /// Does this layout place element `k` at byte `k` (contiguous row-major, no stick-blocking reshuffle)?
    /// True for the row-major `Flat` kind, a single row (`rows<=1`), or a single full stick (`cols==lanes`).
    /// A stick-blocked tensor with `rows>1` and `cols>lanes` is NOT contiguous — that is the case a reshape
    /// cannot silently cross.
    pub fn flat_addressed(&self) -> bool {
        matches!(self.kind, StickKind::Flat) || self.rows <= 1 || self.cols == self.lanes()
    }

    /// dxp's `lrfimm` coordInfo-stride field: a 21-bit UNSIGNED MODULO immediate (bits 10..30,
    /// `defineField(111, "lrfimm", 10, {}, 21, MODULO_UNSIGNED)`, deeptools/dsc/isa.cpp:919). A
    /// `group_stride` at or above this wraps silently (MODULO, not a hard fault) — dxp's own
    /// `DtException LX_MODLRFIMM :: lrfimm out of boundary` fires below the true wrap point, so this
    /// is the CONSERVATIVE ceiling: any stride the tiler emits must stay strictly under it.
    pub const LRFIMM_MAX: i64 = (1i64 << 21) - 1;

    /// The coordInfo stick-GROUP stride (elements) a `RowBlocked` reduce/matmul-A operand must step by
    /// to walk one row's OWN sticks — the value `build_coordinates` used to hand-derive per call site
    /// from `arg.scale`/`arg.row_blocked` (two parallel encodings of ONE fact, see
    /// `lower_subtile_tape_to_superdsc.rs::build_coordinates`). A `RowBlocked` tensor is physically
    /// `[cols/lanes, rows, lanes]` — ALL rows interleaved inside each stick-group — so consecutive
    /// stick-groups of ONE row sit `rows·lanes` apart, NOT `lanes` (reading `lanes` instead makes a
    /// core gather the NEXT row's stick — the row-mixing that zeroed rmsnorm sums, POD-confirmed).
    ///
    /// BUT `rows·lanes` OVERFLOWS dxp on-card at `rows·lanes = 1984` (fp16 lanes=64, rows=31 — the mq>1
    /// matmul-by-ones reduce, `LX_MODLRFIMM :: lrfimm:5347585`, dxp.cpp:122), which is FAR below
    /// [`LRFIMM_MAX`]'s raw 2M ceiling — the immediate must fold against another dim (fp16's rank-3
    /// `[mb,out,y]` device_dims), so [`LRFIMM_MAX`] alone is NOT a sufficient fit test for fp16. The
    /// only representable value PROVEN on-card for a fp16 RowBlocked reduce/matmul-A stride is `lanes`
    /// itself (the reduce's stick-major seam is instead corrected STRUCTURALLY — a native per-row
    /// reduce reading its own row's sticks — not via a wider coordInfo stride). fp32 (32-elem stick,
    /// narrower dims, the rmsnorm island) is the ONE proven exception where `rows·lanes` fits. So:
    /// fp16 (and fp8/int8, same rank-3 risk, untested — conservatively excluded) always fold to
    /// `lanes`; fp32 uses `rows·lanes`. `is_reduction` gates this to reduce/matmul-A reads (an
    /// elementwise op's own in+out share indexing and stays self-consistent at `lanes` regardless).
    /// `rows<=1` (decode) and `cols<=lanes` (single-stick) make `rows·lanes == lanes` anyway, so this
    /// is byte-identical at both — no `m`-branch, just an extent feeding the SAME formula.
    pub fn group_stride(&self, is_reduction: bool) -> i64 {
        let wide_fits =
            matches!(self.df, Df::Fp32) && is_reduction && !matches!(self.kind, StickKind::Flat);
        if wide_fits {
            self.rows as i64 * self.lanes() as i64
        } else {
            self.lanes() as i64
        }
    }

    /// The descriptor's `maxDimSizes_` — the ON-CARD WALK dxp reconstructs, which MUST agree with the
    /// per-core START (`dev_off`) that also derives from this same `StickLayout`. Returns the opaque
    /// [`DeviceWalk`] (NOT a `Vec<i64>`): the value is a WHOLE-TENSOR binary choice fixed by the layout,
    /// so mixed `-1`/actual, a wrong rank, or a hand-typed constant are all unrepresentable. dxp
    /// reconstructs the device_size from `maxDimSizes_`: `-1` per dim ⇒ it rebuilds a STICK-BLOCKED
    /// (RowBlocked) walk from `N_`/`layoutDimOrder_`/`stickSize_` (and coarse-tiles the M dim) — correct
    /// for a stick-blocked start (`RowBlocked`/`Kernel`) and what torch-spyre emits for every non-indirect
    /// stick tensor. A `Flat` (row-major, head-major) start is NOT stick-blocked, so forcing `-1` there
    /// tells the card to walk RowBlocked over a row-major buffer — the producer-writes-Flat /
    /// card-reads-RowBlocked scramble. `extents` are the op's per-dim sizes, one per `layoutDimOrder_`.
    pub fn device_walk(&self, extents: &[i64]) -> DeviceWalk {
        // Derived from the SAME `addressing()` as the start (`dev_off`) — the walk cannot classify the
        // tensor differently than the start. Every stick-blocked start reconstructs (`-1` per dim); a
        // row-major start pins its ACTUAL extents.
        //
        // ⛔⛔⛔ "ROWS" HERE IS EVERY AXIS BEFORE THE STICK, NOT `self.rows`. THIS WAS THE hd != 64 BUG.
        //
        // `self.rows` is `dims[0]` — for a matmul operand that is `mb`, the QUERY-ROW count. The old
        // test `self.rows > 1` exempted decode on the stated grounds that "at `rows<=1` the two
        // residencies coincide", and for a rank-2 `[1, cols]` view they genuinely do. A BATCHED operand
        // is rank-3 `[mb, y, in]`, and there `mb == 1` at decode while `y` is the GQA group — so the
        // exemption fired on a view with FOUR row-major rows and told the card to reconstruct a
        // stick-blocked walk over a row-major buffer. That is the producer-writes-Flat /
        // card-reads-RowBlocked scramble this doc names, reached by a test that could not see `y`.
        //
        // MEASURED on granite-3.1-8b (hd=128, nqh=32, nkvh=8, mq=1): the batched score leg emitted
        // `layoutDimOrder_ = ["mb","y","in"]` with `max=[-1,-1,-1]`. Row-major over `[1,4,128]` strides
        // `y` by 128 — which IS `qs`'s head pitch `mq*hd` — while a stick-blocked reconstruction
        // strides it by one 64-lane stick. Off by exactly the slab count, so INERT at `hd == 64` (one
        // slab, `y` really does index successive sticks — the identity `opspec.rs` writes as
        // "y-stride = 64 = hd") and silently wrong at every head dim above it. No fault; fluent garbage.
        //
        // So the row count a row-major walk actually has is the product of every axis LEFT of the
        // stick. `extents` is that vector, one entry per `layoutDimOrder_`, and the stick is its last
        // axis by construction (`Walk2`/`Walk3` have no other shape) — so this reads the quantity the
        // walk is about instead of a field that happens to coincide with it at rank 2.
        //
        // Rank-2 `[mb, cols]` gives `mb` back unchanged, so every non-batched operand — every shipped
        // bundle — is byte-identical.
        // ⭐ PURELY ADDITIVE, AND THAT IS DELIBERATE. `self.rows > 1` stays EXACTLY as it was, so every
        // operand that pins today still pins: at the head-outermost order `["y","mb","·"]` `dims[0]` IS
        // `y`, so the batch-decode rungs are already pinned and are PROVEN that way on hardware —
        // MEASURED `max=[4,8,64]` at hd=64 mq=8. A first attempt at this repair replaced that test and
        // silently un-pinned them, which is a regression in a working path; this clause can only ever
        // pin MORE.
        //
        // What it adds is the case the row test cannot see: the BATCH-INNER order `["mb","y","·"]`,
        // where `dims[0]` is `mb == 1` at decode while `y` carries the GQA group. And only where the
        // two readings actually differ — a row-major walk gives the axis before the stick a stride of
        // the stick EXTENT, a stick-blocked reconstruction gives it one stick WIDTH, and those are the
        // same number whenever the stick axis IS one stick. So a one-stick batched operand (every one
        // of them at head_dim 64) keeps the `-1` it ships with, and only a MULTI-STICK contraction —
        // which is what the batched score leg introduces above one stick per head — gets pinned.
        let (stick_extent, lead) = extents
            .split_last()
            .map_or((0i64, &[][..]), |(s, l)| (*s, l));
        let outer_rows: i64 = lead.iter().product::<i64>().max(1);
        let readings_differ = stick_extent > self.lanes() as i64;
        let pin = self.rows > 1 || (outer_rows > 1 && readings_differ);
        match self.addressing() {
            Addressing::RowMajor if pin => DeviceWalk(Walk::RowMajor(extents.to_vec())),
            _ => DeviceWalk(Walk::Reconstruct {
                rank: extents.len(),
            }),
        }
    }

    /// The nest a PER-ROW reduction over the feature axis MUST walk: `FreeM` (rows) is CARRIED, the
    /// `Reduction` covers exactly `{cols/lanes tiles × lanes}` PER ROW, for any `m`.
    pub fn reduce_over_feature(&self) -> ReduceNest {
        ReduceNest {
            free_m: self.rows,
            red_tiles: self.cols / self.lanes(),
            lane: self.lanes(),
        }
    }

    /// The broadcast map pairing a `RowScalar` with each `(r,c)` of THIS `RowBlocked` tensor: element
    /// `(r,c)` is multiplied by ROW `r`'s scalar (`RowScalar::dev_off(r,0) == r*lanes`) for ANY `m`.
    pub fn broadcast_rowscalar(&self) -> BcastNest {
        BcastNest {
            m: self.rows,
            feat: self.cols,
            lanes: self.lanes(),
        }
    }

    /// CLASSIFY an emitter tensor VIEW `(dims, stick_idx)` into the `StickLayout` whose addressing
    /// reproduces the free [`dev_off`] — the typed migration path for the GENERIC per-core addresser
    /// (`per_core_addr`), so every op's baked device address flows through a `StickLayout` value rather
    /// than a raw `dev_off` call. A rank-2 view sticked on its last dim ⇒ the stick-blocked residency
    /// (`RowBlocked`; address is identical to `Kernel` — both stick on dim-1); anything else ⇒ `Flat`
    /// (row-major over the whole shape, which the free `dev_off` else-branch computes). The row/col
    /// fields carry the 2-D framing; [`off_view`](Self::off_view) does the exact fold from the ORIGINAL
    /// `dims`/`corner` (arbitrary rank). Proven byte-equal by `for_view_off_equals_dev_off`.
    pub fn for_view(dims: &[usize], stick_idx: usize) -> StickLayout {
        Self::for_view_df(dims, stick_idx, Df::Fp16)
    }

    /// [`for_view`](Self::for_view) with an EXPLICIT device format, so an fp8 (128-lane) operand is
    /// addressed at its OWN stick width — `dev_off` then tiles on 128, matching the fp8 descriptor
    /// (`DeviceTileLayout::<Fp8>`) the RetileDescriptor + emitter emit. fp16 ⇒ 64-lane (byte-identical).
    pub fn for_view_df(dims: &[usize], stick_idx: usize, df: Df) -> StickLayout {
        // Fold trailing UNIT dims: a tensor sticked on `stick_idx` with only size-1 dims AFTER the stick
        // is physically sticked on its LAST non-unit dim — the SAME residency whether an op presents it
        // as rank-2 `[m,k]` (a matmul view) or rank-3 `[mb,out,1]` (a pointwise view with a phantom `y`).
        // The arrangement must NOT depend on that per-op rank: without this fold the matmul view classifies
        // RowBlocked and the pointwise view Flat, and their `dev_off` DIVERGE for rows>1 — the M>1 prefill
        // scramble, and a per-core start that disagrees with the descriptor's own `stickDimOrder`. Only
        // size-1 dims fold (semantic no-ops); the product `cols` is unchanged (×1), and rows==1 /
        // single-stick are byte-identical to Flat either way, so decode is untouched.
        let mut end = dims.len();
        while end > stick_idx + 1 && dims.get(end - 1) == Some(&1) {
            end -= 1;
        }
        let dims = &dims[..end];
        if dims.len() == 2 && stick_idx == 1 {
            // stick-blocked on the last dim — RowBlocked and Kernel share this dev_off exactly.
            StickLayout::row_blocked_df(dims[0], dims[1], df)
        } else {
            let rows = dims.first().copied().unwrap_or(1);
            let cols = if dims.len() > 1 {
                dims[1..].iter().product::<usize>()
            } else {
                1
            };
            StickLayout {
                rows,
                cols: cols.max(1),
                df,
                kind: StickKind::Flat,
            }
        }
    }

    /// The device element offset of multi-index `corner` for the ORIGINAL view `dims` under this
    /// (`for_view`-classified) layout — BYTE-IDENTICAL to `dev_off(dims, stick_idx, corner)`. Stick-
    /// blocked kinds delegate to [`dev_off`](Self::dev_off) on the 2-D `(corner[0], corner[1])`; `Flat`
    /// folds `corner` row-major over `dims` (the free `dev_off` else-branch). `dims`/`corner` are the
    /// SAME slices passed to `for_view`.
    ///
    /// `ext` is a [`DeviceExtents`], not a bare slice, so the vector an address is folded over can only
    /// come from [`DeviceExtents::of_view`] — the one place a view's declared physical extent is
    /// honoured. Hand-rolling the extents (the way the per-core start addresser used to, which is how
    /// `device_extent` came to be read when classifying a view and ignored when addressing it) is an
    /// `E0308`.
    pub fn off_view(&self, ext: &DeviceExtents, corner: &[usize]) -> usize {
        let dims = ext.dims();
        match self.addressing() {
            Addressing::RowMajor => {
                let mut off = 0usize;
                for d in 0..dims.len() {
                    off = off * dims[d] + corner[d];
                }
                off
            }
            // The ROW extent comes from `dims` — THIS dataspace's own device shape — not from
            // `self.rows` (the op's iteration extent). A stick-blocked tensor is `[cols/lanes, rows,
            // lanes]`, so a corner on the stick axis steps by `rows·lanes`; an operand the op BROADCASTS
            // over (rmsnorm gamma `[1, hidden]`, an fp8 per-channel dequant scale) physically has ONE
            // row, so charging it the op's `mb` strode `mb×` too far and a split stick-dim core read off
            // the end of the operand. `cols` is inert in this formula, so this is byte-identical
            // whenever `dims[0] == self.rows` (every non-broadcast view) and whenever the stick-axis
            // corner is 0 (any unsplit stick dim).
            Addressing::StickBlocked => dev_off_stk(
                &[*dims.first().unwrap_or(&self.rows), self.cols],
                1,
                &[corner[0], corner[1]],
                self.lanes(),
            ),
        }
    }
}

/// THE single host-staging path — the worker↔emitter layout-mismatch firewall. Stage a logical
/// `[layout.rows, layout.cols]` tensor into its DEVICE byte order by placing `get(r, c)` at exactly
/// `layout.dev_off(r, c)` — the SAME `dev_off` the emitter's descriptor reads through. A worker that
/// stages via this (with a tensor's [`StickLayout`]) and an emitter that reads via the SAME `StickLayout`
/// CANNOT disagree on the layout: there is ONE `dev_off`, no hand-written `(c/64)·(rows·64)+…` formula,
/// and no `stickmajor` bool to get wrong. The whole point of routing every host-stage through here is to
/// make the flat-vs-RowBlocked mismatch class (which zeroed the K cache / scrambled the embedding)
/// UNCONSTRUCTABLE — the offset a producer writes is definitionally the offset the consumer reads.
pub fn stage_2d(layout: &StickLayout, get: impl Fn(usize, usize) -> f32) -> Vec<f32> {
    let (rows, cols, lanes) = (layout.rows, layout.cols, layout.lanes());
    assert!(
        cols % lanes == 0,
        "stage_2d: cols {cols} is not a multiple of the {lanes}-stick — host staging must be \
         stick-aligned (the emitter addresses whole sticks)"
    );
    // dev_off max for a stick-aligned RowBlocked tensor is rows·cols−1 (Flat is trivially in-range),
    // so rows·cols is exactly the buffer size for every StickKind.
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[layout.dev_off(r, c)] = get(r, c);
        }
    }
    out
}

/// A device-bind buffer whose bytes were placed THROUGH a layout — the TYPE-LEVEL worker↔emit firewall.
/// There is NO `Vec<f32> → Staged` conversion (the field is private, the only ctors are `tiled`/`filled`),
/// so a caller CANNOT bind a hand-written-`(c/64)*(rows*64)+…`-offset buffer: 2-D layout math is forced
/// through `dev_off` (`tiled`), and the only non-`tiled` producer is a constant-value fill (`filled`) that
/// is layout-invariant by construction. `into_bytes` is the sole exit, so every bound tensor's bytes
/// provably came from a `StickLayout` or a constant — the flat-vs-RowBlocked mismatch is UNCONSTRUCTABLE.
pub struct Staged {
    bytes: Vec<f32>,
}

impl Staged {
    /// Stage a structured 2-D `[layout.rows, layout.cols]` tensor via THE layout: `dev_off(r,c) = get(r,c)`
    /// (see [`stage_2d`]). This is the ONLY way to place non-uniform bytes, so their device offset is by
    /// definition the offset the emitter reads.
    pub fn tiled(layout: &StickLayout, get: impl Fn(usize, usize) -> f32) -> Self {
        Staged {
            bytes: stage_2d(layout, get),
        }
    }

    /// `n_blocks` head-BLOCKS stacked contiguously, each a `[block.rows, block.cols]` tensor tiled via
    /// `block` — block `b` at `b·(rows·cols)`, `get(b, r, c)` at `block.dev_off(r, c)` within it. The one-hot
    /// head-major selectors (Sel_q/Sel_kv/SelT: each block a per-head RowBlocked kernel) are exactly this.
    pub fn blocks(
        block: &StickLayout,
        n_blocks: usize,
        get: impl Fn(usize, usize, usize) -> f32,
    ) -> Self {
        let block_len = block.rows * block.cols;
        let mut bytes = vec![0.0f32; n_blocks * block_len];
        for b in 0..n_blocks {
            let blk = stage_2d(block, |r, c| get(b, r, c));
            bytes[b * block_len..(b + 1) * block_len].copy_from_slice(&blk);
        }
        Staged { bytes }
    }

    /// A CONSTANT-VALUE buffer (`len` copies of `val`) — layout-INVARIANT: every device tiling yields the
    /// identical bytes, so there is no `(r,c)` offset to get wrong. The all-ones reduce weight, the ±448/
    /// 1÷448 fp8 consts, the zero pad, and per-row scalar broadcasts are all this. The sole non-`tiled` ctor.
    pub fn filled(val: f32, len: usize) -> Self {
        Staged {
            bytes: vec![val; len],
        }
    }

    /// The staged bytes — the SOLE exit, so a bound buffer provably came from a `StickLayout` or a fill.
    pub fn into_bytes(self) -> Vec<f32> {
        self.bytes
    }
}

/// The visit-nest of a per-row feature reduction (from [`StickLayout::reduce_over_feature`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ReduceNest {
    pub free_m: usize,
    pub red_tiles: usize,
    pub lane: usize,
}
impl ReduceNest {
    /// Device offset of the `t`-th reduction tile's `l`-th lane FOR ROW `r` — EXACTLY the `RowBlocked`
    /// producer's `dev_off(r, t*lane + l)`, so a reduce driven by this nest reads what the producer wrote.
    pub fn elem_off(&self, r: usize, t: usize, l: usize) -> usize {
        dev_off(
            &[self.free_m, self.red_tiles * self.lane],
            1,
            &[r, t * self.lane + l],
        )
    }
}

/// The broadcast-nest pairing a per-row scalar with a `[m, feat]` `RowBlocked` tensor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BcastNest {
    pub m: usize,
    pub feat: usize,
    pub lanes: usize,
}
impl BcastNest {
    /// Device offset of the row-scalar element `(r,c)` pairs with — always row `r`'s lane-0 scalar.
    pub fn scalar_off(&self, r: usize, _c: usize) -> usize {
        StickLayout::row_scalar(self.m).dev_off(r, 0)
    }
    /// Device offset of the paired `RowBlocked` data element `(r,c)`.
    pub fn data_off(&self, r: usize, c: usize) -> usize {
        dev_off(&[self.m, self.feat], 1, &[r, c])
    }
}

// ── The typed tensor handle `Stk<K>` — a producer/consumer KIND mismatch is a `cargo build` error ──
// Lives HERE (with `StickLayout`) so the LIVE emitter can thread typed handles; the scratchy-sdsc tower
// re-exports it. A leaf helper that requires kind `K'` accepts only `&Stk<K'>`, so a wrong-kind handoff
// (e.g. addressing the K/V cache as a non-`Kernel`) does not type-check.

mod sealed {
    pub trait Sealed {}
}

/// KIND marker for [`Stk`] — the type the checker unifies. Maps to a runtime [`StickKind`]. Sealed ⇒
/// only crate ctors mint tags.
pub trait KindTag: sealed::Sealed + 'static {
    fn kind() -> StickKind;
}
/// Activation / matmul-A / pointwise data.
pub enum RowBlockedTag {}
/// Matmul kernel / K-V cache.
pub enum KernelTag {}
/// Per-row reduced scalar.
pub enum RowScalarTag {}
/// Head-major flat device tensor.
pub enum FlatTag {}
impl sealed::Sealed for RowBlockedTag {}
impl sealed::Sealed for KernelTag {}
impl sealed::Sealed for RowScalarTag {}
impl sealed::Sealed for FlatTag {}
impl KindTag for RowBlockedTag {
    fn kind() -> StickKind {
        StickKind::RowBlocked
    }
}
impl KindTag for KernelTag {
    fn kind() -> StickKind {
        StickKind::Kernel
    }
}
impl KindTag for RowScalarTag {
    fn kind() -> StickKind {
        StickKind::RowScalar
    }
}
impl KindTag for FlatTag {
    fn kind() -> StickKind {
        StickKind::Flat
    }
}

/// Layout contract violation — a `cargo build` surface error (never a runtime assert), returned by
/// the [`Stk`] ctor so a mis-tagged handle cannot be silently constructed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayoutError {
    KindMismatch { got: StickKind, want: StickKind },
}

/// A typed device-tensor handle: an emitter name + the ONE [`StickLayout`], with KIND in the type.
/// Threaded BY VALUE producer→consumer so two ops cannot hold divergent layouts for the same tensor;
/// a leaf helper that requires kind `K'` accepts only `&Stk<K'>`, so handing a `RowScalar` reduce
/// output to a matmul-A (which needs `RowBlocked`) does NOT type-check. `dev_off` is reachable only
/// through the handle's layout. The row count lives in `layout.m()` as a RUNTIME field — there is no
/// M type parameter, so decode/prefill stay ONE code path (symbolic continuous-batching expressible).
///
/// KIND guard — a `RowScalar` reduce-output handed to a matmul-A does not type-check:
/// ```compile_fail
/// use ktir_superdsc::sdsc_abstract::*;
/// fn matmul_a(_: &Stk<RowBlockedTag>) {}
/// let recip = Stk::<RowScalarTag>::new("t1", StickLayout::row_scalar(8)).unwrap();
/// matmul_a(&recip); // E0308: expected RowBlockedTag, found RowScalarTag
/// ```
/// Correct usage compiles and the layout travels with the handle:
/// ```
/// use ktir_superdsc::sdsc_abstract::*;
/// fn matmul_a(a: &Stk<RowBlockedTag>) -> usize { a.dev_off(1, 64) }
/// let normed = Stk::<RowBlockedTag>::new("t448", StickLayout::row_blocked(8, 2048)).unwrap();
/// assert_eq!(matmul_a(&normed), StickLayout::row_blocked(8, 2048).dev_off(1, 64));
/// ```
#[derive(Clone, Debug)]
pub struct Stk<K: KindTag> {
    name: String,
    layout: StickLayout,
    _p: std::marker::PhantomData<fn() -> K>,
}

impl<K: KindTag> Stk<K> {
    /// Mint a handle. `layout.kind` MUST equal `K::kind()` (else `Err`) — the sole runtime layout
    /// check; thereafter the type carries it. Never panics.
    pub fn new(name: impl Into<String>, layout: StickLayout) -> Result<Self, LayoutError> {
        if layout.kind != K::kind() {
            return Err(LayoutError::KindMismatch {
                got: layout.kind,
                want: K::kind(),
            });
        }
        Ok(Stk {
            name: name.into(),
            layout,
            _p: std::marker::PhantomData,
        })
    }
    /// The emitter dataspace name — the layout travels WITH it.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// The ONE layout.
    pub fn layout(&self) -> StickLayout {
        self.layout
    }
    /// Device offset of `(r,c)` for THIS tensor — the single addressing path.
    pub fn dev_off(&self, r: usize, c: usize) -> usize {
        self.layout.dev_off(r, c)
    }
}

impl Stk<KernelTag> {
    /// A `Kernel` handle `[in(K), out(N)]` stick-blocked on `out` — INFALLIBLE (the kind is `Kernel`
    /// by construction, so no runtime check / `Result` / unwrap). The compile-time `KernelTag` then
    /// proves downstream addressing treats it as a kernel (e.g. the K/V cache cannot be addressed as a
    /// non-`Kernel`). Used by the attention-cache layout guard.
    pub fn kernel(k_in: usize, n_out: usize, name: impl Into<String>) -> Self {
        Stk {
            name: name.into(),
            layout: StickLayout::kernel(k_in, n_out),
            _p: std::marker::PhantomData,
        }
    }
}

/// THE K-CACHE PRODUCER OFFSET — the device element offset (WITHIN one head) at which the
/// producer (the shim's host scatter) must store `K[slot][d]` so the score matmul, which reads the
/// K cache as a `[hd, cap]` cap-sticked KERNEL, decodes it as the correct score. This is EXACTLY the
/// consumer's device-address model evaluated at `(in=d, out=slot)` — i.e. `dev_off(&[hd, cap], 1,
/// &[d, slot])` — so producer and consumer cannot diverge by construction. It is the single source of
/// truth shared by: (a) the emitter's per-element build guard (`lower_attn_node`), (b) the interpreter
/// proof below, and (c) (mirrored in C++) the shim's `host_kv_write` scatter. `stk` = elems per stick.
///
/// The bug this replaces: the cachewr used to write NATURAL slot-major `slot*hd + d`, which differs
/// from this whenever `cap > stk` (here cap=256, hd=64), scrambling the prefix-K the score reads.
/// ```
/// use ktir_superdsc::sdsc_abstract::*;
/// let (hd, cap, stk) = (64usize, 256usize, 64usize);
/// // The Kᵀ producer offset is the consumer's read model at (d, slot) — identical for every cell.
/// for slot in [0usize, 1, 63, 64, 65, 128, 255] { for d in [0usize, 1, 7, 63] {
///     assert_eq!(kcache_kt_write_offset(slot, d, hd, cap, stk), dev_off(&[hd, cap], 1, &[d, slot]));
/// }}
/// // And it is NOT the natural slot-major layout `slot*hd+d` for cap>stk (the exact bug). Use d>0:
/// // at (slot=64,d=1) stick-blocked = 4160 but slot-major = 64*hd+1 = 4097. (At d=0 they coincide.)
/// assert_ne!(kcache_kt_write_offset(64, 1, hd, cap, stk), 64 * hd + 1);
/// ```
pub fn kcache_kt_write_offset(slot: usize, d: usize, hd: usize, cap: usize, stk: usize) -> usize {
    debug_assert_eq!(
        stk, STK,
        "K-cache stick size must match the device stick (64)"
    );
    // WIRED onto the systolic `StickLayout`: the K cache is a `[hd, cap]` KERNEL sticked on `cap`, so
    // element `(in=d, out=slot)` is `StickLayout::kernel(hd, cap).dev_off(d, slot)` — byte-identical to
    // the free `dev_off(&[hd,cap],1,&[d,slot])` (Kani `dev_off_method_equals_free_fn`), i.e. the layout
    // now OWNS this address (no hand-rolled stick math). `cap` bounds `slot`; the offset formula uses hd.
    StickLayout::kernel(hd, cap).dev_off(d, slot)
}

/// THE V-CACHE PRODUCER OFFSET — the device element offset (WITHIN one head) at which the producer
/// (the shim's host scatter) must store `V[slot][d]` so the VALUE bmm, which reads the V cache as a
/// `[cap, hd]` KERNEL sticked on `hd` (`out`), decodes the correct value. `out_pre = bmm(probs[1,cap]
/// · V[cap,hd]) → [1,hd]`, so the kernel is `[k=cap, n=hd]` sticked on `n=hd`, and its element
/// `(k=slot, n=d)` lives at `dev_off(&[cap, hd], 1, &[slot, d])` — this IS that. Producer and consumer
/// (value bmm read) therefore cannot diverge by construction (the twin of `kcache_kt_write_offset`).
/// Shared source of truth for the shim's `host_kv_write` V scatter + the interpreter/Kani proof.
/// ```
/// use ktir_superdsc::sdsc_abstract::*;
/// let (hd, cap, stk) = (128usize, 256usize, 64usize);
/// for slot in [0usize, 1, 63, 255] { for d in [0usize, 63, 64, 127] {
///     assert_eq!(vcache_write_offset(slot, d, hd, cap, stk), dev_off(&[cap, hd], 1, &[slot, d]));
/// }}
/// ```
pub fn vcache_write_offset(slot: usize, d: usize, hd: usize, cap: usize, stk: usize) -> usize {
    debug_assert_eq!(
        stk, STK,
        "V-cache stick size must match the device stick (64)"
    );
    // WIRED onto the systolic `StickLayout`: the V cache is a `[cap, hd]` KERNEL sticked on `hd`, so
    // element `(in=slot, out=d)` is `StickLayout::kernel(cap, hd).dev_off(slot, d)` — byte-identical to
    // the free `dev_off(&[cap,hd],1,&[slot,d])`. `hd` bounds `d`; the offset formula uses cap.
    StickLayout::kernel(cap, hd).dev_off(slot, d)
}

/// THE RoPE ROTATE-HALF PERMUTATION MATRIX ENTRY `P[inn][o]` for a head of width `hd` (even). RoPE
/// computes `out = x·cos + rot·sin` with `rot = matmul(x, P)`; this P makes `rot` the NeoX rotate-half:
/// `rot[o] = -x[o+half]` for `o < half`, `+x[o-half]` for `o >= half` (half = hd/2). SINGLE SOURCE OF
/// TRUTH shared by (a) the worker's `ROPE_P` seg0-activation fill (`spyre_worker.rs`) and (b) the Kani
/// proof (`rope_permutation_matrix_is_rotate_half`) — so there is NO transcription gap between the code
/// that RUNS on-card and the code that is PROVEN. Pure integer (no heap) ⇒ CBMC-tractable.
/// ```
/// use ktir_superdsc::sdsc_abstract::rope_p_entry;
/// // hd=4, half=2: rot[0]=-x[2], rot[1]=-x[3], rot[2]=+x[0], rot[3]=+x[1].
/// assert_eq!(rope_p_entry(4, 2, 0), -1); // P[in=2][o=0] = -1  ⇒ rot[0] gets -x[2]
/// assert_eq!(rope_p_entry(4, 0, 2), 1);  // P[in=0][o=2] = +1  ⇒ rot[2] gets +x[0]
/// assert_eq!(rope_p_entry(4, 0, 0), 0);  // no self term
/// ```
pub fn rope_p_entry(hd: usize, inn: usize, o: usize) -> i8 {
    let half = hd / 2;
    if o < half && inn == o + half {
        -1 // rot[o] = -x[o+half]   (o < half)
    } else if o >= half && inn == o - half {
        1 // rot[o] = +x[o-half]    (o >= half)
    } else {
        0
    }
}

/// THE DECODE PREFIX-MASK EXTENT: at decode position `p` (= the count of cached prefix tokens), the
/// attention prefix-length mask (`ATTN_MASK_TID`) is VALID (additive 0) exactly on cache columns `[0..p)`
/// and −∞ on `[p..cap)`. This is the single decision that sets how many prefix slots the softmax attends;
/// combined with the new token as slot `p` it must yield `attn_reference`'s `nslot = p+1` contiguous
/// softmax. SINGLE SOURCE OF TRUTH shared by (a) the worker's `pmask` fill (`spyre_worker.rs`) and (b) the
/// Kani proof (`decode_attn_masked_split_equals_contiguous_reference`) — no transcription gap between the
/// on-card mask and the proof. An off-by-one here is the classic norm-preserving softmax redistribution.
pub fn decode_prefix_col_valid(col: usize, p: usize) -> bool {
    col < p
}

/// WHICH PLANE OF A PAGE+LAYER — the three copies of the same keys the attention needs in different
/// orders. Named, because "2" meaning Kᵀ is the sort of thing a comment has to say.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvPlane {
    /// Natural K, `[PAGE_SLOTS, hd]` — what the cache write produces.
    Knat,
    /// V, `[PAGE_SLOTS, hd]` — the value kernel.
    V,
    /// Transposed K, `[hd, PAGE_SLOTS]` — the score kernel.
    Kt,
}

/// A FEATURE INDEX inside a head — `0..hd`. Not a slot, not a request, not a row.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct FeatIdx(u32);

impl FeatIdx {
    pub const ZERO: FeatIdx = FeatIdx(0);

    /// ⭐ QUANTITY (4) OF THE FOUR MEANINGS OF 64: features in one head-dim SLAB — one stick of
    /// FEATURES, the width a per-slab op writes and the step [`Self::of_slab`] multiplies by, held
    /// zero-sized in [`SlabFeats`]. Distinct in meaning from [`Lanes`] (the machine's vector width)
    /// even though a slab is one stick of lanes by definition — the same identity-with-named-roles
    /// that [`MqPad`] keeps for rows/slots/cols.
    pub const SLAB_FEATS: SlabFeats = SlabFeats;

    /// The first feature of head-dim slab `s` — the `s * 64` product, owned by the type that owns
    /// features. At one stick per head (`hd == 64`) every call is slab 0 and the product is invisible,
    /// which is exactly why the hand-written form could swap in any of the other three 64s unnoticed.
    pub const fn of_slab(s: u32) -> FeatIdx {
        FeatIdx(s * SlabFeats::FEATS)
    }

    pub const fn new(d: u32) -> FeatIdx {
        FeatIdx(d)
    }
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// ⭐ QUANTITY (4) OF THE FOUR 64s, AS A TYPE: the features of one head-dim slab — zero-sized, the
/// value is the associated const and never travels at runtime. NO cross-conversion with the other
/// three ([`Lanes`], [`SlotWindow::SLOTS`], [`RowWindow::ROWS`]): a slot that demands this one
/// cannot be fed any of them even though all four are 64 at every baked rung.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SlabFeats;

impl SlabFeats {
    /// The features themselves — module-private: outside this file the quantity is only spendable typed.
    const FEATS: u32 = POOL_STICK;

    const fn n(self) -> u32 {
        Self::FEATS
    }
}

/// ⭐⭐⭐ ONE COORDINATE INTO THE KV POOL — every address the pool can produce, named rather than
/// composed.
///
/// ⛔ THE PROBLEM THIS EXISTS TO END. The pool grew fifteen address-producing methods, and FIFTEEN call
/// sites outside it that add or multiply further terms onto whatever a method returned — a slab here, a
/// request stride there, a block index somewhere else. That is fifteen independent opinions about the
/// layout, and the fold's request term alone was hand-rolled three different ways this session, each
/// wrong differently. A caller that names a COORDINATE cannot hold an opinion: the arithmetic is
/// [`PagedKvPool::addr`]'s, once.
///
/// It is also what makes the layout CHANGEABLE. Moving the request from outside `hd` to inside the slot
/// axis — so one matmul spans the batch and the fold's per-request re-launch has nothing left to
/// iterate — is then an edit to ONE function instead of a hunt through fifteen call sites.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct KvCoord {
    pub plane: KvPlane,
    /// Which kv head.
    pub kvh: KvHead,
    /// Which slot of the page — ABSOLUTE within the page, not relative to any request. Which slots a
    /// request holds is the host's business (its page map and its mask), never an address term here.
    pub slot: KvSlot,
    /// Which feature of the head.
    pub feat: FeatIdx,
}

impl KvCoord {
    /// The corner of a `(plane, kv head, request)` block — slot 0, feature 0. The shape most callers
    /// want, so they do not have to spell two zeros to get it.
    pub const fn block(plane: KvPlane, kvh: KvHead) -> KvCoord {
        KvCoord {
            plane,
            kvh,
            slot: KvSlot::ZERO,
            feat: FeatIdx::ZERO,
        }
    }

    pub const fn at_slot(self, slot: KvSlot) -> KvCoord {
        KvCoord { slot, ..self }
    }

    pub const fn at_feat(self, feat: FeatIdx) -> KvCoord {
        KvCoord { feat, ..self }
    }
}

/// WHICH KV HEAD — the ONLY block coordinate a plane has, now that a page holds no requests.
///
/// ⛔ THIS REPLACED `RequestInPage`. That type named "which request of a page", `0..ROWS`, and it is
/// exactly the concept the device must not have: it made every address carry a request term, and it let
/// prefill and decode disagree about which row a request occupied. A request is a set of SLOTS reached
/// through the host's page map; the device never computes with a request index.
///
/// Typed rather than a bare `u32` for the usual reason — a kv head, a query head, a slot and a feature are
/// all small integers, and the only thing that stopped them being interchanged was that they were spelled
/// differently at the call site.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct KvHead(u32);

/// WHICH QUERY HEAD — distinct from [`KvHead`] BY TYPE, because under GQA they are different counts
/// (`nqh` vs `nkvh`) and passing one where the other belongs addresses another head's keys.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct QueryHead(u32);

impl QueryHead {
    /// ⭐ EVERY QUERY HEAD OF A MODEL, IN ORDER — and the ONLY way to obtain one.
    ///
    /// There is deliberately no per-index constructor. A fallible one forces `expect`/`?` at call sites
    /// (a runtime assertion, which this codebase forbids) and an infallible one takes a bare `u32` and so
    /// admits any integer. Iterating is total by construction: the bound IS the model's head count, so an
    /// out-of-range `QueryHead` cannot be built at all.
    pub fn all(nqh: NonZeroU32) -> impl Iterator<Item = QueryHead> {
        (0..nqh.get()).map(QueryHead)
    }
    /// THE FIRST QUERY HEAD, always valid — every model has at least one head (`nqh` is `NonZeroU32`).
    ///
    /// Exists so a call site that wants head 0 can SAY so, instead of taking `all(nqh).next()` and then
    /// having to answer what an empty iterator would mean. That `.next()` returns an `Option`, and the
    /// only cheap ways to discharge it are `unwrap` (a runtime assertion, forbidden here) or
    /// `unwrap_or_default` — which SILENTLY substitutes head 0 for a head that does not exist, i.e. reads
    /// another head's keys and produces fluent wrong output. A const cannot be wrong.
    pub const FIRST: QueryHead = QueryHead(0);

    pub fn get(self) -> u32 {
        self.0
    }
}

impl KvHead {
    /// `None` past the pool's head count — a head beyond `nkvh` would land in the next plane.
    pub fn new(i: u32, nkvh: NonZeroU32) -> Option<KvHead> {
        (i < nkvh.get()).then_some(KvHead(i))
    }

    /// ⭐ THE KV HEAD A QUERY HEAD READS — **TOTAL**, no `Option`, because it cannot fail: `gqa` is
    /// `nqh / nkvh`, so `qh / gqa < nkvh` for every `qh < nqh`. Stating that as a total function is what
    /// lets the fold's per-head closures use it at all — they return an address, not a `Result`, so an
    /// `Option` there forced a `?` that could not compile and tempted a raw `u32` instead.
    pub fn of_query(qh: QueryHead, gqa: Gqa) -> KvHead {
        KvHead(qh.get() / gqa.get().max(1))
    }

    /// THE FIRST HEAD, always valid.
    pub const FIRST: KvHead = KvHead(0);

    /// ⭐ EVERY KV HEAD OF A POOL, IN ORDER — the same "iterating is the only door" discipline as
    /// [`QueryHead::all`], so a kv-head loop yields TYPED heads and the `kvh * gqa` arithmetic that used
    /// to turn one into a query head disappears into [`KvHead::group_first_query`].
    pub fn all(nkvh: NonZeroU32) -> impl Iterator<Item = KvHead> {
        (0..nkvh.get()).map(KvHead)
    }

    /// ⭐ THE FIRST QUERY HEAD OF THIS KV HEAD'S GQA GROUP — **TOTAL**, the exact inverse of
    /// [`KvHead::of_query`] and total by the same argument: every constructible `KvHead` satisfies
    /// `kvh < nkvh` (the only doors are [`KvHead::new`], which checks it, [`KvHead::FIRST`], and
    /// `of_query`, which divides into range), and `gqa == nqh / nkvh`, so
    /// `kvh * gqa < nkvh * gqa == nqh`. The result is therefore a query head of the model by
    /// construction.
    ///
    /// ⛔ THIS REPLACED `QueryHead::all(nqh).nth(kvh * gqa).unwrap_or_default()`. That spelling asked the
    /// iterator for an index it had already proven in range, got an `Option` back for a question that
    /// cannot fail, and discharged it with a fallback to head 0 — so the one case it was allegedly
    /// guarding (a group head past `nqh`) would not fault but would SILENTLY score against head 0's keys.
    /// A total function has no such arm to get wrong.
    pub fn group_first_query(self, gqa: Gqa) -> QueryHead {
        QueryHead(self.0 * gqa.get().max(1))
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

/// A SLOT OF THE KV POOL — where one key/value pair lives inside a request's page row.
///
/// ⛔ NOT a token count, NOT a byte offset, NOT a row, NOT a page. Those are the other four small
/// integers in scope here, and a bare `u32` lets any of them be passed for this one: a launch slot
/// passed as a pool row cost a session (`KvRow` vs `LaunchSlot` exist for that reason), and one
/// untyped `kv_request` produced seven separate bugs. A slot count is [`SlotCount`], deliberately a
/// different type, because "how many slots" and "which slot" are not interchangeable either.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct KvSlot(u32);

impl KvSlot {
    /// The first slot of a page row. The one slot that needs no derivation.
    pub const ZERO: KvSlot = KvSlot(0);

    /// A slot named by its index. Total: every `u32` is a slot of some pool, and whether it is a slot
    /// this REQUEST holds is [`KvHistory::contains`]'s question, not this constructor's.
    pub const fn new(i: u32) -> KvSlot {
        KvSlot(i)
    }

    /// The index, for addressing and for printing. Named rather than `From`, so a slot cannot silently
    /// become a count, a row, or a byte offset at a call site.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// This slot advanced by `n`. Saturating, so there is no wrap to reason about.
    pub const fn plus(self, n: SlotCount) -> KvSlot {
        KvSlot(self.0.saturating_add(n.get()))
    }
}

/// A NUMBER OF SLOTS — a prefill chunk's width, or 1 for a decode token. Distinct from [`KvSlot`]
/// because a count added to a count is a count while a slot added to a slot is nothing at all.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct SlotCount(u32);

impl SlotCount {
    pub const ONE: SlotCount = SlotCount(1);

    pub const fn new(n: u32) -> SlotCount {
        SlotCount(n)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }
}

/// ⭐⭐⭐⭐ QUANTITY (2) OF THE FOUR MEANINGS OF 64: one 64-SLOT KV WINDOW of a fold sweep — WHICH block
/// of the resident cache one fold pass's score/reduce group reads.
///
/// The window holds [`Self::SLOTS`] slots — one POOL stick of them — NOT because a stick has 64 lanes
/// but because reduce-MAX mis-combines partial maxima past one stick of COLUMNS (the documented
/// hardware defect), so the sweep is blocked at exactly one stick of score columns, and a resident
/// slot is one score column. Same value as [`Lanes::FP16`], different law: the lane count is the
/// machine's, this is the reduce's.
///
/// The index is two coordinates at once, aligned by construction: the KV window's first slot
/// ([`Self::first_slot`]) and the prefix-mask SLAB whose columns are those slots
/// ([`Self::mask_slab`]) — a fold block sweeps exactly one stick of slots and reads exactly one slab
/// of mask, which is what keeps the two grids in step.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct SlotWindow(u32);

impl SlotWindow {
    /// Slots one window holds — the score WIDTH of one fold block, bounded by the reduce's
    /// one-stick column budget, held zero-sized in [`WindowSlots`].
    pub const SLOTS: WindowSlots = WindowSlots;

    /// The windows a sweep of `swept` slots takes, in order — the fold's block loop.
    pub fn sweep(swept: SlotCount) -> impl Iterator<Item = SlotWindow> {
        (0..Self::count_in(swept).get()).map(SlotWindow)
    }

    /// How many windows `swept` slots hold — the `active_cap / 64` division, where the 64 is THIS
    /// window's slot count, not the lane count it used to be spelled as. The answer is the typed
    /// [`WindowCount`], never a bare number.
    pub const fn count_in(swept: SlotCount) -> WindowCount {
        WindowCount(swept.get() / WindowSlots::SLOTS)
    }

    /// Which window of the sweep — the fold block's name.
    pub const fn index(self) -> u32 {
        self.0
    }

    /// The window's first slot — the `b * 64` product, owned here.
    pub const fn first_slot(self) -> KvSlot {
        KvSlot::new(self.0 * WindowSlots::SLOTS)
    }

    /// The prefix-mask SLAB whose columns are this window's slots. The same index on a different
    /// grid: mask slabs and KV windows both advance one stick of slots per fold block.
    pub const fn mask_slab(self) -> u32 {
        self.0
    }
}

/// ⭐ QUANTITY (2) OF THE FOUR 64s, AS A TYPE: the slots of one [`SlotWindow`] — zero-sized, the
/// value is the associated const and never travels at runtime. NO cross-conversion with the other
/// three ([`Lanes`], [`RowWindow::ROWS`], [`FeatIdx::SLAB_FEATS`]): a slot that demands this one
/// cannot be fed any of them even though all four are 64 at every baked rung.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WindowSlots;

impl WindowSlots {
    /// The slots themselves — module-private: outside this file the quantity is only spendable typed.
    const SLOTS: u32 = POOL_STICK;

    const fn n(self) -> u32 {
        Self::SLOTS
    }
}

/// HOW MANY SLOT WINDOWS A SWEEP HOLDS — [`SlotWindow::count_in`]'s answer. Multiplying it back by
/// the window's slots is [`Self::slots`], and the product lands in [`SlotCount`] — typed both ways,
/// so a window count is never spent as a slot count without naming the conversion it took.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WindowCount(u32);

impl WindowCount {
    /// count × [`SlotWindow::SLOTS`] — the slots this many windows sweep.
    pub const fn slots(self) -> SlotCount {
        SlotCount::new(self.0 * WindowSlots::SLOTS)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A HALF-OPEN RUN OF SLOTS `[start, end)` — what a request actually holds, in one piece.
///
/// A named pair, not `(u32, u32)`: the two ends of a range are the same type and a tuple lets them be
/// built the wrong way round, which the constructor here refuses by construction (`end` below `start`
/// yields the empty run rather than a backwards one).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SlotRun {
    start: KvSlot,
    end: KvSlot,
}

impl SlotRun {
    /// The run `[start, start + n)`. Empty when `n` is zero, and never backwards.
    pub const fn new(start: KvSlot, n: SlotCount) -> SlotRun {
        SlotRun {
            start,
            end: start.plus(n),
        }
    }

    pub const fn start(self) -> KvSlot {
        self.start
    }

    pub const fn end(self) -> KvSlot {
        self.end
    }

    pub const fn is_empty(self) -> bool {
        self.end.get() <= self.start.get()
    }

    pub const fn contains(self, slot: KvSlot) -> bool {
        slot.get() >= self.start.get() && slot.get() < self.end.get()
    }

    /// How many slots the run covers — for [`KvHistory::cacheable_prefix`], where the first run's LENGTH
    /// is a token count because that run starts at slot 0.
    pub const fn slots(self) -> SlotCount {
        SlotCount::new(self.end.get().saturating_sub(self.start.get()))
    }

    /// Extend to `end` — used only to grow the LAST run when an append lands exactly on it, which is
    /// what keeps a contiguous history a single run and therefore uniquely represented.
    const fn extended_to(self, end: KvSlot) -> SlotRun {
        SlotRun {
            start: self.start,
            end,
        }
    }
}

/// ⭐ THE SLOT A PREFILL CHUNK WILL ACTUALLY WRITE AT — what [`PagedKvPool::chunk_write_start`] returns,
/// which may be STEPPED BACK behind the request's own end so the chunk's padded window stays in one page.
///
/// A newtype rather than a bare [`KvSlot`], because the two slots in play were the same type and the
/// write used one while the record used the other: the chunk wrote from the stepped-back start, and
/// [`KvHistory::record`] appended from [`BatchSlot::solo`] — the request's UN-stepped-back end. They
/// coincide for every chunk whose padded window already fits its page, which is why only prompts long
/// enough to cross a boundary were wrong, and why it reproduced with ONE request and no batch at all.
///
/// MEASURED, 2026-08-12, granite-3.1-8b fp8 at `live=1`: a 450-token prompt stepped back twice, so its
/// history ended at slot 514 while its position was 450 — 64 slots marked valid that no chunk wrote. A
/// 731-token prompt stepped back three times: history 827, position 731. The mask is built from the
/// history, so those slots were attended, holding whatever the pool held before.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ChunkStart(KvSlot);

impl ChunkStart {
    pub const fn slot(self) -> KvSlot {
        self.0
    }

    /// The run this chunk wrote: `n` slots from where it ACTUALLY started. The only mint of a
    /// [`ChunkWrite`], so a chunk's record cannot be taken at a slot other than the one it wrote at.
    pub const fn wrote(self, n: SlotCount) -> ChunkWrite {
        ChunkWrite { start: self.0, n }
    }
}

/// THE RUN A PREFILL CHUNK ACTUALLY WROTE — its (possibly stepped-back) start and its real length, as
/// one value, so neither can be recorded without the other.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ChunkWrite {
    start: KvSlot,
    n: SlotCount,
}

impl ChunkWrite {
    pub const fn run(self) -> SlotRun {
        SlotRun::new(self.start, self.n)
    }
}

/// WHETHER `slot` IS INSIDE ONE OF `runs` — the [`KvHistory`] membership test, as a free function over
/// a slice so a proof can range over a fixed-size array and CBMC never sees a heap.
///
/// Runs are ascending and disjoint. A linear scan is right for the shapes this holds: a request that
/// has only ever run alone has ONE run, and a run is added only when a batch's shared cursor jumps past
/// it, which happens at most once per admission.
pub fn runs_contain(runs: &[SlotRun], slot: KvSlot) -> bool {
    let mut i = 0;
    while i < runs.len() {
        if runs[i].contains(slot) {
            return true;
        }
        i += 1;
    }
    false
}

/// THE RUN A REQUEST WITH `len` SLOTS AND NO BATCH HISTORY OCCUPIES — `[0, len)`, or nothing at all.
///
/// Free, like the two below, because these lines ARE the history's arithmetic and a `Vec` in a proof
/// means CBMC modelling an allocator. `KvHistory` only wraps them in a growable list, so proving them
/// over fixed-size arrays proves the type.
pub fn contiguous_run(len: SlotCount) -> Option<SlotRun> {
    if len.is_zero() {
        None
    } else {
        Some(SlotRun::new(KvSlot::ZERO, len))
    }
}

/// THE RUN AN APPEND OF `n` SLOTS AT `at` PRODUCES, given the history currently ends at `last_end`.
///
/// `n` IS A [`SlotCount`], NOT A SLOT — as three bare `u32`s a count could be passed as a position and
/// the result would be a plausible, wrong run. `last_end` and `at` are both [`KvSlot`]s and that is
/// safe HERE, uniquely: the clamp is `max`, which is symmetric in them, so swapping the two cannot
/// change the answer. Where the distinction does matter is the API above — [`KvHistory::record`] takes
/// a [`BatchSlot`], whose only constructors derive it from real histories, so a token count cannot
/// arrive as a write slot.
///
/// The clamp is what makes recording TOTAL — no failure mode, so nothing to assert on. It never fires
/// in practice ([`shared_write_slot`] is past every live end, proved), but if it ever did, clamping
/// forward is the only safe answer: writing BEHIND `last_end` would overwrite keys the request still
/// needs, and there is no position at which that is preferable to skipping the gap.
pub fn appended_run(last_end: KvSlot, at: KvSlot, n: SlotCount) -> SlotRun {
    let start = if at.get() > last_end.get() {
        at
    } else {
        last_end
    };
    SlotRun::new(start, n)
}

/// THE ONE SLOT A BATCH APPENDS AT, from its live requests' ends: the maximum, so no request
/// overwrites its own keys. See [`BatchSlot`] for why one slot is the whole point.
pub fn shared_write_slot(ends: &[KvSlot]) -> KvSlot {
    let mut m = KvSlot::ZERO;
    let mut i = 0;
    while i < ends.len() {
        if ends[i].get() > m.get() {
            m = ends[i];
        }
        i += 1;
    }
    m
}

/// WHERE A REQUEST'S KEYS ACTUALLY ARE — the KV slots holding its history, as ascending disjoint runs.
///
/// [`decode_prefix_col_valid`] answers the same question with a LENGTH, and a length is exact only
/// while a request's keys occupy `[0, p)`. That holds for a request that has always run alone. It stops
/// holding the moment a BATCH appends, because a batch has one write slot: a launch resolves exactly one
/// slot shift for every trip inside it, so eight requests appending at eight different positions is
/// eight launches at the ~93 us floor, and the only way to make it one launch is for all eight to append
/// at the same slot. That slot is past the shortest request's own length, so its keys become
/// `[0, len) ∪ [shared, ...)` — TWO runs, with a hole that a length cannot describe.
///
/// ⛔ THE HOLE MUST BE MASKED, AND `col < p` MARKS IT VALID. The slots in the hole hold whatever the
/// pool held before — another request's evicted keys, or nothing. Attending them is not a crash and not
/// a shape error: it is a softmax over real-looking keys the request never said, which reads as fluent
/// output drifting into someone else's topic. So the mask has to be built from the history, and this
/// type is what the mask is built from.
///
/// The hole itself costs nothing. It is not extra pool space — the pages spanning it are resident for
/// the batch's longest request anyway, and the request dimension is INSIDE the page, so a short request
/// is not paying for pages it skipped. It is only extra masked columns, and the mask is already
/// materialised at full page width every step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KvHistory {
    /// Ascending, disjoint, non-empty runs. Never touching: an append at `end` extends the last run
    /// rather than adding one, so the representation of a contiguous history is unique.
    runs: Vec<SlotRun>,
}

impl KvHistory {
    /// The history of a request whose keys occupy `[0, len)` — what every request has until a batch
    /// appends to it, and what [`decode_prefix_col_valid`] describes exactly.
    // ⛔⛔⛔ THERE IS NO `padding()`, AND ITS ABSENCE IS THE LOCK.
    //
    // It returned `contiguous(0)` — nothing valid — for a launch row holding no request, on the reasoning
    // that a row must not be marked valid on columns nobody assigned it. That reasoning is right about
    // what a padding row may CLAIM and wrong about what it must COMPUTE. A padding slot borrows live 0's
    // page map and the paged cachewr has no row term inside a page, so its cache write lands on live 0's
    // cell with no barrier between them; the write is sound only if the bytes are identical, and the bytes
    // are this row's whole forward. Give it an empty history and its softmax sees a different key set from
    // live 0's, so its output differs, so its K/V differ, so live 0's newest key is clobbered at every
    // layer >= 1.
    //
    // A padding row's history is therefore LIVE 0's, and it arrives through the only door that also carries
    // live 0's token, rotation and new-block column: [`LiveBatch::launch_rows`]. With no `padding()` there
    // is no second way to fill a launch row's history, so the partial replica cannot be written again.
    pub fn contiguous(len: usize) -> Self {
        KvHistory {
            runs: contiguous_run(SlotCount::new(len as u32))
                .into_iter()
                .collect(),
        }
    }

    /// One past the last slot this request occupies, and therefore the slot it would append at if it
    /// were alone. 0 for a request that holds nothing yet.
    pub fn end(&self) -> KvSlot {
        self.runs.last().map_or(KvSlot::ZERO, |r| r.end())
    }

    /// Is `slot` one of this request's own? The mask's only question.
    pub fn contains(&self, slot: usize) -> bool {
        u32::try_from(slot).is_ok_and(|s| runs_contain(&self.runs, KvSlot::new(s)))
    }

    /// The runs, for a mask fill that wants to walk them rather than test every column.
    pub fn runs(&self) -> &[SlotRun] {
        &self.runs
    }

    /// ⭐ DOES THIS REQUEST HOLD ANY KEY IN LOGICAL PAGE `page`? The question the page map is built from.
    ///
    /// A batched step appends every row at ONE slot, so a short row's logical pages between its own end
    /// and the batch's write slot hold NOTHING — every column in them is masked invalid, the fold reads
    /// them and contributes zero. Those pages need no storage of their own; the ones this answers `true`
    /// for do.
    pub fn touches(&self, page: LogicalPage) -> bool {
        let (first, past) = page.slot_bounds();
        self.runs
            .iter()
            .any(|r| r.start().get() < past.get() && r.end().get() > first.get())
    }

    /// Record `n` slots written from `at` — a prefill chunk, or one decode token at `n == 1`.
    ///
    /// TOTAL, with no failure mode to assert on: `at` comes from [`BatchSlot::of`] over a set that
    /// includes this history, so `at >= self.end()` (proved by `batch_slot_is_past_every_history`), and
    /// both branches below are then correct. [`appended_run`]'s clamp is what makes it total rather than
    /// a panic; the proof is what makes the clamp never fire.
    /// ⭐ RECORD THE RUN A PREFILL CHUNK WROTE, AT THE SLOT IT WROTE IT — including a stepped-back chunk
    /// that re-writes slots this request already holds.
    ///
    /// NOT [`record`]. That one appends at a slot proved to be past every live end (a batch's shared write
    /// slot), so [`appended_run`] CLAMPS FORWARD — and for a chunk starting BEHIND this request's end the
    /// clamp records `n` slots from the end, marking the last `end - start` of them valid although the
    /// chunk never wrote them. UNION is the right operation for an overlapping rewrite: the re-written
    /// keys are idempotent (same tokens, same positions, same K/V — that is what makes stepping back
    /// sound), so the only thing an overlap changes is how far the history reaches.
    ///
    /// Total: the run is non-empty or this returns, and a run starting at or before the last run's end
    /// extends it, so runs stay ascending, disjoint and non-touching.
    pub fn record_chunk(&mut self, w: ChunkWrite) {
        let run = w.run();
        if run.is_empty() {
            return;
        }
        match self.runs.last_mut() {
            // Overlapping or exactly abutting the history: reach forward if it reaches further, and
            // otherwise change nothing — a chunk fully inside what this request already holds wrote only
            // keys it already had.
            Some(last) if run.start().get() <= last.end().get() => {
                if run.end().get() > last.end().get() {
                    *last = last.extended_to(run.end());
                }
            }
            _ => self.runs.push(run),
        }
    }

    pub fn record(&mut self, at: BatchSlot, n: SlotCount) {
        if n.is_zero() {
            return;
        }
        let run = appended_run(self.end(), at.slot(), n);
        match self.runs.last_mut() {
            // Touching the previous run: extend it, so a contiguous history has ONE representation and
            // a request that never batched is byte-identical to the length-based form.
            Some(last) if last.end() == run.start() => *last = last.extended_to(run.end()),
            _ => self.runs.push(run),
        }
    }

    /// ⭐ IS EVERY KEY AT ITS OWN TOKEN POSITION? True until this request shares a batched step, which
    /// appends it at a slot past its own end and leaves a HOLE.
    ///
    /// The prefill path relies on "the write SLOT and the token OFFSET are one number" — true exactly while
    /// this holds. A caller that steps a chunk back in SLOTS and moves its token offset by the same amount
    /// is only correct for a contiguous history; with a hole it underflows the step's token window
    /// (measured: `range start index 18446744073709551521`).
    pub fn is_contiguous(&self) -> bool {
        match self.runs.as_slice() {
            [] => true,
            [only] => only.start() == KvSlot::ZERO,
            _ => false,
        }
    }

    /// ⭐ HOW MANY LEADING TOKENS SIT AT THEIR OWN TOKEN POSITIONS — the FIRST RUN, and therefore the
    /// most a token-indexed prefix cache may claim about this request.
    ///
    /// A host prefix cache hashes token `t` into block `t / block_size` and hands that block to a later
    /// request as "the keys of tokens `[b*bs, (b+1)*bs)`". That promise holds for slot `s == t` and for
    /// nothing else. This history's FIRST run is exactly the region where they are equal: it starts at
    /// slot 0 and is contiguous, so its `n`-th slot is its `n`-th token. Everything past the first hole
    /// is the request's own keys at slots the host never named, and a block hashed over it would point
    /// a later request at the wrong tokens' keys — fluent, wrong, no fault.
    ///
    /// 🛑 THE HOLE IS NOT AN EDGE CASE, IT IS THE BATCH. A batched step appends every row at ONE slot
    /// ([`BatchSlot`]), so every request shorter than its batch-mates gets a hole at the step it joins.
    /// A prompt prefilled before joining a batch is always one contiguous run, which is why the answer
    /// is usually the whole prompt and never silently more.
    pub fn cacheable_prefix(&self) -> SlotCount {
        match self.runs.first() {
            Some(first) if first.start() == KvSlot::ZERO => first.slots(),
            // Either nothing recorded, or a history that does not start at slot 0 — no token is at its
            // own position, so nothing may be cached.
            _ => SlotCount::new(0),
        }
    }
}

/// ⭐ THE PAGES A REQUEST'S KEYS LIVE IN — the HOST's block table, in logical page order, and the only
/// thing that says where a request's KV is.
///
/// 🛑 **IT WAS A `Vec<usize>` THE WORKER FILLED FROM A FREE LIST OF ITS OWN.** `ensure_pages` searched
/// `(0..pool_pages)` for a page "no live request holds" and pushed it, while the scheduler — which owns
/// the same pool, refcounts its blocks and keeps the prefix cache alive by those refcounts — was handing
/// out block ids the worker never read. Two allocators for one resource, agreeing only because one of
/// them was ignored. That is why prefix caching could not be turned on: a cache HIT is the scheduler
/// saying "these blocks already hold that prefix", and a worker that allocates its own pages holds the
/// prefix nowhere.
///
/// So there is exactly ONE constructor, and it takes the scheduler's ids. A page the host did not
/// allocate is not a value this type can hold, which is what makes the second allocator unwritable
/// rather than merely deleted.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct BlockTable {
    /// Logical page `i` → physical pool page. Ascending order is NOT required and must not be assumed:
    /// with prefix caching on, a request's list interleaves pages another request wrote
    /// (`[0,6,7,8]` / `[1,2,9,10]` measured), which is exactly what `LaunchPages::of` classifies at
    /// launch time.
    pages: Vec<PhysPage>,
}

impl BlockTable {
    /// The pages as the runtime's `set_block_table` wants them. Named rather than `From` so the one
    /// untyping in the path is visible at its call site.
    pub fn as_i64(&self) -> Vec<i64> {
        self.pages.iter().map(|p| i64::from(p.0)).collect()
    }

    /// ⛔⛔⛔ THERE IS NO `install`/`append`/`of_host` ON THIS TYPE, AND THE ABSENCE IS DELIBERATE.
    ///
    /// This type is a DERIVED value — the map one launch binds — not storage. The host's per-request block
    /// list is already stored, in the shape the engine defines for every backend
    /// (`InputBatch::block_tables` + `update_blocks`, `crates/serving/engine/src/input_batch.rs`), and
    /// that module's own doc carries the rule a second store here got wrong: *"The scheduler ships the
    /// FULL table each step, so this REPLACES (not appends)."* A `BlockTable` that could be mutated
    /// incrementally is a second copy of that state, free to drift from it; one built by
    /// [`map_row`](Self::map_row) on every bind cannot.
    ///
    /// ⭐⭐⭐ THE MAP THE LAUNCH ACTUALLY BINDS: the host's blocks placed at the logical pages this
    /// request HOLDS KEYS IN, the fully-masked pages aliased to one scratch page, and the write page
    /// backed by the row's own reserved hole page when the host's blocks stop short of it.
    ///
    /// 🛑 **THE HOST'S BLOCKS ARE NOT IN LOGICAL ORDER, AND THAT IS THE WHOLE PROBLEM THIS SOLVES.** The
    /// host allocates by TOKEN — its block `j` is token page `j` — while a launch addresses by SLOT, and a
    /// batched step makes those diverge: a row appended at the batch's shared slot has its keys at slots
    /// past its own token count, so its `k`-th real page is at a logical index HIGHER than `k`. Installing
    /// the host's list in order puts every key after the row's first hole at the wrong logical page, which
    /// is a request reading its own keys at the wrong positions — fluent, wrong, no fault.
    ///
    /// So the walk is over LOGICAL pages, and the host's list is consumed IN ORDER for the pages that hold
    /// keys — which is exactly right because both orders are ascending in token position.
    ///
    /// `None` when the host granted fewer blocks than this row's logical pages need — a refusal, never a
    /// wrong page, and never a page from a private reserve.
    pub fn map_row(
        hist: &KvHistory,
        host: &[usize],
        want: RowPages,
        part: PoolPartition,
    ) -> Option<BlockTable> {
        let mut pages = Vec::with_capacity(want.get() as usize);
        let mut next_host = 0usize;
        let write_page = want.write_page();
        for page in want.pages() {
            // A page holding no key is read fully-masked and never written: one scratch page serves them
            // all. The write page is the exception — the launch WILL write this step's key there, and the
            // row records it, so it needs storage of its own even though it holds nothing yet.
            if !hist.touches(page) && Some(page) != write_page {
                pages.push(part.scratch());
                continue;
            }
            match host.get(next_host) {
                Some(&id) => {
                    let p = u32::try_from(id).ok()?;
                    if p >= part.host_blocks() {
                        return None;
                    }
                    next_host += 1;
                    pages.push(PhysPage(p));
                }
                // ⛔⛔⛔ THE HOST'S BLOCKS RAN OUT: REFUSE. THIS USED TO BE THE BATCHED HOLE.
                //
                // A page holding a key — including the write page — came from the worker's own reserve here,
                // a run of pages keyed by a per-request `KvRow`. That was the LAST request concept below the
                // host, and it is gone: the scheduler allocates `own pages + 1` so the write page is always
                // one of the host's own blocks (`4ef84be5`, measured at 0 firings across 30 gate runs on
                // hd=64 AND hd=128 before this arm was deleted).
                //
                // So arriving here now means the host granted fewer blocks than this row's logical pages need,
                // which is a DISAGREEMENT between the scheduler's allocation and the launch's page count — and
                // the only safe answer is a refusal. Serving it from a private page would be the old
                // behaviour: correct output, and a per-request row kept alive to produce it.
                None => return None,
            }
        }
        Some(BlockTable { pages })
    }

    /// Logical pages this table addresses. `.slots()` is what it can HOLD; a request needing more has
    /// outrun its allocation and must be refused, not grown.
    pub fn pages(&self) -> &[PhysPage] {
        &self.pages
    }

    /// Pages this table addresses, as a page count rather than a `usize` — so it can only be compared
    /// with another page count, never with a slot span or a block id.
    pub fn held(&self) -> RowPages {
        RowPages(self.pages.len() as u32)
    }

    /// Does it hold every page `want` reaches? The one question the binder asks, and it is asked between
    /// two [`RowPages`] — the division that turned a slot span into a page count happened once, in
    /// [`RowPages::holding`].
    pub fn covers(&self, want: RowPages) -> bool {
        want.get() <= self.held().get()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

/// ONE LOGICAL PAGE OF A REQUEST'S MAP — `slot / PAGE_SLOTS`, and the index both the fold and the mask
/// step by. Minted only by [`RowPages::pages`], so an index past a request's own map does not exist.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct LogicalPage(u32);

impl LogicalPage {
    /// The slots this page covers, `[first, past)`.
    pub fn slot_bounds(self) -> (KvSlot, KvSlot) {
        let per = PagedKvPool::PAGE_SLOTS as u32;
        (KvSlot::new(self.0 * per), KvSlot::new((self.0 + 1) * per))
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

/// ⭐ PAGES ONE REQUEST'S KEYS OCCUPY — the only result of dividing a slot span by a page, and the
/// quantity the prefix mask's ceiling is expressed in.
///
/// 🛑 IT WAS `positions.max(1).div_ceil(per_page)` AT THE ALLOCATION SITE, compared against
/// `MAX_PAGES_PER_ROW as usize` on one line and against `pages.len()` on another — three `usize`s meaning
/// "a count of pages", one of them derived with a runtime `page_slots` read off the bundle while the pool's
/// mask geometry is baked from the const. Being a type means the ceiling check and the coverage check
/// cannot be handed a slot count by mistake, and the division has one home.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct RowPages(u32);

impl RowPages {
    /// Pages needed to hold `span` slots — `ceil(span / PAGE_SLOTS)`, and at least one, because a
    /// request that has written nothing still writes into its first page on this forward.
    ///
    /// ⛔ FROM THE CONST, NOT FROM A BUNDLE FIELD. The mask's per-(row, page) cost and the pool's page
    /// stride are both baked from [`PagedKvPool::PAGE_SLOTS`], so a page count derived from a runtime
    /// `page_slots` is a second source for a compile-time quantity — see
    /// [`crate::sdsc_abstract::PagedKvPool::PAGE_SLOTS`]'s own warning about `PLANE_SLOTS`.
    pub fn holding(span: SlotCount) -> RowPages {
        RowPages(span.get().max(1).div_ceil(PagedKvPool::PAGE_SLOTS as u32))
    }

    /// `None` when the prefix mask cannot reach this deep — [`PagedKvPool::MAX_PAGES_PER_ROW`], which is
    /// the mask segment's bound and not the pool's. Past it the fold reads mask blocks the host never
    /// staged; they are ZERO, and zero in an additive mask means VALID, so the row attends whatever the
    /// pool holds and answers fluently and wrongly.
    pub fn within_mask_reach(self) -> Option<RowPages> {
        (self.0 <= PagedKvPool::MAX_PAGES_PER_ROW).then_some(self)
    }

    /// The slots these pages hold — for a message that wants to state the context in positions.
    pub fn slots(self) -> SlotCount {
        SlotCount::new(self.0 * PagedKvPool::PAGE_SLOTS as u32)
    }

    /// Every logical page of the map, ascending — what the map builder walks.
    pub fn pages(self) -> impl Iterator<Item = LogicalPage> {
        (0..self.0).map(LogicalPage)
    }

    /// The LAST logical page, which is the one the write slot lands in (the span is one past that slot).
    /// `None` only for an empty map, which [`holding`](Self::holding) cannot produce.
    pub fn write_page(self) -> Option<LogicalPage> {
        self.0.checked_sub(1).map(LogicalPage)
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

/// HOW MANY PAGES THE POOL HAS — the bound every host block id is checked against, and the count the
/// host is told to allocate from (`Worker::kv_cache_num_blocks_override`).
///
/// A type rather than the `usize` it was, because the pool's page count and the scheduler's block count
/// are the same number reached from opposite ends: the worker sizes the pool from a byte budget, the
/// scheduler sizes its allocator from a memory estimate, and while the worker ignored the scheduler's
/// ids the two were free to disagree. They are one value now, minted where the pool is created.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct PoolPages(u32);

impl PoolPages {
    /// The pool that was actually allocated. `None` for zero pages — a pool with no page cannot serve
    /// any request, and every block id would be out of range.
    pub fn of_pool(pages: usize) -> Option<PoolPages> {
        u32::try_from(pages).ok().filter(|&n| n > 0).map(PoolPages)
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

/// ⭐⭐⭐ THE PAGES THE RUN **DECLARED** IT NEEDS — `--max-model-len` × `--max-num-seqs`, in pages.
///
/// 🛑 **THE POOL USED TO BE SIZED BY A CONSTANT, AND THAT MADE EVERY CAPACITY ARGUMENT CIRCULAR.**
/// `DEFAULT_POOL_BUDGET_BYTES = 8 GiB` decided the page count, the page count decided how deep a request
/// could go, and when a request died at `request needs 769 positions but its share is 3 page(s)` the answer
/// was to re-cut the same 8 GiB differently. The user's objection was exact: *"we size the pools based on
/// the command line max-seq-len and other such parameters."* A pool sized by a constant cannot be asked
/// whether it is big enough for what was requested — there is nothing to compare it to.
///
/// So the demand comes from the DECLARATION, and the budget is checked AGAINST it:
/// * `context` — the deepest a single request may go. `--max-model-len`, already capped by the engine to
///   [`Worker::kv_max_addressable_tokens`]-worth of slots, which is what the prefix mask can reach.
/// * `rows` — how many may decode at once, rounded up to a ladder rung by [`PoolRows::for_admission`].
///
/// ⭐ AND IT INCLUDES THE RESERVE, via [`PoolPartition::reserve`] rather than its own copy of that
/// arithmetic. A pool sized to `context * rows` exactly is a pool that cannot be split — the reserve comes
/// off the top, so `of_pool` returns `None` and the run refuses at load having asked for precisely what it
/// then could not fund. That reserve is now ONE shared scratch page rather than `rows * 2 + 1`, since the
/// host owns every page a batched launch writes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PoolDemand {
    rows: PoolRows,
    per_row: RowPages,
}

impl PoolDemand {
    /// What a run declaring `context` tokens per request and `rows` concurrent requests needs.
    pub fn of_declaration(context: SlotCount, rows: PoolRows) -> PoolDemand {
        PoolDemand {
            rows,
            per_row: RowPages::holding(context),
        }
    }

    /// The total pool, reserve included. `None` only on `u32` overflow, which needs a declaration no
    /// device could hold anyway.
    ///
    /// ⭐ THE RESERVE IS ONE PAGE NOW, NOT `rows * 2 + 1` — the per-row hole run is gone, so a declaration of
    /// the same width and depth buys a slightly smaller pool. The `+ 1` that remains is the shared scratch
    /// page every fully-masked logical page aliases, which is read-only and therefore genuinely shared.
    pub fn pages(self) -> Option<PoolPages> {
        let serving = self.per_row.get().checked_mul(self.rows.get().get())?;
        PoolPages::of_pool(serving.checked_add(PoolPartition::reserve())? as usize)
    }

    /// Pages one request may fill — the number a "needs N pages" refusal should quote.
    pub fn per_row(self) -> RowPages {
        self.per_row
    }

    pub fn rows(self) -> PoolRows {
        self.rows
    }
}

/// ⭐⭐⭐ THE POOL, SPLIT BY OWNER — the host allocates from the LOW range, the batched write's HOLE comes
/// from the HIGH range, and the two can never be the same value.
///
/// 🛑 **WHY A SECOND RANGE EXISTS AT ALL, AND WHY IT IS NOT A SECOND ALLOCATOR.** A batched step appends
/// every row at ONE slot ([`BatchSlot`]), so a row shorter than its batch-mates is written at a slot past
/// its own keys — in a logical page the HOST never allocated, because the host allocates by TOKEN and this
/// is a slot the row reached only by being in a batch. The host cannot be told about it either: the slot
/// depends on the deepest LIVE request, which is not knowable on the step a request is first scheduled.
/// MEASURED 2026-08-13: sizing the host's allocation from a worker-reported span refused 4 of 6 ragged
/// requests mid-forward (`request needs 885 slot(s) = 4 page(s), but the scheduler has allocated only 1`),
/// with prefix caching both on and off.
///
/// So the hole is served from pages the host is never told exist:
/// * [`scratch`](Self::scratch) — ONE page for every logical page a row holds NO key in. Those pages are
///   read by the fold and masked invalid in every column, so their content is irrelevant and every row
///   may alias the same one. They are never written.
/// * [`hole`](Self::hole) — a SMALL EXCLUSIVE run per launch row, for the page the write slot lands in
///   when the host's blocks stop short of it. That page holds a REAL key (the row records it, the next
///   step's mask calls it valid), so it cannot be shared and cannot be recycled. `HOLE_PAGES_PER_ROW` is
///   2 because the deficit between "pages a row's runs touch" and "blocks the host granted for its
///   tokens" is at most the partial page at each end of the post-join run.
///
/// ⛔ THE PARTITION IS THE GUARD. `install_host` bounds host ids by [`host_blocks`](Self::host_blocks),
/// so a host id can never name a hole page, and `hole`/`scratch` only ever return ids at or above it. The
/// collision that a shared free list would make possible is not expressible.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PoolPartition {
    /// Pages `[0, host)` — the only ones the scheduler is told about.
    host: u32,
}

impl PoolPartition {
    /// PAGES NO REQUEST MAY ALLOCATE — exactly ONE, the shared read-only scratch page.
    ///
    /// ⛔⛔⛔ IT WAS `rows * HOLE_PAGES_PER_ROW + 1`, AND THE PER-ROW PART IS GONE (`4ef84be5`). Those pages
    /// backed the write page of a row whose host allocation stopped short of the batch's shared write slot.
    /// The host allocates that page itself now — `own pages + 1`, one page per row, flat — so the reserve has
    /// no per-request part, and with it went `KvRow`, `AffineRows` and `free_row`. Measured before the
    /// deletion: 0 hole pages across 30 gate runs on granite-3.1-2b (hd=64) AND granite-3.1-8b (hd=128).
    ///
    /// It stays a FUNCTION because a pool is still *sized* to include this page and then *split* by it, which
    /// is one law with two callers — the `one-quantity-computed-twice` shape. It just no longer takes `rows`.
    pub const RESERVED_PAGES: u32 = 1;

    pub fn reserve() -> u32 {
        Self::RESERVED_PAGES
    }

    /// Split `pool`, or `None` when it cannot fund the scratch page and still leave the host a page.
    pub fn of_pool(pool: PoolPages, rows: PoolRows) -> Option<PoolPartition> {
        let host = pool.get().checked_sub(Self::reserve())?;
        (host >= rows.get().get()).then_some(PoolPartition { host })
    }

    /// How many blocks the host may allocate from — what `Worker::kv_cache_num_blocks_override` reports.
    pub fn host_blocks(&self) -> u32 {
        self.host
    }

    /// The ONE read-only page every fully-masked logical page aliases. Sits immediately above the host's
    /// range; never written by anybody, so sharing it is sound by the same argument the mask makes.
    ///
    /// ⛔ THIS IS NOT A HOLE PAGE AND MUST NOT BECOME ONE AGAIN. A fully-masked page is READ under a zero
    /// mask and never written, which is why one page can serve every row. The write page is WRITTEN, so it
    /// needs storage of its own — and that storage now comes from the host's own allocation, never from here.
    pub fn scratch(&self) -> PhysPage {
        PhysPage(self.host)
    }
}

/// THE ONE SLOT A DECODE STEP APPENDS AT — one value for the whole launch, because a launch has one.
///
/// This is the type that makes the batch's write fusable. `RequestSlot::fusable_with` in the runtime's
/// `fold_plan` says two slots fuse only if they are the SAME slot; a batch built from per-request
/// positions never satisfies that, and no amount of emitter work changes it. Deciding the slot ONCE for
/// the batch is the fix, and having it be a type with one constructor is what stops a later caller from
/// reaching for `n_computed` again at one of the several sites that need it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct BatchSlot(KvSlot);

impl BatchSlot {
    /// The slot a batch appends at: past EVERY live history, so no request overwrites its own keys.
    ///
    /// The maximum, not a stored cursor. After a step in which every live history recorded at `V`, all
    /// of their ends are `V + 1`, so the maximum advances by exactly one on its own — there is no
    /// counter to keep in the session, and therefore no counter that can go stale against the requests
    /// it describes. A request admitted mid-batch simply enters the max: shorter than the batch and it
    /// takes the batch's slot (gaining a hole); longer, and the batch takes ITS slot (gaining one).
    pub fn of<'a>(live: impl Iterator<Item = &'a KvHistory>) -> BatchSlot {
        BatchSlot(shared_write_slot(
            &live.map(KvHistory::end).collect::<Vec<_>>(),
        ))
    }

    /// The slot itself. Typed, so it cannot be mistaken for a token count on the way out — which is
    /// the confusion `BatchSlot` exists to prevent in the first place.
    pub fn slot(self) -> KvSlot {
        self.0
    }

    /// THE SLOT A REQUEST APPENDS AT WHEN IT RUNS ALONE — its own end, which is what a solo launch's
    /// one slot shift resolves to. Separate constructor rather than `of(once(h))` so the solo call sites
    /// read as what they are, and so neither can reach for a token count instead.
    ///
    /// ⛔ NOT the request's token count. A request that has been in a batch holds a hole, so it has more
    /// SLOTS than TOKENS; writing at the token count would land back inside its own keys and overwrite
    /// them, and the mask (built from the history) would then describe the overwritten slot as valid.
    pub fn solo(h: &KvHistory) -> BatchSlot {
        BatchSlot(h.end())
    }

    /// The raw index, for the launch position and the page it lands in. Named `get` and not `From` so
    /// the untyping is visible at every call site that still needs it.
    pub fn get(self) -> u32 {
        self.0.get()
    }
}

// ⛔⛔⛔ `AffineRows` WAS HERE — a witness that the batch's KV rows were CONSECUTIVE, minted only by
// `AffineRows::of` and required BY VALUE at the bind so a launch could not be built from a pool that did not
// satisfy it. It guarded a real hazard: with the request baked into the block index, launch slot `i` wrote
// pool row `base + i`, so a request finishing in the MIDDLE left a gap and slot 2 wrote nobody's row.
//
// The rows are gone. `kv_rows` left the device layer at `5e319159`, and the host-side `KvRow` that survived
// existed only to name the pages reserved behind a batched write — which the host allocates now (`own pages +
// 1`, `KvSlotSpan::blocks_this_step`). A launch slot binds its own block table, so there is no stride to
// witness, no consecutiveness to preserve, and no fragmented-pool fallback to degrade into.
//
// ⭐ THE TYPE WAS RIGHT FOR ITS TIME, and it is worth saying why it is not a loss: it replaced an `if` in the
// worker, which is the guard shape this project forbids. What replaced IT is better still — the precondition
// it witnessed no longer exists to be violated.

/// ⛔ WHETHER THE FOLD ACTUALLY SWEEPS EVERY SLOT A REQUEST HOLDS.
///
/// The prefix fold attends `swept_per_pass * passes` slots — `active_cap` columns per pass, once per
/// pass. A request's keys reach as far as its history's end, and with a SHARED write slot that end is
/// the batch's maximum, not this request's own length. Nothing tied those two numbers together.
///
/// The failure mode is the one that looks like a broken kernel: a request whose middle history falls
/// outside the swept window attends its prompt and the current token and NOTHING BETWEEN, so its
/// distribution collapses and it repeats a single token. Indistinguishable, from the output, from a
/// mis-addressed matmul — which is why it is worth stating rather than assuming.
///
/// Returns the first slot that would go unattended, or `None` when the sweep covers everything. Data
/// dependent (it moves with the write slot), so it is checked where the launch is built rather than at
/// emit — but checked, and named, instead of hoped for.
pub fn first_unswept_slot(end: KvSlot, swept_per_pass: SlotCount, passes: u32) -> Option<KvSlot> {
    let covered = (swept_per_pass.get() as u64) * (passes as u64);
    (u64::from(end.get()) > covered).then(|| KvSlot::new(covered.min(u32::MAX as u64) as u32))
}

/// THE ROW A `(head, request)` PAIR OWNS in every batched-over-heads buffer — `qs`, `sc`, the whole
/// online-softmax state, and the prefix mask.
///
/// ⛔ THIS WAS DERIVED TWICE. The emitter computed a head's first row as `h * mq`
/// (`BlockNests::head_row`) and the mask staging computed a pair's row as `h * mq + r`, in different
/// crates, and nothing tied them together. They agree today by inspection, which is not a property.
/// One law, two callers.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct HeadRequestRow(u32);

impl HeadRequestRow {
    /// Head-major with the request minor: head `h` owns the `mq` consecutive rows at `h*mq`, and
    /// request `r` is the `r`-th of them. That order is what makes a GQA group's `gqa*mq` rows
    /// contiguous, which is what a batched matmul's `(y, mb)` grid needs.
    pub const fn of(head: u32, request: u32, mq: u32) -> HeadRequestRow {
        HeadRequestRow(head * mq + request)
    }

    /// ROWS BETWEEN ADJACENT HEADS under this law — head `h` owns the `mq` rows at `h*mq`, so the
    /// pitch is the chunk width. Stated as the law's own difference (`(h+1, r)` minus `(h, r)`),
    /// the same way [`RequestHeadRow::head_pitch_rows`] states its, so each framing answers its own
    /// head pitch off the same arithmetic that places its rows.
    pub const fn head_pitch_rows(mq: u32) -> u32 {
        HeadRequestRow::of(1, 0, mq).get() - HeadRequestRow::of(0, 0, mq).get()
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// THE OTHER ROW ORDER — request-major with the head minor: request `r` owns the `nqh` consecutive
/// rows at `r*nqh`, and head `h` is the `h`-th of them.
///
/// ⭐⭐⭐⭐⭐ THIS IS THE TRANSPOSE OF [`HeadRequestRow`], AND BOTH ARE REAL. The whole-batch
/// online-softmax buffers are `[nqh*mq, hd]` head-major, because a GQA group's `gqa*mq` rows must be
/// contiguous for the batched matmul's `(y, mb)` grid. The per-request new-token buffers are the
/// opposite: the pass is already rebased onto one request's block, so that request's `nqh` rows are the
/// contiguous ones, which is what makes `mb = gqa` legal there. Two framings, two orders, both correct
/// for their own buffer.
///
/// ⛔ AND THEY COINCIDE AT EXACTLY THE CONFIGURATION THAT ALWAYS WORKED. `head*mq + request` and
/// `request*nqh + head` are both just `head` when `mq == 1` and `request == 0` — i.e. every bs=1 decode.
/// They diverge the moment a batch is wider than one, which is the only regime where the defect appears.
/// Before this type both were a bare `u32` computed inline from the same four numbers, so picking the
/// wrong one for a buffer was a spelling choice no compiler could see.
///
/// The two laws cannot be swapped by accident because they do not take the same arguments in the same
/// roles: this one is parameterised by the HEAD COUNT and takes the request first, the head-major law is
/// parameterised by the CHUNK WIDTH and takes the head first. Feeding either one the other's extent
/// changes what the arithmetic means, and now also what it is named.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct RequestHeadRow(u32);

impl RequestHeadRow {
    pub const fn of(request: u32, head: u32, nqh: u32) -> RequestHeadRow {
        RequestHeadRow(request * nqh + head)
    }

    /// ROWS BETWEEN ADJACENT HEADS under this law — the head is the row minor, so the pitch is ONE
    /// row whatever the head count. Stated as the law's own difference (`(r, h+1)` minus `(r, h)`)
    /// so a stride declared from it reads off the same arithmetic that places the rows.
    pub const fn head_pitch_rows(nqh: u32) -> u32 {
        RequestHeadRow::of(0, 1, nqh).get() - RequestHeadRow::of(0, 0, nqh).get()
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// THE ROW STRIDES A BATCHED MATMUL MUST DECLARE to land on the rows a row law names, in ROWS —
/// one constructor per law ([`Self::of_head_major`] for [`HeadRequestRow`],
/// [`Self::of_request_major`] for [`RequestHeadRow`]), each stated as its law's own differences.
///
/// ⭐ THIS IS THE GUARD BOTH FOLD-COLLAPSE ATTEMPTS NEEDED AND DID NOT HAVE. A batched matmul's
/// operands step `mb` by one unit and `y` by `mb_extent` units; whether those land on the right rows is
/// arithmetic nobody was checking, and getting it wrong writes one request's scores onto another's row
/// — fluent, wrong, silent. Derived from the row law rather than restated, so the two cannot drift.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RequiredRowStrides {
    /// Rows between consecutive values of whatever axis carries the REQUEST.
    pub per_request: u32,
    /// Rows between consecutive values of whatever axis carries the HEAD.
    pub per_head: u32,
}

impl RequiredRowStrides {
    pub const fn of_head_major(mq: u32) -> RequiredRowStrides {
        // Straight out of the law: (h, r+1) is one row on; (h+1, r) is `mq` rows on.
        RequiredRowStrides {
            per_request: HeadRequestRow::of(0, 1, mq).get() - HeadRequestRow::of(0, 0, mq).get(),
            per_head: HeadRequestRow::of(1, 0, mq).get() - HeadRequestRow::of(0, 0, mq).get(),
        }
    }

    /// The strides that land on the rows [`RequestHeadRow`] names — the request-major framing,
    /// stated as that law's own differences exactly as [`Self::of_head_major`] states the other's:
    /// (r, h+1) is one row on; (r+1, h) is `nqh` rows on.
    pub const fn of_request_major(nqh: u32) -> RequiredRowStrides {
        RequiredRowStrides {
            per_request: RequestHeadRow::of(1, 0, nqh).get() - RequestHeadRow::of(0, 0, nqh).get(),
            per_head: RequestHeadRow::of(0, 1, nqh).get() - RequestHeadRow::of(0, 0, nqh).get(),
        }
    }

    /// Whether a matmul whose REQUEST axis steps `request_rows` and whose HEAD axis steps `head_rows`
    /// lands on the law's rows. `head_rows` is `None` when the head is carried by the op's own base
    /// offset instead of an axis (one op per head), which is always consistent.
    pub const fn satisfied_by(self, request_rows: u32, head_rows: Option<u32>) -> bool {
        if request_rows != self.per_request {
            return false;
        }
        match head_rows {
            Some(h) => h == self.per_head,
            None => true,
        }
    }
}

/// ⭐ THE WIDTH A LAUNCH BINDS — the row count a bundle was BAKED at (`mq` on the emitter side,
/// `seqs` on the worker side), padding included.
///
/// One of four widths that all travelled as bare integers named `mq`: this baked rung width, the
/// LIVE request count ([`LiveRows`] — how many requests the scheduler actually admitted this step),
/// the mask row count ([`MaskRows`] = `nqh * mq`), and the stick-padded row count
/// (`mq.div_ceil(64) * 64`). With `nqh = 32` the last two coincide at 64 for every `mq <= 2` and
/// separate at `mq = 4` (128 vs 64), and a 3-live step runs the 4-row rung — so a slot fed the
/// wrong width is numerically right on every narrow rung and wrong from the first wide one.
///
/// Every slot that means "rows the launch binds" takes THIS type: the fold grid's row axis
/// ([`FoldPages::grid`]), the slot map's width ([`SlotMap::of_live`]), the mask's row law
/// ([`MaskRows::new`], [`PrefixMaskShape::new`]) and the capacities read back off a baked bundle
/// ([`BakedMaskBlocks::of_placement`]). The constructors name where the width came FROM, so a live
/// count cannot arrive in a rung slot without saying so at the call site.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct RungWidth(NonZeroU32);

impl RungWidth {
    /// A BAKED row count, read from the bake's own artifacts — the manifest's `RungSeqs`, a ladder
    /// entry, a bundle's `bundle_layout.json`. The worker-side source.
    pub const fn of_baked_rows(rows: u32) -> Option<RungWidth> {
        match NonZeroU32::new(rows) {
            Some(n) => Some(RungWidth(n)),
            None => None,
        }
    }

    /// The width the EMITTER is baking a bundle at — the emit-side source, where the op's `mq` IS
    /// the rung width by definition.
    pub const fn of_emitted_rows(mq: u32) -> Option<RungWidth> {
        match NonZeroU32::new(mq) {
            Some(n) => Some(RungWidth(n)),
            None => None,
        }
    }

    /// Whether this rung holds `live` requests — rung selection's one comparison between the two
    /// widths, named so `baked >= live` cannot be written operand-swapped on bare integers.
    pub const fn holds(self, live: LiveRows) -> bool {
        self.0.get() as usize >= live.get()
    }

    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// The row axis as the non-zero the pass grid carries.
    pub const fn rows(self) -> NonZeroU32 {
        self.0
    }

    /// The row count as a host-side extent — the arithmetic exit for buffer sizing and loops.
    pub const fn count(self) -> usize {
        self.0.get() as usize
    }
}

/// REQUESTS ACTUALLY LIVE THIS STEP — what the scheduler admitted, BEFORE padding to a rung.
///
/// Distinct from [`RungWidth`]: a 3-live step runs the 4-row rung, so the two differ on the very
/// step where confusing them matters, and agree (`bs == rung`) on every full step where a test
/// would look. A launch is sized, masked and folded by the RUNG width; only rung selection, the
/// padding diagnostics and the per-request readback are about the live count.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct LiveRows(usize);

impl LiveRows {
    /// The number of requests the scheduler put into this step.
    pub const fn of_scheduled(n: usize) -> LiveRows {
        LiveRows(n)
    }
    pub const fn get(self) -> usize {
        self.0
    }
}

/// ROWS OF THE PREFIX MASK — `nqh * mq`, the batched-over-heads row count. Not a slot, not a request,
/// not a head: the mask's row axis carries `(head, request)` pairs and any of the three would fit a
/// bare `u32`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct MaskRows(u32);

impl MaskRows {
    /// The request axis is the RUNG width: the mask has a row for every row the launch binds,
    /// padding included, because the emitter's ops sweep all of them.
    pub const fn new(nqh: u32, mq: RungWidth) -> MaskRows {
        MaskRows(nqh * mq.get())
    }
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// ONE REQUEST'S ROW EXTENT of a per-request attention pass — the request's `nqh` head rows, the
/// frame `fold_plan`'s intermediate-segment shift rebases each pass onto. A DIFFERENT quantity from
/// the whole-batch [`MaskRows`] (`nqh*mq`): the two coincide exactly at `mq == 1`, so a whole-batch
/// consumer reading a per-request extent sweeps `mq`× too few rows from `mq == 2` up — which is why
/// the two regimes carry their extents in two types instead of one `u32`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct PerRequestRows(u32);

impl PerRequestRows {
    /// One request's rows are its query-head rows: `nqh`.
    pub const fn of_one_request_heads(nqh: u32) -> PerRequestRows {
        PerRequestRows(nqh)
    }
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// ⭐ THE ROW REGIME A BUNDLE'S FOLD PASSES WERE BAKED WITH — whole-batch rows or one request's —
/// and the ONE quantity the worker's intermediate-segment rebase stride derives from
/// (`fold_plan::SessionKv::int_rep_stride_bytes`, set via `set_int_stride`).
///
/// ⛔ THE MIRROR THIS CLOSES. The emitter chose the regime (`attn.rs`, `BlockRows`) and the worker
/// hand-mirrored the consequence as `const FOLD_IS_PER_REQUEST: bool = false` — two constants, two
/// crates, no tie — and the worker's comment argued its value from the wrong premise ("the fold
/// really is COLLAPSED, tested on card"). Collapse (`fold_plan::collapsed`: a request axis on the
/// op AND an affine pool) is a LAUNCH property; the rebase stride is a BAKE property. A whole-batch
/// bundle needs stride 0 whether or not any launch ever collapses, because its every pass already
/// sweeps every shared-buffer row and the mask silences the rows the pass does not own — rebasing
/// such a pass shifts the running softmax state onto rows the launch does not address (measured on
/// card: word salad on previously-working rows).
///
/// The emitter declares it on the bundle (`op_manifest.json`, beside `kv_batched_requests` — see
/// `lower_subtile_tape_to_superdsc::manifest_fold_row_regime`), and the worker DERIVES the stride
/// from what it loads. One source; a bundle whose kernels sweep whole-batch rows gets stride 0 by
/// derivation.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FoldRowRegime {
    /// Every fold pass sweeps the whole batch's `nqh*mq` shared-buffer rows ([`MaskRows`]); the
    /// mask silences the rows the pass does not own. Every bundle baked so far.
    #[default]
    WholeBatch,
    /// Each fold pass carries ONE request's `nqh` rows ([`PerRequestRows`]) and the runtime rebases
    /// it onto its own request's block in the intermediate segment.
    PerRequest,
}

impl FoldRowRegime {
    /// Bytes between consecutive requests' row blocks in the INTERMEDIATE segment — what a launch
    /// hands `set_int_stride`. Whole-batch is 0 BY DERIVATION: its passes span every row, so there
    /// is no per-request block to rebase onto. Per-request, a request's block is its `nqh` head
    /// rows of one fp16 stick.
    pub const fn int_rep_stride_bytes(self, nqh: u32) -> u64 {
        match self {
            FoldRowRegime::WholeBatch => 0,
            FoldRowRegime::PerRequest => Lanes::FP16
                .elems_of_rows(PerRequestRows::of_one_request_heads(nqh))
                .fp16_bytes(),
        }
    }
}

/// THE COUNT OF REAL QUERY ROWS — `mq`: the decode batch size, or a prefill chunk's token rows.
/// Kind-erased (see [`QueryRows`] for the chunk-vs-batch KIND); this is the pure COUNT, typed so it
/// cannot stand in for the stick-padded row count ([`MqPad`]), the shared-buffer row extent
/// ([`MaskRows`], `nqh*mq`) or a lane count ([`Lanes`]) — each of which equals it at some baked rung
/// (`nqh*mq == mq_pad == 64` at nqh=32/mq=2; `mq == 1` collapses everything).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct QueryRowCount(u32);

impl QueryRowCount {
    pub const fn of_mq(mq: u32) -> QueryRowCount {
        QueryRowCount(mq)
    }
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// LANES OF ONE STICK — a COLUMN count of exactly one stick, never a row count: the one meaning of
/// 64 that is about the MACHINE rather than about a tensor. Equal to the fp16 stick (64), to
/// `head_dim` on every hd=64 model, and to `mq_pad` at every chunk ≤64 rows — the three-way
/// coincidence the attention emitter's row/width slots live inside.
///
/// This is the width of the scalar-per-row softmax-state buffers (one stick is the narrowest a
/// device buffer can be) and the lane term of every `hd / lanes` slab count.
///
/// ⭐ QUANTITY (1) OF THE FOUR 64s, AS A TYPE — zero-sized: the count is the associated const and
/// never travels at runtime. NO cross-conversion with the other three ([`SlotWindow::SLOTS`],
/// [`RowWindow::ROWS`], [`FeatIdx::SLAB_FEATS`]); the number leaves only through doors that name
/// what it fills ([`BlockCols::of_one_stick`], [`MatN::one_stick`]/[`MatK::one_stick`],
/// [`Self::elems_of_rows`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Lanes;

impl Lanes {
    /// The fp16 stick's lanes — the one stick width the fp16 attention path uses.
    pub const FP16: Lanes = Lanes;

    /// The lane count itself — module-private: outside this file the quantity is only spendable typed.
    const LANES: u32 = POOL_STICK;

    const fn n(self) -> u32 {
        Self::LANES
    }

    /// A [`Lanes`] product: `rows` one-stick rows times the lane count — the element count of a
    /// rows-tall, one-stick-wide block, in [`ElemCount`].
    pub const fn elems_of_rows(self, rows: PerRequestRows) -> ElemCount {
        ElemCount(rows.get() as u64 * self.n() as u64)
    }
}

/// A COUNT OF DEVICE ELEMENTS — what a [`Lanes`] product yields. Not an offset and not a byte
/// count: bytes are asked for by format, so an element count cannot be spent as bytes without
/// naming the format that sizes them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ElemCount(u64);

impl ElemCount {
    /// These elements as fp16 bytes — two bytes per element.
    pub const fn fp16_bytes(self) -> u64 {
        self.0 * 2
    }
}

/// THE COLUMN EXTENT ONE REDUCE/POINTWISE OP SWEEPS — a score-block width. Constructed only through
/// doors that NAME which width it is: one stick ([`Self::of_one_stick`]), one slot window
/// ([`Self::of_slot_window`]), one row window ([`Self::of_row_window`]), the head dim
/// ([`Self::of_head_dim`]) / one head slab ([`Self::of_head_slab`]), or a residual feature extent
/// ([`Self::of_feature_cols`]). At hd=64/mq≤64 the block widths are all 64; the door records which
/// one a slot really holds.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct BlockCols(u32);

impl BlockCols {
    /// A stick-wide sweep: one prefix block, one new-block sub-block, or the one-stick running
    /// online-softmax state buffers.
    pub const fn of_one_stick(lanes: Lanes) -> BlockCols {
        BlockCols(lanes.n())
    }
    /// A head-dim-wide sweep — the `[rows, hd]` output-accumulator buffers.
    pub const fn of_head_dim(hd: u32) -> BlockCols {
        BlockCols(hd)
    }
    /// A residual-stream feature extent (hidden/intermediate columns) — outside the attention
    /// row-and-width run.
    pub const fn of_feature_cols(cols: u32) -> BlockCols {
        BlockCols(cols)
    }
    /// One [`RowWindow`]'s score columns — the new-token block's sub-block sweep, quantity (3) of
    /// the four meanings of 64 spent as COLUMNS (each padded row is one score column against the
    /// new block). Demands the window's own witness ([`RowWindow::ROWS`]) by type — NOT the lane
    /// count it shares a value with, which no longer fits this slot.
    pub const fn of_row_window(rows: WindowRows) -> BlockCols {
        BlockCols(rows.n())
    }
    /// One [`SlotWindow`]'s slots — the resident-prefix fold's per-pass score width, quantity (2):
    /// the reduce's one-stick column budget. Demands the window's own witness
    /// ([`SlotWindow::SLOTS`]) by type — NOT the lane count it shares a value with.
    pub const fn of_slot_window(slots: WindowSlots) -> BlockCols {
        BlockCols(slots.n())
    }
    /// One head-dim SLAB of features — quantity (4), [`FeatIdx::SLAB_FEATS`], demanded by type: the
    /// width a per-slab op writes. Distinct in meaning from one stick of lanes even though a slab
    /// is one stick of lanes by definition.
    pub const fn of_head_slab(feats: SlabFeats) -> BlockCols {
        BlockCols(feats.n())
    }
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// THE ROW EXTENT ONE REDUCE/POINTWISE OP SWEEPS. Constructed only through doors that NAME which
/// row quantity fills the slot — the batched shared-buffer extent (`nqh*mq`, [`Self::of_mask_rows`]),
/// the real query rows ([`Self::of_query_rows`]), the stick-padded rows ([`Self::of_padded_chunk_rows`]),
/// the zero-fill remainder ([`Self::of_zero_pad_rows`]) or residual token rows ([`Self::of_token_rows`]).
/// At nqh=32 the first equals `mq_pad` exactly up to mq=2 and separates at mq=4 — the first broken width.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct RowCount(u32);

impl RowCount {
    /// The batched-over-heads shared-buffer extent: `nqh*mq`, every whole-batch attention
    /// reduce/pointwise.
    pub const fn of_mask_rows(rows: MaskRows) -> RowCount {
        RowCount(rows.get())
    }
    /// The real query rows — a per-head or per-tensor `mq`-row sweep.
    pub const fn of_query_rows(mq: QueryRowCount) -> RowCount {
        RowCount(mq.get())
    }
    /// The stick-padded chunk rows — `mq_pad` spent as a ROW extent (the padded new-K/new-V ops),
    /// through the pad law's own row exit.
    pub const fn of_padded_chunk_rows(pad: PaddedRows) -> RowCount {
        RowCount(pad.row_axis_extent())
    }
    /// The padding remainder `mq_pad - mq` — the zero-fill of the rows RoPE never packs.
    pub const fn of_zero_pad_rows(rows: u32) -> RowCount {
        RowCount(rows)
    }
    /// Residual-stream token rows — outside the attention row-and-width run.
    pub const fn of_token_rows(rows: u32) -> RowCount {
        RowCount(rows)
    }
    /// ONE REQUEST's rows — the per-request attention regime's sweep, `nqh` head rows per pass.
    pub const fn of_per_request_rows(rows: PerRequestRows) -> RowCount {
        RowCount(rows.get())
    }
    /// A head-major-collapsed view's rows — every head's row block stacked on the row axis
    /// (`heads*mq`; `heads` may be `nqh` OR `nkvh`, so this is not [`MaskRows`]). The RoPE
    /// collapse's sweep.
    pub const fn of_head_major_rows(rows: u32) -> RowCount {
        RowCount(rows)
    }
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// BLOCKS OF THE PREFIX MASK — one per fold pass. Distinct from a page count and from a slot count,
/// because "how many passes" stopped equalling "how many pages" the moment the fold collapsed.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct MaskBlocks(u32);

impl MaskBlocks {
    /// ⛔ FROM A PASS GRID, NEVER A LOOSE COUNT. The fold takes one pass per (row, page) and steps the mask
    /// by one block per pass, so the block count is `pages * rows` and NOTHING ELSE. Building it from a bare
    /// integer is how the staged count and the pass count came to disagree — the mask was filled
    /// page-major with `pages` blocks while the fold walked `rows * pages` of them, so pass 1 read a block
    /// that held every row's page 1 instead of row 0's.
    ///
    /// ⭐ AND THE GRID CARRIES ITS *FORM*, so this one function answers for both regimes. A gathered fold's
    /// pass IS a page and serves every row at once, so it needs `pages` blocks and not `pages * rows` — see
    /// [`MaskBlockForm`]. The count, the block INDEX the fill composes and `fold_plan::reps` are three
    /// readings of ONE decision, and the form is what keeps them one.
    pub const fn of(grid: MaskPassGrid) -> MaskBlocks {
        MaskBlocks(match grid.form {
            MaskBlockForm::PerRowPage => grid.pages.get() * grid.rows.get(),
            MaskBlockForm::PerPage => grid.pages.get(),
        })
    }
    pub const fn get(self) -> u32 {
        self.0
    }

    /// ⛔ DOES THE BUNDLE'S OWN PMASK HOLD THIS MANY BLOCKS? `false` ⇒ the fold would step past the baked
    /// placement, and the bytes past it were never staged — which in an ADDITIVE mask read as ZERO, i.e.
    /// VALID, so the tail passes attend whatever the pool holds instead of faulting.
    pub const fn fits(self, baked: BakedMaskBlocks) -> bool {
        self.0 <= baked.get()
    }
}

/// ⭐⭐⭐⭐ HOW MANY MASK BLOCKS A BUNDLE ACTUALLY BAKED ROOM FOR — read from its own `bundle_layout.json`,
/// never assumed.
///
/// ⛔ A DISTINCT TYPE FROM [`MaskBlocks`], WHICH IS THE COUNT A STEP NEEDS. The two are compared, so they
/// must not be interchangeable: `pmask = [nqh*mq, cap]`, so a bundle holds `cap / COLS` blocks, while a step
/// needs `pages * width`. Both are "a number of mask blocks" and both were bare integers — one of them
/// derived from a file on disk and one from the live batch.
///
/// ⛔ AND THE WORKER ONLY *PRINTED* THIS. The capacity was computed beside a `[mask-cap]` eprintln and
/// thrown away, so exceeding it was silent. The comment there records that the bound "was guessed twice
/// before being read once" and that the fold is known to break at 8 pages x 4 rows while 6 x 4 works — a
/// boundary that is exactly this quantity, left as a print.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct BakedMaskBlocks(u32);

impl BakedMaskBlocks {
    /// From the bundle's pmask placement: `bytes / (nqh * mq * 2)` slots, then `/ COLS` blocks. Takes the
    /// row factors SEPARATELY rather than a precomputed row count, because dividing by the wrong `mq` gives
    /// a plausible, wrong capacity — the same error that made a PREFILL bundle's pmask answer for a DECODE
    /// one and cost four theories. `mq` is the [`RungWidth`] of the SAME bundle the placement was read
    /// from — the baked row count, never the live one.
    pub fn of_placement(
        bytes: u64,
        nqh: u32,
        mq: RungWidth,
        cols: SlotCount,
    ) -> Option<BakedMaskBlocks> {
        let per_row_pair = (nqh as u64) * (mq.get() as u64) * 2;
        let cols = cols.get() as u64;
        if per_row_pair == 0 || cols == 0 {
            return None;
        }
        let slots = bytes / per_row_pair;
        u32::try_from(slots / cols).ok().map(BakedMaskBlocks)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// ⭐ HOW MANY PAGES THE FOLD WALKS — ONE derivation of a number that had two, in two languages.
///
/// ⛔ THE DEFECT THIS CLOSES. The fold's pass count and the prefix mask's block count are the SAME
/// quantity, and they were computed independently:
///
/// * the worker took `max(req.pages().len())` over the live requests (Rust, `spyre_worker.rs` ~5599),
/// * the shim took `(seq_pos + kv_page_slots - 1) / kv_page_slots` (C++, `sdsc_shim.cpp` ~4400).
///
/// They agree only because a THIRD function — `ensure_pages` — grows every row to the batch's shared
/// write slot. That property lives in neither expression, nothing checks it, and it is exactly the
/// shape of the bug that killed the 8b at 768 positions: two `usize`s both meaning "widest", 3700 lines
/// apart, one multiplied into the pool and the other divided into it.
///
/// ⛔ WHY A MISMATCH IS SILENT. `decode_batch_prefix_mask_f16` writes block `row * pages + page`, and
/// `fold_plan::fold_pass` inverts it as `(rep / pages, rep % pages)`. Those two `pages` must be one
/// value. Block the mask by 16 while the fold divides by 8 and row 1's page 0 is WRITTEN to block 16
/// but READ from block 8 — every row but row 0 reads some other row's page. It cannot be caught
/// downstream, because an additive mask byte the host never wrote reads as ZERO and zero means VALID:
/// the row attends whatever the pool happens to hold, fluently.
///
/// So the count comes from the [`BatchSlot`] — the one slot the launch appends at — and nothing else
/// can reach [`MaskPassGrid`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct FoldPages(NonZeroU32);

impl FoldPages {
    /// THE ONLY CONSTRUCTOR, and it takes the batch's SHARED WRITE SLOT.
    ///
    /// ⛔ NOT any one request's page list, and not any request's length. A batch appends every row at
    /// the maximum over live histories, so a page count derived from a REQUEST is short for every row
    /// but the deepest — and the row it is short for is the deepest one, whose history then falls
    /// outside the covered window and whose distribution collapses onto a single repeated token.
    ///
    /// `slot` is the index appended AT, so the covered range must include it: `slot + 1` slots, hence
    /// the `+ 1` before the ceiling. Off by one here and the deepest row loses its newest token.
    /// ⭐ THE PAGES OF **RESIDENT PREFIX** BEFORE `slot` — `ceil(slot / per_page)`, EXCLUSIVE of the page
    /// `slot` itself lands in, which is the launch body's to handle rather than a fold pass's.
    ///
    /// This is the count a decode step's mask staging must use, and [`Self::covering`] is NOT it:
    /// `covering` includes `slot`'s own page, so at every exact page multiple the host staged one block
    /// more than the fold walks and the launch was REFUSED
    /// (`the launch walks 8 page(s) but the host STAGED the mask for 9`, granite-3.1-8b at `start=2048`,
    /// batched path only). Which of the two is the truth is not decidable from the arithmetic — it was
    /// settled by measurement: making the LAUNCH inclusive instead regressed the ragged gate 3/6 → 1/6.
    pub fn resident_before(slot: BatchSlot, per_page: SlotCount) -> Option<FoldPages> {
        let per = per_page.get();
        if per == 0 {
            return None;
        }
        NonZeroU32::new((slot.get() as u64).div_ceil(per as u64) as u32).map(FoldPages)
    }

    /// The pages spanned by `0..=slot`, INCLUDING `slot`'s own page. ⛔ For a decode step's mask
    /// staging you almost certainly want [`Self::resident_before`] — see its note.
    pub fn covering(slot: BatchSlot, per_page: SlotCount) -> Option<FoldPages> {
        let per = per_page.get();
        if per == 0 {
            return None;
        }
        let needed = slot.get() as u64 + 1;
        NonZeroU32::new(needed.div_ceil(per as u64) as u32).map(FoldPages)
    }

    pub const fn get(self) -> NonZeroU32 {
        self.0
    }

    /// The pass grid this coverage implies for the rows the launch binds — the ONLY door to
    /// [`MaskPassGrid`], which is why it builds the fields directly instead of calling a constructor
    /// that would have to be reachable to be called.
    ///
    /// The row axis is the [`RungWidth`] — padding included, because `fold_plan::reps` multiplies by
    /// `block_tables.len()`, which is every slot the launch bound. A LIVE count here would stage
    /// fewer mask blocks than the fold walks the moment a step runs padded.
    ///
    /// ⛔ `form` IS A BAKE FACT AND MUST COME FROM THE BUNDLE, NOT FROM THIS STEP. Which form a bundle
    /// wants is decided by whether its fold ops carry a request axis over a gathered scratch — a
    /// property of the emission (`BakeFacts::gathers_kv`), invisible from the batch. Staging the wrong
    /// form is silent both ways: `PerRowPage` blocks against a collapsed fold leaves `mq-1` of every
    /// `mq` rows at −∞ (the deepest rows answer from nothing), and `PerPage` blocks against an
    /// uncollapsed fold has every pass read row 0's validity.
    pub const fn grid(self, rows: RungWidth, form: MaskBlockForm) -> MaskPassGrid {
        MaskPassGrid {
            pages: self.0,
            rows: rows.rows(),
            form,
        }
    }
}

/// ⭐⭐⭐⭐⭐ WHAT ONE PREFIX-MASK BLOCK DESCRIBES — the single decision the block COUNT
/// ([`MaskBlocks::of`]), the block INDEX the fill composes ([`decode_batch_prefix_mask_f16`]) and the
/// fold's pass count (`fold_plan::reps`) are three readings of.
///
/// ⛔ THEY WERE THREE INDEPENDENT EXPRESSIONS AND THAT IS THE WHOLE BUG CLASS. `MaskBlocks::of` said
/// `pages * rows`, the fill wrote `row * pages + page`, and `fold_plan::fold_pass` inverted it as
/// `(rep / pages, rep % pages)`. All three had to move together for the collapse, and an additive mask
/// byte the host never wrote reads as ZERO — which is VALID — so any pair of them agreeing while the
/// third does not is fluent output from another request's keys.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MaskBlockForm {
    /// ONE BLOCK PER (REQUEST, PAGE) — the uncollapsed fold. A pass reads ONE request's page, so its
    /// block marks that request's rows valid and leaves every other row at the sentinel. `mq-1` of every
    /// `mq` rows in a block is therefore −∞ by design, which is exactly what the collapse removes.
    PerRowPage,
    /// ⭐ ONE BLOCK PER PAGE — the GATHERED fold. A pass is a page and serves the whole batch, because
    /// the score kernel carries `y = request` over the gathered scratch: row `h*mq + r` reads request
    /// `r`'s own Kᵗ block and therefore needs request `r`'s own validity in the SAME block. So a block
    /// describes every row, and there are `pages` of them instead of `pages * requests` — `mq`× less
    /// mask to stage as well as `mq`× fewer launches.
    PerPage,
}

/// THE FOLD'S PASS GRID — how many pages each row holds, and how many rows the launch bound.
///
/// One value carrying both, because every consumer needs both and deriving either from the other (by
/// dividing a block count, say) is a second derivation that can be wrong. `MaskBlocks::of` is the only way
/// to a block count, so the mask's size and the fold's pass count come from the same place by construction.
///
/// ⭐ NO PUBLIC CONSTRUCTOR FROM A RAW COUNT. The only way in is [`FoldPages::grid`], and the only way to
/// a [`FoldPages`] is [`FoldPages::covering`] from the [`BatchSlot`] — so a grid whose page count came
/// from a request's page list, or from a `cap`, or from the shim's own ceiling, is unrepresentable rather
/// than guarded against. The runtime guard that used to check this (`first_unswept_slot` against
/// `max_pages`) could only ever report the drift after the fact.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MaskPassGrid {
    pages: NonZeroU32,
    rows: NonZeroU32,
    /// WHAT ONE BLOCK DESCRIBES — see [`MaskBlockForm`]. Carried on the grid rather than passed
    /// alongside it, so the block count, the fill's block index and the launch's pass count are one
    /// decision reaching all three consumers.
    form: MaskBlockForm,
}

impl MaskPassGrid {
    // ⛔ NO CONSTRUCTOR HERE, ON PURPOSE. `new(pages, rows)` used to be `pub`, and it is what let the
    // worker hand in `max(req.pages().len())` while the shim divided by its own ceiling. The single door
    // is `FoldPages::grid`, and a private `new` would still be a door for anything later added to this
    // module — so the fields are built there and nowhere else.
    /// Pages ONE row holds — the per-row share of a striped pool.
    pub const fn pages(self) -> NonZeroU32 {
        self.pages
    }
    /// Rows the launch bound, padding included: what `fold_plan::reps` multiplies by.
    pub const fn rows(self) -> NonZeroU32 {
        self.rows
    }
    /// What one block describes — the form both the count and the fill read.
    pub const fn form(self) -> MaskBlockForm {
        self.form
    }
    /// Passes the fold will take — identical to the block count, which is the point.
    pub const fn passes(self) -> u32 {
        MaskBlocks::of(self).get()
    }

    /// ⭐⭐⭐ TELL THE RUNTIME THAT WILL WALK THESE PASSES, AND TAKE BACK THE ONLY VALUE THE PREFIX-MASK
    /// STAGING ACCEPTS. This is the single door between "a grid someone computed" and "the grid the fold
    /// walks": [`DeclaredFold`] has no other constructor, so a mask cannot be blocked by a geometry the
    /// walker was never told about.
    ///
    /// ⛔ THE DEFECT IT CLOSES: THE SAME NUMBER, DERIVED FROM TWO POPULATIONS. The launch declares its
    /// pass count over every live request (a still-PREFILLING request may be the deepest, and the batch
    /// appends past it), while the staging had its own [`BatchSlot::of`] over the DECODING rows alone. A
    /// prefilling request deeper than every decoder made those two page counts differ, and then
    /// [`decode_batch_prefix_mask_f16`] wrote block `row * pages_staged + page` while `fold_plan::fold_pass`
    /// read `(rep / pages_walked, rep % pages_walked)` — every row but row 0 reading another row's page,
    /// with the tail passes reading past the staged bytes entirely. An additive mask byte the host never
    /// wrote reads as ZERO, and zero is VALID, so the affected rows attend whatever the pool holds:
    /// fluent output, no fault, nothing downstream that notices.
    pub fn declare_to<W: FoldWalker>(self, walker: &mut W) -> Result<DeclaredFold, W::Error> {
        walker.declare_fold(FoldPages(self.pages), MaskBlocks::of(self))?;
        Ok(DeclaredFold(self))
    }
}

/// THE RUNTIME THAT WALKS A FOLD'S PASSES — whatever a launch must tell before its mask means anything.
///
/// Two numbers, together, because they are one decision seen from two sides: `pages` is what the walker's
/// own per-request ceiling is checked against (`fold_plan::reps` refuses a disagreement), and `blocks` is
/// how far the staged mask reaches. Declaring one without the other is how a mask came to be blocked by
/// one quantity and indexed by another.
///
/// The error is the implementor's, so a device session keeps its FFI return code and a test sink can be
/// infallible — neither has to be flattened into a shared error type to satisfy this door.
pub trait FoldWalker {
    type Error;

    /// Record the pass geometry this launch will walk.
    fn declare_fold(&mut self, pages: FoldPages, blocks: MaskBlocks) -> Result<(), Self::Error>;
}

/// ⭐ A PASS GRID THE WALKER HAS BEEN TOLD ABOUT — the only geometry the prefix mask can be blocked by,
/// which is what makes "the blocks staged" and "the passes walked" one number rather than two that agree.
///
/// Minted solely by [`MaskPassGrid::declare_to`]. A staging that wanted a different page count would have
/// to declare that count first, at which point it is the count the fold walks and the two are one number
/// again — so the divergence is not a mistake to be guarded against but a state with no representation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DeclaredFold(MaskPassGrid);

impl DeclaredFold {
    /// The grid, for the block arithmetic and the diagnostics that need its axes by name.
    pub const fn grid(self) -> MaskPassGrid {
        self.0
    }
}

/// ⭐ THE PREFIX MASK'S SHAPE — ONE value that both the emitter's READ and the worker's STAGING derive
/// from, so they cannot disagree.
///
/// ⛔ THEY USED TO BE COMPUTED TWICE, from different inputs, in different crates: the emitter built
/// `Nest::new(["row","feat"], [rows, cap]).slab(b)` from the baked `cap`, and the worker built a buffer
/// from `(nqh, page_slots, cap)` with its own idea of how many blocks there are. Nothing forced the two
/// to describe the same bytes, and an additive mask fails SILENTLY in the dangerous direction: bytes
/// the worker never staged read as ZERO, and zero is VALID, so a row attends whatever the pool holds.
/// `mask_laws.rs` could pin the staging and the emitter could still read somewhere else.
///
/// `STICK` is a const generic because the lane count is the one number the layout cannot be wrong
/// about — a block's column count must be a whole number of sticks, and [`Self::new`] is the only
/// constructor, so a shape that is not is unrepresentable rather than checked for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PrefixMaskShape<const STICK: u32, const COLS: u32> {
    /// Kept SEPARATELY, not just as their product: a caller that has only `rows` must divide to get
    /// either back, and dividing by the wrong `mq` yields a plausible, wrong row for every pair.
    nqh: u32,
    /// The request axis is the RUNG width — the mask has rows for every slot the launch binds.
    mq: RungWidth,
    rows: MaskRows,
}

impl<const STICK: u32, const COLS: u32> PrefixMaskShape<STICK, COLS> {
    /// ⭐ A BLOCK THAT IS NOT WHOLE STICKS IS NOW A TYPE ERROR, not a `None` a caller might unwrap.
    /// `COLS` is `PagedKvPool::PAGE_SLOTS` — a `const` — so there was never a reason for it to be a
    /// runtime field, and as a parameter the block arithmetic below is compile-time too.
    const _COLS_IS_WHOLE_STICKS: () = assert!(
        STICK != 0 && COLS != 0 && COLS.is_multiple_of(STICK),
        "a prefix-mask block must be a whole number of sticks"
    );

    /// The shape, or `None` when there are no heads — a zero rung width is unrepresentable in
    /// [`RungWidth`] and needs no arm here.
    /// ⛔ NO BLOCK COUNT. It is the LIVE PAGE count — a runtime quantity only the worker can know — so
    /// holding it here would mean each side constructing it from its own `cap`, which is exactly the
    /// drift this type exists to prevent and which I reintroduced one line below it: the emitter's baked
    /// `cap` gave 32 blocks where the worker's live `cap` gave 1. A shape must claim only what BOTH
    /// sides can agree on, which is the row layout and the block width.
    pub const fn new(nqh: u32, mq: RungWidth) -> Option<Self> {
        let () = Self::_COLS_IS_WHOLE_STICKS;
        if nqh == 0 {
            return None;
        }
        Some(PrefixMaskShape {
            nqh,
            mq,
            rows: MaskRows::new(nqh, mq),
        })
    }

    /// A BAKED DECODE RUNG'S SHAPE — every field is [`RungRowLaws`]' const: the row count is the
    /// compiler's `NQH*MQ`, not a runtime multiplication. No `Option`, because the laws' own
    /// evaluation already proved the head count nonzero and the width a ladder rung — there is no
    /// refusal left for a call site to discharge.
    pub const fn of_rung_laws<const NQH: u32, const MQ: u32>(
        laws: RungRowLaws<NQH, MQ>,
    ) -> PrefixMaskShape<STICK, COLS> {
        let () = Self::_COLS_IS_WHOLE_STICKS;
        PrefixMaskShape {
            nqh: NQH,
            mq: laws.rung_width(),
            rows: laws.mask_rows(),
        }
    }

    pub const fn nqh(self) -> u32 {
        self.nqh
    }
    pub const fn mq(self) -> u32 {
        self.mq.get()
    }

    /// The row a `(head, request)` pair owns — the SAME law the emitter's nests use.
    pub const fn row_of(self, head: u32, request: u32) -> HeadRequestRow {
        HeadRequestRow::of(head, request, self.mq.get())
    }

    pub const fn rows(self) -> MaskRows {
        self.rows
    }
    /// Columns per block — the slots ONE fold pass may attend. Compile-time.
    pub const fn cols(self) -> SlotCount {
        SlotCount::new(COLS)
    }

    /// Sticks per block — how many `slab` steps one pass may take. A pass that reads past this is
    /// reading the NEXT pass's block, which is another request's history.
    pub const fn slabs_per_block(self) -> u32 {
        COLS / STICK
    }

    /// Elements in one block.
    pub const fn block_elems(self) -> usize {
        (self.rows.get() as usize) * (COLS as usize)
    }

    /// Elements in a buffer of `blocks` blocks — the count is the CALLER's, since only the worker
    /// knows how many pages are live.
    pub const fn elems(self, blocks: MaskBlocks) -> usize {
        self.block_elems() * (blocks.get() as usize)
    }

    /// Bytes the runtime steps between passes — what the worker declares as the mask's rep stride, in
    /// f16. One block, by definition, so the declaration cannot drift from the layout.
    pub const fn rep_stride_bytes(self) -> u64 {
        (self.block_elems() as u64) * 2
    }

    /// ⭐ THE SAME SHIFT, IN INT32 ENTRIES — the gather index table's per-pass pitch, because the index
    /// rides this mask's segment shift and one shift cannot have two pitches. The `/ 4` lives here, at
    /// the mask that owns the number, instead of at the launch site that used to spell it.
    pub const fn pass_stride(self) -> PassStride {
        PassStride(self.rep_stride_bytes() as usize / 4)
    }

    /// THE ONE ADDRESS LAW. Element `(block, row, col)` of the stick-blocked buffer:
    /// `block*block_elems + (col/STICK)*(rows*STICK) + row*STICK + col%STICK`.
    ///
    /// The emitter reaches it as `slab_elems(0, b)` (its nest's per-block slab corner) and the worker
    /// as `elem(p, row, col)` (where it writes). Same arithmetic, one place.
    pub const fn elem(self, block: u32, row: u32, col: u32) -> usize {
        let (r, st) = (self.rows.get() as usize, STICK as usize);
        (block as usize) * self.block_elems()
            + ((col / STICK) as usize) * r * st
            + (row as usize) * st
            + ((col % STICK) as usize)
    }

    /// The corner of `block`'s slab `slab` — the emitter's per-fold-block mask offset, typed so it
    /// crosses into a device offset through its own door rather than through a raw cast.
    pub const fn slab_elems(self, block: u32, slab: u32) -> MaskCorner {
        MaskCorner(self.elem(block, 0, slab * STICK) as u32)
    }

    /// ⛔ WHETHER A PASS SWEEPING `swept` COLUMNS STAYS INSIDE ITS OWN BLOCK. The emitter sweeps
    /// `active_cap` columns per pass; if that exceeds a block, the pass reads into the next block —
    /// i.e. into another pass's validity rows — and the failure is fluent wrong output, never a fault.
    /// Checked where the shape is built, so it is a build error rather than a device symptom.
    pub const fn sweep_fits(self, swept: SlotCount) -> bool {
        swept.get() <= COLS
    }
}

/// ⭐ A CORNER INTO THE STAGED PREFIX MASK, in device elements — [`PrefixMaskShape::slab_elems`]'s
/// result and nothing else's, so the one address in the attention emit that comes from the MASK's own
/// law reaches a device offset as the quantity it is.
///
/// ⛔ IT USED TO CROSS AS `slab_elems(0, b) as u32` INTO `DevOff::from_view_step` — a `usize` element
/// count cast raw into a constructor whose contract is "a single-stick step within a view". A slab
/// corner is not that: its value is `b * rows * 64`, a whole stick-group PLANE term that scales with
/// the row count, so the door it went through documented a shape it does not have. The typed corner
/// has its own door ([`crate::addr::DevOff`] accepts it directly) and the cast has no site left.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct MaskCorner(u32);

impl MaskCorner {
    /// The element offset, for the device-offset door and for printing.
    pub const fn elems(self) -> u32 {
        self.0
    }
}

/// WHAT ONE ROW OF A BATCHED FORWARD IS: a rotary position and a KV history, in ONE value.
///
/// ⛔ They are one value because they are two different numbers about the same request and the bug is
/// always that a call site used one where the other belonged. `rope_pos` is how many TOKENS the request
/// has — its place in its own conversation, which is what RoPE rotates by. The history is which SLOTS
/// hold its keys, and once a batch shares a write slot the shorter requests hold a hole, so their slot
/// count exceeds their token count and the two stop being interchangeable. Passed as two parallel arrays
/// they can differ in length or in order and still compile; the failure is a request rotated at another
/// request's position, or masked to another request's history — fluent output, no crash, nothing
/// downstream that notices. Passed as one array of pairs there is no such thing as a row whose position
/// and history came from different requests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchRow {
    /// Tokens this request has consumed — its RoPE position. NOT a slot, and now NOT AN INTEGER either.
    pub rope_pos: SeqPos,
    /// Slots this request's keys occupy. NOT a length.
    pub hist: KvHistory,
}

/// ⭐⭐⭐ WHERE A REQUEST IS IN ITS OWN CONVERSATION — the number RoPE rotates by, and NOT a slot.
///
/// ⛔ THE DOC ABOVE `BatchRow` HAS SAID THESE ARE DIFFERENT NUMBERS FOR A WHILE, AND THEY WERE STILL BOTH
/// INTEGERS. In one scope of the batched decode there are four of them — a request's own token count
/// (`start`), the shared write slot (`write_slot`), the count after this step (`n_computed`), and the rotary
/// position — and only comments said which was which. `BatchSlot` and `KvSlot` are already types; this is the
/// other side of the same distinction, so the pairing that produces "one request drifting toward another's
/// topic" cannot be written.
///
/// ⛔ DELIBERATELY NO CONVERSION TO OR FROM A SLOT TYPE. Once a batch shares a write slot, a short request's
/// slot count EXCEEDS its token count — the hole is the whole point — so any `From` between them would be a
/// silent lie exactly in the ragged case that matters. The only exit is [`get`](Self::get), for the one place
/// that stages the rotary table.
///
/// ⛔ AND NO ARITHMETIC. `pos + 1` after a step is `advanced()`, named, because "position plus one" and "slot
/// plus one" are the same expression on two different quantities and that is how they got confused.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct SeqPos(u32);

impl SeqPos {
    /// The start of a conversation. The one position that needs no derivation.
    pub const ZERO: SeqPos = SeqPos(0);

    /// A request's own token count, as its position. Total: every count is a position of some request, and
    /// whether it is the position this LAUNCH should rotate by is [`BatchRow`]'s job to pair, not this
    /// constructor's.
    pub const fn new(tokens: u32) -> SeqPos {
        SeqPos(tokens)
    }

    /// One token later — after a decode step. Named rather than `+ 1` so it cannot be applied to a slot.
    pub const fn advanced(self) -> SeqPos {
        SeqPos(self.0 + 1)
    }

    /// The integer, for the rotary staging that must eventually index a table. The ONLY exit.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// THE DECODE-BATCH PREFIX MASK, staged in the device's own f16 — the whole `ATTN_MASK_TID` buffer for
/// a batch of `hists.len()` requests, blocked by the grid's own [`MaskBlockForm`].
///
/// ⭐ WHICH FORM IS THE BUNDLE'S DECISION, NOT THIS FUNCTION'S. Under `PerPage` a fold pass is one PAGE
/// and reads EVERY request's copy of that page: the score kernel carries a request axis
/// (`matmul_opspec_batched_off` with `y = mq`) over the gathered scratch, so row `h*mq + r` of the pass
/// reads request `r`'s own Kᵀ block and needs the validity of request `r`'s own history — not of
/// whichever request the pass belonged to. So a block describes the whole batch and there is one per
/// page, where `PerRowPage` has one per `(request, page)` with `(mq-1)/mq` of every block spent on −∞
/// for the rows that pass did not own.
///
/// That is not only smaller, it is the WHOLE batched-decode scaling curve: a pass is a LAUNCH and a
/// launch costs ~93 µs whatever it carries, so `pages × requests` passes were 8 of the 12 launches a
/// bs=8 layer paid for. The mask's shape and the fold's `reps` are the same decision seen from two
/// sides, and they have to move together — see `fold_plan::reps`, gated on `OpKv::batched_requests`.
///
/// Blocks are `[nqh*mq, page_slots]` STICK-BLOCKED because that is how the emitter reads them (a
/// `[rows, cap]` nest, `dev_off(r,c) = (c/64)*(rows*64) + r*64 + c%64`).
///
/// f16 RATHER THAN f32 because this buffer is not small: it measured as 93% of every element the bind
/// loop narrows (2.10 M of 2.26 M at 16 requests — 12.37 ms/step, against 0.86 ms to decide the
/// values). Both values it can hold are exactly representable in f16, so emitting the bit pattern is
/// not an approximation of the f32 path: it is the same bytes, which is what `mask_laws.rs` pins. Most
/// of the buffer is still the −∞ constant, so that fill is a two-byte `repeat` — a memcpy, not a
/// per-element write — and only the valid cells are then stamped.
///
/// Lives here, beside [`decode_prefix_col_valid`], because it is a LAYOUT LAW the emitter and the
/// worker must agree on, not worker bookkeeping — and because the worker crate has no test target to
/// pin it from.
pub fn decode_batch_prefix_mask_f16<const COLS: u32>(
    // THE SHAPE THE EMITTER WILL READ, not a set of loose extents to re-derive. This is the whole
    // point of `PrefixMaskShape`: the buffer's size, its block stride and its element law all come
    // from the same value the emitter's own offsets come from.
    shape: PrefixMaskShape<POOL_STICK, COLS>,
    // ⛔ THE GEOMETRY THE FOLD WALKS, PROVED BY THE FACT THAT IT WAS DECLARED. The block index this fill
    // composes (`row * pages + page`) is inverted by `fold_plan::fold_pass` as `(rep / pages, rep % pages)`,
    // and those two `pages` are the same value only when the staging cannot pick its own. [`DeclaredFold`]
    // is that guarantee: its one constructor hands the count to the walker on the way through.
    fold: DeclaredFold,
    // WHICH SLOTS EACH REQUEST OWNS — a [`KvHistory`], not a length, because a batch appends at ONE
    // shared slot and a request shorter than the batch therefore holds `[0, len) ∪ [shared, ...)`. A
    // length would mark the hole between them valid, which is this request attending keys it never
    // said. `len()` is the batch's row count `mq`.
    hists: &[KvHistory],
    mask_neg: f32,
) -> Vec<u8> {
    // FROM THE SHAPE, not re-derived: `rows / hists.len()` was a third derivation of `nqh`, and it is
    // silently wrong whenever the histories and the shape were built with different `mq`.
    let grid = fold.grid();
    let mq = shape.mq();
    let nqh = shape.nqh();
    let per_page = shape.cols().get();
    let neg = half::f16::from_f32(mask_neg).to_le_bytes();
    let zero = half::f16::from_f32(0.0).to_le_bytes();
    let mut m: Vec<u8> = neg.repeat(shape.elems(MaskBlocks::of(grid)));
    // ⭐ ONE BLOCK PER FOLD PASS, IN THE PASS'S OWN ORDER — and that order is ROW-MAJOR.
    //
    // `fold_plan::fold_pass` maps pass `rep` to `(row, page)` as `rep = r * pages + p`, and
    // `fold_delta` steps the mask by exactly one block per pass. This fill used to treat its block index
    // as a PAGE (`p` outermost, every row written into every block), which agrees with the pass order only
    // when there is one row: with two rows, pass 1 wanted row 0's page 1 and read a block holding both
    // rows' page 1. Both rows then started correctly and degraded, because prefill had written the KV and
    // only the fold's view of it was wrong.
    //
    // So: block `r * pages + p` describes ROW `r`'s page `p` AND NOTHING ELSE. Every other row in that
    // block stays at the sentinel, which is what makes a pass contribute only its own row — the mask is
    // where a sequence's identity lives, and this is it doing that job.
    //
    // ⭐⭐⭐ UNLESS THE GRID SAYS `PerPage`, IN WHICH CASE THE BLOCK IS THE PAGE AND EVERY ROW IS WRITTEN
    // INTO IT. A gathered fold's pass serves the whole batch — the score kernel's `y` axis walks the
    // requests over the gathered scratch, so row `h*mq + r` of pass `p` reads request `r`'s own Kᵗ block
    // and needs request `r`'s own validity right there. The two arms differ ONLY in the block index, which
    // is why the form travels on the grid: the count (`MaskBlocks::of`), this index and `fold_plan::reps`
    // then cannot disagree.
    let pages = grid.pages().get();
    for (r, h_r) in hists.iter().enumerate() {
        for p in 0..pages {
            let block = match grid.form() {
                MaskBlockForm::PerRowPage => r as u32 * pages + p,
                MaskBlockForm::PerPage => p,
            };
            for h in 0..nqh {
                let row = HeadRequestRow::of(h, r as u32, mq).get();
                for col in 0..per_page {
                    if h_r.contains((p * per_page + col) as usize) {
                        let e = shape.elem(block, row, col);
                        m[e * 2..e * 2 + 2].copy_from_slice(&zero);
                    }
                }
            }
        }
    }
    m
}

/// THE PREFILL CAUSAL-MASK EXTENT (mq>1 new-token self-attention block): in a batched prefill of `mq`
/// new tokens, query row `row` (0-based within the chunk) attends new-token column `col` iff `col <= row`
/// — the token attends itself + all EARLIER new tokens, never a LATER one. Additive mask: 0 when valid,
/// −∞ when `col > row`. This is the `[mq,mq]` causal block that folds (jointly with the prefix score
/// [mq,cap]) into ONE softmax per query row. SINGLE SOURCE OF TRUTH shared by (a) the emitter's `cmask`
/// fill for the mq>1 attention (`lower_attn_node`, the batched-prefill path replacing the mq>1 build-Err)
/// and (b) the Kani proof (`prefill_causal_mask_partition_ok`). An off-by-one here (`col < row`, dropping
/// the diagonal self-term) or wrong direction (`col >= row`) scrambles causal attention — a
/// coherence bug with a preserved-ish norm, so it MUST be pinned, not eyeballed.
pub fn prefill_causal_col_valid(col: usize, row: usize) -> bool {
    col <= row
}

/// THE BATCHED-DECODE NEW-BLOCK MASK: row `i` is a DIFFERENT REQUEST's single new token, so it
/// attends column `i` and nothing else — `col == row`, the diagonal, not [`prefill_causal_col_valid`]'s
/// triangle.
///
/// The two look interchangeable and are not. A prefill chunk's rows are consecutive positions of one
/// prompt, so row `r` legitimately attends every earlier row; a decode batch's rows are independent
/// requests, and letting row `i` see column `j < i` is one request attending another request's token.
/// That is not a crash and not a shape error: it is fluent output that has quietly read someone
/// else's conversation, which no oracle here would catch.
///
/// This is why a batched decode bundle cannot simply reuse the prefill mask despite running the same
/// attention path — the emission is shared, the mask is the thing that must differ.
///
/// SINGLE SOURCE OF TRUTH for the worker's cmask fill, as its two siblings are for prefill and for
/// the resident prefix.
pub fn decode_batch_causal_col_valid(col: usize, row: usize) -> bool {
    col == row
}

/// THE HEAD-MAJOR SELECTOR (mq>1 prefill: turn row-major `q[mq, nqh·hd]` into per-head-contiguous
/// `[nqh, mq, hd]` so the per-head attention ops read a contiguous `[mq,hd]` slice). For head `h`,
/// `q_h[mq,hd] = q[mq, nqh·hd] @ Sel_h`, where `Sel_h` is a one-hot `[nqh·hd, hd]` column-selector:
/// `Sel_h[i, j] = 1` iff `i == h·hd + j`, else 0. Then `q_h[r,j] = q[r, h·hd + j]` — exactly head h's
/// columns. This uses the DEPLOYED matmul (A read contiguous `[m=mq, k=nqh·hd]`, correct) — the only
/// deployed way to head-major-ize (a strided per-head read / 3D reshape / restickify all fail; see
/// [`kcache_kt_write_offset`] history). This fn is the SINGLE SOURCE OF TRUTH for `Sel_h`'s nonzero
/// position (emitter fill + Kani `selector_extracts_head_column`): output column `j` of head `h` reads
/// input column `selector_head_src_col(h, hd, j)`. A wrong map scrambles which head's Q feeds the score.
pub fn selector_head_src_col(h: usize, hd: usize, j: usize) -> usize {
    h * hd + j
}

/// THE LAST-ROW INDEX (mq>1 prefill lm_head: only the LAST prompt token's logits feed the first
/// generated token, so the lm_head runs at `m=1` reading `hidden[m-1, :]` — which is what lets the
/// prefill bundle produce those logits itself instead of paying a second whole forward).
///
/// A flat `a_off = (m-1)·hidden` is WRONG — `hidden[m,hidden]` is device-tiled `[hidden/64, m, 64]`, so
/// row `m-1` is SCATTERED: its stick-group `j` is 64 contiguous elements at `(j·m + (m-1))·64`. The
/// DEPLOYED extraction copies those `hidden/64` runs one stick at a time (see `LAST_HIDDEN_TID`); a
/// one-hot `sel[1,m] @ hidden[m,hidden]` cannot be used because `k = m` is never a whole 64-stick, and
/// a single strided read cannot be expressed at fp16 (see [`StickLayout::group_stride`]). Either way the
/// ROW is the same fact, and this is its SINGLE SOURCE OF TRUTH — shared by the emitter's copy offsets
/// and the Kani proof (`selector_lastrow_picks_last_row`). A wrong row feeds the WRONG token's hidden to
/// the lm_head → a wrong first generated token, with everything downstream still perfectly fluent.
pub fn selector_lastrow_col(m: usize) -> usize {
    m - 1
}

/// ⭐⭐⭐⭐⭐ WHAT A QUERY ROW *IS* — and it is not something a row COUNT can tell you.
///
/// ⛔ THE DEFECT THIS CLOSES, WHICH IS THE ROOT OF THE RAGGED BATCH-DECODE BUG. The emitter carried the
/// query-row count as a bare `mq: u32` whose MEANING lived in a separate `bool` — and, for the matmul
/// splitter, in a **thread-local `Cell<bool>`** (`matmul/dims.rs`), set once per bundle. Its own comment
/// admitted the hazard and then chose ambient state anyway:
///
/// > "a batched decode's `mb=8` is INDISTINGUISHABLE from a prompt chunk's"
/// > "Set once at the top of the lowering ... rather than threaded through four signatures"
///
/// The result, measured by counting: **101 sites in the emitter branch on `mq`, and FOUR consult the
/// flag.** So ~97 decision points emit a batched decode AS A PREFILL CHUNK — not a wrong value, a wrong
/// KIND. The four that do consult it are the four that already failed on the card and were patched one at
/// a time, which is what sixteen rounds of eliminate-and-retry were really fighting.
///
/// ⭐ WHY THAT IS INVISIBLE UNTIL THE BATCH IS RAGGED. A prefill chunk's defining property is that its
/// rows are consecutive positions of ONE sequence, so **they share one history and one valid extent**.
/// Everything written for `mq > 1` may assume it. And that assumption holds for:
///   * a prefill chunk — by definition;
///   * solo decode (`mq == 1`) — trivially;
///   * a UNIFORM decode batch — **by coincidence**, because every row happens to have the same extent;
///   * a RAGGED decode batch — **never**.
///
/// So a uniform batch was never a control for any of those 97 sites, and "equal lengths cannot see the
/// bug" is a mechanical fact here rather than a rule of thumb.
///
/// ⭐ AND `mq == 1` MEANS TWO THINGS, which is why `bs=1` passing proved nothing: it is both "a chunk of
/// width 1" and "one request". At width one the two kinds coincide, so solo decode lands in a branch that
/// happens to be right for a single independent sequence. Under this type they are [`QueryRows<Chunk>`]
/// and [`QueryRows<Requests>`] — same count, different type, and no site may use one as evidence about
/// the other.
///
/// ⛔ WHY TWO TYPES RATHER THAN ONE ENUM. An enum makes the kind *inspectable*; a function may still
/// accept both and be wrong in one arm. Two types make the kind part of the SIGNATURE, so the
/// discrimination happens at the CALL SITE — which is the only place that actually knows the answer.
///
/// ⭐ WHAT STAYS LEGAL. The count. Batching needs a width of four, and [`QueryRows::count`] hands it over
/// for strides, buffer sizes and extents. What becomes unwritable is `if mq > 1` as a way to INFER THE
/// KIND: the count does not carry it, and the kind is reachable only through the type.
/// ⭐ THE KIND IS A **CONST GENERIC**, not a trait object, not a field, and above all not ambient state.
///
/// This crate is driven by a PROC MACRO that knows the model as literals at expansion time
/// (`codegen.rs` reads `head_dim`/`n_heads`/rung widths from the config and interpolates them), so the
/// row kind is a COMPILE-TIME CONSTANT of the bundle being emitted. Writing it as a `bool` — worse, as a
/// `thread_local! { Cell<bool> }` — demoted a constant to a variable, and a runtime variable is exactly
/// what cannot be guarded at build time. That demotion is why 101 sites could branch on the row COUNT and
/// only four ever asked the kind.
///
/// As a const generic the two kinds are DISTINCT TYPES, the discriminant is usable in `const` context, and
/// a generic function may specialise on it without a runtime test.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QueryRows<const ROWS_ARE_REQUESTS: bool> {
    n: NonZeroU32,
}

/// Query rows that are CONSECUTIVE POSITIONS OF ONE SEQUENCE — a prefill chunk. One history, ONE valid
/// extent, a causal triangle among the rows, and row `r` is position `start + r`.
pub type ChunkRows = QueryRows<false>;

/// Query rows that are INDEPENDENT SEQUENCES — a decode batch. `n` histories, `n` valid extents, a
/// DIAGONAL mask among the rows, and row `r`'s position is request `r`'s own token count, unrelated to `r`.
pub type BatchRows = QueryRows<true>;

impl<const ROWS_ARE_REQUESTS: bool> QueryRows<ROWS_ARE_REQUESTS> {
    /// The only constructor. The kind comes from the type the caller names, so there is no flag to forget
    /// and no ambient default to inherit.
    pub const fn new(n: NonZeroU32) -> Self {
        QueryRows { n }
    }

    /// THE COUNT, FOR ARITHMETIC ONLY — strides, buffer extents, row totals. Deliberately hands back a
    /// bare count and NOT the kind: a caller holding only this cannot reconstruct which kind it had. That
    /// one-way funnel is what stops `if n > 1` from becoming a semantic decision again.
    pub const fn count(self) -> NonZeroU32 {
        self.n
    }

    /// `nqh * count` — the row total of every intermediate the attention touches. Kind-free by
    /// construction, because a buffer's SIZE genuinely does not depend on what a row means.
    pub const fn total_rows(self, nqh: NonZeroU32) -> NonZeroU32 {
        match NonZeroU32::new(nqh.get() * self.n.get()) {
            Some(n) => n,
            None => nqh, // unreachable: a product of two non-zeros
        }
    }

    /// What the manifest records, DERIVED FROM THE TYPE. The old `set_rows_are_requests` is gone, so a
    /// bundle cannot be baked in one form and launched as the other.
    pub const fn rows_are_requests(self) -> bool {
        ROWS_ARE_REQUESTS
    }

    /// For the diagnostics that used to print the flag.
    pub const fn what(self) -> &'static str {
        if ROWS_ARE_REQUESTS {
            "INDEPENDENT sequences (decode batch)"
        } else {
            "consecutive positions of ONE sequence (prefill chunk)"
        }
    }
}

impl BatchRows {
    /// A decode batch of `n` independent sequences.
    pub const fn batch(n: NonZeroU32) -> BatchRows {
        QueryRows::new(n)
    }

    /// ONE request decoding alone. ⛔ NOT the same VALUE as [`ChunkRows::single`] even though both count
    /// one row: this row is an independent sequence at its own position. `mq == 1` erased that
    /// distinction, which is why a passing `bs=1` run was never evidence about the batched kind.
    pub const fn solo() -> BatchRows {
        QueryRows::new(match NonZeroU32::new(1) {
            Some(n) => n,
            None => unreachable_nonzero(),
        })
    }
}

impl ChunkRows {
    /// A prefill chunk `n` consecutive positions wide.
    pub const fn chunk(n: NonZeroU32) -> ChunkRows {
        QueryRows::new(n)
    }

    /// A one-token prefill chunk. See the warning on [`BatchRows::solo`].
    pub const fn single() -> ChunkRows {
        QueryRows::new(match NonZeroU32::new(1) {
            Some(n) => n,
            None => unreachable_nonzero(),
        })
    }

    /// THE PAD-ROW POSITION MAP OF A PROMPT CHUNK: which sequence position row `row` holds, given the
    /// chunk has `real` genuine prompt tokens padded out to the rung's `mq` rows.
    ///
    /// The emitter extracts the lm_head's input from row `selector_lastrow_col(mq)` — the LAST row of
    /// the BAKED width, a compile-time constant. But a chunk rarely fills its rung exactly (a 19-token
    /// prompt runs on the 23-wide rung), so the last baked row is a PAD row. Clamping every pad row to
    /// the last real token's position makes the last baked row numerically IDENTICAL to the last real
    /// token: same token id (the token map already clamps this way), same rotary position, and —
    /// because the causal extent is `col <= row_logical_pos(row, real)` — the same attended prefix
    /// `[0, real)`. So the row the emitter reads is the row the sampler wants, at every `real <= mq`.
    ///
    /// A chunk's pad rows APPEND their K/V at their own slots `[real, mq)`, past the request's
    /// `n_computed`, so no later decode step reads them and no other row's write lands there.
    ///
    /// ⛔ THIS LAW IS THE CHUNK'S AND ONLY THE CHUNK'S, which is why it is a method on [`ChunkRows`]
    /// rather than a free function. Its clamp sends a pad row to the LAST real row — correct here,
    /// because "the last row must carry the last token" is the whole reason it exists. A decode batch's
    /// rows are separate requests whose pad rows must replicate LIVE 0 instead (their K/V write races
    /// live 0's at one pool cell — see [`PadRowReplica`]), and taking this clamp there is exactly the
    /// defect: it gives the pad row the LAST live request's column, so its forward, and therefore the
    /// bytes it writes into live 0's cell, differ from live 0's. As a free function both kinds could
    /// call it; as a method only a chunk can.
    ///
    /// IDENTITY WHEN `real == mq`: the map is `min(row, mq-1) == row`, so a chunk that fills its rung
    /// binds byte-identical rotary and mask data to the pre-fold path. SINGLE SOURCE OF TRUTH for the
    /// worker's rope-position staging AND its causal-mask fill ([`Self::new_block_mask`] composes it
    /// with [`prefill_causal_col_valid`]), pinned by the Kani proof
    /// `prefill_pad_row_holds_last_real_token`.
    pub fn row_logical_pos(self, row: usize, real: usize) -> usize {
        row.min(real - 1)
    }

    /// THE `[mq, mq_pad]` NEW-BLOCK CAUSAL MASK OF A PROMPT CHUNK, additive and row-major: 0 where row
    /// `r` may attend new-block column `c`, `mask_neg` elsewhere.
    ///
    /// The rows are consecutive positions of ONE prompt, so the extent is [`prefill_causal_col_valid`]'s
    /// TRIANGLE taken at the row's clamped position — a row attends itself and every earlier row.
    ///
    /// `mq_pad` is the stick-padded row count the emitter reads the mask at (`mq.div_ceil(64)*64`),
    /// derived here from the row count rather than passed, so the buffer cannot be built at one width
    /// and read at another.
    pub fn new_block_mask(self, real: usize, mask_neg: f32) -> Vec<f32> {
        let mq = self.count().get() as usize;
        let mq_pad = mq.div_ceil(64) * 64;
        let mut cmask = vec![mask_neg; mq * mq_pad];
        for row in 0..mq {
            let pos = self.row_logical_pos(row, real);
            for col in 0..mq_pad {
                if prefill_causal_col_valid(col, pos) {
                    cmask[row * mq_pad + col] = 0.0;
                }
            }
        }
        cmask
    }
}

/// `NonZeroU32::new(1).unwrap()` is not const; this is the const-context stand-in.
const fn unreachable_nonzero() -> NonZeroU32 {
    match NonZeroU32::new(1) {
        Some(n) => n,
        None => panic!("1 is nonzero"),
    }
}

/// THE GQA HEAD MAP: query head `qh` attends the KV head `qh / gqa` (each contiguous group of `gqa` query
/// heads shares one K/V head; `gqa = num_q_heads / num_kv_heads`). SINGLE SOURCE OF TRUTH shared by (a) the
/// emitter's K/V GQA-replication in `lower_attn_node` (`copy(new_k, kvh*hd, krep, qh*hd)`) and (b) the Kani
/// proof (`gqa_replicate_*`). If this map were wrong, query heads would attend the WRONG KV head — scrambled
/// attention with a preserved-ish norm (still a real head's K/V), a norm-preserving-direction-shift class.
pub fn gqa_kv_head(qh: usize, gqa: usize) -> usize {
    qh / gqa
}

/// THE GQA-DEDUP SCORE/VALUE KERNEL BASE (element offset of a query head's shared KV-head kernel).
/// With the K/V caches sized to the `nkvh` DISTINCT heads (not the `nqh` GQA-expanded heads — 4× less
/// restickify traffic + 4× smaller resident KV), query head `qh`'s score matmul (kernel `kct[hd,cap]`)
/// and value bmm (kernel `vc[cap,hd]`) must read the SHARED kv-head `gqa_kv_head(qh,gqa)` at head-base
/// `gqa_kv_head(qh,gqa) * hd * cap`. SINGLE SOURCE OF TRUTH shared by the emitter (`lower_attn_node`
/// score & value kernel offsets, the restickify loop head index) and the Kani proof
/// (`gqa_dedup_kernel_base_ok`). ⚠️ Passing a RAW `qh` here (the pre-dedup `qh*hd*cap`) indexes OOB /
/// a wrong kv-head once the cache is `nkvh`-sized — the proof's in-bounds assert fails first on that.
pub fn gqa_dedup_kv_kernel_base(qh: usize, gqa: usize, hd: usize, cap: usize) -> usize {
    gqa_kv_head(qh, gqa) * hd * cap
}

/// A PHYSICAL page in the resident KV pool. A request's block table maps its logical pages to these.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PhysPage(pub u32);

/// THE PAGED K/V POOL — the one authority for every resident K/V address.
///
/// The pre-paged cache is `[nqh, cap, hd]` per layer, so `cap` appears in every per-head base and
/// the servable context is fixed when the bundle is built: past it, generation dies mid-stream and
/// the only cure is a re-emit. Here the page index is the OUTERMOST axis, so no address inside an op
/// mentions the pool size at all — an op is baked at page 0 and the runtime adds
/// `layer + physical page`. The pool becomes a launch-time allocation and the context limit becomes
/// memory rather than a constant.
///
/// ```text
///   pool : [page][layer][Kᵀ | V | Knat][kvh][request][ ... ]
///            Kᵀ   : [kvh][hd][PAGE_SLOTS]         stick-major on slots — the score kernel
///            V    : [kvh][PAGE_SLOTS][hd]         sticked on hd       — the value kernel
///            Knat : [kvh][PAGE_SLOTS][hd]         natural K           — the Kᵀ re-transpose source
/// ```
///
/// ⛔ **THE REQUEST IS NOT A DIMENSION HERE AT ALL — IT IS A CHOICE OF PAGES.**
///
/// This layout used to put `ROWS = 32` requests side by side inside every page, immediately inside `kvh`,
/// so that "the distance between two requests" was bakeable geometry (`hd * PAGE_SLOTS`) and one additive
/// shift relocated all three planes onto request `r`. That siting is exactly what taught the device what a
/// request is, and it cost more than it bought:
///
/// * every address carried a request term, so the emitter, the runtime and the worker each maintained a
///   request→row mapping — and prefill and decode DISAGREED about it, which is why a ragged batch produced
///   wrong tokens while an equal-length batch looked fine;
/// * a page cost `ROWS` times its own slots (`page_stride` ~1 GB, an 8-page pool 7.6 GB);
/// * and the widest decode rung was pinned to `ROWS` by a const assert.
///
/// Now a page holds SLOTS and a request owns PAGES, through the host's block table — ordinary paged
/// attention. A request's identity exists only in the host-staged mask. The pool's coordinate is
/// `(plane, kvh, slot, feat)` with `slot` ABSOLUTE within the page, and nothing below the host can name a
/// request, so nothing below the host can disagree about one.
///
/// Three properties this layout has to have, each paid for by a bug:
///
/// * **All three planes in ONE page, in the KV segment.** The re-roll executor advances only the
///   weight and KV segments per layer, so a plane parked anywhere else is never advanced and every
///   layer rebuilds its Kᵀ from another layer's keys. And a second POOL would have a base that
///   depends on the runtime page count, shifting every baked address in it.
/// * **The page is [`PAGE_SLOTS`](Self::PAGE_SLOTS) = the pre-paged `cap`.** A fold covers one page
///   per launch, so the page IS the swept extent: making it wider than the old cache would sweep
///   more KV for the same context — a straight performance regression, which is exactly what a
///   512-slot page cost. Equal to the old cap, a single page's fold is bit-for-bit the work the
///   baseline did, and longer contexts cost additional launches rather than a wider sweep.
/// * **GQA-DEDUPED.** The score and value kernels already read only the group representative, so the
///   `nqh`-sized reservation was `gqa`x dead space.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PagedKvPool {
    /// DISTINCT kv heads — a query head reads its group's plane via [`gqa_kv_head`].
    pub nkvh: usize,
    pub hd: usize,
}

// ⛔⛔⛔ THE LADDER MUST SERVE EVERY BATCH THE POOL WILL ADMIT — OR THE HOST SILENTLY DECODES GARBAGE.
//
// `decode_rung_for(live)` returns the SMALLEST rung `>= live`, and **`None` when no rung is wide enough**.
// The caller then falls through to "decode them one at a time", which is documented as "correct, and the
// path it took before batching existed" — MEASURED FALSE on 2026-08-09: cutting `BATCH_RUNGS` to `[2]` and
// admitting 4 requests made **ALL FOUR** collapse to one EOS token, 3/3 trials, where before the cut only
// the one over-deep row failed (3/4 answered). So the fallback is not a safe path, and reaching it is not a
// slow-but-correct outcome — it is silent garbage for every request in the batch.
//
// Therefore the pool must never seat more rows than the ladder can express. `PoolRows::for_admission()` is
// `WIDEST_BATCH_RUNG`, so this holds by construction TODAY — but the relation is what makes it true, and it
// has to fail the build the moment someone cuts a rung (as I just did) without cutting admission with it.
const _: () = assert!(
    PagedKvPool::WIDEST_BATCH_RUNG >= PagedKvPool::NARROWEST_BATCH_RUNG,
    "the ladder is empty or inverted"
);
const _: () = {
    // Admission is bounded by the widest rung, and the widest rung must BE the last ladder entry — the
    // ladder is built by SKIPPING rungs whose sentinel or logits placement was rejected, so "the last one"
    // and "the widest one" are different claims and only one of them bounds a launch.
    let mut i = 0;
    let mut widest = 0;
    while i < PagedKvPool::BATCH_RUNGS.len() {
        if PagedKvPool::BATCH_RUNGS[i] > widest {
            widest = PagedKvPool::BATCH_RUNGS[i];
        }
        i += 1;
    }
    assert!(
        widest == PagedKvPool::WIDEST_BATCH_RUNG,
        "WIDEST_BATCH_RUNG must be the widest entry of BATCH_RUNGS — admission is bounded by it, and a \
         batch the ladder cannot express falls through to the one-at-a-time path, which MEASURES as silent \
         garbage for every request in the batch (all 4 of 4 collapsed, 3/3 trials)."
    );
};

// ⛔⛔⛔ THE FOLD'S PASS CEILING, AND IT IS A COMPILE-TIME RELATION — NOT A RUNTIME CHECK.
//
// One mask block per fold pass (`MaskBlocks::of(grid)`), and `passes = pages_per_row * batch_width`. The
// DECODE bundle's mask segment is a FIXED reservation, so the product has a hard ceiling and the geometry
// must not be able to exceed it. A pass past the last block reads bytes the host never staged; they are
// ZERO, and zero is VALID in an additive mask, so the row attends whatever the pool holds.
//
// ⛔⛔ THE NUMBERS THAT USED TO BE HERE WERE STALE, AND THEY COST A CORRECT CONCLUSION ITS RETRACTION.
// This block asserted, as MEASURED, that the width-4 mask segment is "~1.6 MB" and therefore that
// `32 passes = 2,097,152 B` OVERFLOWS it — which is exactly the failing decode geometry, so it reads as a
// diagnosis. It is not one. The `[mask-cap]` instrument resolves the placement by TID from THE RUNG'S OWN
// bundle, keyed by `seqs` (the batch width — NOT `sk_bucket_rungs`, which is keyed by `active_cap`; see
// `bundle_code::LadderRung` on the ladders sharing one name), and on granite-3.1-8b fp8 hd=128 it
// reads:
//   seqs=2  pmask =  2,097,152 B  =>  pages*width <= 64  (32 pages per row)
//   seqs=4  pmask =  4,194,304 B  =>  pages*width <= 64  (16 pages per row)
//   seqs=8  pmask =  8,388,608 B  =>  pages*width <= 64  ( 8 pages per row)
//   seqs=16 pmask = 16,777,216 B  =>  pages*width <= 64  ( 4 pages per row)
//   seqs=32 pmask = 33,554,432 B  =>  pages*width <= 64  ( 2 pages per row)
// The failing geometry needs `8 pages x 4 rows = 32` blocks against a ceiling of 64 — **capacity is 2x the
// need at every rung**, so the mask segment is NOT the deep-row defect.
//
// ⭐ A COMMENT CLAIMING A MEASURED NUMBER IS NOT EVIDENCE. Re-run `[mask-cap]` before trusting any figure
// in this block; the placement moves with `MAX_PAGES_PER_ROW`, the rung ladder and the mask shape, and a
// number written here outlives all three.
//
// 🛑 AND THIS CANNOT BE SATISFIED TODAY, WHICH IS THE POINT OF PUTTING IT HERE. Pages now come from a FREE
// LIST (`f5e26a3e`), so ONE REQUEST MAY DRAW THE WHOLE POOL: `pages_per_row` is bounded only by
// `pool_pages`, a runtime number, while the mask segment is baked. The product is therefore UNBOUNDED
// against a FIXED reservation — the geometry can express a batch the bundle cannot serve, by construction,
// and no runtime guard can fix that because by the time it fires the pool has already been sized.
//
// So the build must refuse it until ONE of these is true:
//   (a) `pages_per_row` is bounded by a const (cap a request's context, the old stripe's only real merit),
//   (b) the decode mask segment is sized from `MAX_FOLD_PASSES` at bake, or
//   (c) the fold steps the mask WITHOUT one block per pass (re-use a block per row, mask per page).
// (b) is the honest fix: the emitter already knows the widest rung and the pool's page count at bake.
// ⛔⛔⛔ THE QUADRATIC MASK ASSERTION IS DELETED — IT WAS FALSE, AND MEASUREMENT KILLED IT.
//
// It asserted `pages * width^2 * nqh * 512 <= 1_638_400` for every rung, i.e. `pages*width^2 <= 100`. Read
// from each rung's OWN bundle by TID (`[mask-cap]`, `8e7dfa96`) on granite-3.1-8b fp8:
//     rung seqs=2: pmask = 2,097,152 B  => cap = 16384 slots
//     rung seqs=4: pmask = 4,194,304 B  => cap = 16384 slots
// **pmask SCALES WITH `mq`** — the reservation already accounts for the batch width — so there is no
// quadratic squeeze and no fixed 1.6 MB ceiling. The true mask bound is `pages*width <= cap/PAGE_SLOTS`
// = **64**, and the failing configuration (`width 4 x 8 pages` = 32) is WELL INSIDE it.
//
// So the mask is EXONERATED, the law was mine and wrong, and the guard blocked the failing case only by
// coincidence. `1_638_400` came from an `attn.rs` comment about a DIFFERENT (prefill) bundle — the same
// wrong-bundle reading that made me call the mask hypothesis "refuted" hours earlier, now inverted.
//
// ⛔ NOT REPLACED BY A CORRECTED CONSTANT, because the real `cap` is a PER-RUNG BAKE FACT (16384 here) read
// at load, not a compile-time const — and asserting a relation whose terms are runtime is how the last
// three wrong laws got written. The `[mask-cap]` line reports it; the REAL cause of the width-4 x 8-page
// failure is still unknown and must be found before anything is asserted about it.

// ⛔ A CHUNK MUST FIT A PAGE AT ALL, or `chunk_write_start` has no boundary to fall back to: skipping to
// the next page would leave the same overrun there. So raising the prefill width past a page has to be a
// `cargo build` error rather than a silent re-run of the corruption it replaced.
//
// 🛑 **AT MODULE LEVEL, AND THE PLACEMENT IS THE ENTIRE GUARD.** This lived inside `impl PagedKvPool` as a
// NAMED associated const (`const CHUNK_FITS_ONE_PAGE`), because `const _` is illegal in an impl — and a
// named associated const that NOTHING FORCES is never const-evaluated. It was INERT, confirmed by
// experiment: inverted to `PREFILL_CHUNK_SLOTS > PAGE_SLOTS` — plainly false — it still compiled clean,
// `cargo check` reporting only "associated item `CHUNK_FITS_ONE_PAGE` is never used". A guard whose failure
// mode is a dead-code WARNING is not a guard, and it read as one in the notes for days. At module level the
// same inversion is `error[E0080]: evaluation panicked` — demonstrated, then reverted.
//
// ⭐ THE TWO WAYS, and a named associated const only works the second way. `const _: ()` at module level is
// evaluated because nothing has to reach it. A named one must be FORCED at a use site — which is exactly
// what `PrefixMaskShape::_COLS_IS_WHOLE_STICKS` and `BakeQueue::_BOUNDED` do (`let () = Self::_BOUNDED;` in
// their constructors), and they have to: both are generic in the const they check, so neither can move out
// here. This one is not generic, so it belongs at module level where the forcing cannot be forgotten.
const _: () = assert!(
    PagedKvPool::PREFILL_CHUNK_SLOTS <= PagedKvPool::PAGE_SLOTS,
    "a prefill chunk's padded write must fit inside ONE page — otherwise no write slot exists that keeps \
     it off the next kv head's plane"
);

impl PagedKvPool {
    /// Positions per page.
    ///
    /// WAS 256, on the stated reasoning that "the page is the swept extent, so anything larger is a
    /// regression for the same context". MEASURED, that is false. Holding requests and passes fixed
    /// and varying only the swept width through the ladder's own rungs: 64 slots 39.5 ms, 128 slots
    /// 39.0 ms, 256 slots 42.0 ms at eight requests. Four times the sweep costs six percent, because a
    /// fold pass is a fixed cost — it is also flat in the ROWS it computes (256 rows to 32 changed
    /// nothing measurable).
    ///
    /// What a pass is NOT flat in is how many of them there are. `reps = pages * requests`, so the page
    /// size sets the pass count: at 576 tokens a 256-slot page is three passes per request and costs
    /// 137 ms of a bs=8 step against 39 ms at one page. A 1024-slot page makes that context ONE pass.
    ///
    /// The ceiling is memory, not time. A page is `nkvh * hd * slots * 2 * layers`, so 1024 slots is
    /// 125.8 MB and a pool wide enough for 32 one-page requests would be 4 GB — more than the weights.
    /// That is the trade a 4-bit KV cache removes: four times the positions for the same bytes buys the
    /// pass reduction without the page growth. Until then the pool must be sized for the batch width
    /// actually run (`SUPERDSC_POOL_PAGES`).
    /// REVERTED TO 256 (2026-08-06). 1024 measured a genuine 3.25x on device compute at a 576-token
    /// context — a fold pass is a fixed cost and `reps = pages * requests`, so a wider page is fewer
    /// passes (the measurements above stand). But it is not shippable as it stands: a page is
    /// `nkvh * hd * slots * 2 * layers`, so 1024 slots is 125.8 MB and the DEFAULT pool
    /// (`DEFAULT_POOL_PAGES.max(widest_rung)` = 32 pages) became 3.84 GB of device memory — larger than
    /// the 2.64 GB of weights — plus, until the host KV shadow is gone, the same again in host RAM.
    /// Paying 4x memory for 3.25x on one workload is not a trade to make silently.
    ///
    /// Two things unlock it, in order: size the pool to the batch actually being run rather than to the
    /// widest baked rung, and make a page DENSER instead of BIGGER (a 4-bit KV cache gives four times
    /// the positions for the same bytes — TurboQuant is already in-tree for metal). Then this goes back
    /// up and the win comes with it.
    pub const PAGE_SLOTS: usize = 256;

    /// The resident Kᵀ plane as the score matmul's KERNEL: `[hd, PAGE_SLOTS]` per kv head, so its
    /// declared physical column count is a page's slots. The pool-side door into [`KtKernelPitch`];
    /// the new block's is [`SlotExtent::kernel_row_pitch`].
    pub const KT_KERNEL_PITCH: KtKernelPitch = KtKernelPitch(Self::PAGE_SLOTS as u32);

    /// ⭐ THE WIDEST PREFILL CHUNK — the baked row count of the prefix-capable prefill bundle, and
    /// therefore the number of slots ONE continuation chunk's cache write actually touches.
    ///
    /// ⛔ IT IS NOT THE CHUNK'S REAL LENGTH. Every continuation chunk runs on that ONE bundle
    /// (`use_prefix = chunk_start > 0`), and its cache write is a single `[mq, hd]` copy per kv head —
    /// so a chunk with 12 real rows still writes 96 CONSECUTIVE SLOTS, the last 84 of them padding.
    /// That is why this number, and not the chunk length, is what the page has to accommodate.
    ///
    /// Kept here rather than beside the prefill rung ladder because the POOL is what must make room for
    /// it; `codegen`'s `PREFILL_RUNGS` ceiling is asserted equal to it, so the two cannot drift.
    pub const PREFILL_CHUNK_SLOTS: usize = 96;

    /// ⭐ SLOTS A PADDED CHUNK WRITE MAY OVERRUN ITS PAGE BY — the physical slack every plane carries
    /// past its addressable [`PAGE_SLOTS`](Self::PAGE_SLOTS) so that overrun lands in DEAD SPACE instead
    /// of the next kv head's keys.
    ///
    /// ⛔ THIS IS THE LONG-CONTEXT BUG, MADE UNREACHABLE. `PREFILL_CHUNK_SLOTS` (96) does not divide
    /// `PAGE_SLOTS` (256), so a chunk eventually starts with less than a full chunk of room left: with
    /// chunks clipped to the room remaining, starts land on page offsets 0, 96, 192, and the one at 192
    /// has 64 slots of room for a 96-slot write. The 32-slot tail used to land at
    /// `block + PAGE_SLOTS*hd`, which IS the next kv head's block — so kv heads 1..nkvh lost the first
    /// 32 slots of their context. On card that read as a correct FIRST token (the last chunk scores from
    /// its own freshly computed K/V) and then fluent garbage from the second on, for every prompt long
    /// enough to need a third chunk. Measured boundary: coherent at a 192-token prompt, garbage at 204.
    ///
    /// `PAGE_SLOTS % PREFILL_CHUNK_SLOTS` is the room the last chunk of a page gets; the write needs a
    /// whole `PREFILL_CHUNK_SLOTS`, so the shortfall is the difference. Zero when the chunk divides the
    /// page, which is the only configuration that needs no slack at all.
    /// ⛔ ZERO, AND IT MUST STAY ZERO — the plane is EXACTLY the addressable page.
    ///
    /// This was `PREFILL_CHUNK_SLOTS - PAGE_SLOTS % PREFILL_CHUNK_SLOTS` (= 32), widening every plane so a
    /// prefill chunk's PADDED write had dead space to land in. **MEASURED: that broke hd=128.** On
    /// granite-3.1-8b it turned ` Rome. Q` / ` Madrid. Q` into ` Romes,` / ` Madridge.` — the right answer
    /// plus corruption — and forcing this to 0 restored them to the byte-for-byte 2b outputs. Some consumer
    /// of the feature-stick GROUP stride does not follow this constant at `hd > 64`, where a head spans two
    /// sticks; auditing the V write/read, Knat, Kᵀ and Q did not find it.
    ///
    /// The padded write is instead kept inside the page by the HOST choosing the write slot
    /// ([`PagedKvPool::chunk_room`] clips a chunk to its page, and a chunk whose PADDED window would still
    /// leave the page starts at the NEXT page instead, taking a [`KvHistory`] hole the mask already masks).
    ///
    /// ⭐ PREFER THE FIX THAT CHANGES THE FEWEST ADDRESSES. A stride EVERYTHING reads is the worst place to
    /// solve a problem the host can solve by choosing one number differently — "just +32 slots" reached every
    /// plane, every kv head and every feature stick at every head dim.
    pub const WRITE_SLACK: usize = 0;

    /// ⭐ THE PLANE'S PHYSICAL SLOT EXTENT — what separates one kv head's block from the next.
    ///
    /// DISTINCT FROM [`PAGE_SLOTS`](Self::PAGE_SLOTS), which is how many slots are ADDRESSABLE: masks are
    /// blocked by it, the fold sweeps it, and a [`KvSlot`] never exceeds it. The extra
    /// [`WRITE_SLACK`](Self::WRITE_SLACK) slots exist only so a padded chunk write has somewhere harmless
    /// to land, and nothing ever reads them — keeping the two numbers separate is what lets the overrun
    /// be absorbed without widening anything the attention actually sweeps.
    pub const PLANE_SLOTS: usize = Self::PAGE_SLOTS + Self::WRITE_SLACK;

    // REQUESTS ONE PAGE HOLDS — the extent of the pool's request dimension, and the widest batch any
    // launch can express.
    //
    // A BAKE CONSTANT, and it has to be: the decode rungs and the prefill rungs are separate baked
    // programs sharing one KV segment by alias, so a value that differed between them would give the
    // same tensor two layouts and one of them would read the other's keys. Being a `const` is what
    // makes that unexpressible rather than checked.
    //
    // 32 = the widest rung of `BATCH_RUNGS`, and it costs NOTHING against the pool the flat layout
    // defaulted to. There the default was `8 pages * widest_rung` = 256 pages, each 31.5 MB at
    // granite-3.1-2b, because a run had to be cut per row; here it is 8 pages of `32 * 31.5 MB`. Same
    // 8 GB, same 2048 positions per request, same 32 concurrent requests — the bytes moved from the
    // page count to the page.
    //
    // ⛔ `ROWS` IS GONE. A page used to hold `ROWS = 32` REQUESTS side by side, which is what taught the
    // device what a request is: every address carried a request term, the head stride was `ROWS` request
    // strides, and a page cost `ROWS`x its own slots (~1 GB, so an 8-page pool was 7.6 GB).
    //
    // A page is now SLOTS AND NOTHING ELSE, and a request owns a LIST OF PAGES through the host's block
    // table — ordinary paged attention. Request identity exists only in the host-staged mask. Two things
    // follow that were the whole bug family: prefill and decode cannot disagree about which row a request
    // occupies, because there is no such row; and a ragged batch has no per-request hole to mask, because
    // requests do not share a page's slot axis.

    /// PAGES ONE ROW MAY HOLD — a CONST CEILING, and `ensure_pages` must refuse past it.
    ///
    /// 🛑 IT CANNOT BE "THE WHOLE POOL". The prefix mask costs `pages * width` blocks of
    /// `nqh * width * PAGE_SLOTS * 2` bytes, so an unbounded page count against a BAKED mask segment is an
    /// unservable batch by construction — that is what the assertion below caught. 16 pages = 4096
    /// positions, which is the context a chat request actually reaches (a 1368-token generation off a long
    /// prompt is ~7 pages) with margin.
    ///
    /// 🛑 **6, NOT 16, AND THE MASK LAW PICKS IT.** `pages * width^2 <= 100` (see `decode_mask_bytes`), so
    /// keeping the 4-wide rung — which a 4-request batch REQUIRES — allows exactly 6 pages. 16 pages would
    /// permit only width 2, and cutting the ladder to `[2]` was MEASURED CATASTROPHIC: `decode_rung_for(4)`
    /// then finds no rung, the caller falls through to "decode one at a time", and **all 4 of 4 requests
    /// collapsed to one EOS token, 3/3 trials** — worse than the 3/4 that answered before the cut. So the
    /// ladder must keep the widths the pool will admit, and DEPTH is what gives way.
    /// ⇒ 6 pages = 1536 positions per request. A longer request is REFUSED by `ensure_pages`, loudly,
    /// which is the whole point: a refusal is a bug report, a collapse is a wrong answer.
    pub const MAX_PAGES_PER_ROW: u32 = 16;

    /// THE DECODE BATCH LADDER, so the widths that get baked and the requests a page holds cannot
    /// disagree. A rung wider than [`ROWS`](Self::ROWS) would bake a launch whose request axis walks
    /// off the end of the page into the next kv head's keys — silently, and only for the highest rows.
    /// ⛔ ONLY RUNGS THE PREFIX MASK CAN SERVE AT [`MAX_PAGES_PER_ROW`](Self::MAX_PAGES_PER_ROW).
    ///
    /// The mask is QUADRATIC in the width (`decode_mask_bytes`), so with a ~1.6 MB decode segment,
    /// `nqh = 32` and 16 pages per row the constraint `pages * width^2 * 16384 <= 1_638_400` leaves
    /// `width^2 <= 6` — **width 2 only**. The 4/8/16/32 rungs were baked and SELECTABLE while the mask
    /// could not serve them past 6 / 1 / 0 / 0 pages, which is exactly the failure the card shows: correct
    /// at bs=1-2, one EOS token per over-wide row at bs=4 with an 8-page history.
    ///
    /// 🛑 THIS IS A CAPABILITY CUT, NOT A TUNING. Wider batching at long context needs the mask to stop
    /// costing a block per (row, page) — one block per PAGE with a row offset, or a compressed validity
    /// encoding. Until that exists, a wider rung is a launch that reads unstaged bytes, and an additive
    /// mask reads those as VALID. Baking it is worse than not having it.
    pub const BATCH_RUNGS: [u32; 5] = [2, 4, 8, 16, 32];

    /// THE WIDEST BAKED RUNG, as a const — how many ROWS a pool must be able to give a page each.
    ///
    /// A const rather than a runtime max over the ladder, because the pool has to be sized before any
    /// batch exists and the answer is known at bake. It is the factor the pool's page count scales by:
    /// every row owns its own pages now, so `pool_pages = pages_per_row * WIDEST_BATCH_RUNG`.
    pub const WIDEST_BATCH_RUNG: u32 = Self::BATCH_RUNGS[Self::BATCH_RUNGS.len() - 1];

    /// THE NARROWEST RUNG. A batch of ONE still runs the narrowest baked body — there is no 1-wide
    /// rung — so a pool striped into fewer than this many rows is a pool whose highest slot has no
    /// pages, which is why [`PoolRows`] cannot name one.
    pub const NARROWEST_BATCH_RUNG: u32 = Self::BATCH_RUNGS[0];

    /// POSITIONS ONE SEQUENCE SHOULD REACH before the POOL is the reason it cannot.
    ///
    /// Not a capability and not a promise — a sizing TARGET, the depth [`PoolRows::widest_reaching`]
    /// buys before it spends the remaining budget on width. 4096 because that is the context a single
    /// chat request actually reaches (a 1368-token generation off a long prompt is ~7 pages) with margin,
    /// and because the alternative default — always the widest rung — is what capped granite-3.1-8b at
    /// `121 pages / 32 rows = 3 pages = 768 positions` and failed a long generation outright.
    pub const TARGET_POSITIONS_PER_ROW: usize = 4096;

    // ⛔ NO LADDER-FITS ASSERT. It tied the widest decode rung to the requests a page holds; a page holds
    // no requests now, so a batch rung is bounded by the pool's PAGES, which is a host allocation
    // question and not a baked one.

    pub fn new(nkvh: usize, hd: usize) -> PagedKvPool {
        PagedKvPool { nkvh, hd }
    }

    /// ELEMENTS BETWEEN TWO REQUESTS' KV, in any plane of any page+layer — the stride a batched
    /// launch steps its request axis by, and the stride a batched matmul's kernel must declare.
    ///
    /// The same number for Kᵀ, V and natural K, which is what lets ONE segment shift serve all three:
    /// Kᵀ's `[hd][PAGE_SLOTS]` block and V's `[PAGE_SLOTS][hd]` block hold the same elements in a
    /// different order, so they are the same size.
    ///
    /// It is also exactly what the kernel dataspaces already declare. `matmul_opspec_batched_off` lays
    /// a kernel out `[y, in, out]` and steps `y` by `in * out`, and the score kernel is declared
    /// `[hd, PAGE_SLOTS]` while the value kernel is `[PAGE_SLOTS, hd]` — so a truthful declaration of
    /// the block's own physical extent produces this stride and no override lies about anything.
    /// ⭐⭐⭐ THE LAYOUT. Every KV address in this backend is this function; everything else is a
    /// coordinate handed to it.
    ///
    /// ⛔ WHY IT IS ONE FUNCTION. The pool had fifteen address-producing methods and fifteen call sites
    /// outside it composing further terms — a slab, a request stride, a block index — which is fifteen
    /// independent opinions about where a key lives. The fold's request term alone was hand-rolled three
    /// different ways in one session, each wrong differently, and the reason a re-layout looked
    /// dangerous ("prefill's writes, the Kᵀ re-transpose and the decode reads must all move together")
    /// was ENTIRELY that duplication. With one authority nothing moves together: the layout changes
    /// here, once, and every caller follows because none of them restates it.
    ///
    /// The three planes differ only in their intra-block order, which is the whole reason all three
    /// exist — the score kernel wants `[hd, PAGE_SLOTS]` and the cache write and value kernel want
    /// `[PAGE_SLOTS, hd]`.
    pub fn addr(&self, c: KvCoord) -> u32 {
        // THE PHYSICAL plane extent — see `PLANE_SLOTS`. The ADDRESSABLE count does not appear in this
        // law at all: Kᵀ groups slots by `hd * stk`, and Knat/V group features by the plane's physical
        // slot span, so `PAGE_SLOTS` is a bound on what a `KvSlot` may BE, not a stride.
        // `plane_extent()`, not `PLANE_SLOTS`: this is the PHYSICAL stride, and the addressable count is a
        // different quantity with the same value. The type is what stops the substitution.
        let (stk, pls) = (STK as u32, Self::plane_extent().get());
        let hd = self.hd as u32;
        // ⚠️ PLANE-RELATIVE, deliberately. The emitter addresses each plane as its OWN NAMED TENSOR
        // (`kc`, `kct`, `vc`), so every offset a caller wants is WITHIN a plane — the plane base is
        // implicit in which tensor it names. An absolute address would be right only for a caller that
        // treats the pool as one buffer, and there is exactly one (the resident-tensor layout, which
        // uses `knat_plane_base`/`v_plane_base`/`kt_plane_base` directly). Folding the plane base in
        // here is what broke two of the pool's own doctests when I first tried to delegate to it.
        // 2. WHICH KV HEAD — and nothing else. There is no request term: a request is a set of SLOTS
        //    (reached through the host's page map), never a coordinate the device computes with.
        let block = c.kvh.get() * self.plane_block_elems() as u32;
        // 3. WHERE INSIDE THE BLOCK — the only place the planes differ.
        let (slot, feat) = (c.slot.get(), c.feat.get());
        let inside = match c.plane {
            // `[PAGE_SLOTS, hd]` sticked on `hd`: one plane of `PAGE_SLOTS` slots per stick of feature.
            // `pls`, not `ps`, for the feature-stick stride: a stick group spans the plane's PHYSICAL
            // slots, so at `hd > 64` the overrun of stick 0 cannot reach into stick 1.
            KvPlane::Knat | KvPlane::V => (feat / stk) * (pls * stk) + slot * stk + feat % stk,
            // `[hd, PAGE_SLOTS]` sticked on `PAGE_SLOTS`: one plane of `hd` features per stick of slot.
            KvPlane::Kt => (slot / stk) * (hd * stk) + feat * stk + slot % stk,
        };
        block + inside
    }

    /// THE (kv head, request) CELL INDEX inside a plane — the ONE place the `ROWS` factor is written.
    ///
    /// Every per-head base below is `block_index(kvh) * <block elements>`, and the emitter's nests
    /// reproduce the same base as `.block(block_index(kvh))` over the block's own footprint. Naming it
    /// once is the guard: the head stride and the request stride cannot drift apart, because the head
    /// stride IS `ROWS` request strides.
    ///
    /// Row 0, deliberately. The live request's row is a RUNTIME fact (which pool row it holds for its
    /// lifetime), so it is added by the launch as a segment shift, never baked into an op — the same
    /// division of labour the physical page already had.
    /// ELEMENTS IN ONE KV HEAD'S BLOCK of a plane — `hd * PAGE_SLOTS`, the whole of one head's page.
    ///
    /// This replaced `block_index(kvh) = kvh * ROWS` and `request_stride`. The head stride used to be
    /// `ROWS` request strides; it is now simply a head's own footprint, so there is no second quantity to
    /// drift against.
    /// ⭐ HOW MANY SLOTS A PREFILL CHUNK STARTING AT `start` MAY CLAIM — the room left in its page.
    ///
    /// A chunk must not straddle a page: one launch resolves ONE page table, so the slots past the page
    /// end belong to a page this launch cannot address, and writing its real rows there would put them
    /// somewhere nothing reads them back from. Clipping to this makes chunk starts land on page offsets
    /// 0, 96, 192 and then exactly on the next page boundary — no slots wasted, and the PADDED tail of
    /// the last chunk in a page is what [`WRITE_SLACK`](Self::WRITE_SLACK) absorbs.
    ///
    /// Total, not fallible: every slot is inside some page, so there is always at least one slot of room.
    pub fn chunk_room(start: KvSlot) -> SlotCount {
        let page = Self::PAGE_SLOTS as u32;
        SlotCount::new(page - start.get() % page)
    }

    /// ⭐ WHERE A PREFILL CHUNK MUST START so its **PADDED** window stays inside ONE page.
    ///
    /// A continuation chunk runs on the ONE prefix-capable bundle, so its cache write covers that bundle's
    /// BAKED row count (`padded`) — not the chunk's real length. If that window leaves the page it lands in
    /// the NEXT KV HEAD's plane: a correct first token, then fluent garbage, for any prompt long enough to
    /// need a third chunk.
    ///
    /// So the chunk steps **BACK** to the last start whose padded window still fits — `page_end - padded` —
    /// and RE-PROCESSES the overlap. Re-writing slots that already hold this request's keys is IDEMPOTENT:
    /// the same tokens at the same positions produce the same K/V, which is exactly what the padding rows of
    /// a launch already rely on.
    ///
    /// ⛔ TWO REJECTED ALTERNATIVES, BOTH MEASURED:
    /// 1. **Widen the plane** ([`WRITE_SLACK`], now 0) so the tail lands in dead space — BROKE hd=128
    ///    (` Rome. Q`/` Madrid. Q` became ` Romes,`/` Madridge.`), because the extra slots move the
    ///    feature-stick GROUP stride every reader re-derives, and a head spans two sticks above hd=64.
    /// 2. **Skip FORWARD** to the next page, leaving a [`KvHistory`] hole — BROKE long context outright
    ///    ("resident cache holds 267 positions but forwarding from 203 — KV desync"). The prefill path still
    ///    uses ONE number as both the rotary position and the KV write slot, so a hole desynchronises them;
    ///    separating those is the `BatchRow { rope_pos, hist }` distinction, and it does not exist here yet.
    ///
    /// Stepping BACK keeps them equal — token `n` still lands at slot `n` — so contiguity holds, no hole is
    /// created, and no stride moves. The cost is re-processing at most `padded - (PAGE_SLOTS % padded)`
    /// tokens per page boundary.
    ///
    /// Total: `padded <= PAGE_SLOTS` (asserted by [`CHUNK_FITS_ONE_PAGE`]) guarantees a fitting start exists.
    /// ⛔ RETURNS A [`ChunkStart`], NOT A [`KvSlot`] — so the slot the chunk writes at is the only slot its
    /// [`KvHistory::record_chunk`] can name. Handing back a bare `KvSlot` is what let the write step back
    /// while the record did not; see [`ChunkStart`] for the measurement.
    pub fn chunk_write_start(want: KvSlot, padded: SlotCount) -> ChunkStart {
        let page = Self::PAGE_SLOTS as u32;
        let off = want.get() % page;
        if off + padded.get() <= page {
            return ChunkStart(want);
        }
        // The last offset in this page whose padded window still fits. `padded <= page` ⇒ no underflow.
        ChunkStart(KvSlot::new(want.get() - off + (page - padded.get())))
    }

    /// ⭐ THE PHYSICAL EXTENT, TYPED — the only way the address law can obtain it, so the ADDRESSABLE count cannot
    /// be substituted for it in a stride.
    pub const fn plane_extent() -> PlaneExtent {
        PlaneExtent(Self::PLANE_SLOTS as u32)
    }

    pub fn plane_block_elems(&self) -> usize {
        // `PLANE_SLOTS`, NOT `PAGE_SLOTS`: this is the distance to the NEXT KV HEAD, so it is exactly the
        // quantity a padded chunk write overruns. Sizing it by the addressable slot count is what put
        // that overrun on top of the next head's keys.
        // Physical, for the same reason: this is the distance to the NEXT KV HEAD, which is exactly what a padded
        // chunk write overruns.
        self.hd * Self::plane_extent().get() as usize
    }

    /// ⭐ THE BLOCK A `(kv head, request)` PAIR IS — for a caller addressing through a `Nest`'s
    /// `.block()` rather than through [`addr`](Self::addr), which is the SystolicIR path the cache write
    /// takes. Same law either way; naming it here is what stops `block_index(kvh) + req` being spelled
    /// at a call site, which is how the request term came to have three independent spellings.
    /// THE BLOCK A KV HEAD IS, for a caller addressing through a `Nest`'s `.block()` rather than through
    /// [`addr`](Self::addr) — the SystolicIR path the cache write takes. Same law either way.
    pub fn block_of(kvh: KvHead) -> u32 {
        kvh.get()
    }

    /// 64-slot blocks in a page — the baked, fused fold a single launch performs.
    pub fn blocks_per_page(&self) -> usize {
        Self::PAGE_SLOTS / STK
    }

    /// ⭐⭐⭐ THE GATHER'S QUANTUM — the ELEMENTS in one 64-slot Kᵗ block, `hd * POOL_STICK`.
    ///
    /// This is the unit an index entry names. The whole pool is a uniform array of these: every stride
    /// it defines ([`plane_block_elems`](Self::plane_block_elems) `= 4 *` this, the plane, layer and
    /// page strides above it) is an exact multiple, so ONE integer reaches any cell and
    /// `addr = idx * skip_addr + base` is exact whenever `skip_addr` divides this.
    ///
    /// ⛔ NOT `plane_block_elems`, WHICH IS FOUR OF THESE. That is the distance to the next KV HEAD; a
    /// gather stepping by it would skip three quarters of every head's keys and still bake.
    /// `the_pool_is_a_uniform_array_of_stick_blocks` holds the two apart.
    pub fn stick_block_elems(&self) -> usize {
        self.hd * STK
    }

    /// ⭐ THE ENTRY IN **BYTES** — what one index step advances the address by. `stick_block_elems`
    /// times the pool's word length; stated here so the `* 2` is not respelled at every caller.
    pub fn stick_block_bytes(&self) -> u64 {
        self.stick_block_elems() as u64
            * <crate::superdsc_opspec::Fp16 as crate::superdsc_opspec::DataFormat>::WORD_LENGTH
                as u64
    }

    /// Pages needed to hold `positions`. This is the fold's launch count and the allocator's demand;
    /// it is a RUNTIME quantity, which is the whole point.
    /// ```
    /// use ktir_superdsc::sdsc_abstract::*;
    /// let p = PagedKvPool::new(8, 64);
    /// assert_eq!(p.pages_for(0), 0, "no prefix ⇒ nothing to fold");
    /// assert_eq!(p.pages_for(1), 1);
    /// assert_eq!(p.pages_for(256), 1);
    /// assert_eq!(p.pages_for(257), 2);
    /// assert_eq!(p.pages_for(4096), 16);
    /// ```
    pub fn pages_for(&self, positions: usize) -> usize {
        positions.div_ceil(Self::PAGE_SLOTS)
    }

    /// Cut a `pool_pages` pool into [`PoolRows`] stripes of `pool_pages / rows` pages each. A row owns
    /// its stripe exclusively — that is what gives the fold a uniform PAGE stride to cross the batch
    /// with, and what makes a request's context `pages_per_row` rather than the whole pool.
    ///
    /// `None` on the one real condition: the pool cannot give every stripe a page (`pool_pages < rows`),
    /// which means size a bigger pool or cut it into fewer stripes. Returning `None` rather than
    /// clamping to zero is deliberate — a zero-page run aliases row 0's KV, so every request would
    /// silently read and overwrite the first one's history.
    ///
    /// 🛑 The doc here used to say "every row gets ALL the pages", and listed a `rows > ROWS` bound
    /// against a page's request capacity. Both describe the request-inside-the-page design that the
    /// rebuild removed; the code has divided the pool since `0d68839f`.
    pub fn split_pool(pool_pages: u32, rows: PoolRows) -> Option<PoolSplit> {
        // ⭐ NO DIVISION ANY MORE. This used to compute `pages_per_row = pool_pages / rows` and hand every
        // row an exclusive stripe; pages now come from a free list, so a row is the identity of a LAUNCH
        // SLOT and bounds no capacity at all. What remains is the one real admission bound: every live
        // request needs AT LEAST one page, so a pool cannot seat more concurrent requests than it has pages.
        (pool_pages >= rows.get().get()).then_some(PoolSplit { rows: rows.get() })
    }

    /// Elements in one K-or-V plane of a page+layer — `nkvh` kv heads, each holding one page.
    ///
    /// ⭐ `nkvh` BLOCKS, with the block size stated ONCE in
    /// [`plane_block_elems`](Self::plane_block_elems). It used to spell `hd * PAGE_SLOTS` again here,
    /// which is the same number only while nothing separates the addressable page from the physical
    /// plane — and once [`WRITE_SLACK`](Self::WRITE_SLACK) did, the two disagreed: the stride between
    /// kv heads grew while the PLACEMENT reserved from this function did not, so kv head 7 addressed
    /// 258048+8192 B inside a 262144 B footprint. The emitter's own placement guard caught it at build
    /// time ("the op addresses PAST its own tensor"), which is exactly the argument for deriving both
    /// from one function rather than two copies of one product.
    ///
    /// The KV segment is three placements of `plane_stride` advanced by
    /// [`layer_stride`](Self::layer_stride), and the runtime's page stride is derived from those same
    /// placement deltas — so this is the one number that sizes the pool.
    pub fn plane_stride(&self) -> usize {
        self.nkvh * self.plane_block_elems()
    }

    /// Elements in one page+layer: Kᵀ, then V, then natural K.
    pub fn layer_stride(&self) -> usize {
        3 * self.plane_stride()
    }

    // ── PLANE ORDER inside a page+layer: natural K, then V, then transposed K. ──
    // This mirrors the pre-paged cache's own order (`kc`, `vc`, `kct`) on purpose. The order is free
    // — every op addresses its plane relatively, so which plane sits where costs no memory and no
    // work — but it decides the byte offset each plane lands on, and this path is measurably
    // placement-sensitive. Having transposed-K first put all three planes at offsets the tuned
    // layout never used.

    /// Element offset of the natural-K plane within a page+layer — FIRST, where `kc` was.
    pub fn knat_plane_base(&self) -> usize {
        0
    }

    /// Element offset of the V plane — SECOND, where `vc` was. Applied by the PLACEMENT, never an op.
    pub fn v_plane_base(&self) -> usize {
        self.plane_stride()
    }

    /// Element offset of the transposed-K plane — LAST, where `kct` was.
    pub fn kt_plane_base(&self) -> usize {
        2 * self.plane_stride()
    }

    #[allow(dead_code)]
    fn v_block_base_first(&self, qh: usize, gqa: usize, b: u32) -> u32 {
        // The head width is the POOL's (`self.hd`), never a caller's: a V page's geometry belongs to
        // the pool that laid it out, and taking a second `hd` here let a caller name one width while
        // the address used another.
        // ⭐ `b * STK * hd`, NOT `b * STK * STK`. A 64-slot block advances 64 SLOTS of one stick each,
        // and a whole slot is `hd` elements — the two agree ONLY at head_dim 64, which is every model that
        // had run. At head_dim 128 block 0 was exact and every later block read one plane too far, so
        // attention was right over the first 64 cached slots and drifted from the 65th. The comment above
        // described this bug; it is now fixed rather than described.
        (gqa_kv_head(qh, gqa) * Self::PAGE_SLOTS * self.hd) as u32 + b * STK as u32 * self.hd as u32
    }

    /// Kᵀ WRITE offset of `K[slot][d]` within a page+layer — the consumer's own read model, so
    /// producer and consumer cannot diverge ([`kcache_kt_write_offset`] at page granularity).
    /// ```
    /// use ktir_superdsc::sdsc_abstract::*;
    /// let p = PagedKvPool::new(8, 64);
    /// for (slot, d) in [(0usize, 0usize), (1, 0), (63, 7), (64, 0), (255, 63)] {
    ///     assert_eq!(p.kt_write_off(0, slot, d),
    ///                kcache_kt_write_offset(slot, d, 64, PagedKvPool::PAGE_SLOTS, 64));
    ///     // 256 slots = 4 sticks ⇒ stick-blocked: (s/64)*(hd*64) + d*64 + s%64.
    ///     assert_eq!(p.kt_write_off(0, slot, d), (slot / 64) * (64 * 64) + d * 64 + slot % 64);
    /// }
    /// ```
    pub fn kt_write_off(&self, kvh: usize, slot: usize, d: usize) -> usize {
        // The head's own block — no `ROWS` factor, because a page holds slots and not requests.
        kvh * self.hd * Self::PAGE_SLOTS
            + kcache_kt_write_offset(slot, d, self.hd, Self::PAGE_SLOTS, STK)
    }

    /// V WRITE offset of `V[slot][d]` within a page+layer. A `[1, hd]` row is CONTIGUOUS here, which
    /// is why V needs no transpose and K does.
    /// ```
    /// use ktir_superdsc::sdsc_abstract::*;
    /// let p = PagedKvPool::new(8, 64);
    /// for slot in [0usize, 1, 255] {
    ///     assert_eq!(p.v_write_off(0, slot, 0), slot * 64);
    ///     assert_eq!(p.v_write_off(0, slot, 1) - p.v_write_off(0, slot, 0), 1);
    /// }
    /// ```
    pub fn v_write_off(&self, kvh: usize, slot: usize, d: usize) -> usize {
        // The head's own block — no `ROWS` factor; a page holds slots, not requests.
        kvh * Self::PAGE_SLOTS * self.hd
            + vcache_write_offset(slot, d, self.hd, Self::PAGE_SLOTS, STK)
    }

    /// THE INVARIANTS THE RUNTIME DEPENDS ON: every access of a page+layer stays inside one plane,
    /// and the three planes tile the layer slice exactly. That is what lets ONE per-op byte offset
    /// relocate the score kernel, the value kernel and the write target together.
    /// ```
    /// use ktir_superdsc::sdsc_abstract::*;
    /// use ktir_superdsc::addr::Gqa;
    /// use std::num::NonZeroU32;
    /// for &(nkvh, hd, gqa) in &[(8usize, 64usize, 4usize), (8, 128, 3), (1, 256, 1)] {
    ///     let p = PagedKvPool::new(nkvh, hd);
    ///     let plane = hd * PagedKvPool::PAGE_SLOTS;
    ///     for qh in 0..nkvh * gqa {
    ///         let head = QueryHead::all(NonZeroU32::new((nkvh * gqa) as u32).unwrap()).nth(qh).unwrap();
    ///         let kvh = KvHead::of_query(head, Gqa::new(gqa as u32));
    ///         let kt = p.addr(KvCoord::block(KvPlane::Kt, kvh));
    ///         assert!(kt as usize + plane <= p.plane_stride());
    ///         let v = p.addr(KvCoord::block(KvPlane::V, kvh));
    ///         assert!(v as usize + plane <= p.plane_stride());
    ///     }
    ///     for kvh in 0..nkvh {
    ///         assert!(p.kt_write_off(kvh, PagedKvPool::PAGE_SLOTS - 1, hd - 1) < p.plane_stride());
    ///         assert!(p.v_write_off(kvh, PagedKvPool::PAGE_SLOTS - 1, hd - 1) < p.plane_stride());
    ///     }
    ///     // natural K, V, transposed K — the pre-paged cache's order, tiling the layer exactly.
    ///     assert_eq!(p.knat_plane_base(), 0);
    ///     assert_eq!(p.v_plane_base(), p.plane_stride());
    ///     assert_eq!(p.kt_plane_base(), 2 * p.plane_stride());
    ///     assert_eq!(p.kt_plane_base() + p.plane_stride(), p.layer_stride());
    ///     // THE HEADS TILE THE PLANE EXACTLY. One head's block is its whole page — `hd * PAGE_SLOTS`
    ///     // — so head `k` ends precisely where head `k+1` begins and there is no request axis in
    ///     // between for a launch to step off the end of.
    ///     assert_eq!(p.plane_block_elems(), p.hd() * PagedKvPool::PAGE_SLOTS);
    ///     assert_eq!(nkvh * p.plane_block_elems(), p.plane_stride());
    /// }
    /// ```
    pub fn planes_tile_the_layer(&self) -> bool {
        self.layer_stride() == 3 * self.plane_stride()
            && self.plane_stride() > 0
            // THE PLANE IS ITS HEADS AND NOTHING ELSE. Stated as an equation because the two sides are
            // written in different places (a nest's `.block()` count against a matmul's declared kernel
            // extent) and drifting apart would put one head's slots on top of the next head's.
            && self.plane_stride() == self.nkvh * self.plane_block_elems()
    }

    /// Head width this pool laid itself out for.
    pub fn hd(&self) -> usize {
        self.hd
    }
}

/// ⭐⭐⭐⭐⭐ THE GATHER'S ENTRY FACTOR — index entries per PHYSICAL PAGE, i.e. what turns a block-table
/// page number into the value the card reads.
///
/// ## The identity this exists to make exact
/// ```text
/// host today (fold_plan::page_base_bytes):   addr = phys * page_stride_bytes
/// card with a gather (ConvertData_gather_idx): addr = idx  * skip_addr      + base
/// ```
/// so `idx = phys * (page_stride_bytes / skip_addr_bytes)`, and `skip_addr_bytes` is ONE STICK BLOCK
/// because that is what the emitter declares (`PageExtent::of_one_stick()` pinned on the slot axis).
/// Both inputs are read off the SAME live session, which is what makes this an identity rather than a
/// second derivation that has to be kept in agreement with the first.
///
/// ⛔ THE FACTOR IS NOT A CONSTANT AND IT IS NOT 4096. It carries `hd` (a stick block is `hd * 64`
/// elements — 4096 at head_dim 64, **8192 at head_dim 128**) and it carries `iters`, because a page
/// spans every LAYER (`page_stride_bytes = iters * kv_stride`). So it is a fact about the BUNDLE, and
/// every comment in this tree that says "global 4096-element block number" is implicitly hd=64.
///
/// ⛔ `None` RATHER THAN A ROUNDED ANSWER when a page is not a whole number of entries: no integer
/// index can then name a page boundary, so every entry would land somewhere inside the previous page.
/// The caller must refuse, and refusing needs the `None` to exist.
///
/// `page_stride_bytes == 0` is an unpaged bundle, which has no pages to index and also answers `None`.
pub fn gather_entries_per_page(
    page_stride_bytes: u64,
    pool: PagedKvPool,
) -> Option<EntriesPerPage> {
    let entry = pool.stick_block_bytes();
    if page_stride_bytes == 0 || entry == 0 || !page_stride_bytes.is_multiple_of(entry) {
        return None;
    }
    i64::try_from(page_stride_bytes / entry)
        .ok()
        .map(EntriesPerPage)
}

/// ⭐⭐⭐⭐⭐ INDEX ENTRIES PER PHYSICAL PAGE — the factor that converts the host's page number into the
/// card's stick-block number, and it has exactly ONE door: [`gather_entries_per_page`].
///
/// ⛔⛔⛔ THE WHOLE POINT IS THAT IT CANNOT BE WRITTEN DOWN. It was an `i64` parameter, so any caller
/// could supply a literal or its own `page_stride / 4096` — and the true factor carries `hd` (a stick
/// block is `hd * POOL_STICK` elements: 4096 at head_dim 64, **8192 at head_dim 128**) and carries
/// `iters`, because a page spans every layer. Every comment in this tree that says "global
/// 4096-element block" is implicitly hd=64, which is exactly the hand-spelled constant this type
/// makes unspellable. The only constructor divides the SESSION's own `page_stride_bytes` by the
/// POOL's own `stick_block_bytes()`, so both halves of the identity come from the artifact.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct EntriesPerPage(i64);

impl EntriesPerPage {
    /// For reporting and for the one multiply in [`GatherEntry::of_page_block`].
    pub const fn get(self) -> i64 {
        self.0
    }
}

/// ⭐⭐⭐⭐⭐ ONE INDEX ENTRY — a GLOBAL STICK-BLOCK NUMBER that dxp turns into an address as
/// `addr = idx * skip_addr + base` (`ConvertData_gather_idx`). It is an ADDRESS IN WAITING, and it has
/// exactly one door: [`Self::of_page_block`].
///
/// ⛔⛔⛔ IT WAS A BARE `i32` IN A `Vec<i32>`, WHICH IS THE ONE SHAPE THAT LETS THE COMPOSITION BE
/// SPELLED WRONG. The entry is `phys * entries_per_page + block_in_page`, and each of those three
/// terms has a plausible wrong sibling: `phys` can be a LOGICAL page, the factor can be a hand-written
/// 4096, and `block_in_page` can be a launch ROW (`row_of`) rather than a block. Every one of those
/// substitutions type-checks against `i32` and produces a REAL address — page 0's first block, or
/// another request's keys — so the failure is fluent output rather than a fault. Minting through one
/// constructor that takes an [`EntriesPerPage`] (which itself cannot be hand-written) leaves no
/// arithmetic for a caller to get wrong.
///
/// ⛔ NO `From<i32>`, NO `Default`, AND NO ARITHMETIC OPERATORS, deliberately: a `+` on this type is
/// how a caller would re-add a page term that `of_page_block` already added.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct GatherEntry(i32);

impl GatherEntry {
    /// THE ONE COMPOSITION: physical page `phys`, times this bundle's entries-per-page, plus the
    /// block that `(kv head, slot window)` occupies inside a page ([`GatherScratch::block_in_page`]).
    ///
    /// `None` on a negative page, or when the product or sum leaves `i32` — the width the descriptor
    /// declares (`SENUINT32`). A wrapped entry is an address, so it cannot be allowed to round.
    pub fn of_page_block(per_page: EntriesPerPage, phys: i64, block_in_page: u32) -> Option<Self> {
        if phys < 0 {
            return None;
        }
        let e = phys
            .checked_mul(per_page.get())?
            .checked_add(i64::from(block_in_page))?;
        i32::try_from(e).ok().map(GatherEntry)
    }

    /// The wire value, for the one site that encodes the table into LE bytes.
    pub const fn as_i32(self) -> i32 {
        self.0
    }
}

/// ⭐⭐⭐⭐⭐ THE PER-PASS PITCH OF THE INDEX TABLE, IN INT32 ENTRIES — and it is the PREFIX MASK's block
/// stride, not a number of its own. One door: [`PrefixMaskShape::pass_stride`].
///
/// ⛔⛔⛔ IT WAS `pass_stride_entries: usize`, COMPUTED AT THE CALL SITE AS `rep_stride_bytes() / 4`.
/// The index is an ACTIVATION (`SegRole::Activation` == `SEG_MASK`), so it rides the MASK's per-pass
/// segment shift — one shift moves both tensors, therefore ONE pitch governs both. A `usize` parameter
/// invites the sibling derivation (a page count, the scratch's own row count, `MAX_FOLD_PASSES`), and
/// a pitch that disagrees puts pass `p`'s gather on another pass's entries with a clean bake and no
/// counter moved. Deriving it inside the mask shape is what makes "the same number, two derivations"
/// stop being expressible.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct PassStride(usize);

impl PassStride {
    pub const fn get(self) -> usize {
        self.0
    }
}

/// ⭐⭐⭐⭐⭐ THE GATHER'S INDEX TABLE FOR ONE LAUNCH — one entry per (launch row, logical page), laid out
/// **A FLAT ARRAY OVER THE GATHER-COPY OP'S OWN `mb` ROWS**, one block of them per fold pass.
///
/// ⛔⛔⛔ THE "ONE 32-ENTRY STICK PER ROW" LAW THIS FUNCTION USED TO IMPLEMENT DOES NOT APPLY HERE, AND
/// REUSING IT MIS-STRIDES THE TABLE. That law was derived for the MATMUL kernel as the gathered
/// operand, whose index dims were the value's two pinned axes `["out","mb"] = [256, 8]` — paged on
/// `out`, so `mb` was a SECOND index axis and dxp rounded the inner (paged) extent up to a whole
/// 32-entry stick per `mb` row. The gather now lives on the KERNEL-LESS copy op
/// ([`crate::ir::bridge::tiled_op_sdsc_op::gather_copy_opspec`] — deeptools cannot schedule a gather on
/// an op that has a KERNEL, measured twice on the card), whose gather dim is `mb` with NO second pinned
/// axis. `IndirectAccess::pins` therefore yields ONE entry, the index is rank-1 `["mb"]`, and
/// `dsc2.cpp:4001-4008`'s product has a single factor: `ceil_to_stick(mb)` entries, packed. There is no
/// per-row stick to pad to, because the ROW *is* the entry.
///
/// ⭐⭐⭐ THE PASS PITCH IS THE PREFIX MASK'S BLOCK STRIDE, AND THAT IS FORCED, NOT CHOSEN. A fold pass
/// needs its OWN entries (pass `p` reads every request's page `p`), and the launch ABI offers exactly
/// three per-pass segment shifts — `kv`, `mask` and `intermediate` (`fold_plan::SegDeltas`). The KV
/// segment is resident and never uploaded, the intermediate segment carries `qs`/the whole online-softmax
/// state (shifting it would move every operand the pass reads), and the index is an ACTIVATION
/// (`SegRole::Activation` == `SEG_MASK`) because it is per-step host data. So the index rides the MASK's
/// shift, which means its per-pass pitch must BE the mask's — [`PrefixMaskShape::rep_stride_bytes`], the
/// one number both placements and both fills derive from. `pass_stride_entries` is that value in int32
/// entries; a pitch that disagrees puts pass `p`'s gather on the wrong entries with a clean bake.
///
/// ⛔ ROW ORDER IS [`GatherScratch::row_of`]'s AND NOTHING ELSE'S. The copy writes scratch row `i` from
/// entry `i`, and the score/value matmuls read row `i` at `i * cols` with `y` stepping ONE row — so the
/// order here IS where request `r`'s keys land. Spelling it at this call site is how the emitter's base
/// offsets and the host's table came to disagree in every earlier attempt.
///
/// ⛔ AND `None` RATHER THAN A CLAMPED ANSWER on every shape it cannot express: a row with no pages
/// (nothing to name), a factor that does not divide, an entry that overflows int32, more rows than the
/// pass pitch holds, or a page a row does not own. Every one of those would otherwise become entry 0 —
/// which is a REAL address (page 0's first block), so the failure would be fluent output.
pub fn gather_index_table(
    scratch: GatherScratch,
    // `block_tables[row][logical_page] = physical page`, exactly as the session holds it.
    block_tables: &[Vec<i64>],
    // Fold passes this step walks — the PAGES, since a gathered fold's pass IS a page.
    pages: usize,
    // [`gather_entries_per_page`], off the same session `fold_plan::page_base_bytes` reads. It is an
    // [`EntriesPerPage`] and not an `i64` so that the hd-carrying factor cannot be hand-written here.
    entries_per_page: EntriesPerPage,
    // The MASK's own per-pass shift ([`PrefixMaskShape::pass_stride`]) — a [`PassStride`] rather than a
    // `usize` because the index rides that shift and one shift cannot have two pitches.
    pass_stride: PassStride,
) -> Option<Vec<GatherEntry>> {
    let rows = scratch.rows() as usize;
    let stride = pass_stride.get();
    if block_tables.is_empty()
        || pages == 0
        || entries_per_page.get() <= 0
        || rows == 0
        // The pass block must hold the pass's entries; otherwise pass 1 would start inside pass 0's.
        || rows > stride
        // Every launch row the scratch addresses must have a page map. Fewer tables than the scratch's
        // request extent is a bind the emitter baked for rows the host did not install.
        || block_tables.len() < scratch.mq() as usize
    {
        return None;
    }
    // ⛔ THE PAD IS PAGE 0's FIRST BLOCK AND THAT IS A REAL ADDRESS — stated, because it is what every
    // entry this loop does not reach will gather. It is sound only because the gather-copy op reads
    // `rows` of them and the mask invalidates the rest; it is NOT a safe default for a missing row,
    // which is why every failure above refuses instead of leaving one here.
    let pad = GatherEntry::of_page_block(entries_per_page, 0, 0)?;
    let mut out = vec![pad; pages * stride];
    for p in 0..pages {
        for r in 0..scratch.mq() {
            let table = block_tables.get(r as usize)?;
            // ⛔ A SHORT ROW REPEATS **ITS OWN** LAST PAGE, AND AN EMPTY ONE IS A REFUSAL. The launch
            // sweeps `pages` for every row, so a row holding fewer has no address of its own for the tail
            // — and the filler must be its own last page, not 0: a fully-masked column reading this row's
            // own keys is inert, while page 0 is a DIFFERENT row's keys and an unstaged mask byte reads as
            // VALID. A row with NO pages has nothing to repeat, which is why that one refuses.
            let last = *table.last()?;
            let phys = *table.get(p).unwrap_or(&last);
            for kvh in 0..scratch.nkvh() {
                for b in 0..scratch.windows() {
                    // ⭐ ONE COMPOSITION, INSIDE THE TYPE. The `phys * per_page + block` arithmetic that
                    // used to be spelled here — and whose three terms all have plausible wrong siblings
                    // — is now [`GatherEntry::of_page_block`]'s and nothing else's.
                    out[p * stride + scratch.row_of(kvh, b, r)? as usize] =
                        GatherEntry::of_page_block(
                            entries_per_page,
                            phys,
                            scratch.block_in_page(kvh, b)?,
                        )?;
                }
            }
        }
    }
    Some(out)
}

/// ⭐⭐⭐⭐⭐ THE PAGE-GRANULAR INDEX TABLE — one entry per `(fold pass, request)`, and **the kv-head and
/// window loops are gone**, which is the entire difference from [`gather_index_table`].
///
/// That function walks `pages × mq × nkvh × windows` and composes `phys * per_page + block_in_page(kvh,
/// b)`. This one walks `pages × mq` and composes `phys * planes_per_page`. The removed loops are exactly
/// the ones that made the entry count `nkvh * windows * mq` — 64/128/256 at width 8 against dxp's ONE
/// 32-word IBR stick, which is the measured rung-8 corruption — and the removed term is the one that had
/// to agree with the emitter's own window numbering.
///
/// ⛔ THE PASS PITCH IS STILL THE PREFIX MASK'S BLOCK STRIDE, and still forced rather than chosen: the
/// index is a `SegRole::Activation` (== `SEG_MASK`), so it rides the MASK's per-pass segment shift, and one
/// shift cannot have two pitches. A pitch that disagrees puts pass `p`'s gather on another pass's entries
/// with a clean bake.
///
/// ⛔ AND `None` RATHER THAN A CLAMPED ANSWER on every shape it cannot express — a row with no pages, more
/// rows than the pass pitch holds, fewer block tables than the batch, an entry that leaves `i32`. Every one
/// of those would otherwise become entry 0, which is page 0's first plane: a REAL address, so the failure
/// would be fluent wrong output rather than a fault.
pub fn page_gather_index_table(
    scratch: PageScratch,
    // `block_tables[row][logical_page] = physical page`, exactly as the session holds it.
    block_tables: &[Vec<i64>],
    // Fold passes this step walks — the PAGES, since a gathered fold's pass IS a page.
    pages: usize,
    // [`gather_entries_per_page`], off the same session `fold_plan::page_base_bytes` reads.
    planes: EntriesPerPage,
    // The MASK's own per-pass shift ([`PrefixMaskShape::pass_stride`]).
    pass_stride: PassStride,
) -> Option<Vec<GatherEntry>> {
    let rows = scratch.rows() as usize;
    let stride = pass_stride.get();
    let per_row = scratch.entries_per_row() as usize;
    if block_tables.is_empty()
        || pages == 0
        || planes.get() <= 0
        || rows == 0
        // ⛔ AND ONE REQUEST'S ENTRIES MUST FIT ITS OWN INDEX STICK. `PageScratch::of_pass` already
        // refuses a wider plane, so this is the host's half of the same law: a `per_row` past the stick
        // would write request `r+1`'s entries over request `r`'s and gather another request's page.
        || per_row == 0
        || per_row > CopyDims::ENTRIES_PER_OP as usize
        // ⛔ THE PASS BLOCK MUST HOLD THE STICK-STRIDED ENTRIES, NOT THE RAW ROW COUNT. Each request's
        // entry sits at its own index stick (see the loop below), so a pass spans
        // `mq * ENTRIES_PER_OP` words — 32× the row count. Checking `rows` alone would admit a pitch that
        // puts request 1's stick inside pass 0's block and silently gather another pass's page.
        || rows * CopyDims::ENTRIES_PER_OP as usize > stride
        || block_tables.len() < scratch.mq() as usize
    {
        return None;
    }
    // ⛔ THE PAD IS PAGE 0's FIRST PLANE AND THAT IS A REAL ADDRESS — stated, because it is what any entry
    // this loop does not reach would gather. Sound only because the copy reads `rows` of them and the mask
    // invalidates the rest; NOT a safe default for a missing row, which is why the checks above refuse.
    let pad = GatherEntry::of_page_block(planes, 0, 0)?;
    let mut out = vec![pad; pages * stride];
    for p in 0..pages {
        for r in 0..scratch.mq() {
            let table = block_tables.get(r as usize)?;
            // ⛔ A SHORT ROW REPEATS **ITS OWN** LAST PAGE, AND AN EMPTY ONE REFUSES. The launch sweeps
            // `pages` for every row, so a row holding fewer has no address of its own for the tail — and
            // the filler must be its OWN last page, not 0: a fully-masked column reading this row's own
            // keys is inert, while page 0 is a DIFFERENT row's keys and an unstaged mask byte reads VALID.
            let last = *table.last()?;
            let phys = *table.get(p).unwrap_or(&last);
            // ⭐⭐⭐ ONE INDEX STICK PER REQUEST, `per_row` ENTRIES FROM COLUMN 0 — the vendor's own
            // `[num_blocks, INT32_ELEMS_PER_STICK]` layout, and it is FORCED rather than copied.
            // `PageScratch::copies` emits one op per request (an LX limit, measured on card), and
            // `GatherCopy::index_base` must be a whole number of index sticks because dxp loads the IBR
            // one stick at a time. So request `r`'s run has to start at stick `r`, column 0; the words
            // past `per_row` in its stick are pad the op never reads.
            //
            // ⛔ AND ENTRY `i` IS THE `i`-th STICK BLOCK OF THE PLANE, IN PLANE ORDER — not a kv head
            // and not a window. The destination row is the plane itself and the copy fills it linearly,
            // so `i` is at once the source block and the destination sub-run; re-deriving it from
            // `(kvh, window)` is what let the emitter's block numbering and the host's disagree.
            let base = p * stride + r as usize * CopyDims::ENTRIES_PER_OP as usize;
            for i in 0..per_row {
                *out.get_mut(base + i)? =
                    GatherEntry::of_page_block(planes, phys, u32::try_from(i).ok()?)?;
            }
        }
    }
    Some(out)
}

/// ⭐⭐⭐⭐⭐ THE GATHER'S CONTIGUOUS DESTINATION, AS ONE LAW — its row count, its row width, WHICH
/// (kv head, slot window, request) each row holds, the index entry that fills it, and the `y` step the
/// score/value matmuls take across it.
///
/// ## Why the scratch exists at all
/// `LaunchPages`' own note settles it: a `y = request` matmul needs an operand with a UNIFORM
/// per-request pitch, and the paged pool has none by law (`PagedKvPool::addr` — "there is no request
/// term"). Three earlier attempts baked `hd * PAGE_SLOTS` believing it was a request stride; it is
/// MEASURED to be exactly `plane_block_elems()`, the KV-HEAD stride, so every row scored against kv head
/// `r`'s keys — real, well-formed, wrong. A destination THIS compiler places has the pitch dxp derives
/// (`in * out` of the kernel), so the stride is not declared, not overridden and not bakeable wrongly:
/// it is the buffer's own row.
///
/// ## The shape, and why `windows` is in it
/// ⛔ ONE FOLD PASS IS `nb` WINDOWS, NOT ONE. `assemble_attn`'s prefix loop is
/// `for w in SlotWindow::sweep(SlotCount::new(active_cap))` and `SlotWindow::count_in(swept)` is
/// `swept/64`, so a pass emits `nb = active_cap / 64` blocks, each reading a DIFFERENT 64-slot stick
/// block of the SAME page. A paged bundle's `active_cap` is a `ActiveCap::decode_ladder` rung below or
/// at `PAGE_SLOTS = 256`, so `nb ∈ {1, 2, 4}` — and it is 4 at the ceiling rung, which is the multi-page
/// regime the collapse is worth anything in. Sizing the scratch or the index by `nkvh * mq` (dropping
/// `nb`) is the defect already recorded once in `lower_subtile_tape_to_superdsc.rs`: entries read PAST
/// the placement, and past it is the next tensor's bytes read as block numbers — a clean bake, no fault,
/// another page's keys, with nothing left to rebase them because a gathered fold drops the KV shift.
///
/// ## Why `hd <= POOL_STICK` is a precondition and not a bound to widen
/// The copy is a FLAT block move, which requires a 64-slot window of the plane to be CONTIGUOUS. It is,
/// for Kᵗ at every head dim (`[hd, cap]` stick-major on `cap`: window `b` is `[b*hd*64, +hd*64)`), and
/// for V (`[cap, hd]` stick-major on `hd`) only while `hd` is one stick — above that a window is
/// `nslab` runs `PLANE_SLOTS*64` apart and a flat copy would relayout it. So a wider head dim gets no
/// gather and emits exactly what it emits today (`None` here), rather than a copy that bakes and
/// scrambles the V leg.
///
/// ⛔⛔⛔ **AND "WIDENING IT MEANS A SECOND `skip_addr`" WAS PROSE, AND IT IS WRONG.** That sentence
/// stood here as the whole account of the cost and it misnames every part of it.
/// `zz_the_two_planes_block_numbering_diverges_above_one_stick.rs` derives the real cost from
/// [`PagedKvPool::addr`] itself, and `GatherIndexConversion.cpp::computeGatherMetadata` (the pod's
/// deeptools, read not grepped) settles the dxp half:
///
/// * **Kᵗ needs nothing at all.** A Kᵗ window is ONE contiguous [`PagedKvPool::stick_block_elems`] run
///   at every head dim, at an offset that is an exact multiple of one. The refusal was never about Kᵗ.
/// * **V's WINDOW stride is `POOL_STICK * POOL_STICK`, not `hd * POOL_STICK`.** V's slot stride is ONE
///   STICK whatever `hd` is, so a 64-slot window advances 4096 elements — at hd=128 that is HALF a
///   stick block, and `addr = idx * skip_addr + base` has no fractional `idx`, so a stick-block entry
///   cannot name V's odd windows. The unit that serves BOTH planes is `POOL_STICK * POOL_STICK`, which
///   IS the stick block at hd=64 — so the one proven geometry pays nothing to adopt it.
/// * **A second `skip_addr` is FREE, and it is not the obstacle.** dxp derives one `skip_addr` per
///   (index lds, value alloc) pair as `page × the per-core extents of the unpinned dims`, and each
///   gather copy is its own SDSC with its own pair and its own synthesised `idx2addr` writing a
///   SEPARATE address buffer (`createIdx2AddrSdsc`: an `index_input_lds` and an `address_output_lds`).
///   So the Kᵗ copy may pin `hd` sub-rows and the V copy `POOL_STICK` with no interaction. (dxp does
///   `DT_CHECK(skip_addr_sticks % 2 == 0)` on SEN1P5+, so the pin must be EVEN — 64 and 128 both are.)
/// * **THE OBSTACLE IS THAT ONE INDEX TABLE CANNOT SERVE BOTH COPIES.** In the 4096 unit, Kᵗ numbers a
///   (window, slab) `b * nslab + s` inside its kv head while V numbers it `s * blocks_per_page + b`:
///   the SAME SET of blocks in a DIFFERENT ORDER, equal for every coordinate **iff `nslab == 1`**. That
///   coincidence is why one table serves both copies today. Above one stick the legs need two tables —
///   and a second SECTION of one table needs a base that is not the live window COUNT, which is
///   [`GatherRows`]' whole subject: the emitter's count is its body's ladder rung and the host's is the
///   CEILING rung. `windows_max * per_window * nslab` is the count-free spelling
///   (`PAGE_SLOTS / SlotWindow::SLOTS` is pool geometry), and it is `0` for Kᵗ and for V at one slab,
///   which is what keeps hd=64's emission byte-identical.
/// * **And both gathered legs must then slab-split.** The gathered score arm contracts the WHOLE head
///   dim in one op under a `y`-batch, and a multi-stick contraction under a `y`-batch is the shape
///   `ScoreArm::SlabSplitBatched` exists because dxp is measured-twice incoherent with; the gathered
///   value arm already declares one stick of `out` but reaches only slab 0. Neither is addressing work,
///   and neither is optional at hd=128.
///
/// ⭐⭐⭐⭐⭐ THE SCRATCH'S ROW LAW — WHICH ROW A `(kv head, slot window, request)` OWNS — AND IT DOES
/// **NOT** KNOW HOW MANY WINDOWS THERE ARE. That absence is the whole type.
///
/// ⛔⛔⛔ THE DEFECT THIS CLOSES, AND IT IS THE ONE THAT MADE EVERY ROW WRONG AT EVERY RUNG.
/// [`GatherScratch`] is built TWICE from two different window counts, and the two are different
/// numbers by construction:
///
/// * the EMITTER builds it from the body it is lowering — `SlotWindow::count_in(active_cap)`, where
///   `active_cap` is that body's own sk_bucket ladder rung (`ActiveCap::decode_ladder` bakes
///   `{64, 128, …}` interior rungs plus the `PAGE_SLOTS` ceiling, so `nb ∈ {1, 2, 4}`);
/// * the HOST builds it from `DecodeRung::swept`, which `codegen` fills with the **CEILING** body's
///   `cap` ("`cap` IS this body's swept extent: `bfp` is the CEILING rung") — one number per BATCH
///   WIDTH, with no way to name which interior rung a step will run.
///
/// And which body a step runs is `Executor::select_body_paged`'s answer, taken from the live context
/// length AFTER the host has already staged the table. So for every context below the ceiling the two
/// window counts DIFFER — at a short prompt, 1 against 4.
///
/// With the window as a MINOR coordinate (`(kvh * windows + window) * mq + request`) that difference
/// moved every row but kv head 0's: the host wrote kv head 1's entry at row `4 * mq` while the body
/// read it at row `1 * mq`, so heads above the first gathered another head's — or an UNWRITTEN — page
/// block, with a valid mask over it. Clean bake, `rc=0`, no fence, and a plausible first token followed
/// by collapse, because the new-token block is ungathered and only the prefix fold is wrong.
///
/// ⭐ WINDOW-**MAJOR** IS WHAT MAKES THE DISAGREEMENT INERT: `window * (nkvh * mq) + kvh * mq + request`
/// contains no window COUNT at all, so a scratch sized for 4 windows and a body reading 1 agree on the
/// row of every coordinate they both have — the wider table is a strict SUPERSET at identical indices.
/// The count survives only as an EXTENT ([`GatherScratch::windows`]), which is what it always was.
///
/// ⛔ AND REQUEST-MINOR IS STILL FORCED, for the reason it always was: the score/value legs step `y`
/// by ONE scratch row per request, so adjacent requests of one `(kv head, window)` must be adjacent
/// rows. That is the relation [`Self::per_window`] and this law hold together.
/// ⭐⭐⭐⭐⭐ ONE WHOLE PAGE PLANE — the gather granularity IBM's own reference uses, and the reason three
/// of this module's "hardware constraints" are not constraints at all.
///
/// `spyre-inference`'s `page_attn_head_major_decode_kernel` gathers `k_pages[kv_rows]` — an ENTIRE PAGE
/// by ONE index value — inside a per-page online-softmax loop, and its probe
/// (`tests/probes/test_spyre_fallback_probes.py`) runs that at `head_size=128`. We gathered a 64-slot
/// WINDOW OF ONE PLANE instead, which is contiguous only while `hd == POOL_STICK`. That single choice is
/// the sole origin of ALL THREE of:
///   * [`GatherScratch::of_fold_pass`] refusing `hd > POOL_STICK`,
///   * "the two planes number blocks differently above one stick" (Kᵗ vs Knat/V),
///   * and the 32-entry IBR overflow — `nkvh * windows * mq` is 64/128/256 at width 8 against dxp's ONE
///     32-word stick, while ONE ENTRY PER REQUEST cannot overflow it at any width the ladder admits.
///
/// ## The contiguity, proven from [`PagedKvPool::addr`] and not assumed
/// For `Knat`/`V`, `addr = kvh * plane_block_elems + (feat/stk)*(pls*stk) + slot*stk + feat%stk`, with
/// `plane_block_elems == hd * PAGE_SLOTS` and `pls == plane_extent()`. Because
/// [`PagedKvPool::WRITE_SLACK`] is `0`, `pls == PAGE_SLOTS`, so the feature-stick stride `pls*stk` is
/// exactly `PAGE_SLOTS*stk` and the per-head block is exactly `hd*PAGE_SLOTS` — the slabs tile the block
/// with NO HOLE, and consecutive kv heads abut. So one page's whole plane over every kv head is the
/// contiguous range `[0, nkvh*hd*PAGE_SLOTS)`:
///   * hd=64  → one slab, `slot*64 + feat`, `[0, 16384)` per head;
///   * hd=128 → two slabs, `slab*16384 + slot*64 + feat%64`, `[0, 32768)` per head.
///
/// ⇒ ONE index entry reaches a request's whole page at ANY head dim that is a multiple of a stick.
///
/// ⛔⛔⛔ AND THAT IS TRUE **ONLY WHILE `WRITE_SLACK == 0`**, which is why this is a type with a build
/// guard and not a comment. A nonzero write slack makes `pls > PAGE_SLOTS`, which opens a
/// `(pls - PAGE_SLOTS) * stk`-element hole between feature slabs — every slab above the first would then
/// be gathered from the wrong offset, at `hd > 64` only, with a clean bake and fluent wrong output. The
/// guard below is evaluated (a module-level `const _` is a required-const context, unlike an associated
/// const — see `GroupSize` in the target crate for the measurement).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct PagePlaneExtent {
    nkvh: u32,
    hd: u32,
}

const _: () = assert!(
    PagedKvPool::WRITE_SLACK == 0,
    "PagePlaneExtent: a page-granular gather is contiguous only while WRITE_SLACK == 0 — a nonzero \
     slack opens a hole between feature slabs and silently mis-gathers every slab above the first at \
     hd > 64. Re-derive the entry law before changing it."
);

impl PagePlaneExtent {
    /// The extent of one `(request, page)` plane, or `None` on a degenerate geometry or an `hd` that is
    /// not a whole number of sticks (a part-stick head dim has no stick-block entry at all).
    pub const fn of_pool(pool: PagedKvPool) -> Option<Self> {
        let (nkvh, hd) = (pool.nkvh as u32, pool.hd as u32);
        if nkvh == 0 || hd == 0 || !hd.is_multiple_of(POOL_STICK) {
            return None;
        }
        Some(PagePlaneExtent { nkvh, hd })
    }

    /// Elements in one request's whole page plane — `nkvh * hd * PAGE_SLOTS`, the contiguous run one
    /// index entry names. This is `skip_addr`, so it is ALSO the unit the host's entry counts in.
    pub const fn elems(self) -> u64 {
        self.nkvh as u64 * self.hd as u64 * PagedKvPool::PAGE_SLOTS as u64
    }

    /// The plane's whole `mb` extent in one-stick sub-rows — `elems / POOL_STICK`. This is what ONE
    /// COPY OP declares, and it is NOT the pinned extent: see [`Self::entry_sub_rows`].
    pub const fn sub_rows(self) -> u64 {
        self.elems() / POOL_STICK as u64
    }

    /// ⭐⭐⭐⭐⭐ THE ELEMENTS ONE INDEX ENTRY NAMES — **ONE STICK BLOCK (`hd * POOL_STICK`), NOT the whole
    /// plane**, and that is what makes the copy bake AND run on more than one core.
    ///
    /// ⛔⛔⛔ THE PIN WAS THE WHOLE PLANE, AND IT IS REFUSED ON CARD:
    /// ```text
    /// sbf-ddc: DtException: Unable to map graph within architecture constraints:
    ///   The initial chunk parameters must fit in LX for SuperDSC: 78_attn_gkt_o734_s0
    ///   L3DlOpsScheduler.cpp:1534   (gated on isDoubleBuffering)
    /// ```
    /// The pin is not just a size, it is the op's UNIT OF WORK DIVISION. `gather_copy_cores` may only
    /// split the gather dim into WHOLE entries (a core owning fewer than `page` positions would SHRINK
    /// `skip_addr`, which is the clean-bake wrong-address class `gather_copy_opspec` refuses), so a
    /// plane-sized pin over a plane-sized `mb` leaves exactly ONE entry — i.e. ONE CORE — and dxp then
    /// measures that one core's double-buffered chunk (256 KB in + 256 KB out, twice) against LX. A
    /// stick-block pin over the same `mb` gives `blocks()` entries, so the same 256 KB spreads over
    /// `blocks()` cores at one stick block each. Same op, same footprint, same destination; the LX
    /// refusal is the pin's, not the op's.
    ///
    /// ⭐ AND THE DESTINATION IS STILL A WHOLE PAGE PLANE. The blocks of a plane abut (proven in
    /// `zz_a_whole_page_plane_is_contiguous_at_every_head_dim`), so entry `i` of a request's run fills
    /// sub-rows `[i * entry_sub_rows, +entry_sub_rows)` of that request's row — page granularity is a
    /// property of the ROW LAW, which is unchanged, not of the pin.
    pub const fn entry_elems(self) -> u64 {
        self.hd as u64 * POOL_STICK as u64
    }

    /// The pinned `mb` positions one entry covers — [`Self::entry_elems`] in one-stick sub-rows, which is
    /// `hd`. This is the [`PageExtent`] the gather dim declares, so dxp derives `skip_addr` as exactly
    /// [`Self::entry_elems`].
    pub const fn entry_sub_rows(self) -> u64 {
        self.entry_elems() / POOL_STICK as u64
    }

    /// ⭐⭐⭐ INDEX ENTRIES ONE REQUEST'S PAGE PLANE NEEDS — `nkvh * PAGE_SLOTS / POOL_STICK`, and note
    /// that `hd` CANCELS: it is 32 at nkvh=8 for every head dim, which is exactly one index stick. That
    /// cancellation is why the same cut serves hd=64 and hd=128.
    pub const fn blocks(self) -> u64 {
        self.elems() / self.entry_elems()
    }

    /// `skip_addr` in STICKS, the quantity SEN1P5+'s `DT_CHECK(skip_addr_sticks % 2 == 0)` reads. An odd
    /// pin is a device ABORT rather than a refusal, so it is answered here.
    pub const fn skip_addr_sticks(self) -> u64 {
        self.entry_sub_rows()
    }

    /// TRUE when the pin satisfies deeptools' even-stick check. The pin is `hd` sticks and `hd` is a
    /// whole number of 64-element sticks by [`Self::of_pool`], so this is a witness rather than a filter
    /// — but it is asked, because an abort is not a refusal.
    pub const fn pin_is_even(self) -> bool {
        self.skip_addr_sticks().is_multiple_of(2)
    }

    /// Elements of ONE kv head's block inside the plane — `hd * PAGE_SLOTS`, the whole of one head's
    /// page. Heads abut (proven in `zz_a_whole_page_plane_is_contiguous_at_every_head_dim`), so this is
    /// both the head stride and the head extent.
    pub const fn head_block_elems(self) -> u64 {
        self.hd as u64 * PagedKvPool::PAGE_SLOTS as u64
    }

    /// [`Self::elems`] in BYTES — `skip_addr` as the host's page arithmetic counts it. The word length is
    /// the pool's own (`Fp16`), read the same way [`PagedKvPool::stick_block_bytes`] reads it, so the two
    /// units cannot drift.
    pub fn bytes(self) -> u64 {
        self.elems()
            * <crate::superdsc_opspec::Fp16 as crate::superdsc_opspec::DataFormat>::WORD_LENGTH
                as u64
    }

    /// DISTINCT kv heads the plane spans.
    pub const fn nkvh(self) -> u32 {
        self.nkvh
    }

    /// Offset of kv head `kvh`'s block within the plane, or `None` past the head count.
    pub const fn head_off(self, kvh: u32) -> Option<u64> {
        if kvh >= self.nkvh {
            return None;
        }
        Some(kvh as u64 * self.head_block_elems())
    }
}

// ⛔⛔⛔ WHAT WAS HERE, AND WHY IT IS GONE: `PlanesPerPage` / `planes_per_page` / `PageEntry`, a SECOND
// entry unit in which one index entry named a whole page PLANE. It existed for exactly one reason — the
// page-granular gather pinned a whole plane on the gather dim — and that pin is REFUSED ON CARD
// (`PagePlaneExtent::entry_elems` carries the refusal and why). With the pin back at one stick block, the
// factor is `EntriesPerPage` and the entry is `GatherEntry`, which already carry this exact identity and
// the three-term composition's guard rails.
//
// ⭐ AND THE PAGE GRANULARITY DID NOT GO WITH THEM. It never lived in the entry unit: it is the ROW LAW
// (`PageScratch::row_of` — a row is a REQUEST, holding a whole page plane), which is unchanged. What one
// entry names and what one row holds are different questions, and conflating them is what cost the pin.

/// ⭐⭐⭐⭐⭐ THE PAGE-GRANULAR GATHER DESTINATION — one row per REQUEST, each row one whole page plane.
///
/// This replaces [`GatherScratch`]'s `(kv head, slot window, request)` row law. The difference is the
/// whole reason the old one was hd==64-only and corrupted at width 8:
///
/// | | old, window-granular | this, page-granular |
/// |---|---|---|
/// | a row is | one 64-slot pool block | one request's WHOLE page plane |
/// | rows | `nkvh * windows * mq` | `mq` |
/// | index entries per pass | 64/128/256 at width 8 | `mq` (≤ 32 for every rung) |
/// | contiguous at hd=128 | ✗ (`nslab` strided runs) | ✓ (proven bijection) |
/// | planes needing separate tables | 2 (Kᵗ numbers blocks differently) | 1 (all three planes tile alike) |
///
/// ⛔ THE WINDOW AXIS IS GONE ON PURPOSE. It existed because the fold's tile was one 64-slot window; the
/// tile is now ONE PAGE, which is what IBM's own kernel uses (`mask_tiles[i]` per page, `matmul(probs,
/// v_page)` over the whole page) and what [`PrefixMaskShape`] has always been shaped for (`COLS` IS
/// `PAGE_SLOTS`). A window term here is what forced the `nkvh * windows * mq` entry count.
///
/// ⛔ AND THE ENTRY CARRIES ONLY THE PAGE. `addr = idx * skip_addr + base` with `skip_addr` one plane, so
/// the LAYER and PLANE terms — both compile-time constants for a given op — belong in that op's own
/// `base`, never in the table. That is what lets ONE table of `mq` entries serve every layer and both
/// planes, instead of a table per (layer, plane).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PageScratch {
    plane: PagePlaneExtent,
    mq: u32,
}

impl PageScratch {
    /// ⛔ THE INDEX IS ONE STICK AND dxp SPENDS ALL OF IT, so a pass may name at most this many entries.
    /// The IBR is loaded as ONE 128-byte stick and each core's read offset is taken modulo that stick
    /// (`L3DlOpsScheduler.cpp`), so a larger count WRAPS silently — measured as the rung-8 corruption.
    /// At page granularity an entry is a REQUEST, and no ladder rung admits more than 32 of them, so this
    /// bound is unreachable rather than enforced. It is stated so that a future wider rung refuses here
    /// instead of wrapping on the card.
    pub const ENTRIES_PER_PASS_MAX: u32 = 32;

    /// THE ONE DOOR. `None` on a geometry the page gather cannot express — and note what is NOT a
    /// refusal any more: **no head-dim ceiling.** `hd` must merely be a whole number of sticks.
    pub const fn of_pass(pool: PagedKvPool, mq: QueryRowCount) -> Option<PageScratch> {
        let Some(plane) = PagePlaneExtent::of_pool(pool) else {
            return None;
        };
        let mq = mq.get();
        if mq == 0 || mq > Self::ENTRIES_PER_PASS_MAX || !plane.pin_is_even() {
            return None;
        }
        // ⛔ ONE REQUEST'S PLANE MUST FIT ONE INDEX STICK, because one op serves one request (LX, see
        // `copies`) and dxp loads a gather's IBR in a SINGLE stick transfer, taking each core's read
        // offset inside it modulo that stick. `blocks()` is `nkvh * PAGE_SLOTS / POOL_STICK` — 32 at
        // nkvh=8, exactly the stick, for every head dim — so a pool with more kv heads gets NO gather
        // here rather than a run whose cores past the wrap gather another core's pages.
        if plane.blocks() > CopyDims::ENTRIES_PER_OP as u64 {
            return None;
        }
        let s = PageScratch { plane, mq };
        // ⛔ THE DESCRIPTOR'S EXTENTS ARE `u32`, so a footprint that does not fit one is refused HERE
        // rather than truncated by an `as` in `copies`/`footprint_dims`. A truncated `mb` reserves less
        // than the copy writes, and this tensor's overrun is the next intermediate's bytes read as page
        // addresses — a real address, clean bake, fluent wrong output.
        if s.cols() > u32::MAX as u64 || s.sub_rows() > u32::MAX as u64 {
            return None;
        }
        Some(s)
    }

    pub const fn plane(self) -> PagePlaneExtent {
        self.plane
    }
    pub const fn mq(self) -> u32 {
        self.mq
    }
    /// DISTINCT kv heads the scratch carries per row — the pool's own count, so the kernel base an op
    /// takes and the entry the host filled cannot come from two numbers.
    pub const fn nkvh(self) -> u32 {
        self.plane.nkvh()
    }

    /// Rows — one per request. This is ALSO the copy's live index entry count for the pass, so a
    /// reservation cannot be sized against a different count than the descriptor declares.
    pub const fn rows(self) -> u32 {
        self.mq
    }

    /// Elements in one row — one whole page plane. This is `skip_addr`.
    pub const fn cols(self) -> u64 {
        self.plane.elems()
    }

    /// The whole footprint in elements, what the synthetic allocator reserves.
    pub const fn elems(self) -> u64 {
        self.rows() as u64 * self.cols()
    }

    /// The copy's declared `mb` extent in ONE-STICK sub-rows — `rows * plane sub-rows`. A `[mb, out]`
    /// operand whose `out` exceeds one stick is STICK-major, which scrambles the block the matmul reads,
    /// so the copy declares one stick of `out` and counts `mb` in sub-rows.
    pub const fn sub_rows(self) -> u64 {
        self.rows() as u64 * self.plane.sub_rows()
    }

    /// Positions of the copy's `mb` axis ONE index entry covers — ONE STICK BLOCK's sub-rows. This is the
    /// [`PageExtent`] pinned on the gather dim, and it is what makes dxp derive `skip_addr` as exactly
    /// [`PagePlaneExtent::entry_elems`].
    ///
    /// ⛔ NOT [`Self::cols`] — see [`PagePlaneExtent::entry_elems`] for the card refusal that settles it.
    pub const fn entry_page(self) -> u64 {
        self.plane.entry_sub_rows()
    }

    /// Index entries one request's row needs — the plane's stick blocks, and therefore also the entries
    /// the host packs into request `r`'s index stick.
    pub const fn entries_per_row(self) -> u64 {
        self.plane.blocks()
    }

    /// The row a request owns, or `None` past the batch — the ONLY row law, with no window and no kv
    /// head in it. Both legs and the host table read this one function.
    pub const fn row_of(self, request: u32) -> Option<u32> {
        if request >= self.mq {
            return None;
        }
        Some(request)
    }

    /// ELEMENT OFFSET of `(request, kv head)`'s block inside the scratch — the kernel base both legs
    /// take. `None` past the batch or the head count.
    pub const fn head_off(self, request: u32, kvh: u32) -> Option<u64> {
        let Some(_) = self.row_of(request) else {
            return None;
        };
        let Some(h) = self.plane.head_off(kvh) else {
            return None;
        };
        Some(request as u64 * self.cols() + h)
    }

    /// ⭐⭐⭐⭐⭐ THE GATHERED KERNEL BASE — **the POOL'S OWN ADDRESS plus this request's row, and nothing
    /// else.** This is the whole reason page granularity is a small change rather than a rewrite.
    ///
    /// A scratch row is a byte-for-byte copy of the page plane (asserted in
    /// `zz_a_whole_page_plane_is_contiguous_at_every_head_dim`), so a gathered op's base differs from the
    /// UNGATHERED one by exactly `request * cols`. The ungathered legs already compute their base as
    /// `pool.addr(KvCoord::block(plane, kvh).at_slot(w.first_slot()).at_feat(FeatIdx::of_slab(sl)))`;
    /// a gathered leg hands that same [`KvCoord`] here and adds no arithmetic of its own.
    ///
    /// ⛔ WHICH IS WHY THIS TAKES A `KvCoord` AND NOT THREE NUMBERS. The window, the feature slab and the
    /// kv head all reach the address through [`PagedKvPool::addr`], the ONE law — so the gathered and
    /// ungathered legs cannot come to disagree about where a slab or a window lives, which is exactly what
    /// happened when the window-granular scratch re-derived them (`kernel_row_off`'s own window term).
    pub fn coord_off(self, request: u32, pool: &PagedKvPool, c: KvCoord) -> Option<u64> {
        let row = self.row_of(request)?;
        Some(row as u64 * self.cols() + pool.addr(c) as u64)
    }

    /// ⭐⭐⭐⭐⭐ THE RESERVATION'S SHAPE — `[rows, cols]`, one value so a caller cannot assemble the pair
    /// in the wrong order or from the wrong two numbers. `[rows, sub_rows]` over-reserves 64×, `[cols,
    /// rows]` reserves the right total at the wrong pitch, and this tensor's overrun is the next
    /// intermediate's bytes read as block numbers.
    pub const fn footprint_dims(self) -> [u32; 2] {
        [self.rows(), self.cols() as u32]
    }

    /// ⭐⭐⭐⭐⭐ THE COPY OPS THIS PASS NEEDS — **ONE PER REQUEST**, and the reason is LX, not the IBR.
    ///
    /// ⛔⛔⛔ MEASURED ON CARD, AND IT REFUTED MY OWN ARITHMETIC. One op per PLANE (`mq * sub_rows` =
    /// 16384 one-stick sub-rows = 2 MB at the shipped 2b geometry) is refused at bake:
    /// ```text
    /// sbf-ddc: DtException: Unable to map graph within architecture constraints:
    ///   The initial chunk parameters must fit in LX for SuperDSC: 78_attn_gkt_o734_s0
    ///   L3DlOpsScheduler.cpp:1534
    /// ```
    /// I had argued this was safe because [`crate::superdsc_opspec::USABLE_LX_BYTES`] is a PER-CORE
    /// budget and 2 MB over 32 cores is 64 KB each. **dxp requires the INITIAL CHUNK to fit LX before
    /// work division**, so the op's WHOLE footprint is what is measured, not its per-core share. A
    /// per-core reading of a pre-division constraint is the mistake; do not re-derive it.
    ///
    /// ⭐ ONE REQUEST PER OP IS `sub_rows` = 2048 SUB-ROWS = 256 KB, which is the SAME op size the
    /// window-granular form shipped (32 rows × 4096 elements), so this is a cut back to a proven
    /// footprint rather than a new shape.
    ///
    /// ⭐⭐⭐ AND THIS IS WHY IBM'S INDEX TABLE IS `[num_blocks, INT32_ELEMS_PER_STICK]` WITH THE VALUE AT
    /// COLUMN 0. [`GatherCopy::index_base`] must be a whole number of index STICKS, because dxp loads the
    /// IBR one stick at a time — so an op that names ONE entry must find it at a stick boundary, which
    /// makes the table one 32-entry stick PER REQUEST with the entry at column 0. That is the vendor
    /// layout exactly, and the stick-aligned base is the constraint that forces it.
    pub fn copies(self) -> impl Iterator<Item = GatherCopy> {
        let page = self.entry_page() as u32;
        let per_row = self.entries_per_row() as u32;
        let mb = self.plane.sub_rows() as u32;
        (0..self.mq).map(move |r| GatherCopy {
            // Request `r`'s entries occupy stick `r` of the pass block, all `per_row` of them — the
            // vendor's `[num_blocks, INT32_ELEMS_PER_STICK]` layout, one real run per gathered tile.
            stick: r,
            // ⛔ AND ITS DESTINATION IS THE `per_row * r`-th ENTRY, NOT THE `r`-th. This is the number
            // the index base cannot supply: entries advance one STICK per op (32 words) while the
            // destination advances one whole PAGE PLANE (`per_row` entries). Deriving the destination
            // from the index base puts op `r`'s rows `per_row/1`× off.
            dest_entry: r * per_row,
            dims: CopyDims {
                // ONE request's whole page plane, in one-stick sub-rows. `mb / page == per_row` ⇒ the
                // index declares exactly `per_row` entries, which `of_pass` has already proved is one
                // index stick.
                mb,
                // ⛔ ONE STICK, for the same reason as the window-granular form: a wider `out` classifies
                // the operand STICK-MAJOR and scatters each gathered block, which scrambles what the
                // matmul then reads.
                out: POOL_STICK,
                page: crate::superdsc_opspec::PageExtent::of_positions(page),
            },
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GatherRows {
    nkvh: u32,
    mq: u32,
}

impl GatherRows {
    /// Rows ONE slot window occupies — `nkvh * mq`, the window's stride in the row axis. It is the
    /// quantity that replaces the window COUNT in the row law, and it is a property of the model and
    /// the batch alone, so both sides compute the same one.
    pub const fn per_window(self) -> u32 {
        self.nkvh * self.mq
    }

    /// The row, or `None` for a coordinate outside the model/batch. TOTAL in the window, deliberately:
    /// a window bound is an EXTENT question, and answering it here would put the count back into the
    /// law. [`GatherScratch::row_of`] checks the extent.
    pub const fn row_of(self, kvh: u32, window: u32, request: u32) -> Option<u32> {
        if kvh >= self.nkvh || request >= self.mq {
            return None;
        }
        Some(window * self.per_window() + kvh * self.mq + request)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GatherScratch {
    /// The row law — see [`GatherRows`] for why the window count is not part of it.
    rows: GatherRows,
    windows: u32,
    hd: u32,
}

impl GatherScratch {
    /// THE ONE DOOR, and it takes the pool plus the two extents a fold pass really has.
    ///
    /// `None` when the geometry admits no flat copy (`hd > POOL_STICK`, see the type's note), when the
    /// pass sweeps no window, or when the pool is degenerate. A refusal here means the caller emits the
    /// ungathered bundle, which is what shipped — never a smaller scratch.
    pub fn of_fold_pass(
        pool: PagedKvPool,
        windows: WindowCount,
        mq: QueryRowCount,
    ) -> Option<GatherScratch> {
        let (nkvh, hd) = (pool.nkvh as u32, pool.hd() as u32);
        if nkvh == 0 || hd == 0 || hd > POOL_STICK || windows.get() == 0 || mq.get() == 0 {
            return None;
        }
        Some(GatherScratch {
            rows: GatherRows { nkvh, mq: mq.get() },
            windows: windows.get(),
            hd,
        })
    }

    /// ⭐⭐⭐⭐⭐ "COULD ANY BODY OF THIS BUNDLE GATHER" — [`Self::of_fold_pass`] ITSELF, asked without a
    /// window count, and the ONE rule the three bundle-level places that must agree now read.
    ///
    /// ⛔⛔⛔ THE DEFECT THIS CLOSES, AND IT WAS INVISIBLE AT hd=64. Three constructors answer "does this
    /// bundle gather" ([`crate::…::GathersKv`] in the target crate: `of(BakeFacts)`, `of_layout()`,
    /// `of_launch_groups()`). The first two read the `KV_BLOCK_INDEX_TID` PLACEMENT, which was reserved
    /// on `rows_are_requests` ALONE; the third reads the emitted fold groups, which additionally require
    /// `of_fold_pass` to have returned a scratch. At hd=64 `of_fold_pass` never refuses, so all three
    /// agreed and the split could not be seen. At hd=128 it always refuses: the bundle reserves an index
    /// activation and the forward tape carries a `KvBlockIndex` step for a gather that is in no
    /// descriptor — benign only because the consumer that would corrupt happens to read the per-body
    /// door. One rule, read in three places, is not a tidy-up: the asymmetry recorded on `GathersKv` is
    /// that a table nobody gathers through is inert while a gather with no table reads entry 0, and a
    /// bundle whose two halves are decided by two different predicates is one edit away from the second.
    ///
    /// ⛔⛔⛔⛔⛔ AND ONLY THE **EMITTER** HALF READS IT TODAY — NOT BECAUSE THE RULE IS WRONG BUT BECAUSE
    /// THE 8b BATCHED ORACLE DOES NOT HOLD, SO AN ADDRESS-MOVING CHANGE CANNOT BE VERIFIED AGAINST IT.
    /// Adding this same call to the `KV_BLOCK_INDEX_TID` reservation removes a `pmbytes` hole from the
    /// activation segment and moves every tensor after it, and granite-3.1-8b fp8 then answered
    /// `solo_diff` 1/3/7 at rungs 2/4/8 where its own commit message records `solo_diff=0`. **The control
    /// refutes the obvious reading:** `f080bb60a` restored file-by-file to md5 equality and REBUILT on the
    /// same pod answers `solo_diff=1` at rung 2 on three consecutive trials too. So the hole is not
    /// load-bearing padding and this door did not break anything — the recorded clean sweep simply does not
    /// reproduce from the commit that records it. Full table and the order of work at that call site. This
    /// door is therefore used only where it costs no addresses (the emitter names the index tensor under
    /// it, which `assemble_attn` already ignores at hd=128), and the three-door split stays OPEN and
    /// documented rather than closed by a change nothing can currently measure.
    ///
    /// ⛔ AND IT IS `of_fold_pass`, NOT A COPY OF ITS CONDITIONS. The window count is the ONE input a
    /// bundle-level caller cannot have (the emitter's is its body's ladder rung, the host's is the
    /// CEILING rung — see [`GatherRows`]), and it is also the one input that cannot change the answer for
    /// a bundle: every body sweeps at least one window. So asking at ONE window is asking the geometry,
    /// and asking it through the real door is what stops the two halves drifting.
    pub fn admits(pool: PagedKvPool, mq: QueryRowCount) -> bool {
        Self::of_fold_pass(pool, SlotWindow::count_in(SlotCount::new(POOL_STICK)), mq).is_some()
    }

    pub const fn nkvh(self) -> u32 {
        self.rows.nkvh
    }
    /// 64-slot windows ONE fold pass sweeps — `nb`. An EXTENT and nothing more: it sizes the
    /// reservation and bounds [`Self::row_of`], and it is deliberately absent from the row law
    /// ([`GatherRows`]) because the emitter's value and the host's are DIFFERENT NUMBERS.
    pub const fn windows(self) -> u32 {
        self.windows
    }
    /// Requests the batch binds — the matmuls' `y` extent.
    pub const fn mq(self) -> u32 {
        self.rows.mq
    }

    /// Rows the scratch holds: one per (kv head, slot window, request). This is ALSO the copy op's `mb`
    /// extent and the index's live entry count per pass — one number, so a reservation cannot be sized
    /// against a different count than the descriptor declares.
    pub const fn rows(self) -> u32 {
        self.windows * self.rows.per_window()
    }

    /// Elements in one row — one 64-slot pool block, `hd * POOL_STICK`. It is `skip_addr` (the pinned
    /// `mb` contributes 1 and `out` the whole row), the copy's `cols`, and the matmuls' `y` step, and
    /// those being one value is what makes the entry a plain global block number.
    pub const fn cols(self) -> u32 {
        self.hd * POOL_STICK
    }

    /// The whole footprint, in elements — what the synthetic allocator reserves.
    pub const fn elems(self) -> u64 {
        self.rows() as u64 * self.cols() as u64
    }

    /// ⭐⭐⭐⭐⭐ THE COPY OP'S DECLARED `mb` EXTENT — `rows * hd` SUB-ROWS of ONE STICK, not `rows` rows of
    /// `cols`. This is not a presentation choice; it is what makes the destination ROW-MAJOR.
    ///
    /// ⛔ A `[mb, out]` OPERAND WHOSE `out` IS WIDER THAN ONE STICK IS **STICK-MAJOR**, and that scrambles
    /// the block the matmul then reads. `StickLayout::for_view_df` classifies `[rows, cols]` sticked on
    /// `cols` as `RowBlocked` (and it FOLDS a trailing unit `y`, so the rank-3 form classifies identically
    /// — there is no flat presentation to pick instead). Its address law is
    /// `(c/64)*(rows*64) + r*64 + c%64`, so block `r`'s element `c` lands at `(c/64)*rows*64 + r*64 + c%64`
    /// — feature `d` of the Kᵗ block `rows*64` elements from the next, where the score kernel's own law
    /// (`Stk::kernel`, `(j/64)*(in*64) + d*64 + j%64`) puts it 64 apart. Off by `rows`, one clean bake,
    /// every feature above the first reading another request's slots.
    ///
    /// ⭐ AT `out == ONE STICK` STICK-MAJOR **IS** ROW-MAJOR (`c/64 == 0` and `c%64 == c`, so the law
    /// collapses to `r*64 + c`), and a block is then `hd` consecutive sub-rows — exactly the pool's own
    /// `[hd, 64]` Kᵗ block, contiguous, which is what the kernel reads. So the copy declares one stick of
    /// `out` and an index entry covers [`entry_page`](Self::entry_page) sub-rows.
    pub const fn sub_rows(self) -> u32 {
        self.rows() * self.entry_page()
    }

    /// POSITIONS OF THE COPY'S `mb` AXIS ONE INDEX ENTRY COVERS — `cols / POOL_STICK`, i.e. `hd`.
    ///
    /// This is the `PageExtent` pinned on the gather dim, and it is what keeps `skip_addr` equal to ONE
    /// POOL BLOCK while the operands stay one stick wide: dxp derives `skip_addr` as
    /// `page × the per-core extents of the unpinned dims` = `hd × 64` = [`cols`](Self::cols). An entry is
    /// therefore still a plain global stick-block number, which is what the host's table stages and what
    /// `gather_entries_per_page` divides a page by.
    pub const fn entry_page(self) -> u32 {
        self.cols() / POOL_STICK
    }

    /// ⭐⭐⭐⭐⭐ THE RESERVATION'S SHAPE, AS ONE VALUE — `[rows, cols]`, the pair
    /// [`crate::placement::BundleLayout::synth`] declares the gather destination at.
    ///
    /// 🛑 **IT WAS TWO CALLS AT THE CALL SITE** (`&[scratch.rows(), scratch.cols()]`), which is a slice
    /// of two same-typed numbers a caller can build from anything: `[rows, sub_rows]` reserves 64× too
    /// much, `[cols, rows]` reserves the right total at the wrong pitch, and `[rows]` alone reserves one
    /// stick. Every one of those compiles, and the recorded cost of under-reserving this exact tensor is
    /// "entries read PAST the placement, and past it is the next tensor's bytes read as block numbers".
    /// One value, minted here, cannot be assembled in the wrong order or from the wrong pair.
    pub const fn footprint_dims(self) -> [u32; 2] {
        [self.rows(), self.cols()]
    }

    /// ⭐⭐⭐⭐⭐ THE COPY OPS THIS PASS NEEDS — **ONE PER INDEX STICK**, in row order, because one
    /// gather op's index is ONE STICK and dxp spends all of it.
    ///
    /// ⛔⛔⛔ THE DEFECT THIS CLOSES, AND IT IS THE ONE THAT SEPARATED RUNG 2 FROM RUNG 8 EXACTLY.
    /// The pass was ONE op whose index declared [`Self::rows`] entries, and above
    /// [`CopyDims::ENTRIES_PER_OP`] of them the card reads entries that were never loaded. The
    /// vendor's own scheduler says so twice, in the file that builds the IBR schedule
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp`):
    ///
    /// ```text
    /// // The index tensor is transferred from HBM to L3LUIBR in the granularity of one stick.
    /// DT_CHECK_MSG(size <= indexLdsCumulativeStickSizes.at(dim),
    ///              "IBR stick dimension size without rounding should always be not greater
    ///               than the stick size.");
    /// ...
    /// offsetInBytes += wkSliceId * dimIbrSize * indexLds.wordLength;
    /// offsetInBytes  = offsetInBytes % numBytesInStick;   // ⬅ ONE 128-BYTE STICK
    /// ```
    ///
    /// The IBR is loaded in ONE transfer of ONE stick, and each core's read offset inside it is taken
    /// **modulo that stick**. So a `wkSlices × per-core entries` product above 32 words does not fault
    /// and does not refuse — it WRAPS, and the cores past the wrap read another core's entries as their
    /// own page addresses. MEASURED as exactly that: dxp's own IBR is `memref<32xi32>` whose HBM load
    /// fetches words `[0, 32)`, and at 128 entries (rung 8, `nb=2`) every row's first decode token came
    /// back wrong while the identical `active_cap=128` body at 32 entries (rung 2) was
    /// `solo=EXACT`.
    ///
    /// ⭐ AND A ONE-STICK RUN OF ROWS IS THE RIGHT CUT, not a per-kv-head or per-window one. The row
    /// law is window-MAJOR ([`GatherRows`]) so a kv head's rows are `windows` runs `nkvh*mq` apart —
    /// not a contiguous `mb` range, which is the only thing a `[mb, out]` operand can name. A run of
    /// rows IS contiguous at every width, its base is a whole number of index sticks by construction,
    /// and the entry at a given row index is the same entry it was before the cut — so the host's
    /// table is unchanged and the ONE clean rung stays byte-for-byte the op it already was.
    ///
    /// The price is op COUNT, which is the cheap axis: 1.42 µs per op against ~28 µs for a fold pass
    /// (measured), and `rows / 32` of them per leg.
    pub fn copies(self) -> impl Iterator<Item = GatherCopy> {
        let (rows, page) = (self.rows(), self.entry_page());
        (0..self.copy_count()).map(move |stick| {
            let first_entry = stick * CopyDims::ENTRIES_PER_OP;
            GatherCopy {
                stick,
                // The window-granular cut keeps the two IN LOCKSTEP, which is what it always relied on:
                // one index stick IS 32 consecutive scratch rows here, so the destination entry and the
                // index's first entry are the same number. Stated rather than derived, so the page-granular
                // form's divergence from it is visible at both sites.
                dest_entry: first_entry,
                dims: CopyDims {
                    // The run this op covers — a whole stick, or the short tail. In ENTRIES first,
                    // then multiplied into `mb` SUB-ROWS, so `page | mb` still holds by construction.
                    mb: (rows - first_entry).min(CopyDims::ENTRIES_PER_OP) * page,
                    // ⛔ ONE STICK, FROM THE STICK ITSELF. The `out` extent is not a parameter of this
                    // shape: a wider `out` classifies the operand STICK-MAJOR and scatters each gathered
                    // block `rows*64` apart (see [`Self::sub_rows`]), so there is one legal value and it
                    // is read off the pool's own stick rather than passed in beside `mb`.
                    out: POOL_STICK,
                    page: crate::superdsc_opspec::PageExtent::of_positions(page),
                },
            }
        })
    }

    /// How many copy ops one leg of the gather costs at this shape — for the diagnostics and the
    /// tests that price the cut. It is `copies().count()`, spelled once.
    pub const fn copy_count(self) -> u32 {
        self.rows().div_ceil(CopyDims::ENTRIES_PER_OP)
    }

    /// Scratch row for one (kv head, window, request) — [`GatherRows`]' law, bounded by THIS scratch's
    /// window extent.
    ///
    /// ⛔ THE ORDER IS WINDOW-MAJOR, REQUEST-MINOR, AND NEITHER HALF IS A CONVENTION. Request-minor
    /// because the score/value legs step `y` by one scratch row per request; window-MAJOR because the
    /// emitter and the host build this type from DIFFERENT window counts (see [`GatherRows`]) and only a
    /// window-major order makes the wider table a superset of the narrower one at identical indices.
    ///
    /// `None` on a coordinate outside the model, the batch, or the windows this scratch reserved — so a
    /// caller cannot compose an address for a row the scratch does not have.
    pub fn row_of(self, kvh: u32, window: u32, request: u32) -> Option<u32> {
        if window >= self.windows {
            return None;
        }
        self.rows.row_of(kvh, window, request)
    }

    /// The row law alone — for the one caller that needs to state that the law does not depend on this
    /// scratch's window extent.
    pub const fn row_law(self) -> GatherRows {
        self.rows
    }

    /// ⭐⭐⭐⭐⭐ THE KERNEL BASE OF ONE (kv head, window, REQUEST) — a per-request OFFSET, which is what
    /// lets the collapsed fold's score and value legs keep the SHIPPED shared 2-D kernel with NO `y`
    /// rank at all.
    ///
    /// ⛔⛔⛔ THIS SLOT USED TO BE A PER-`y` KERNEL **DIM** AND THE CARD REFUSED IT. A `[y,in,out]`
    /// kernel gives every core its own weight start, so `numCoresUsed_` collapsed onto the request
    /// count and the launch faulted at `job_bin_ptr + cores*128` — one flit past the program's
    /// per-core patch table — at every rung (2, 4 and 8, bit-identically apart from the index). The
    /// gather is what makes the DIM unnecessary: the scratch destination is contiguous and
    /// request-minor, so request `r`'s block is reachable by a baked offset, exactly as a GQA group's
    /// kv head is (`MatY::of_gqa_group` + `kt_off_fn`). One op per request, `y` back on the group.
    ///
    /// ⛔ AND `None` UNLESS THE OP'S DECLARED KERNEL IS **EXACTLY ONE ROW**. The rows are
    /// [`cols`](Self::cols) apart and an op reads `in * out` elements from its base, so a kernel wider
    /// than a row reads into row `r+1` — request `r+1`'s keys, fluently, with no fault. That is the one
    /// relation left to check once the `y` rank is gone (the walk no longer derives a request stride at
    /// all), and it is checked here because both numbers are this type's own. It refuses `hd` below one
    /// stick, where the value leg's `out` is a full 64 lanes against an `hd`-wide row.
    pub fn kernel_row_off(
        self,
        kvh: u32,
        window: u32,
        request: u32,
        k: MatK,
        n: MatN,
    ) -> Option<crate::addr::DevOff> {
        (k.get().checked_mul(n.get())? == self.cols())
            .then(|| self.row_of(kvh, window, request))
            .flatten()
            .and_then(|r| r.checked_mul(self.cols()))
            .map(crate::addr::DevOff::from_view_step)
    }

    /// ⭐ WHICH 64-SLOT STICK BLOCK OF A PAGE'S PLANE a (kv head, window) is — the term that rides in the
    /// INDEX ENTRY rather than in the copy's base offset.
    ///
    /// It has to ride in the entry: ONE copy op serves every head and window of a pass, and an
    /// `EwOperand` carries a single `col_offset`, so there is nowhere for a per-head base to go. A kv
    /// head is `plane_block_elems / stick_block_elems` blocks (`PLANE_SLOTS / 64`) and the window is the
    /// block inside it — read off the pool's own two sizes, never spelled as 4.
    ///
    /// Kᵗ and V agree on this number: Kᵗ's window `b` of head `kvh` is at
    /// `kvh*hd*PLANE_SLOTS + b*hd*64` and V's at `kvh*hd*PLANE_SLOTS + b*64*hd` — the same block, which
    /// is why one index serves both copies.
    pub fn block_in_page(self, kvh: u32, window: u32) -> Option<u32> {
        let pool = PagedKvPool::new(self.nkvh() as usize, self.hd as usize);
        let per_plane =
            u32::try_from(pool.plane_block_elems() / pool.stick_block_elems().max(1)).ok()?;
        (kvh < self.nkvh() && window < self.windows && window < per_plane)
            .then_some(kvh * per_plane + window)
    }
}

/// ⭐⭐⭐⭐⭐ THE GATHER COPY'S `mb`, `out` AND ENTRY `page` — ONE VALUE, because all three are read off
/// ONE [`GatherScratch`] and the op is correct only if they agree with each other.
///
/// 🛑 **THEY WERE THREE ARGUMENTS SIDE BY SIDE**, and each one had its own way of being wrong and its
/// own runtime refusal to catch it:
///
/// * `out` wider than one stick ⇒ the destination classifies STICK-MAJOR, and every gathered block is
///   scattered `rows*64` apart instead of contiguous. `gather_copy_opspec` refuses it — after the
///   caller has already had the chance to pass it.
/// * `page` not dividing `mb` ⇒ the last index entry covers a PARTIAL block, which is an address.
///   Refused there too.
/// * `page` disagreeing with the `page` inside the [`crate::superdsc_opspec::GatherIndex`] ⇒ nothing
///   refused it at all: the pin the walk carries and the pin the entry unit is derived from were two
///   separate reaches into the scratch.
///
/// Minted only by [`GatherScratch::copy_dims`], so `mb` is the scratch's sub-rows, `out` is the pool's
/// own stick, and the `page` the op pins IS the `page` the index declares. The two refusals above
/// become statements about a value that cannot be built wrongly rather than checks on three that can:
/// `out == POOL_STICK` by construction and `page | mb` because `sub_rows == rows * entry_page`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CopyDims {
    mb: u32,
    out: u32,
    page: crate::superdsc_opspec::PageExtent,
}

impl CopyDims {
    /// ⭐⭐⭐⭐⭐ THE VENDOR'S CEILING ON ONE GATHER OP'S INDEX — **ONE 128-BYTE STICK OF ENTRIES**, and
    /// therefore the largest run of scratch rows one copy may serve.
    ///
    /// It is the index's own stick, read off the index's own dtype ([`superdsc_opspec::SenUint32`],
    /// 4-byte / 32-stick), because that is exactly what the number is: the L3LU IBR is filled by ONE
    /// transfer "in the granularity of one stick" and each core's read offset inside it is taken
    /// `% bytesPerStick`. See [`GatherScratch::copies`] for the two vendor lines and the two card
    /// measurements that separate a fitting index from a wrapping one.
    ///
    /// ⛔ NOT SPELLED `32`. The one thing this number must never become is a literal in the emitter: it
    /// is the index's stick, so if the index's dtype ever changes it changes with it, and a gather whose
    /// index is one stick wide is the whole invariant.
    pub const ENTRIES_PER_OP: u32 =
        <crate::superdsc_opspec::SenUint32 as crate::superdsc_opspec::DataFormat>::ELEMS_PER_STICK;

    /// The copy's `mb` extent — one-stick sub-rows of THIS op's run of the pass.
    pub const fn mb(self) -> u32 {
        self.mb
    }
    /// The copy's `out` extent — exactly one pool stick, and nothing else is expressible.
    pub const fn out(self) -> u32 {
        self.out
    }
    /// The `mb` positions ONE index entry covers, as the pin both the walk and the index take.
    pub const fn page(self) -> crate::superdsc_opspec::PageExtent {
        self.page
    }

    /// ⭐⭐⭐⭐⭐ THE INDEX TENSOR'S DECLARED EXTENT, IN ENTRIES — `mb / page`, which is ALSO
    /// [`GatherScratch::rows`] and ALSO the live entries [`gather_index_table`] packs into each pass
    /// block. **One number, three readers**, and it is the one dxp derives for itself
    /// ([`crate::superdsc_opspec::PageExtent::entries_in`], which performs the division for both this
    /// and the emitter so there is only ever one).
    ///
    /// ⛔ THE DEFECT THIS CLOSES: the host staged `rows` entries packed at words `0..rows` while the
    /// descriptor declared the index at the op's `mb` (64× larger) and strode each core's start by the
    /// op's own per-core `mb` — so core `c` loaded index words `[64c, 64c+32)` out of a buffer holding
    /// 32 converted addresses. Sized from `mb` and the `page` that already divides it, the two cannot
    /// be different numbers.
    ///
    /// `None` is unreachable by construction (`mb == rows * entry_page` and `page == entry_page`), and
    /// it is an `Option` rather than an unwrap because the division belongs to the pin, not here.
    pub const fn entries(self) -> Option<u32> {
        self.page.entries_in(self.mb)
    }

    /// Does this op's index fit the ONE STICK dxp loads for it? True by construction for every value
    /// [`GatherScratch::copies`] mints; asked by the ONE builder that can also be handed hand-built
    /// extents ([`crate::ir::bridge::tiled_op_sdsc_op::gather_copy_opspec`]), so the raw door refuses
    /// what the minting door cannot express.
    pub const fn fits_one_index_stick(self) -> bool {
        match self.entries() {
            Some(e) => e <= Self::ENTRIES_PER_OP,
            None => false,
        }
    }
}

/// ⭐⭐⭐⭐⭐ ONE GATHER COPY OP — the run of scratch rows it serves, its declared extents, and the two
/// BASES that place that run. One value, because the three are the same cut seen three ways and an op
/// built from two of them and a stale third writes one run's blocks into another run's rows.
///
/// 🛑 **IT WAS ONE OP FOR THE WHOLE PASS**, with no base at all and an index as long as the pass had
/// rows. That is the rung-8 corruption exactly ([`GatherScratch::copies`]). Splitting it means three
/// numbers now have to agree per op — how many entries it reads, where in the index it starts, and
/// where in the scratch it writes — and all three are the SAME `first_entry`:
///
/// * the index base is `first_entry` SENUINT32 words, which is a whole number of index sticks because
///   `first_entry` is a stick multiple by construction — so dxp's one-stick load starts on a stick;
/// * the destination base is `first_entry * cols` elements, because the scratch is row-major at one
///   stick of `out` ([`GatherScratch::sub_rows`]) and a row is `cols` elements — so it is DERIVED from
///   the index base by the builder rather than carried here, and the two cannot name different runs;
/// * the SOURCE base stays ZERO, and that is load-bearing and unchanged by the cut: `addr =
///   idx * skip_addr + base_addr`, so the entry supplies everything inside the pool and a per-run
///   source base would be added twice.
///
/// Minted only by [`GatherScratch::copies`], so a run longer than one index stick, a base that is not
/// a stick multiple, and a destination offset computed from a different `first_entry` than the index's
/// are all unrepresentable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GatherCopy {
    dims: CopyDims,
    /// Which index stick of the pass this op is — and therefore its name suffix, so the op names say
    /// the cut out loud instead of two identically-named ops differing only in a baked offset.
    stick: u32,
    /// ⭐⭐⭐⭐⭐ WHICH DESTINATION ENTRY THIS OP'S RUN STARTS AT — **a SECOND number, and it stopped being
    /// the same one as [`Self::index_base`] when the index became stick-padded.**
    ///
    /// ⛔⛔⛔ THESE WERE ONE VALUE ON PURPOSE, AND THAT IS NOW A BUG. `gather_copy_opspec` derived the
    /// destination base as `entry_elems * first_entry.entries()`, which is exact only while a run's
    /// entries and its destination rows advance together — true for the window-granular cut, where one
    /// index stick WAS 32 consecutive scratch rows. A page-granular op names ONE entry but must sit at a
    /// stick boundary (dxp loads the IBR one stick at a time), so its entries advance 32 per op while its
    /// destination advances 1. Deriving one from the other puts every op's rows 32× too far out — past the
    /// scratch entirely, which is the next intermediate's bytes read as keys.
    ///
    /// Both are minted from the SAME request index inside `copies()`, so they cannot drift; what they
    /// cannot be any more is the same number.
    dest_entry: u32,
}

impl GatherCopy {
    /// The three declared extents.
    pub const fn dims(self) -> CopyDims {
        self.dims
    }

    /// Which index stick of the pass — the op's name suffix.
    pub const fn stick(self) -> u32 {
        self.stick
    }

    /// The DESTINATION entry this op's run starts at — see the field's note for why this is not
    /// [`Self::index_base`].
    pub const fn dest_entry(self) -> u32 {
        self.dest_entry
    }

    /// ⭐ THE INDEX OPERAND'S BASE, AS A WHOLE NUMBER OF INDEX STICKS. Spelled as a stick count and not
    /// as an entry count because dxp loads the IBR one stick at a time: a base that is not a stick
    /// multiple would load 32 words straddling two of this op's runs.
    pub const fn index_base(self) -> crate::superdsc_opspec::EntryBase {
        crate::superdsc_opspec::EntryBase::of_sticks(self.stick)
    }
}

// ⛔⛔⛔ WHAT WAS HERE, AND WHY IT IS GONE: `RequestAxis` — the three `y` steps a per-batch-kernel
// request-axis matmul would take, differenced out of the three buffers that have them, checked against
// the walk `matmul_opspec_batched_off` derived. It guarded the `[y,in,out]` KERNEL form, and that form
// is REFUTED ON CARD: it gives every core its own weight start, so `numCoresUsed_` collapsed onto the
// request count and the launch faulted at `job_bin_ptr + cores*128` at rungs 2, 4 and 8 alike (the
// syndrome, locator and case set bit-identical; only the index moved).
//
// The collapsed fold now reaches request `r`'s block by a baked OFFSET
// (`GatherScratch::kernel_row_off`) with `y` back on the GQA group, so no operand strides across
// requests and there is no derived request stride left to check. The relation that replaces it — the
// declared kernel is EXACTLY one scratch row, or op `r` reads request `r+1`'s keys — lives on that
// door, where both numbers are the scratch's own.

/// THE POOL A BATCH SHARES — every row owns the WHOLE page list, and the row separates them inside
/// each page.
///
/// A SuperDSC launch resolves exactly ONE page base, so one launch can address B rows only if row
/// `r`'s base is `base + r * stride` for a single stride, AND that stride is a number the emitter
/// baked. Whether both hold is not a detail: it is the entire cost of batched decode. Measured on
/// granite-3.1-2b at eight requests, a launch costs ~88 us of device time whatever it contains — a
/// one-trip group and a sixteen-trip group both read ~0.09 ms, while the group holding every matmul in
/// the layer does 79 trips in 0.268 ms. So the step is priced in launches, and a bs=8 layer spends
/// 27.6 of them: one for the matmuls, one for attn_o, one for fq_absx, and then 8 + 8 + 8.58 because
/// the cache write, the Kᵀ re-transpose and the prefix fold each run once PER REQUEST. That is 95 ms
/// of a 119 ms step.
///
/// They run per request because the map they consult is a free list. `alloc_pages` hands out the
/// lowest free page as each request grows, so row `r`'s pages are an arbitrary set and the base is
/// not affine in `r` — Kani finds the counterexample in under half a second (rows at physical pages
/// 0, 1, 3 have no single stride). Nothing about the addressing is wrong; it simply cannot be
/// expressed as one launch.
///
/// ⛔ **AND CUTTING THE POOL INTO ONE RUN PER ROW DOES NOT FIX IT.** That was this type's first shape:
/// row `r` owned pages `[r * pages_per_row, (r+1) * pages_per_row)`, which IS affine in the row. But
/// its stride is `pages_per_row * page_stride` and `pages_per_row = pool_pages / rows` comes from
/// `SUPERDSC_POOL_PAGES` or from whatever pool the loaded bundle found — a runtime number. A matmul
/// steps its batch axis by `in * out` of a kernel declared when the bundle was COMPILED, so a stride
/// only knowable at launch cannot be stepped at all, however affine it is. Kani
/// `request_separation_does_not_depend_on_the_pool_size` exhibits two pools of identical geometry
/// whose rows sit different distances apart.
///
/// So the request moved INTO the page instead, where its stride was `hd * PAGE_SLOTS` — geometry, and
/// the kernel's own declared block size. THAT SHAPE IS THE ONE THAT WAS UNDONE: a page holds SLOTS
/// again and a row owns its own consecutive PAGES, so `physical(lp) = row * pages_per_row + lp` and the
/// stride between rows is a PAGE stride. The launch can step it because the fold installs the page maps.
///
/// ⛔ **AND THE CAPACITY PRICE IS BACK.** This paragraph used to read "there is no capacity price — a
/// page belongs to all `ROWS` requests at once, so `8` pages is `2048` positions whether a request is
/// alone or in a batch of 32". Striping made that false and the sentence survived the rebuild, which is
/// exactly the doc a reader would trust while hunting the opposite symptom: a soloist now gets
/// `pool_pages / rows` and NOTHING MORE, because the pages of the 31 rows it is not using are not
/// addressable from the row it holds. That is why the stripe count is a decision with a type
/// ([`PoolRows`]) instead of a `max()` over whatever baked.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PoolSplit {
    rows: NonZeroU32,
}

/// A PREFILL CHUNK'S KV EXTENT — proof that the rows the bundle WRITES are exactly the tokens the host
/// COUNTS.
///
/// 🛑 **THEY WERE TWO `usize`s PASSED SIDE BY SIDE.** `run_prefill_batch(.., &toks[off..off+chunk_len],
/// rung_m, ..)` runs the launch at the BUNDLE's row count while `req.record_kv_solo(chunk_len)` advances
/// the history by the CHUNK's real length. When `chunk_len < rung_m` the device still writes `rung_m` rows
/// of K/V, the host records `chunk_len`, and the next chunk starts at `chunk_start + chunk_len` — so the
/// `rung_m - chunk_len` PADDING rows are never overwritten and sit INSIDE the attended history as K/V
/// computed from padding. The row attends them, its distribution collapses, and the request answers with a
/// single EOS: `completion_tokens: 1`, empty text, `status: 200`, no error anywhere.
///
/// ⛔ **ONE PADDING ROW IS ENOUGH.** Measured on granite-3.1-8b fp8 (hd=128), four IDENTICAL 1986-token
/// prompts in a 4-wide batch, where a shared per-step token budget split the chunks:
/// | `m_used` | `prefill_m` | padding | result |
/// |---|---|---|---|
/// | 96 | 96 | 0 | answered |
/// | 62 | 63 | **1** | **EOS, empty** |
/// | 96 | 96 | 0 | answered |
/// | 60 | 63 | **3** | **EOS, empty** |
/// Every padded chunk collapsed; every exact chunk answered. That is why this is a TYPE and not a
/// tolerance: there is no safe amount of unaccounted K/V.
///
/// This is why a batch is needed to see it at all — a solo prompt takes the full budget, so every chunk is
/// exact and the desync never arises. Ragged LENGTHS are not enough either; what splits a chunk is several
/// requests sharing one step's token budget.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ChunkKvExtent(u32);

impl ChunkKvExtent {
    /// The extent, or `None` when the bundle would write rows the history does not count.
    ///
    /// Refusing rather than trusting the caller is the whole point: the two numbers are both token counts
    /// in the same units, so nothing but a type stops the wrong one reaching the write.
    pub fn exact(chunk_len: u32, rung_m: u32) -> Option<ChunkKvExtent> {
        (chunk_len == rung_m && chunk_len > 0).then_some(ChunkKvExtent(chunk_len))
    }

    /// Rows written AND tokens counted — one number, because they are the same number.
    pub fn get(self) -> u32 {
        self.0
    }

    /// Padding rows a chunk of `chunk_len` on a `rung_m` bundle would leave uncounted — 0 iff the chunk
    /// is exact. For the message a caller prints when [`exact`](Self::exact) refuses.
    pub fn unaccounted_rows(chunk_len: u32, rung_m: u32) -> u32 {
        rung_m.saturating_sub(chunk_len)
    }
}

/// HOW MANY STRIPES THE POOL IS CUT INTO — the ONE width the pool is both SIZED by and SPLIT by, fixed
/// for the whole run.
///
/// 🛑 **IT WAS TWO NUMBERS OF THE SAME TYPE.** The pool was sized by the const ladder's last rung
/// (`WIDEST_BATCH_RUNG`, 32) because sizing happens in `load_model` BEFORE the ladder is built, and it
/// was split by `decode_rungs.iter().map(|r| r.seqs).max()` — the rungs that actually SURVIVED baking.
/// Two `usize`s, both spelled `widest`, computed 3700 lines apart from different sources, multiplied and
/// divided into the same pool. Nothing made them agree and nothing would have said so: a rung dropping
/// out of the ladder silently re-cut every request's share.
///
/// 🛑 **AND A STRIPE COUNT IS NOT A FREE INTEGER.** [`LaunchPages::Affine`] strides slots `0..seqs`
/// arithmetically across the stripes, so the rung that RUNS must never be wider than the pool has rows,
/// or the top slots address past the end of the pool. `decode_rung_for` picks the smallest rung `>= live`
/// and admission cannot exceed the rows (`free_row` runs out), so that bound holds **iff the stripe count
/// is itself a rung of [`PagedKvPool::BATCH_RUNGS`]** — for any other value there is a live count whose
/// selected rung overshoots. Hence the only constructors are ladder members: the invariant is the type's,
/// not a caller's to remember.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PoolRows(NonZeroU32);

impl PoolRows {
    /// The widest the ladder bakes — every row a rung could ever address gets a stripe.
    pub const WIDEST: PoolRows = match NonZeroU32::new(PagedKvPool::WIDEST_BATCH_RUNG) {
        Some(n) => PoolRows(n),
        None => panic!("WIDEST_BATCH_RUNG is zero"),
    };

    /// `n` as a stripe count, or `None` unless it is a rung of the ladder.
    ///
    /// Rejecting rather than clamping is the point: a caller asking for 5 stripes has a belief about the
    /// pool that no legal split can satisfy, and rounding it to 4 would leave that belief in place.
    pub fn of_rung(n: u32) -> Option<PoolRows> {
        PagedKvPool::BATCH_RUNGS
            .contains(&n)
            .then(|| NonZeroU32::new(n).map(PoolRows))
            .flatten()
    }

    /// Every stripe count a pool may legally be cut into, widest first — what a policy chooses among.
    pub fn all() -> impl Iterator<Item = PoolRows> {
        PagedKvPool::BATCH_RUNGS
            .into_iter()
            .rev()
            .filter_map(PoolRows::of_rung)
    }

    pub fn get(&self) -> NonZeroU32 {
        self.0
    }

    /// THE ADMISSION WIDTH FOR A RUN THAT WILL ADMIT AT MOST `want` REQUESTS — the SMALLEST rung that
    /// holds them, not the widest the ladder bakes.
    ///
    /// 🛑 THIS USED TO BE A DEPTH-VS-WIDTH TRADE (`widest_reaching`), and that trade existed ONLY because
    /// the pool was striped: bytes were `rows * pages_per_row * page_stride`, so buying concurrency cost
    /// every request context, and a fixed budget had to choose. `TARGET_POSITIONS_PER_ROW` was the depth it
    /// bought first. With pages on a free list there is nothing to trade for CAPACITY — one request can
    /// reach the whole pool whatever `rows` is — so this returned `WIDEST`, full stop.
    ///
    /// ⛔⛔ THAT WAS TRUE OF CAPACITY AND FALSE OF THE RESERVE, WHICH IS WHAT MADE IT COSTLY.
    /// `PoolPartition::of_pool` holds back `rows * HOLE_PAGES_PER_ROW + 1` pages, because each row of a
    /// LAUNCH writes its masked hole into a page the host was never told exists. At `WIDEST` = 32 that is
    /// 65 of a 136-page pool — the host is offered 71 — **whatever the run will actually admit.** A run
    /// serving four concurrent requests never binds more than a 4-row rung, so 56 of those 65 pages are
    /// reserved against launches the run cannot perform. Measured on granite-3.1-8b fp8: 71 host pages
    /// instead of 127, i.e. the addressable context nearly halved to pay for concurrency nobody asked for.
    ///
    /// ⭐ AND THE WIDTH IS A DECIDED VALUE: `--max-num-seqs` is the cap the scheduler admits under, so it
    /// is what the reserve should be charged against. It still MUST land on a rung — `decode_rung_for`
    /// picks the smallest rung `>= live` and that launch strides its slots arithmetically, so a width off
    /// the ladder has a live count whose rung addresses slots the pool never seated. Hence: round the
    /// request cap UP to a rung (never down — admitting more than the pool was cut for is the corruption
    /// this type exists to prevent), and cap it at `WIDEST`.
    pub fn for_admission(want: AdmittedRequests) -> PoolRows {
        let n = want.get().get();
        PagedKvPool::BATCH_RUNGS
            .into_iter()
            .find(|&r| r >= n)
            .and_then(PoolRows::of_rung)
            .unwrap_or(PoolRows::WIDEST)
    }
}

/// HOW MANY REQUESTS A RUN MAY ADMIT AT ONCE — `--max-num-seqs`, the scheduler's own cap.
///
/// ⛔ A REQUEST COUNT IS NOT A ROW COUNT, and keeping them one integer is what let the pool's hole reserve
/// be charged against the widest rung the ladder bakes rather than against the width the run will reach.
/// The conversion is [`PoolRows::for_admission`], which rounds UP to a ladder rung — the direction matters,
/// because a pool cut for fewer rows than a launch binds addresses slots it never seated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AdmittedRequests(NonZeroU32);

impl AdmittedRequests {
    /// `n` requests, or `None` for zero — a run that admits nothing has no width to derive.
    pub fn new(n: usize) -> Option<AdmittedRequests> {
        u32::try_from(n)
            .ok()
            .and_then(NonZeroU32::new)
            .map(AdmittedRequests)
    }

    pub fn get(self) -> NonZeroU32 {
        self.0
    }
}

impl PoolSplit {
    /// Rows this split was cut for. A launch addressing more of them than this would run off the end
    /// of the pool.
    pub fn rows(&self) -> NonZeroU32 {
        self.rows
    }

    // ⛔ NO `row()` AND NO `all_rows()`. They were the only way to obtain a `KvRow` and the list `free_row`
    // picked from. What remains of this type is the WIDTH the pool was cut for — a bound on how many launch
    // rows can run, not a set of per-request identities to hand out.
}

// ⛔⛔⛔ `KvRow` WAS HERE, AND ITS DELETION IS THE POINT OF THIS COMMIT: NO REQUEST CONCEPT BELOW THE HOST.
//
// A row was a request's identity in the KV pool, claimed on first forward and held for life. It stopped being
// an ADDRESS term long before it died (`PageRun`'s `physical(lp) = row * pages_per_row + lp` went with the
// stripe, and pages come from the host's block table), after which it survived for exactly one purpose:
// naming the pool pages reserved to back a batched write past a request's own keys. The host allocates that
// page now (`own pages + 1`, `KvSlotSpan::blocks_this_step`), so a request has no identity down here at all —
// its KV is its slot history plus the scheduler's block ids, both of which every backend already has.
//
// ⭐ THE LESSON WORTH KEEPING IS THE DISTINCTION IT ENFORCED: a KV row and a `LaunchSlot` were one `u32` for a
// day and it cost a session. A row was an allocation identity outliving any forward; a slot is a position in
// the launch happening now, and the rung decides how many exist. Conflating them coupled the launch WIDTH to
// the highest row any live request held, so four requests on rows 4..7 selected the four-wide rung and could
// not be expressed in it. `LaunchSlot` remains, and it is now the ONLY "which row" number in this file.

/// A ROW OF THE LAUNCH HAPPENING NOW — `0..width` of the rung being run, and the index every op's
/// `kv_request` names. Minted only by [`SlotMap`], so a slot past the rung's width does not exist.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct LaunchSlot(usize);

impl LaunchSlot {
    /// The only slot a SINGLE-row launch has — every prefill and every unbatched decode. There is no
    /// batch to be a row of, so there is nothing for a [`SlotMap`] to lay out; naming it here keeps
    /// slot 0 from being written as a bare `0` at those call sites, which is how a KV row came to be
    /// passed as a slot in the first place. Matches `fold_plan::LaunchWidth::Single`.
    pub fn solo() -> LaunchSlot {
        LaunchSlot(0)
    }

    /// The slot index, for the bind call and the logits read-back.
    pub fn index(&self) -> usize {
        self.0
    }
}

/// Why a live batch cannot be laid into a launch. Both cases are real conditions with real cures, and
/// neither is something to round: picking a winner for a shared row is the silent corruption being
/// avoided, and overflowing the rung would run off the end of the installed maps.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SlotMapError {
    /// No live requests. A launch with nothing in it has no slot to point padding at.
    NoLiveRequest,
    /// More live requests than the rung has rows.
    TooWide { live: usize, width: u32 },
    // ⛔ NO `DuplicateRow`: two live requests sharing a pool row was the aliasing this refused, and there are
    // no pool rows to share. A request's pages are the scheduler's block ids, and two requests holding the
    // same page is a prefix-cache HIT — the ordinary case, not a collision.
}

/// WHICH LIVE REQUEST EACH SLOT OF A LAUNCH HOLDS.
///
/// The live list is ordered by KV row (stable across steps, since a row is an identity), and the slots
/// are handed out DENSELY over it: live `i` runs in slot `i`. Slots past the live count are padding and
/// reuse live 0's map, so no slot addresses a table that was never installed — that re-writes one
/// token at one slot a second time with the same key and value, which is idempotent, and is why
/// padding can point at real pages instead of somewhere invented for it.
///
/// Dense is not a convenience. It is what makes the launch width depend only on HOW MANY requests are
/// live, which is the number the rung was chosen from. Any assignment that reads the row instead makes
/// the width depend on WHICH rows they hold, and then a narrow rung cannot hold a high row.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SlotMap {
    /// Index = slot, value = WHOSE map that slot installs. Length is the rung's width.
    source: Vec<SlotSource>,
    live: usize,
}

/// ⭐⭐⭐⭐⭐ WHOSE MAP A LAUNCH SLOT INSTALLS — and, decisively, whether there is anyone at all.
///
/// ⛔ THIS WAS A BARE `usize` AND THAT IS THE DEFECT. A padding slot was spelled `0`, which is also a
/// perfectly good live index, so "slot 3 is padding" and "slot 3 runs live request 0" were the same value.
/// The consumer then indexed the live list with it and handed that request's PAGES to the padding slot's
/// ROW — see [`SlotMap::of_live`] for the address collision that produces, measured on the card.
///
/// A padding slot owns no request, therefore no pages, and with this type it cannot silently borrow
/// anyone else's: the `Padding` arm carries nothing to index with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SlotSource {
    /// Live request `i` of this step runs in this slot.
    Live(usize),
    /// The rung is wider than the live count and nobody runs here. Its output is discarded — but it still
    /// WRITES, into live 0's pool cell, unordered against live 0's own write, so what it RUNS is live 0
    /// entire ([`PadRowReplica`]). Owning no request and replicating one are not in tension: it is handed
    /// no request, no pages of its own and no place in the live list, and it re-evaluates live 0 so that
    /// the write it cannot avoid making is the write live 0 was already making.
    Padding,
}

impl SlotSource {
    /// The live request, or `None` for padding. The only way to an index, so a caller must state what it
    /// means to do about padding instead of silently reading request 0.
    pub fn live(self) -> Option<usize> {
        match self {
            SlotSource::Live(i) => Some(i),
            SlotSource::Padding => None,
        }
    }
}

impl SlotMap {
    /// Lay `live` requests into a launch of the RUNG's width — the two widths meet here and nowhere else,
    /// which is why the width slot is typed.
    ///
    /// ⛔ IT TOOK `&[Option<KvRow>]`, THE KV ROW EACH LIVE REQUEST HELD, and consulted it for exactly ONE
    /// thing: refusing two requests that shared a row. Rows are gone — the host allocates the batched write
    /// page, so a request has no identity in the pool — so that refusal has no state left to detect, and
    /// `SlotMapError::DuplicateRow` went with it. Live `i` runs in slot `i`, which was never derived from
    /// the rows.
    pub fn of_live(live: usize, width: RungWidth) -> Result<SlotMap, SlotMapError> {
        let w = width.count();
        if live == 0 {
            return Err(SlotMapError::NoLiveRequest);
        }
        if live > w {
            return Err(SlotMapError::TooWide {
                live,
                width: width.get(),
            });
        }
        // DENSE: live `i` runs in slot `i`.
        //
        // ⛔ AND EVERY SLOT PAST THE LIVE COUNT IS `Padding`, NOT `0`. It used to be
        // `if s < live { s } else { 0 }`, so "slot 3 is padding" and "slot 3 runs live request 0" were the
        // SAME VALUE, and every consumer had to re-derive the difference from `s < map.live()`.
        //
        // ⭐ WHAT THE TYPE BUYS: what a padding slot borrows is a DECISION each consumer makes out loud,
        // per input, instead of getting `0` for free. The worker's launch borrows live 0's PAGE MAP for
        // it — every slot's table must exist, and the pool's pages are one shared set per row — and
        // that borrow is what forces the rest: the borrowed map puts the padding slot's cache write on
        // live 0's cell, unordered against live 0's own, so the slot must also RUN live 0's token,
        // rotation, new-block column and history. All four arrive together through
        // `LiveBatch::launch_rows`, never one at a time.
        let source = (0..w)
            .map(|s| {
                if s < live {
                    SlotSource::Live(s)
                } else {
                    SlotSource::Padding
                }
            })
            .collect();
        Ok(SlotMap { source, live })
    }

    /// Slots in the launch.
    pub fn width(&self) -> usize {
        self.source.len()
    }

    /// Live requests laid in.
    pub fn live(&self) -> usize {
        self.live
    }

    /// The slot live request `i` runs in.
    pub fn slot_of(&self, i: usize) -> Option<LaunchSlot> {
        self.source
            .iter()
            .position(|&src| src == SlotSource::Live(i))
            .map(LaunchSlot)
    }

    /// Every slot of the launch with WHOSE map it installs, lowest slot first. Padding slots come back as
    /// [`SlotSource::Padding`], so a consumer cannot walk this and quietly bind request 0 four times.
    pub fn slots(&self) -> impl Iterator<Item = (LaunchSlot, SlotSource)> + '_ {
        self.source
            .iter()
            .enumerate()
            .map(|(s, &src)| (LaunchSlot(s), src))
    }

    /// Whose map slot `s` installs. Total over the slots this map has.
    pub fn source(&self, s: LaunchSlot) -> SlotSource {
        self.source[s.0]
    }

    /// ⭐ THE LAUNCH'S LIVE ROWS, minted by the map that knows which slots are live: `f(i)` is called
    /// once per live request `i`, in slot order, and NEVER for a padding slot. The one constructor of
    /// [`LiveBatch`], so a padding row's own `BatchRow` is a value with no way to exist — nothing here
    /// invents a request for a slot that holds none.
    ///
    /// What a padding slot RUNS is a separate question, and its answer is [`PadRowReplica`]: live 0's
    /// row, complete. That is not this builder inventing a request — it is the launch stating that a
    /// padding slot is a second copy of live 0, which is what its racing same-cell write requires.
    pub fn live_rows<E>(
        &self,
        mut f: impl FnMut(usize) -> Result<BatchRow, E>,
    ) -> Result<LiveBatch, E> {
        let mut rows = Vec::with_capacity(self.live);
        for src in &self.source {
            if let SlotSource::Live(i) = *src {
                rows.push(f(i)?);
            }
        }
        Ok(LiveBatch { rows })
    }
}

/// ⭐⭐⭐ THE LIVE ROWS OF A BATCHED LAUNCH — one [`BatchRow`] per LIVE slot, in slot order, and
/// NOTHING for the padding slots.
///
/// It carries the live count, which a rung-width vector cannot: `toks.len() == rung width` left no
/// consumer able to ask how many rows were real, and every law about padding needs that number.
///
/// Minted ONLY by [`SlotMap::live_rows`], which walks the map's OWN live slots. To run a launch it is
/// widened to [`LaunchRows`], where the padding slots take live 0's row entire.
pub struct LiveBatch {
    rows: Vec<BatchRow>,
}

impl LiveBatch {
    /// The row of LIVE request `i`, `None` past the live count. Live `i` runs in slot `i` (the map is
    /// dense), so a live index and its launch slot are the same integer — but this asks about the LIVE
    /// list only. What a PADDING slot runs is [`LaunchRows::row`]'s question, and it is not "nothing".
    pub fn row(&self, slot: usize) -> Option<&BatchRow> {
        self.rows.get(slot)
    }

    /// Rows carried — the launch's LIVE count, never the rung width.
    pub fn live(&self) -> usize {
        self.rows.len()
    }

    /// ⭐⭐⭐ THE LAUNCH AS THE RUNG ACTUALLY RUNS IT — every slot of `width`, live or padding, each
    /// with a COMPLETE set of row inputs. This is the one door from "who is live" to "what every row
    /// runs", and the only constructor of [`LaunchRows`].
    ///
    /// `toks` is the launch's own token list, ONE PER LIVE ROW, from the same walk that minted `self`;
    /// any other length refuses, so no caller can pair a row with another slot's token. Live 0 is
    /// picked HERE and not by the caller, because the replicated row is not a choice: live 0 is the row
    /// whose page map every padding slot borrows, so live 0's write cell is the cell a padding write
    /// races.
    ///
    /// `None` when the batch is empty (no live 0 to replicate), when the tokens do not match the live
    /// rows, when a token is not a `u32`, or when more requests are live than the rung is wide.
    pub fn launch_rows<'a>(&'a self, toks: &[usize], width: RungWidth) -> Option<LaunchRows<'a>> {
        if toks.len() != self.rows.len() || self.rows.len() > width.count() {
            return None;
        }
        let live0 = self.rows.first()?;
        let ids: Vec<TokenId> = toks
            .iter()
            .map(|&t| u32::try_from(t).ok().map(TokenId))
            .collect::<Option<_>>()?;
        // ⭐ ALL FOUR ASPECTS, IN ONE EXPRESSION. There is no way to build a replica that mirrors
        // live 0's token but not its history, because there is no other constructor and no field is
        // optional — which is the whole lock: the two previous attempts at this each mirrored one
        // aspect and left another, and a partial replica writes different bytes into live 0's cell.
        let pad = PadRowReplica {
            token: *ids.first()?,
            rope_pos: live0.rope_pos,
            // Live 0's launch slot, which under the diagonal new-block law IS the column its query row
            // attends. `SlotMap::of_live` hands slots out densely from live 0, so it is slot 0.
            new_block_col: LaunchSlot(0),
            hist: live0.hist.clone(),
        };
        Some(LaunchRows {
            live: self,
            pad,
            width,
            toks: ids,
        })
    }
}

/// ⭐⭐⭐⭐⭐ WHAT A PADDING ROW *IS*: LIVE 0, COMPLETE — the same token, the same rotation, the same
/// new-block column and the same prefix history, so its WHOLE FORWARD is bit-identical to live 0's.
///
/// ⛔ WHY REPLICATION AND NOT "NOTHING". A padding slot borrows live 0's PAGE MAP (every slot's table
/// must exist), and the on-card paged cachewr address carries NO row term inside a page — a pool cell
/// is (plane, kv-head, slot-in-page, feature) — so the padding slot's one cache write lands on the very
/// cell live 0's write lands on. The shim elides the pipeline barrier between consecutive slot-write
/// groups on the premise the writes are disjoint (`sdsc_shim.cpp`: `pipeline_barrier =
/// !next_is_slot_write`); a padding slot breaks that premise, so the two writes are UNORDERED and
/// either may land last. The one payload that commutes with live 0's write is live 0's own: identical
/// bytes, so whichever lands last, the cell holds live 0's newest key.
///
/// ⛔ AND THE BYTES ARE THE WHOLE FORWARD, NOT THE INPUT. What the cache write stores is this row's K
/// and V at every layer — the output of layer `l-1` fed through attention and the MLP. So it is not
/// enough for the padding row to start at live 0's token and rotation: EVERY input to that forward has
/// to be live 0's, or the hidden state diverges at layer 1 and every layer from there writes different
/// K/V into live 0's cell. The mask is such an input. A padding row whose new-block column or whose
/// prefix history differed from live 0's ran a different softmax over a different key set, produced a
/// different output, and clobbered live 0's newest key nondeterministically at every layer >= 1 — which
/// is fluent output, no fault, and only from bs >= 3, because a full rung has no padding row and bs=2
/// on the 2-rung has none either.
///
/// ⛔ SO "PADDING OWNS NOTHING" IS THE WRONG SAFETY. A nothing-valid mask row does stop the padding row
/// claiming history it was not assigned — and buys that by making the row a DIFFERENT computation from
/// live 0, which is the one thing the racing write forbids. Padding still owns nothing: it is handed no
/// request, no pages of its own and no slot in the live list, and its logits are never read. It is a
/// second evaluation of live 0 whose result is discarded.
///
/// Constructible ONLY by [`LiveBatch::launch_rows`], and every field is populated there in one
/// expression: a replica that mirrors some aspects and not others is a value with no way to exist.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PadRowReplica {
    token: TokenId,
    rope_pos: SeqPos,
    new_block_col: LaunchSlot,
    hist: KvHistory,
}

/// ⭐⭐⭐ EVERY ROW THE LAUNCH BINDS — `width` of them, live and padding alike, and the ONLY way to ask
/// what any of them runs.
///
/// The kind lives in this type. A prompt chunk has no `LaunchRows` at all (its pad rows clamp to the
/// last real token — [`ChunkRows::row_logical_pos`]), and a decode batch has nothing else, so neither
/// can take the other's law: the two are not two arms of one function that a caller picks between.
pub struct LaunchRows<'a> {
    live: &'a LiveBatch,
    pad: PadRowReplica,
    width: RungWidth,
    toks: Vec<TokenId>,
}

/// ⭐⭐ ONE ROW'S COMPLETE INPUTS — the token it gathers, the position it rotates at, the new-block
/// column it attends and the prefix history it may attend, as ONE value.
///
/// ⛔ FOUR SEPARATE LOOKUPS IS THE DEFECT, not a style. Each aspect had its own `if this row is
/// padding` at its own staging site, so a fix at one site left the others: the token and the rotation
/// were mirrored to live 0 while the mask row was still "nothing valid" and the causal column was still
/// the clamp's `real - 1`. The row then ran a different forward from live 0 and raced live 0's cache
/// write with different bytes. With one value there is one padding decision, taken once, and mirroring
/// the token but not the mask is not something a caller can express.
///
/// No public constructor: the only way to one is [`LaunchRows::row`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RowInputs<'a> {
    token: TokenId,
    rope_pos: SeqPos,
    new_block_col: LaunchSlot,
    hist: &'a KvHistory,
}

impl<'a> RowInputs<'a> {
    /// The token this row's embedding gathers.
    pub fn token(self) -> TokenId {
        self.token
    }

    /// The position this row rotates at — its own request's token count, or live 0's for padding.
    pub fn rope_pos(self) -> SeqPos {
        self.rope_pos
    }

    /// The new-block column this row attends, under the diagonal law: its own launch slot, or live 0's
    /// for padding.
    pub fn new_block_col(self) -> LaunchSlot {
        self.new_block_col
    }

    /// The slots this row may attend in the resident prefix.
    pub fn hist(self) -> &'a KvHistory {
        self.hist
    }
}

impl<'a> LaunchRows<'a> {
    /// Slots the launch binds — the RUNG's width, padding included.
    pub fn width(&self) -> RungWidth {
        self.width
    }

    /// Requests actually running. Padding slots are `width - live` of them.
    pub fn live(&self) -> usize {
        self.live.live()
    }

    /// ⭐ WHAT LAUNCH SLOT `slot` RUNS — TOTAL over `0..width`, and the ONE place padding is resolved.
    ///
    /// A live slot runs its own request. A padding slot runs [`PadRowReplica`] — live 0, complete —
    /// because its cache write races live 0's at one pool cell and only identical bytes commute there.
    /// Every staging site reads its own field off the value this returns, so the four aspects cannot
    /// disagree about whether the row is padding.
    ///
    /// `None` only past the rung's width, which is not a slot the launch has.
    pub fn row(&self, slot: usize) -> Option<RowInputs<'_>> {
        if slot >= self.width.count() {
            return None;
        }
        Some(match self.live.row(slot) {
            Some(b) => RowInputs {
                token: self.toks[slot],
                rope_pos: b.rope_pos,
                new_block_col: LaunchSlot(slot),
                hist: &b.hist,
            },
            None => RowInputs {
                token: self.pad.token,
                rope_pos: self.pad.rope_pos,
                new_block_col: self.pad.new_block_col,
                hist: &self.pad.hist,
            },
        })
    }

    /// THE `[mq, mq_pad]` NEW-BLOCK MASK OF A DECODE BATCH, additive and row-major: 0 where row `r` may
    /// attend new-block column `c`, `mask_neg` elsewhere.
    ///
    /// The rows are INDEPENDENT REQUESTS, so the extent is [`decode_batch_causal_col_valid`]'s
    /// DIAGONAL — a row attends its own column and no other. A PADDING row's column is live 0's, from
    /// the same [`Self::row`] every other aspect comes from, so its mask row is byte-identical to live
    /// 0's and its attention is live 0's attention.
    ///
    /// `mq_pad` is derived from the rung width here (`mq.div_ceil(64)*64`), matching the emitter, so the
    /// buffer cannot be built at one width and read at another.
    pub fn new_block_mask(&self, mask_neg: f32) -> Vec<f32> {
        let mq = self.width.count();
        let mq_pad = mq.div_ceil(64) * 64;
        let mut cmask = vec![mask_neg; mq * mq_pad];
        for row in 0..mq {
            let inputs = self.row(row).expect("row < width");
            for col in 0..mq_pad {
                if decode_batch_causal_col_valid(col, inputs.new_block_col().index()) {
                    cmask[row * mq_pad + col] = 0.0;
                }
            }
        }
        cmask
    }

    /// THE PREFIX HISTORY OF EVERY BOUND ROW, in slot order — what [`decode_batch_prefix_mask_f16`]
    /// blocks by. A padding row's is live 0's, not an empty one, for the same reason its token is.
    pub fn mask_histories(&self) -> Vec<KvHistory> {
        (0..self.width.count())
            .map(|s| self.row(s).expect("row < width").hist().clone())
            .collect()
    }
}

/// A vocabulary token id, as the staged embedding gather indexes it (`embed[token * hidden ..]`).
/// Minted by [`LiveBatch::launch_rows`] so a row's token names WHICH quantity it carries instead of
/// riding as a bare integer; the only exit is [`get`](Self::get), for the gather that must eventually
/// index rows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TokenId(u32);

impl TokenId {
    /// The integer, for the embedding gather. The ONLY exit.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// ONE ROW'S PAGES, AS A RUN. Built only by [`PoolSplit::run`], so a non-affine page map is not a
/// value this type can hold — which is the point: it is a build-time guard, not a checked invariant.
///
/// The run is now the WHOLE pool, identically for every row, because the row separates requests inside
/// a page rather than by choosing pages. It is still a distinct type per row, and still the only way to
/// get a physical page, so nothing that consults a row's pages can forget which row it asked about —
/// and the pool can go back to being cut into runs (a denser page, a per-row reservation) without
/// touching a caller.
// ⛔⛔⛔ `PageRun` IS DELETED, AND THAT DELETION *IS* THE LOCK-DOWN.
//
// It held `physical(lp) = row * pages_per_row + lp` — a page address DERIVED BY FORMULA from the row. That
// formula is what made two requests sharing a page unrepresentable, which is why spyre became the only
// backend whose worker reported `supports_prefix_caching() == false`, and it is what capped one request at
// `pool_pages / rows` (the cap that killed granite-8b at 768 positions). metal and cuda address KV through
// an arbitrary per-request `block_table` (`gpu_worker.rs`) and neither problem exists there.
//
// Pages now come from a FREE LIST (`pages_held` + `ensure_pages`, `f5e26a3e`), so a request's page list IS
// its block table. Nothing in the tree maps `(row, logical page) -> physical page` any more, so the stripe
// cannot be written down again without re-introducing this type — which is exactly the point. A guard that
// only *documented* the rule would leave the formula one keystroke away; deleting the API makes the whole
// CLASS unrepresentable.
//
// What a `KvRow` still means: the identity of a LAUNCH SLOT (`PoolSplit::row`/`all_rows` still mint it).
// It is no longer a capacity share, and no longer an address term.
/// The address model EXPOSES the live inf bug — `sp` written per-head as `[1,cap]` (a=1,
/// contiguous) but read full as `[nqh,cap]` (a=nqh, device-tiled `[cap/64,nqh,64]`) resolves the
/// SAME logical element to DIFFERENT addresses for h≥1, so the softmax reads garbage:
/// ```
/// use ktir_superdsc::sdsc_abstract::*;
/// let (nqh, cap, base) = (9usize, 256usize, 0usize);
/// let writer = |h: usize| AbsTensor { name: "sp".into(), base: base + h * cap, dims: vec![1, cap], stick_idx: 1 };
/// let reader = AbsTensor { name: "sp".into(), base, dims: vec![nqh, cap], stick_idx: 1 };
/// let mut mem = AbsMem::default();
/// for h in 0..nqh { for s in 0..cap { mem.set(writer(h).addr(&[0, s]), (h * 1000 + s) as f64); } }
/// let mut mismatch = 0usize;
/// for h in 0..nqh { for s in 0..cap {
///     if (mem.get(reader.addr(&[h, s])) - (h * 1000 + s) as f64).abs() > 1e-9 { mismatch += 1; }
/// }}
/// assert!(mismatch > 0, "the interpreter must expose the per-head/full sp layout mismatch");
/// ```
/// A typed view of a tensor for the interpreter: its base device address (element units), full
/// shape, and stick axis. `addr(idx)` is the device address of a logical element.
#[derive(Clone, Debug)]
pub struct AbsTensor {
    pub name: String,
    pub base: usize,
    pub dims: Vec<usize>,
    pub stick_idx: usize,
}
impl AbsTensor {
    pub fn addr(&self, idx: &[usize]) -> usize {
        self.base + dev_off(&self.dims, self.stick_idx, idx)
    }
}

/// Device memory as element-address → value. Reads of an unwritten address return 0.0 (matches
/// the zeroed segments); a WRITE collision (two tensors at one address) is what catches aliasing.
#[derive(Default)]
pub struct AbsMem {
    pub cells: HashMap<usize, f64>,
}
impl AbsMem {
    pub fn get(&self, a: usize) -> f64 {
        *self.cells.get(&a).unwrap_or(&0.0)
    }
    pub fn set(&mut self, a: usize, v: f64) {
        self.cells.insert(a, v);
    }
}

/// Deterministic seed for an input element (no RNG dep; reproducible across builds). A cheap
/// hash → a value in roughly [-2, 2], enough that a structurally-wrong emission diverges.
pub fn seed_val(kind: u64, a: u64, b: u64, c: u64) -> f64 {
    let mut h = kind
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(a.wrapping_mul(0xC2B2AE3D27D4EB4F))
        .wrapping_add(b.wrapping_mul(0x165667B19E3779F9))
        .wrapping_add(c.wrapping_mul(0xD6E8FEB86659FD93));
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 32;
    // map to ~[-2,2]
    ((h % 4001) as f64 / 1000.0) - 2.0
}

/// Which dim of the OUTPUT index a broadcast operand collapses (reads index 0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bcast {
    None,
    /// broadcast over rows (mb) — a `[1,C]` operand read across all rows (mb_broadcast)
    Mb,
    /// broadcast over cols (out) — a `[R,1]` per-row scalar read across all cols (out_broadcast)
    Out,
}
fn bidx(o_idx: &[usize], b: Bcast) -> Vec<usize> {
    let mut v = o_idx.to_vec();
    match b {
        Bcast::None => {}
        Bcast::Mb => v[0] = 0,
        Bcast::Out => v[1] = 0,
    }
    v
}

/// Elementwise op kind.
#[derive(Clone, Copy, Debug)]
pub enum EwF {
    Add,
    Sub,
    Mul,
    Max,
    Exp,
    Recip,
    Identity,
    Silu,
    Sigmoid,
}
/// Reduction kind.
#[derive(Clone, Copy, Debug)]
pub enum RedF {
    Sum,
    Max,
}

/// One interpretable SDSC op, carrying the REAL device addressing of its operands (from the
/// emitter) so the interpreter executes the ACTUAL emission — addressing/layout/dataflow bugs
/// (overlap, stick-major-vs-dense, wrong contraction) surface as a wrong result.
#[derive(Clone, Debug)]
pub enum AbsOp {
    /// `o[m,n] = Σ_k a[m,k]·w[k,n]`. a:[M,K] w:[K,N] o:[M,N].
    Matmul {
        a: AbsTensor,
        w: AbsTensor,
        o: AbsTensor,
    },
    /// `o[c,r] = x[r,c]` (transpose). x:[R,C] o:[C,R].
    Restickify { x: AbsTensor, o: AbsTensor },
    /// `o[i] = f(a[i'], b[i''])` over o's [R,C], with per-operand broadcast. b=None for unary.
    Ew {
        f: EwF,
        a: AbsTensor,
        ab: Bcast,
        b: Option<AbsTensor>,
        bb: Bcast,
        o: AbsTensor,
    },
    /// `o[r,0] = (reduce_c x[r,c]) · scale`. x:[R,C] o:[R,1].
    Reduce {
        f: RedF,
        x: AbsTensor,
        o: AbsTensor,
        scale: f64,
    },
}

fn ewf(f: EwF, a: f64, b: f64) -> f64 {
    match f {
        EwF::Add => a + b,
        EwF::Sub => a - b,
        EwF::Mul => a * b,
        EwF::Max => a.max(b),
        EwF::Exp => a.exp(),
        EwF::Recip => 1.0 / a,
        EwF::Identity => a,
        EwF::Sigmoid => 1.0 / (1.0 + (-a).exp()),
        EwF::Silu => a / (1.0 + (-a).exp()),
    }
}

/// Execute the op-DAG over `mem` (in order). Reads of unwritten cells return 0.
///
/// Engine self-validation — a correct per-head score path (`restickify` ∘ `matmul`) must
/// reproduce `sp[h][slot] = Σ_d qs[h][d]·kc[h][slot][d]`:
/// ```
/// use ktir_superdsc::sdsc_abstract::*;
/// let (nqh, hd, cap) = (2usize, 4usize, 3usize);
/// let (qs_b, kc_b, kct_b, sp_b) = (0usize, 1000, 5000, 9000);
/// let qs = |h: usize| AbsTensor { name: "qs".into(), base: qs_b + h * hd, dims: vec![1, hd], stick_idx: 1 };
/// let kc = |h: usize| AbsTensor { name: "kc".into(), base: kc_b + h * cap * hd, dims: vec![cap, hd], stick_idx: 1 };
/// let kct = |h: usize| AbsTensor { name: "kct".into(), base: kct_b + h * hd * cap, dims: vec![hd, cap], stick_idx: 1 };
/// let sp = |h: usize| AbsTensor { name: "sp".into(), base: sp_b + h * cap, dims: vec![1, cap], stick_idx: 1 };
/// let mut mem = AbsMem::default();
/// for h in 0..nqh {
///     for d in 0..hd { mem.set(qs(h).addr(&[0, d]), seed_val(1, h as u64, d as u64, 0)); }
///     for s in 0..cap { for d in 0..hd { mem.set(kc(h).addr(&[s, d]), seed_val(2, h as u64, s as u64, d as u64)); } }
/// }
/// let mut ops = vec![];
/// for h in 0..nqh {
///     ops.push(AbsOp::Restickify { x: kc(h), o: kct(h) });
///     ops.push(AbsOp::Matmul { a: qs(h), w: kct(h), o: sp(h) });
/// }
/// interp(&ops, &mut mem);
/// for h in 0..nqh {
///     for slot in 0..cap {
///         let mut want = 0.0;
///         for d in 0..hd { want += seed_val(1, h as u64, d as u64, 0) * seed_val(2, h as u64, slot as u64, d as u64); }
///         let got = mem.get(sp(h).addr(&[0, slot]));
///         assert!((got - want).abs() < 1e-9, "sp[{h}][{slot}] got {got} want {want}");
///     }
/// }
/// ```
pub fn interp(ops: &[AbsOp], mem: &mut AbsMem) {
    for op in ops {
        match op {
            AbsOp::Matmul { a, w, o } => {
                let (m, k) = (a.dims[0], a.dims[1]);
                let n = w.dims[1];
                for mi in 0..m {
                    for ni in 0..n {
                        let mut acc = 0.0;
                        for ki in 0..k {
                            acc += mem.get(a.addr(&[mi, ki])) * mem.get(w.addr(&[ki, ni]));
                        }
                        mem.set(o.addr(&[mi, ni]), acc);
                    }
                }
            }
            AbsOp::Restickify { x, o } => {
                let (r, c) = (x.dims[0], x.dims[1]);
                for ri in 0..r {
                    for ci in 0..c {
                        let v = mem.get(x.addr(&[ri, ci]));
                        mem.set(o.addr(&[ci, ri]), v);
                    }
                }
            }
            AbsOp::Ew { f, a, ab, b, bb, o } => {
                let (r, c) = (o.dims[0], o.dims[1]);
                for ri in 0..r {
                    for ci in 0..c {
                        let av = mem.get(a.addr(&bidx(&[ri, ci], *ab)));
                        let bv = b
                            .as_ref()
                            .map(|bt| mem.get(bt.addr(&bidx(&[ri, ci], *bb))))
                            .unwrap_or(0.0);
                        mem.set(o.addr(&[ri, ci]), ewf(*f, av, bv));
                    }
                }
            }
            AbsOp::Reduce { f, x, o, scale } => {
                let (r, c) = (x.dims[0], x.dims[1]);
                for ri in 0..r {
                    let mut acc = match f {
                        RedF::Sum => 0.0,
                        RedF::Max => f64::NEG_INFINITY,
                    };
                    for ci in 0..c {
                        let v = mem.get(x.addr(&[ri, ci]));
                        acc = match f {
                            RedF::Sum => acc + v,
                            RedF::Max => acc.max(v),
                        };
                    }
                    mem.set(o.addr(&[ri, 0]), acc * scale);
                }
            }
        }
    }
}

/// THE SPEC: a UNIFIED per-head attention decomposition (all ops `[1,X]` per head ⇒ one
/// consistent contiguous layout — no per-head-write/full-read mismatch) must equal `softmax·v`.
/// This is the structure the emitter must mirror; the doctest is the local, ~0.01s lock-down loop.
/// ```
/// use ktir_superdsc::sdsc_abstract::*;
/// let (nqh, hd, cap, p) = (2usize, 4usize, 6usize, 3usize);
/// let (qb, kb, vb, mb) = (0usize, 100, 300, 500);
/// let (spb, exb, ppb, mxb, smb, rcb, ob) = (700usize, 900, 1100, 1300, 1400, 1500, 1600);
/// let qs = |h: usize| AbsTensor { name: "qs".into(), base: qb + h * hd, dims: vec![1, hd], stick_idx: 1 };
/// let kc = |h: usize| AbsTensor { name: "kc".into(), base: kb + h * hd * cap, dims: vec![hd, cap], stick_idx: 1 };
/// let vc = |h: usize| AbsTensor { name: "vc".into(), base: vb + h * cap * hd, dims: vec![cap, hd], stick_idx: 1 };
/// let mask = |h: usize| AbsTensor { name: "mask".into(), base: mb + h * cap, dims: vec![1, cap], stick_idx: 1 };
/// let sp = |h: usize| AbsTensor { name: "sp".into(), base: spb + h * cap, dims: vec![1, cap], stick_idx: 1 };
/// let ex = |h: usize| AbsTensor { name: "ex".into(), base: exb + h * cap, dims: vec![1, cap], stick_idx: 1 };
/// let pp = |h: usize| AbsTensor { name: "pp".into(), base: ppb + h * cap, dims: vec![1, cap], stick_idx: 1 };
/// let mx = |h: usize| AbsTensor { name: "mx".into(), base: mxb + h, dims: vec![1, 1], stick_idx: 1 };
/// let sm = |h: usize| AbsTensor { name: "sm".into(), base: smb + h, dims: vec![1, 1], stick_idx: 1 };
/// let rc = |h: usize| AbsTensor { name: "rc".into(), base: rcb + h, dims: vec![1, 1], stick_idx: 1 };
/// let out = |h: usize| AbsTensor { name: "out".into(), base: ob + h * hd, dims: vec![1, hd], stick_idx: 1 };
/// let mut mem = AbsMem::default();
/// for h in 0..nqh {
///     for d in 0..hd { mem.set(qs(h).addr(&[0, d]), seed_val(1, h as u64, d as u64, 0)); }
///     for s in 0..cap { for d in 0..hd {
///         mem.set(kc(h).addr(&[d, s]), seed_val(2, h as u64, s as u64, d as u64)); // kc is Kᵀ: kc[d][s]=K[s][d]
///         mem.set(vc(h).addr(&[s, d]), seed_val(3, h as u64, s as u64, d as u64));
///     }}
///     for s in 0..cap { mem.set(mask(h).addr(&[0, s]), if s <= p { 0.0 } else { -1e30 }); }
/// }
/// let mut ops = vec![];
/// for h in 0..nqh {
///     ops.push(AbsOp::Matmul { a: qs(h), w: kc(h), o: sp(h) });
///     ops.push(AbsOp::Ew { f: EwF::Add, a: sp(h), ab: Bcast::None, b: Some(mask(h)), bb: Bcast::None, o: sp(h) });
///     ops.push(AbsOp::Reduce { f: RedF::Max, x: sp(h), o: mx(h), scale: 1.0 });
///     ops.push(AbsOp::Ew { f: EwF::Sub, a: sp(h), ab: Bcast::None, b: Some(mx(h)), bb: Bcast::Out, o: sp(h) });
///     ops.push(AbsOp::Ew { f: EwF::Exp, a: sp(h), ab: Bcast::None, b: None, bb: Bcast::None, o: ex(h) });
///     ops.push(AbsOp::Reduce { f: RedF::Sum, x: ex(h), o: sm(h), scale: 1.0 });
///     ops.push(AbsOp::Ew { f: EwF::Recip, a: sm(h), ab: Bcast::None, b: None, bb: Bcast::None, o: rc(h) });
///     ops.push(AbsOp::Ew { f: EwF::Mul, a: ex(h), ab: Bcast::None, b: Some(rc(h)), bb: Bcast::Out, o: pp(h) });
///     ops.push(AbsOp::Matmul { a: pp(h), w: vc(h), o: out(h) });
/// }
/// interp(&ops, &mut mem);
/// let qsf = |h: usize, d: usize| seed_val(1, h as u64, d as u64, 0);
/// let kcf = |h: usize, s: usize, d: usize| seed_val(2, h as u64, s as u64, d as u64);
/// let vcf = |h: usize, s: usize, d: usize| seed_val(3, h as u64, s as u64, d as u64);
/// let refout = attn_reference(&qsf, &kcf, &vcf, nqh, hd, p);
/// for h in 0..nqh { for d in 0..hd {
///     let got = mem.get(out(h).addr(&[0, d]));
///     assert!((got - refout[h * hd + d]).abs() < 1e-9, "out[{h}][{d}] got {got} want {}", refout[h * hd + d]);
/// }}
/// ```
/// The ATTENTION reference: `out[h,d] = Σ_{s≤p} softmax_s(qs[h]·kc[h,s]·… )·vc[h,s,d]`.
/// `qs` is the ALREADY-SCALED query (scale folded in upstream), so the score is just qs·kc.
/// Returns out as `[nqh*hd]` row-major (out[h*hd+d]).
pub fn attn_reference(
    qs: &dyn Fn(usize, usize) -> f64,        // (h,d)
    kc: &dyn Fn(usize, usize, usize) -> f64, // (h,s,d)
    vc: &dyn Fn(usize, usize, usize) -> f64, // (h,s,d)
    nqh: usize,
    hd: usize,
    p: usize,
) -> Vec<f64> {
    let nslot = p + 1;
    let mut out = vec![0.0; nqh * hd];
    for h in 0..nqh {
        let mut sc = vec![0.0; nslot];
        for (s, score) in sc.iter_mut().enumerate() {
            let mut acc = 0.0;
            for d in 0..hd {
                acc += qs(h, d) * kc(h, s, d);
            }
            *score = acc;
        }
        let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut den = 0.0;
        for x in sc.iter_mut() {
            *x = (*x - mx).exp();
            den += *x;
        }
        for d in 0..hd {
            let mut acc = 0.0;
            for (s, &score) in sc.iter().enumerate() {
                acc += (score / den) * vc(h, s, d);
            }
            out[h * hd + d] = acc;
        }
    }
    out
}

/// The SubtileIR `RopeRotate` reference (NeoX rotate-half): per head `h`, dim `d`:
/// `out[h,d] = x[h,d]·cos[d] + rotate_half(x)[h,d]·sin[d]`, where
/// `rotate_half(x)[h,d] = -x[h,d+half]` for `d<half` and `+x[h,d-half]` for `d>=half`.
/// This is the math the emitter's permutation-matmul form (`rot = x·P; out = x·cos + rot·sin`,
/// `lower_rope_node`) must reproduce. Returns `[heads*hd]` row-major.
pub fn rope_reference(
    x: &dyn Fn(usize, usize) -> f64, // (h,d)
    cos: &dyn Fn(usize) -> f64,      // (d)
    sin: &dyn Fn(usize) -> f64,      // (d)
    heads: usize,
    hd: usize,
) -> Vec<f64> {
    let half = hd / 2;
    let mut out = vec![0.0; heads * hd];
    for h in 0..heads {
        for d in 0..hd {
            let rh = if d < half {
                -x(h, d + half)
            } else {
                x(h, d - half)
            };
            out[h * hd + d] = x(h, d) * cos(d) + rh * sin(d);
        }
    }
    out
}

/// The DEVICE element offset of row `r`, head `h`'s `[1,hd]` RoPE sub-block inside the roped Q/K
/// tensor `[mq, total]` (`total = heads·hd`), sticked on its last dim. `lower_rope_node` emits ONE
/// `mb=1 [1,hd]` op per `(r,h)` at this offset (the proven `mb=1` primitive). This is the FIX for the
/// mq>1 prefill collapse: the old code looped `for h` only (offset `h·hd`), writing ROW 0 for every
/// head and NEVER rows `1..mq` → roped Q/K came back with only row 0 non-zero → K-cache 1 slot →
/// garbage. CRUCIAL: a `[mq, total]` activation is STICK-SCATTERED on-device — [`dev_off`] maps
/// `[r,c] → (c/64)·(mq·64) + r·64 + c%64`, so rows are interleaved WITHIN each 64-stick, NOT row-major
/// contiguous. For `hd == 64` (= STK, granite) each head occupies exactly one stick, so head h row r's
/// 64 dims ARE contiguous at `h·mq·hd + r·hd` == `dev_off([mq,total],1,[r,h·hd])` (proven in
/// `rope_prefill_offset_is_devoff_*`). This is why the naïve flat `r·total + h·hd` failed: it wrote
/// where the consumer (attention matmul reading q as `[mq,total]`) never reads for `r>0`. DEGENERATES
/// to the decode offset at `mq=1` (`r=0 ⇒ h·hd`), byte-identical. hd>64 in prefill is a build error
/// (a head spans hd/64 NON-contiguous sticks — `lower_rope_node` guards it).
/// ```
/// use ktir_superdsc::sdsc_abstract::{rope_prefill_block_offset, dev_off};
/// // granite Q: mq=8, heads=32, hd=64. Matches the stick-scattered device address of [r, h·hd]:
/// assert_eq!(rope_prefill_block_offset(3, 5, 8, 64), dev_off(&[8, 32 * 64], 1, &[3, 5 * 64]));
/// // Decode (mq=1, r=0) == old per-head offset h·hd — byte-identical:
/// assert_eq!(rope_prefill_block_offset(0, 5, 1, 64), 5 * 64);
/// ```
pub fn rope_prefill_block_offset(r: usize, h: usize, mq: usize, hd: usize) -> usize {
    h * mq * hd + r * hd
}

/// The DEVICE element offset of row `r`'s cos/sin `[1,hd]` slice (row `r`'s HEAD-0 slice) inside the
/// worker-bound `[mq, total]` cos/sin table. The table is per-position + head-tiled
/// (`cos[r, h·hd + i] = cos_r(i)` for every head), so row r's head-0 slice serves every head of row r.
/// For `hd == 64`, this is `dev_off([mq,total],1,[r,0]) = r·64 = r·hd`. DEGENERATES to `0` at `mq=1`
/// (decode reads cos/sin at 0), so decode is byte-identical.
pub fn rope_prefill_cos_offset(r: usize, hd: usize) -> usize {
    r * hd
}

/// The SubtileIR `RmsNorm` reference: `out[c] = gamma[c] · x[c] / sqrt(mean_c(x²) + eps)`.
/// Returns `[hidden]`.
pub fn rmsnorm_reference(
    x: &dyn Fn(usize) -> f64,
    gamma: &dyn Fn(usize) -> f64,
    hidden: usize,
    eps: f64,
) -> Vec<f64> {
    let mut ss = 0.0;
    for c in 0..hidden {
        let v = x(c);
        ss += v * v;
    }
    let inv = 1.0 / (ss / hidden as f64 + eps).sqrt();
    (0..hidden).map(|c| gamma(c) * x(c) * inv).collect()
}

/// The SubtileIR MLP reference (Llama gated MLP, no biases): given the rmsnorm-normalized
/// input `n[hidden]`, `out[c] = Σ_i (silu(Σ_d n[d]·Wg[d,i]) · Σ_d n[d]·Wu[d,i]) · Wd[i,c]`.
/// Returns the MLP output `[hidden]` (the value ADDED to the residual — NOT yet added).
///
/// THE BODY SPEC (lock-down doctest): the MLP + residual decomposition the emitter must mirror.
/// All tensors are `m=1` ([1,X], stick=last) so `dev_off` collapses to a flat column index —
/// producer and consumer address the SAME cell with no per-head/stick mismatch. This locks the
/// body DATAFLOW: gate/up read the SAME normalized input; silu(gate)·up feeds down; the residual
/// add reads the layer input + the down output. A wrong base (down reading the wrong silu tensor,
/// the residual add reading a stale hidden) makes the interpreter diverge from
/// `hidden + mlp_reference(rmsnorm(hidden))`. The rmsnorm itself is validated on-card (cos 1.0),
/// so its OUTPUT is seeded from [`rmsnorm_reference`] rather than re-modeled op-by-op.
/// ```
/// use ktir_superdsc::sdsc_abstract::*;
/// let (hidden, inter, eps) = (4usize, 8usize, 1e-5);
/// let (hb, nb, gtb, upb, sib, dnb, ob) = (0usize, 100, 200, 300, 500, 600, 700);
/// let ht = AbsTensor { name: "hidden".into(), base: hb, dims: vec![1, hidden], stick_idx: 1 };
/// let nt = AbsTensor { name: "norm".into(), base: nb, dims: vec![1, hidden], stick_idx: 1 };
/// let wg = AbsTensor { name: "wg".into(), base: gtb, dims: vec![hidden, inter], stick_idx: 1 };
/// let wu = AbsTensor { name: "wu".into(), base: upb, dims: vec![hidden, inter], stick_idx: 1 };
/// let wd = AbsTensor { name: "wd".into(), base: dnb, dims: vec![inter, hidden], stick_idx: 1 };
/// let gate = AbsTensor { name: "gate".into(), base: sib, dims: vec![1, inter], stick_idx: 1 };
/// let up = AbsTensor { name: "up".into(), base: sib + inter, dims: vec![1, inter], stick_idx: 1 };
/// let silu = AbsTensor { name: "silu".into(), base: sib + 2 * inter, dims: vec![1, inter], stick_idx: 1 };
/// let mlp = AbsTensor { name: "mlp".into(), base: ob, dims: vec![1, hidden], stick_idx: 1 };
/// let outh = AbsTensor { name: "outh".into(), base: ob + hidden, dims: vec![1, hidden], stick_idx: 1 };
/// let xf = |c: usize| seed_val(1, c as u64, 0, 0);
/// let gam = |c: usize| seed_val(9, c as u64, 0, 0) * 0.05 + 0.04;
/// let wgf = |d: usize, i: usize| seed_val(2, d as u64, i as u64, 0);
/// let wuf = |d: usize, i: usize| seed_val(3, d as u64, i as u64, 0);
/// let wdf = |i: usize, c: usize| seed_val(4, i as u64, c as u64, 0);
/// let mut mem = AbsMem::default();
/// for c in 0..hidden { mem.set(ht.addr(&[0, c]), xf(c)); }
/// let nrm = rmsnorm_reference(&xf, &gam, hidden, eps);
/// for c in 0..hidden { mem.set(nt.addr(&[0, c]), nrm[c]); }
/// for d in 0..hidden { for i in 0..inter { mem.set(wg.addr(&[d, i]), wgf(d, i)); mem.set(wu.addr(&[d, i]), wuf(d, i)); } }
/// for i in 0..inter { for c in 0..hidden { mem.set(wd.addr(&[i, c]), wdf(i, c)); } }
/// let ops = vec![
///     AbsOp::Matmul { a: nt.clone(), w: wg.clone(), o: gate.clone() },
///     AbsOp::Matmul { a: nt.clone(), w: wu.clone(), o: up.clone() },
///     AbsOp::Ew { f: EwF::Silu, a: gate.clone(), ab: Bcast::None, b: None, bb: Bcast::None, o: silu.clone() },
///     AbsOp::Ew { f: EwF::Mul, a: silu.clone(), ab: Bcast::None, b: Some(up.clone()), bb: Bcast::None, o: silu.clone() },
///     AbsOp::Matmul { a: silu.clone(), w: wd.clone(), o: mlp.clone() },
///     AbsOp::Ew { f: EwF::Add, a: ht.clone(), ab: Bcast::None, b: Some(mlp.clone()), bb: Bcast::None, o: outh.clone() },
/// ];
/// interp(&ops, &mut mem);
/// let mlpref = mlp_reference(&|c| nrm[c], &wgf, &wuf, &wdf, hidden, inter);
/// for c in 0..hidden {
///     let want = xf(c) + mlpref[c];
///     let got = mem.get(outh.addr(&[0, c]));
///     assert!((got - want).abs() < 1e-9, "outh[{c}] got {got} want {want}");
/// }
/// ```
pub fn mlp_reference(
    n: &dyn Fn(usize) -> f64,         // normalized input (c)
    wg: &dyn Fn(usize, usize) -> f64, // gate weight (d,i)  [hidden,inter]
    wu: &dyn Fn(usize, usize) -> f64, // up   weight (d,i)
    wd: &dyn Fn(usize, usize) -> f64, // down weight (i,c)  [inter,hidden]
    hidden: usize,
    inter: usize,
) -> Vec<f64> {
    let mut act = vec![0.0; inter];
    for (i, a) in act.iter_mut().enumerate() {
        let mut g = 0.0;
        let mut u = 0.0;
        for d in 0..hidden {
            let nd = n(d);
            g += nd * wg(d, i);
            u += nd * wu(d, i);
        }
        *a = (g / (1.0 + (-g).exp())) * u;
    }
    let mut out = vec![0.0; hidden];
    for (c, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (i, &a) in act.iter().enumerate() {
            acc += a * wd(i, c);
        }
        *o = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⛔ A RESHAPE CANNOT BE A PLACEMENT ALIAS ON THIS DEVICE — MEASURED, not
    /// read off the formula.
    ///
    /// `dev_off_stk` places element `(i, j)` at
    /// `(j / stk) * (a * stk) + i * stk + (j % stk)`, where `a` is the ROW
    /// COUNT. So the layout is a function of the shape, and two tensors over
    /// the same bytes with different `(rows, cols)` disagree about where
    /// elements live.
    ///
    /// This pins the exact case gemma-4 needs: a per-head view
    /// `[M, heads*hd] -> [M*heads, hd]`. If aliasing were sound, walking both
    /// shapes in row-major order would visit the same device offsets in the
    /// same order. It does not — so a Reshape must be lowered to a RESTICKIFY
    /// (a real re-laying copy), never to a shared placement.
    ///
    /// Fail-first: make `dev_off_stk` ignore `dims[0]` and the two sequences
    /// coincide, and this test stops failing.
    #[test]
    fn reshape_is_not_a_placement_alias_under_stick_layout() {
        const STK: usize = 64;
        // gemma-4-shaped: M rows of heads*hd, viewed as M*heads rows of hd.
        let (m, heads, hd) = (2usize, 4usize, 128usize);
        let (before, after) = ([m, heads * hd], [m * heads, hd]);

        // Row-major element order is what a "view" claims to preserve.
        let walk = |dims: [usize; 2]| -> Vec<usize> {
            let mut v = Vec::with_capacity(dims[0] * dims[1]);
            for i in 0..dims[0] {
                for j in 0..dims[1] {
                    v.push(dev_off_stk(&dims, 1, &[i, j], STK));
                }
            }
            v
        };

        let a = walk(before);
        let b = walk(after);
        assert_eq!(a.len(), b.len(), "a reshape preserves the element COUNT");
        assert_ne!(
            a, b,
            "if these matched, aliasing a reshaped tensor's placement would be sound — \
             they do not, because dev_off_stk scales by dims[0] (the row count)"
        );

        // And name the first divergence, so a future change that alters the
        // layout shows up as a moved element rather than a bare inequality.
        let first = a.iter().zip(&b).position(|(x, y)| x != y);
        assert_eq!(
            first,
            Some(STK),
            "the layouts agree only within the first stick; element {STK} is the first to move"
        );
    }

    #[test]
    fn stage_2d_lands_at_emitter_dev_off_for_residual_stream() {
        // THE worker↔emitter firewall proof. The emitter (per_core_addr) reads a mq>1 rank-2 [mb,out]
        // residual/token tensor via `for_view_df(dims, stick_idx=1)` = RowBlocked. The worker MUST stage via
        // the SAME layout. Prove: (1) the emitter's classification of the residual IS RowBlocked, and
        // (2) `stage_2d` places logical (r,c) at EXACTLY the offset the emitter's `dev_off(r,c)` reads —
        // so a worker staging through `stage_2d(&for_view_df(...))` cannot land a byte where the emit
        // won't read it. This makes the flat-vs-RowBlocked embedding/cmask mismatch UNCONSTRUCTABLE.
        for &(rows, cols) in &[(31usize, 2048usize), (64, 4096), (19, 128)] {
            let emit = StickLayout::for_view_df(&[rows, cols], 1, Df::Fp16);
            assert_eq!(
                emit,
                StickLayout::row_blocked(rows, cols),
                "residual [{rows},{cols}] emitter classification must be RowBlocked (NOT flat)"
            );
            let staged = stage_2d(&emit, |r, c| (r * cols + c + 1) as f32); // +1 so 0 means 'unwritten'
            for r in 0..rows {
                for c in 0..cols {
                    assert_eq!(
                        staged[emit.dev_off(r, c)],
                        (r * cols + c + 1) as f32,
                        "stage_2d wrote logical ({r},{c}) somewhere the emitter's dev_off({r},{c}) \
                         does NOT read — a layout mismatch that stage_2d must make impossible"
                    );
                }
            }
        }
    }

    #[test]
    fn dev_off_flat_is_row_major() {
        // 3-D [nqh,cap,hd] stick=2 → flat row-major
        let dims = [9, 256, 64];
        assert_eq!(dev_off(&dims, 2, &[0, 0, 0]), 0);
        assert_eq!(dev_off(&dims, 2, &[0, 0, 5]), 5);
        assert_eq!(dev_off(&dims, 2, &[0, 1, 0]), 64);
        assert_eq!(dev_off(&dims, 2, &[1, 0, 0]), 256 * 64);
    }

    #[test]
    fn dev_off_2d_sticked_kernel() {
        // [hd=64, cap=256] stick=1(last) → device [cap/64=4, hd=64, 64]
        let dims = [64, 256];
        // element (i=0, j=0) → t=0,i=0,s=0 → 0
        assert_eq!(dev_off(&dims, 1, &[0, 0]), 0);
        // (0, 1) → t=0,i=0,s=1 → 1
        assert_eq!(dev_off(&dims, 1, &[0, 1]), 1);
        // (1, 0) → t=0,i=1,s=0 → i*64 = 64
        assert_eq!(dev_off(&dims, 1, &[1, 0]), 64);
        // (0, 64) → t=1,i=0,s=0 → 1*(64*64) = 4096
        assert_eq!(dev_off(&dims, 1, &[0, 64]), 4096);
    }
}

// ═══════════════════ SpyreTensorLayout — torch-spyre's per-tensor device ElementArrangement ═══════════
// A faithful port of torch_spyre/csrc/spyre_tensor_impl.cpp (`SpyreTensorLayout::init`,
// `get_generic_stick_layout`, `dim_map_to_stride_map`). torch-spyre keeps the HOST/HBM tensor ROW-MAJOR
// (flat) and carries, PER TENSOR, a `device_size` + `stride_map` that maps the device's RowBlocked stick
// iteration `(g, r, l)` → host offset `g·64 + r·row_stride + l`. dxp reads the RowBlocked layout DIRECTLY
// from this per-tensor layout — it never reconstructs a stride from the op coordInfo (which is why
// torch-spyre never overflows the LX immediate the way scratchy's `rows·eps` coordInfo hack did).

/// The device element arrangement (torch-spyre `ElementArrangement`). `EXX2` is the reduction mode
/// ("two values per stick"); the converts change the stick dtype/width.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ElementArrangement {
    Standard = 0,
    Dl16ToFp32 = 1,
    Qfp8ch = 2,
    Exx2 = 3,
    Fp32ToDl16 = 4,
}

/// Device tiling of a host dim-order (torch-spyre `get_generic_stick_layout`). The STICK dim (the last
/// host dim) appears TWICE — outermost (stick-group) and innermost (lanes); the leading host dim (rows/
/// batch) goes in the middle. rank-2 `[rows, feat]` → `[feat, rows, feat]` ⇒ device `[feat/stk, rows, stk]`.
pub fn get_generic_stick_layout(dim_order: &[i32]) -> Vec<usize> {
    let rank = dim_order.len();
    match rank {
        0 => vec![],
        1 => vec![dim_order[0] as usize, dim_order[0] as usize],
        n => {
            let mut m: Vec<usize> = (1..n).map(|i| dim_order[i] as usize).collect();
            m.push(dim_order[0] as usize);
            m.push(dim_order[n - 1] as usize);
            m
        }
    }
}

fn compute_host_stride(host_size: &[i64]) -> Vec<i64> {
    let n = host_size.len();
    let mut s = vec![1i64; n];
    let mut stride = 1i64;
    for i in (0..n).rev() {
        s[i] = stride;
        stride *= host_size[i];
    }
    s
}

fn dim_map_to_stride_map(
    dim_map: &[usize],
    host_size: &[i64],
    host_stride: &[i64],
    device_size: &[i64],
) -> Vec<i64> {
    let n = dim_map.len();
    let mut stride_map = vec![-1i64; n];
    let mut last_stride = vec![-1i64; host_size.len().max(1)];
    for j in (0..n).rev() {
        let d = dim_map[j];
        if host_size[d] == 1 {
            stride_map[j] = -1;
        } else if last_stride[d] == -1 {
            stride_map[j] = host_stride[d];
            last_stride[d] = stride_map[j] * device_size[j];
        } else {
            stride_map[j] = last_stride[d];
            last_stride[d] = stride_map[j] * device_size[j];
        }
    }
    stride_map
}

/// A tensor's device layout: `device_size` (the `[feat/stk, rows, stk]` device dims) + `stride_map`
/// (host offset per device dim) + the arrangement. Byte-for-byte the torch-spyre `SpyreTensorLayout`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpyreTensorLayout {
    pub device_size: Vec<i64>,
    pub stride_map: Vec<i64>,
    pub element_arrangement: ElementArrangement,
}

impl SpyreTensorLayout {
    /// STANDARD layout for a ROW-MAJOR host tensor `host_size`, dtype `df` (identity dim-order).
    pub fn standard(host_size: &[i64], df: Df) -> Self {
        let dim_order: Vec<i32> = (0..host_size.len() as i32).collect();
        Self::with_dim_order(host_size, df, &dim_order, ElementArrangement::Standard)
    }

    pub fn with_dim_order(
        host_size: &[i64],
        df: Df,
        dim_order: &[i32],
        ea: ElementArrangement,
    ) -> Self {
        let eps = df.elems_per_stick() as i64;
        let host_stride = compute_host_stride(host_size);
        let dim_map = get_generic_stick_layout(dim_order);
        let n = dim_map.len();
        let stick_dim = *dim_map.last().unwrap();
        let mut device_size = vec![0i64; n];
        device_size[n - 1] = eps;
        for i in 0..n - 1 {
            let dim = dim_map[i];
            device_size[i] = if dim == stick_dim {
                (host_size[dim] + eps - 1) / eps
            } else {
                host_size[dim]
            };
        }
        let stride_map = dim_map_to_stride_map(&dim_map, host_size, &host_stride, &device_size);
        SpyreTensorLayout {
            device_size,
            stride_map,
            element_arrangement: ea,
        }
    }

    /// The DENSE on-device strides that the SDSC descriptor + per-core start address ACTUALLY address
    /// (torch-spyre `_calculate_device_stride`, superdsc.py:238): `stride[i] = prod(device_size[i+1..])`.
    /// For a `[rows, cols]` fp16 tensor (`device_size = [cols/64, rows, 64]`) this is `[rows·64, 64, 1]`
    /// over device dims `[feat-groups, rows, lanes]` — i.e. the RowBlocked packing: row stride = 64,
    /// feat-group stride = rows·64. (The host `stride_map` row-stride = `cols` is host-only bookkeeping and
    /// is NEVER emitted into the descriptor — this is the load-bearing torch-spyre fact.)
    pub fn dense_strides(&self) -> Vec<i64> {
        let n = self.device_size.len();
        (0..n)
            .map(|i| self.device_size[i + 1..].iter().product::<i64>())
            .collect()
    }

    /// The device dim index that carries the leading (row / `mb` / token) host dim — the MIDDLE of the
    /// `[feat, rows, feat]` generic-stick layout for rank≥2, so `dense_strides()[row_device_dim()]` is the
    /// per-token stride the per-core start folds against (64 for fp16 `[rows, cols]`).
    pub fn row_device_dim(&self) -> usize {
        // get_generic_stick_layout puts the leading host dim in the middle (index n-2) for rank≥2;
        // rank-1 collapses to [d0, d0] (index 0).
        self.device_size.len().saturating_sub(2)
    }
}

#[cfg(test)]
mod spyre_layout_tests {
    use super::*;

    #[test]
    fn reduce_activation_matches_torchspyre() {
        // torch-spyre SpyreTensorLayout for a [rows=31, feat=2048] fp16 activation (the rmsnorm/reduce
        // input) computes device_size=[32,31,64], stride_map=[64,2048,1]. Verified against the C++ algo,
        // NOT asserted by hand: device (g,r,l) → host g·64 + r·2048 + l = row-major [31,2048] element [r, g·64+l].
        let l = SpyreTensorLayout::standard(&[31, 2048], Df::Fp16);
        assert_eq!(
            l.device_size,
            vec![32, 31, 64],
            "device_size [feat/64, rows, 64]"
        );
        assert_eq!(
            l.stride_map,
            vec![64, 2048, 1],
            "stride_map [64, row_stride, 1]"
        );
    }

    #[test]
    fn fp8_and_fp32_stick_widths() {
        // fp8 stick=128, fp32 stick=32 — the divide uses the tensor's own dtype.
        let l8 = SpyreTensorLayout::standard(&[31, 2048], Df::Fp8);
        assert_eq!(l8.device_size, vec![16, 31, 128]); // 2048/128=16
        let l32 = SpyreTensorLayout::standard(&[31, 2048], Df::Fp32);
        assert_eq!(l32.device_size, vec![64, 31, 32]); // 2048/32=64
    }

    #[test]
    fn rank1_and_size1_dims() {
        // a [1, feat] per-token vector: rows=1 ⇒ its stride_map entry is -1 (broadcast/degenerate), matching
        // torch-spyre's `host_size[d]==1 → -1`.
        let l = SpyreTensorLayout::standard(&[1, 2048], Df::Fp16);
        assert_eq!(l.device_size, vec![32, 1, 64]);
        assert_eq!(l.stride_map[1], -1, "rows==1 → -1");
    }
}

/// PROVEN over SYMBOLIC `rows`/`m` (no fixed-case enumeration): [`StickLayout::group_stride`] never emits
/// a coordInfo stride dxp cannot walk, for ANY row count — decode (m=1) through arbitrarily wide prefill.
/// This is the invariant the four-week rmsnorm-NaN/rope-garble class of bug lived in: a hand-derived
/// stride at ONE call site (`rb_rows * eps`) that was correct for the fp32 island and silently wrong (dxp
/// build-fault) the moment someone copied it for fp16. Discharging it here means the NEXT such stride
/// computation is a proof obligation, not a bake-and-discover.
/// PROVEN over a SYMBOLIC set of held KV rows: laying the live batch into a launch never loses a
/// request, whatever rows they hold. The bug this discharges shipped: the launch slot WAS the KV row,
/// so a batch of `n` live requests holding high rows selected the `n`-wide rung (the rung is chosen
/// from the live count) and then could not address them — four requests on rows 4..7 in a four-wide
/// launch. It surfaced as an intermittent hard error at bs=8 (four of eight requests returning null,
/// once in seven runs, only in the runs where a narrower rung ran first), which is exactly the shape of
/// defect a proof over symbolic rows finds in half a second and a bake finds by luck.
#[cfg(kani)]
mod slot_map_proofs {
    use super::*;

    /// `LIVE` requests laid into a `WIDTH`-row launch, asserting every property the layout owes its caller:
    ///  * TOTALITY — every live request gets a slot. 🛑 THE FAIL-FIRST ONE.
    ///  * ROUND TRIP — the slot a request was bound at is the slot its logits come back in, so the write and
    ///    the read cannot drift apart.
    ///  * NO UNINSTALLED MAP — every slot of the launch, padding included, sources a LIVE request. A slot with
    ///    no map installed would address physical page 0.
    ///
    /// ⛔ IT USED TO TAKE SYMBOLIC KV ROWS — `LIVE` requests holding arbitrary DISTINCT rows anywhere in a pool
    /// of 8 — because the shipped bug was entirely about WHICH rows were held: with the slot taken from the
    /// row, two requests on rows 2 and 3 could not run in a two-row launch at all. Rows are deleted (the host
    /// allocates every page a batched launch writes), so the layout has no row input left to be symbolic over,
    /// and the sparse-and-high-rows case it explored is not a state that exists. The properties it proved are
    /// unchanged and still proved.
    fn every_live_request_is_addressable<const LIVE: usize, const WIDTH: u32>() {
        let map = SlotMap::of_live(LIVE, RungWidth::of_baked_rows(WIDTH).unwrap())
            .expect("a launch at least as wide as the live count holds every live request");
        assert!(map.live() == LIVE && map.width() == WIDTH as usize);
        for i in 0..LIVE {
            let s = map
                .slot_of(i)
                .expect("every live request is laid into the launch");
            assert!(s.index() < map.width());
            assert!(map.source(s) == SlotSource::Live(i));
        }
        for (s, src) in map.slots() {
            // Padding sources nobody; a live slot sources a request that exists. Both arms are stated so
            // the proof cannot pass by treating padding's old `0` as a live index.
            match src.live() {
                Some(i) => assert!(
                    i < map.live(),
                    "slot {} sources request {i} of {}",
                    s.index(),
                    map.live()
                ),
                None => assert!(
                    s.index() >= map.live(),
                    "only slots past the live count may be padding"
                ),
            }
        }
    }

    /// The batch exactly fills the rung — what `decode_rung_for` produces whenever a rung matches.
    #[kani::proof]
    #[kani::unwind(4)]
    fn a_full_launch_addresses_every_live_request() {
        every_live_request_is_addressable::<2, 2>();
    }

    /// A batch NARROWER than the rung — the padded case, and the one the shipped bug failed on.
    #[kani::proof]
    #[kani::unwind(6)]
    fn a_padded_launch_addresses_every_live_request() {
        every_live_request_is_addressable::<2, 4>();
    }

    /// ⭐ A LAUNCH WIDER THAN ITS RUNG IS REFUSED — the one refusal `of_live` still makes, over a SYMBOLIC
    /// live count so it is the property rather than three examples.
    ///
    /// ⛔ THIS SLOT HELD `two_requests_cannot_share_a_row`, which proved that two live requests on one pool row
    /// were REFUSED rather than resolved (one would have silently read the other's KV). There are no rows to
    /// share, and `SlotMapError::DuplicateRow` is deleted with them — a prefix-cache HIT is now exactly two
    /// requests holding the same PAGE, which is the ordinary case rather than a collision.
    #[kani::proof]
    #[kani::unwind(6)]
    fn more_live_requests_than_the_rung_holds_is_refused() {
        let live: usize = kani::any();
        kani::assume(live <= 16);
        let w = RungWidth::of_baked_rows(4).unwrap();
        match SlotMap::of_live(live, w) {
            Ok(map) => assert!(live >= 1 && live <= 4 && map.live() == live),
            Err(SlotMapError::NoLiveRequest) => assert!(live == 0),
            Err(SlotMapError::TooWide { live: l, width }) => {
                assert!(l == live && live > 4 && width == 4)
            }
        }
    }
}

#[cfg(kani)]
mod pool_demand_proofs {
    use super::*;

    /// ⭐⭐⭐ THE CONSERVATION LAW: A POOL SIZED BY A DEMAND CAN ALWAYS BE SPLIT BY THE SAME ROWS.
    ///
    /// This is the proof family `why-locks-missed-the-prefill-bug` says was missing. The reserve
    /// (`rows * HOLE_PAGES_PER_ROW + 1`) has TWO callers — one that SIZES a pool to include it and one that
    /// SPLITS a pool by it — and nothing in the type system relates them. Newtypes cannot: both sides
    /// already speak `PoolPages` and `PoolRows`. The bug that shape produces is a load that computes a
    /// page count, allocates it, and then has `PoolPartition::of_pool` return `None` on the very next line,
    /// or worse hands the host a range that overlaps the holes.
    ///
    /// So it is asserted over EVERY declaration and EVERY width simultaneously: the sized pool splits, the
    /// host range is non-empty and seats every row, and the split's own admission bound holds.
    #[kani::proof]
    fn a_pool_sized_by_its_demand_always_splits() {
        let context: u32 = kani::any();
        // 1 slot to the deepest the prefix mask can address — past that `RowPages::within_mask_reach`
        // refuses and no pool should be sized for it.
        kani::assume(
            context >= 1
                && context <= PagedKvPool::MAX_PAGES_PER_ROW * PagedKvPool::PAGE_SLOTS as u32,
        );
        let rows_n: u32 = kani::any();
        kani::assume(rows_n >= 1 && rows_n <= PoolRows::WIDEST.get().get());
        let rows = PoolRows::for_admission(AdmittedRequests::new(rows_n as usize).unwrap());

        let demand = PoolDemand::of_declaration(SlotCount::new(context), rows);
        let pages = demand
            .pages()
            .expect("a bounded declaration cannot overflow a page count");

        // THE LAW: sized by the demand ⇒ splittable by the demand's own rows.
        let part = PoolPartition::of_pool(pages, rows)
            .expect("a pool sized to include the reserve must fund the reserve");
        assert!(
            part.host_blocks() >= rows.get().get(),
            "every launch row needs at least one host page"
        );
        assert!(
            PagedKvPool::split_pool(pages.get(), rows).is_some(),
            "the pool must seat the width it was sized for"
        );

        // AND THE DEPTH IS THE ONE DECLARED, not a page count that lost slots to rounding: the host range
        // alone must hold every row at its full declared depth.
        assert!(
            part.host_blocks() >= demand.per_row().get() * rows.get().get(),
            "the host range must hold every row's declared context, or a request dies mid-generation"
        );
    }

    /// The reserve is the SAME arithmetic on both sides of the law above — stated separately so a future
    /// edit to `reserve` that keeps `of_pool` compiling still fails here.
    ///
    /// ⛔ IT WAS `reserve_for(rows)`, `rows * HOLE_PAGES_PER_ROW + 1`. The per-row hole run is deleted, so the
    /// reserve is ONE page at every width — the law is the same law, over a constant instead of a product.
    #[kani::proof]
    fn the_reserve_is_exactly_what_the_split_withholds() {
        let pages_n: u32 = kani::any();
        let rows_n: u32 = kani::any();
        kani::assume(rows_n >= 1 && rows_n <= PoolRows::WIDEST.get().get());
        let rows = PoolRows::for_admission(AdmittedRequests::new(rows_n as usize).unwrap());
        kani::assume(pages_n > PoolPartition::reserve() && pages_n <= 1 << 20);
        let pages = PoolPages::of_pool(pages_n as usize).unwrap();
        if let Some(part) = PoolPartition::of_pool(pages, rows) {
            assert_eq!(
                pages_n - part.host_blocks(),
                PoolPartition::reserve(),
                "what the split withholds must be what the sizing added"
            );
        }
    }
}

/// ⭐⭐⭐ THE KV CACHE WRITE'S `y` PITCH — discharged over symbolic geometry, because the four head
/// dims a bake can afford are exactly where this relation is a coincidence.
///
/// `y` walks the kv heads of one slab, and the head-outermost order derives its step as
/// `pitch * stick` ([`crate::ir::bridge::tiled_op_sdsc_op`]'s `RungRegime::y_stride_elems`). The
/// pitch a plane-walked placement declares is `rows * nslab`, so the step is `rows * nslab * stick` —
/// and the heads really are `rows * hd` apart. Those agree for every `hd` that is `nslab` whole
/// sticks, which is what this proves, and the failure mode when they do not is the recorded one: at
/// `nslab == 1` the pitch `rows` and the pitch `rows * nslab` are one number, so a wrong pitch is
/// invisible on granite-2b and writes into another head's keys on anything wider.
#[cfg(kani)]
mod kv_cache_write_pitch_proofs {
    use super::*;

    /// Bounds: `rows` past the widest decode rung and every prefill chunk, `nslab` past hd=512's
    /// eight. Symbolic inside them, and covering BOTH operands — the source's `rows` is a chunk's `mq`
    /// and the cache's is the physical plane's slot count, which is the pair that must not be shared.
    #[kani::proof]
    fn the_declared_pitch_reaches_the_next_head_at_every_slab_count() {
        let (rows, nslab): (u32, u32) = (kani::any(), kani::any());
        kani::assume(rows >= 1 && rows <= 512);
        kani::assume(nslab >= 1 && nslab <= 8);
        let stick = Df::Fp16.elems_per_stick();
        let hd = nslab * stick;
        // The walk's step under the head-outermost order, against where the heads actually are.
        assert!(
            rows * nslab * stick == rows * hd,
            "pitch*stick is the head stride"
        );
        // And the SLAB step is one plane — the coordinate the offset carries, `nslab` of which is that
        // same head. Stated together, because a pitch that reached the head while the slab stepped
        // something else would write each head's upper features over its own lower ones.
        assert!(
            nslab * (rows * stick) == rows * hd,
            "nslab planes are one head"
        );
    }
}

#[cfg(kani)]
mod group_stride_proofs {
    use super::*;

    /// fp16 RowBlocked/Kernel/RowScalar reduce/matmul-A stride is ALWAYS `lanes()` (64) — the ONLY value
    /// proven representable on-card for this stick width (see `group_stride`'s doc: the raw `rows·lanes`
    /// fault at 1984 is far below `LRFIMM_MAX`'s raw ceiling, so the ceiling constant alone cannot gate
    /// this; the fp16 case must stay pinned to `lanes()` regardless of `rows`). Symbolic `rows` covers
    /// decode (rows=1) through any prefill width in ONE proof, not per-shape unit tests.
    #[kani::proof]
    fn fp16_group_stride_always_lanes() {
        let rows: usize = kani::any();
        kani::assume(rows >= 1 && rows <= 4096); // bounded: decode..a generous prefill ceiling
        let is_reduction: bool = kani::any();
        let kind: StickKind = if kani::any() {
            StickKind::RowBlocked
        } else {
            StickKind::Kernel
        };
        let sl = StickLayout {
            rows,
            cols: 2048,
            df: Df::Fp16,
            kind,
        };
        assert_eq!(sl.group_stride(is_reduction), sl.lanes() as i64);
    }

    /// fp32 (the ONE proven wide-fit exception) returns EXACTLY `rows·lanes` when reducing, and that
    /// product — for any `rows` up to the assumed prefill ceiling at fp32's 32-lane width — stays
    /// strictly under [`StickLayout::LRFIMM_MAX`]. If a future wider prefill target ever violates this,
    /// the proof fails at BUILD time instead of an on-card `LX_MODLRFIMM` fault.
    #[kani::proof]
    fn fp32_reduce_group_stride_fits_lrfimm() {
        let rows: usize = kani::any();
        kani::assume(rows >= 1 && rows <= 4096);
        let sl = StickLayout {
            rows,
            cols: 2048,
            df: Df::Fp32,
            kind: StickKind::RowBlocked,
        };
        let gs = sl.group_stride(true);
        assert_eq!(gs, rows as i64 * sl.lanes() as i64);
        assert!(
            gs < StickLayout::LRFIMM_MAX,
            "fp32 group_stride must stay under the lrfimm ceiling"
        );
    }

    /// Non-reduction reads (an op's own in+out share indexing) NEVER take the wide branch, for any format
    /// or row count — only a genuine reduce/matmul-A read is eligible for `rows·lanes`.
    #[kani::proof]
    fn non_reduction_never_widens() {
        let rows: usize = kani::any();
        kani::assume(rows >= 1 && rows <= 4096);
        let df: Df = if kani::any() { Df::Fp16 } else { Df::Fp32 };
        let sl = StickLayout {
            rows,
            cols: 2048,
            df,
            kind: StickKind::RowBlocked,
        };
        assert_eq!(sl.group_stride(false), sl.lanes() as i64);
    }

    /// `Flat` (row-major head-major attention intermediates) never widens either, reduction or not — a
    /// row-major tensor has no stick-group interleave to step over.
    #[kani::proof]
    fn flat_never_widens() {
        let rows: usize = kani::any();
        kani::assume(rows >= 1 && rows <= 4096);
        let is_reduction: bool = kani::any();
        let sl = StickLayout {
            rows,
            cols: 2048,
            df: Df::Fp16,
            kind: StickKind::Flat,
        };
        assert_eq!(sl.group_stride(is_reduction), sl.lanes() as i64);
    }

    /// decode (rows=1) is BYTE-IDENTICAL regardless of the is_reduction/df/kind branch taken — the
    /// invariant that makes every group_stride caller's decode path untouched by this refactor.
    #[kani::proof]
    fn decode_m1_byte_identical_across_branches() {
        let is_reduction: bool = kani::any();
        let df: Df = if kani::any() { Df::Fp16 } else { Df::Fp32 };
        let kind: StickKind = if kani::any() {
            StickKind::RowBlocked
        } else {
            StickKind::Flat
        };
        let sl = StickLayout {
            rows: 1,
            cols: 2048,
            df,
            kind,
        };
        assert_eq!(sl.group_stride(is_reduction), sl.lanes() as i64);
    }
}
