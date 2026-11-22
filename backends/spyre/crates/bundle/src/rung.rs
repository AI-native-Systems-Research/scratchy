/// ⭐ THE POOL'S STICK AS A CONST GENERIC ARGUMENT — so `PrefixMaskShape<POOL_STICK, COLS>` says WHICH 64 it is.
///
/// ⛔ EVERY INSTANTIATION SPELLED IT `<64, …>`. The type had the const generic right — `STICK` is a parameter
/// precisely so a block whose column count is not a whole number of sticks is unrepresentable — and then every
/// call site filled it with a literal, in a position where `mq_pad` (ALSO 64 today) or the fp8 stick (128, and
/// equal to `head_dim` at the model we run) would type-check identically. A const generic only separates
/// quantities if its ARGUMENTS are named.
use std::num::NonZeroU32;

pub const POOL_STICK: u32 = FP16_LANES;

/// fp16 elements per stick: the generation's stick bytes over fp16's two.
pub const FP16_LANES: u32 =
    (<ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::FLIT_BYTES / 2) as u32;

/// ⭐ AN ACTIVATION'S ROW AS THE DEVICE HOLDS IT — a `[rows, cols]` activation is row-major, each
/// row whole sticks of its format: `cols` rounded up to the stick's `lanes`. The placement reserves
/// rows of this pitch, the kernels' views address them, and a host read of one steps by it.
pub const fn row_pitch(cols: u64, lanes: u64) -> u64 {
    cols.next_multiple_of(lanes)
}

/// ⭐⭐⭐⭐⭐ A HEAD'S WIDTH, STATED IN THE POOL'S OWN STICKS — never a bare `head_dim` number.
///
/// THE LAW: a head on this device is a WHOLE NUMBER of fp16 sticks — `head_dim = sticks · 64`,
/// for any `sticks ≥ 1` (64, 128, 256, 512, …). The whole domain, stated once, in the only
/// constructor ([`HeadWidth::of_whole_sticks`]); no arm per width, no upper bound, nothing shaped
/// by which model happens to run.
///
/// ⛔ WHAT IT REPLACES: the raw `hd: u32` parameter, at every constructor and emitter site that
/// then re-derived "is a head one stick here?" (`hd == 64`), "how many slabs?" (`hd / 64`), or
/// nothing at all — the last being the hazard this file's newtypes exist for, because at
/// head_dim 128 the width EQUALS the fp8 stick width (see [`POOL_STICK`]'s table) and a missing
/// slab multiplier type-checks identically. With this type, `hd == 64` is [`Self::sticks`] `== 1`
/// (a proven-emit predicate, askable by name), `hd / 64` IS [`Self::sticks`] (never re-derived),
/// and a width that is not whole sticks is UNREPRESENTABLE rather than checked at the reader.
///
/// The model-facts `HeadDim` (scratchy-ir) stays positive-only — the stick law is a DEVICE lane
/// fact, and this is the one door where it is minted.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct HeadWidth(NonZeroU32);

impl HeadWidth {
    /// ⭐ THE LAW, WRITTEN ONCE: a head width is admitted iff it is a whole number of the pool's
    /// fp16 sticks — a head never straddles a stick boundary. Every door reads this fn and
    /// none restates the check.
    pub const fn try_whole_sticks(hd: u32) -> Option<Self> {
        match (hd.is_multiple_of(POOL_STICK), NonZeroU32::new(hd)) {
            (true, Some(nz)) => Some(Self(nz)),
            _ => None,
        }
    }

    /// The const door over [`Self::try_whole_sticks`]: the refusal is a literal panic (a const
    /// fn cannot format), so a const context — the bake, a `static` tape — gets it as a BUILD
    /// ERROR, which is the only time this fact exists to be checked.
    pub const fn of_whole_sticks(hd: u32) -> Self {
        match Self::try_whole_sticks(hd) {
            Some(w) => w,
            None => panic!(
                "HeadWidth: a head is a whole number of fp16 sticks — a width that straddles a \
                 stick boundary has no emit form"
            ),
        }
    }

    /// The bake-time door over [`Self::try_whole_sticks`]: for the value the MODEL'S OWN BOUNDS
    /// state (a `u64`), so the head_dim reaches this door as its own typed fact and the one
    /// conversion — bounds to width — happens here, with a panic that NAMES the model and the
    /// width. This is the one door the `#[forward]` bake reads; a model whose head_dim is not
    /// whole sticks is a BUILD ERROR, not a runtime refusal.
    pub fn of_whole_sticks_at_bake(hd: u64, model: &str) -> Self {
        let hd = u32::try_from(hd).unwrap_or_else(|_| {
            panic!("{model}: head_dim {hd} does not fit a u32, let alone a head width")
        });
        match Self::try_whole_sticks(hd) {
            Some(w) => w,
            None => panic!(
                "{model}: head_dim {hd} is not a whole number of fp16 sticks ({POOL_STICK} lanes) \
                 — a head that straddles a stick boundary has no emit form"
            ),
        }
    }

    /// The width, in elements.
    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// The width as a non-zero — the storage already is one, so no caller re-proves it.
    pub const fn nonzero(self) -> NonZeroU32 {
        self.0
    }

    /// The width as a `usize` — the accessor for the host-side element arithmetic that counts
    /// in `usize` (lossless: a head width fits a `u32`). The ONE widening past the type, as the
    /// page types' own `as_usize`; no use site spells `as usize` on the raw value.
    pub const fn as_usize(self) -> usize {
        self.0.get() as usize
    }

    /// How many fp16 sticks one head spans — `hd / 64`, proven whole by construction. This is
    /// the slab count every feature-axis stride multiplies by, and the `== 1` case is the
    /// one-stick emit form's own predicate — askable by name instead of `hd == 64`.
    pub const fn sticks(self) -> u32 {
        self.0.get() / POOL_STICK
    }
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

    /// ONE row — a decode batch's per-request width: every row of a batch's graph is a
    /// request's one token ([`QueryExtent`]'s decode law), the width a body's pair takes
    /// wherever the rows are a batch's. The `S = 1` counterpart of [`RungCap::ONE`].
    pub const ONE: RungWidth = RungWidth(NonZeroU32::MIN);
}

/// SEQUENCES A PREFILL RUNG HOLDS — the S axis of the prefill ladder.
///
/// A prefill rung is keyed by TWO quantities that both fit a bare `u32`: its chunk WIDTH
/// ([`RungWidth`] — rows of one sequence's chunk) and how many SEQUENCES it packs. The packed
/// rung's non-attention tape runs at `width · cap` rows (the decode rungs' own device — the
/// same tape at a wider row count), so a width handed where a capacity is read bakes a rung
/// at 2·w rows for one sequence and silently doubles every chunk. Typed apart, as the decode
/// ladder's own keyed-by-type lesson demands (its `RungSeqs`): three lists once shared a bare
/// `u32` and two quantities were mixed type-safely by mistake.
///
/// ONE is today's every rung — one sequence per chunk, the only shape the bake emits until
/// the packed prefill rung (M2) lands. Selection reads (per-request width, live sequence
/// count) against the (width, cap) pairs: the smallest rung holding both.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct RungCap(NonZeroU32);

impl RungCap {
    /// A BAKED sequence count, read from the bake's own artifacts — a ladder entry. The
    /// worker-side source. The bake's codegen is the one caller: a ladder literal states its
    /// rung's capacity as this type (today always `RungCap::ONE`), and nothing on the host
    /// mints a capacity a bake never stated.
    pub const fn of_baked_seqs(seqs: u32) -> Option<RungCap> {
        match NonZeroU32::new(seqs) {
            Some(n) => Some(RungCap(n)),
            None => None,
        }
    }

    /// ONE sequence — today's every rung, and the value the S = 1 axis bakes at.
    pub const ONE: RungCap = RungCap(NonZeroU32::MIN);

    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// The sequence count as a host-side extent — the arithmetic exit.
    pub const fn count(self) -> usize {
        self.0.get() as usize
    }
}

/// THE ATTENTION QUERY'S EXTENT PAIR — the rung's own `(width, capacity)`, never a division of the
/// graph's rows.
///
/// A packed prefill rung's graph runs `width · capacity` rows, but its attention kernel serves ONE
/// request's `width` rows per request (the reference's own law: the varlen batch is one query
/// buffer, attention is called once per sequence with `PADDED_QUERY_LEN = aligned_s` —
/// `spyre_attn.py`'s `_run_page_attn`/`_build_query_row_tables`). The two counts were one `mq`
/// until S > 1, and the splice spent it as BOTH extents; this type is the pair that separates them.
///
/// ⛔ MINTED ONCE, FROM THE RUNG — [`Self::of_baked_rung`] is the only mint, and the graph's row
/// count is the product HELD by the type, not a division any reader performs: nothing may recover
/// the per-request width by dividing graph rows back (a width a later reader re-derived could
/// disagree with the one the bake minted). The splice CHECKS the held product against the graph it
/// lowers and refuses by name on disagreement, so a mint that does not describe the graph is a
/// bake refusal, never a silently-wrong extent.
///
/// The decode mints state their own true pair — every row a request's one token: per-request one
/// row, requests the baked row count — which the decode and batch kernel arms read exactly as they
/// read the graph's rows today (one row per request means the two agree by construction).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QueryExtent {
    /// One request's query rows — the prefill chunk width the rung bakes.
    per_request: RungWidth,
    /// How many requests the rung packs — the S axis.
    requests: RungCap,
    /// `per_request · requests`, minted once and held: the graph's own row count. The mint checks
    /// the product, so this is a read, never a recompute.
    rows: RungWidth,
}

impl QueryExtent {
    /// THE MINT — the rung's own pair. `None` only when the product does not fit a `u32` (a rung
    /// wider than four billion rows is a bake bug, named by the caller that holds the rung).
    pub const fn of_baked_rung(per_request: RungWidth, requests: RungCap) -> Option<Self> {
        match per_request.get().checked_mul(requests.get()) {
            Some(rows) => match RungWidth::of_baked_rows(rows) {
                Some(w) => Some(Self {
                    per_request,
                    requests,
                    rows: w,
                }),
                // `rows` is nonzero whenever `checked_mul` returned `Some` over nonzero factors,
                // so this arm is unreachable; it exists because `of_baked_rows` states the
                // nonzero law and the const context cannot `debug_assert` it away.
                None => None,
            },
            None => None,
        }
    }

    /// One request's query rows — the prefill kernel's `PADDED_QUERY_LEN` and its walk's extent.
    pub const fn per_request(self) -> RungWidth {
        self.per_request
    }

    /// How many requests the rung packs.
    pub const fn requests(self) -> RungCap {
        self.requests
    }

    /// The graph's row count — `per_request · requests`, held since the mint. The whole-batch
    /// query buffer's `NUM_TOKENS`.
    pub const fn graph_rows(self) -> RungWidth {
        self.rows
    }
}

/// ⭐ THE PREFILL TAIL'S ANSWER ROWS — one per request window of a packed prefill rung: request
/// `r`'s answer row is `r·w + selector_lastrow_col(w)`, its window's last BAKED row — the row
/// whose hidden the lm-head fold extracts, [`selector_lastrow_col`]'s own law stated per
/// request. Minted ONCE, from the graph's [`QueryExtent`] at the bake; every tail site reads
/// it, none re-derives a row. The host's window law ([`crate::paged_step`]'s `RowSpan::last`)
/// states the SAME law at the slot — the two are one statement in two spellings, pinned equal
/// by fixture, never two laws that agree by accident.
///
/// At one request (`S = 1`, every rung today) the first of `rows()` is `selector_lastrow_col(w)`
/// with `w = mq` — exactly today's single-row law, so every tail site is byte-identical there.
///
/// ⛔ THE COUNT IS THE RUNG'S CAPACITY, NOT THE LAUNCH'S LIVE REQUESTS — this is the bake's
/// own statement of the rung's windows. A launch that seats fewer requests than the rung
/// packs leaves EMPTY WINDOWS (no request's), which produce NO answer rows: the readback reads
/// the LIVE request count off the launch's own slot list and maps pick `r` to slot `r` by
/// order. The bake names the rows; the launch names how many are read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AnswerRows {
    extent: QueryExtent,
}

impl AnswerRows {
    /// ⭐ THE ONE MINT — the graph's QueryExtent, the same pair the attention's extents read:
    /// one `(w, S)` statement per rung, three readers, one bake fact.
    pub const fn of_extent(extent: QueryExtent) -> Self {
        Self { extent }
    }

    /// The rung's answer rows, one per packed window, in request order — the ONLY mint of a
    /// [`RequestIndex`], so a request past the capacity is UNSPELLABLE: no reader passes a bare
    /// request index, and the readback takes `rows().take(live)` with the live count off the
    /// launch's own slot list (an empty window's row is never read).
    pub fn rows(self) -> impl ExactSizeIterator<Item = (RequestIndex, usize)> {
        (0..self.count()).map(move |r| {
            (
                RequestIndex(r),
                r * self.extent.per_request().get() as usize
                    + selector_lastrow_col(self.extent.per_request().get() as usize),
            )
        })
    }

    /// How many answer rows the rung bakes — the request count `S` (the capacity, not the
    /// launch's live requests; see the type's doc).
    pub const fn count(self) -> usize {
        self.extent.requests().count()
    }

    /// The extent the rows were minted from — the tail's agreement check against the graph it
    /// lowers, the same check the attention's extents take.
    pub const fn extent(self) -> QueryExtent {
        self.extent
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

/// THE PREFILL CAUSAL-MASK EXTENT (mq>1 new-token self-attention block): in a batched prefill of `mq`
/// new tokens, query row `row` (0-based within the chunk) attends new-token column `col` iff `col <= row`
/// — the token attends itself + all EARLIER new tokens, never a LATER one. Additive mask: 0 when valid,
/// −∞ when `col > row`. This is the `[mq,mq]` causal block that folds (jointly with the prefix score
/// [mq,cap]) into ONE softmax per query row. SINGLE SOURCE OF TRUTH for the host's mask fill
/// ([`CausalKey`]) and the Kani proof (`prefill_causal_mask_partition_ok`). An off-by-one here (`col < row`, dropping
/// the diagonal self-term) or wrong direction (`col >= row`) scrambles causal attention — a
/// coherence bug with a preserved-ish norm, so it MUST be pinned, not eyeballed.
pub fn prefill_causal_col_valid(col: usize, row: usize) -> bool {
    col <= row
}

/// The fp16-safe additive "-inf" every staged attention mask carries: half the largest finite fp16
/// magnitude, negated, so `exp(score + mask)` underflows to 0 and the sum stays finite. A named
/// constant because [`CausalKey::mask`] has no other input: the value the mask carries can no more
/// come from a caller than its rows can.
pub const ATTN_MASK_NEG: f32 = -(half::f16::MAX.to_f32_const() / 2.0);

/// ⭐⭐⭐ EVERY FACT A LAUNCH'S NEW-BLOCK CAUSAL MASK IS A FUNCTION OF, AS ONE VALUE — and
/// [`Self::mask`] takes nothing else, so the mask cannot depend on a fact this key leaves out.
///
/// The mask is `[rows, rows_pad]` additive (0 where row `r` may attend new-block column `c`), and it
/// is a function of TWO facts on each path, never of the tokens, the positions or the prefix:
///
/// * a prompt chunk — the rung's rows and how many of them are `real` (pad rows clamp to the last real
///   row, [`ChunkRows::row_logical_pos`]: the lm_head reads the LAST baked row, which must attend
///   exactly `[0, real)`). A FULL chunk (`real == rows`) is the plain triangle, so every full chunk of
///   a rung binds the same mask;
/// * a decode batch — the rung's rows and how many are `live` (live slot `s` attends its own column;
///   a padding slot is live 0 complete, [`PadRowReplica`], so it attends column 0).
///
/// ⭐ SO IT IS A STAGED CONSTANT, KEYED. A session holds the mask of ONE key at a time. A launch whose
/// key matches what the session holds re-binds nothing: the host never wrote anything else to that
/// placement, and the device never writes segment 3 (measured: no compute op of the 252 granite-2b
/// bundles names a seg3 dataspace among its outputs). One equality on two small integers decides it:
/// no content compare and no per-step rebuild of a 16 MB tile. That rebuild was 26–28 ms of host work
/// per 512-row chunk on granite-3.1-2b, measured.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CausalKey {
    /// A prompt chunk: `real` genuine rows of a `rows`-row rung.
    Chunk { rows: RungWidth, real: NonZeroU32 },
    /// A decode batch: `live` requests on a `rows`-row rung.
    Batch { rows: RungWidth, live: NonZeroU32 },
}

impl CausalKey {
    /// A prompt chunk's key. `None` unless `1 <= real <= rows`: a chunk with no real row, or more
    /// real rows than its rung, is not a launch.
    pub fn chunk(rows: RungWidth, real: usize) -> Option<CausalKey> {
        let real = NonZeroU32::new(u32::try_from(real).ok()?)?;
        (real.get() <= rows.get()).then_some(CausalKey::Chunk { rows, real })
    }

    /// The rows the mask has, and the stick-padded column count the emitter reads it at
    /// (`rows.div_ceil(64) * 64`), so the buffer cannot be built at one width and read at another.
    pub fn rows(self) -> usize {
        match self {
            CausalKey::Chunk { rows, .. } | CausalKey::Batch { rows, .. } => rows.count(),
        }
    }

    pub fn rows_pad(self) -> usize {
        self.rows().div_ceil(64) * 64
    }

    /// The new-block column row `row` attends up to (a chunk's clamped position) or exactly (a
    /// batch's column).
    fn row_col(self, row: usize) -> usize {
        match self {
            CausalKey::Chunk { rows, .. } => {
                ChunkRows::chunk(rows.rows()).row_logical_pos(row, self.real_or_live())
            }
            CausalKey::Batch { .. } if row < self.real_or_live() => row,
            CausalKey::Batch { .. } => LaunchSlot::solo().index(),
        }
    }

    fn real_or_live(self) -> usize {
        match self {
            CausalKey::Chunk { real, .. } => real.get() as usize,
            CausalKey::Batch { live, .. } => live.get() as usize,
        }
    }

    /// THE `[rows, rows_pad]` NEW-BLOCK MASK, additive and row-major, from this key alone: the
    /// [`prefill_causal_col_valid`] triangle for a chunk, the [`decode_batch_causal_col_valid`]
    /// diagonal for a batch, `0` where valid and [`ATTN_MASK_NEG`] elsewhere.
    pub fn mask(self) -> Vec<f32> {
        let (mq, mq_pad) = (self.rows(), self.rows_pad());
        let mut cmask = vec![ATTN_MASK_NEG; mq * mq_pad];
        for row in 0..mq {
            let at = self.row_col(row);
            for col in 0..mq_pad {
                let valid = match self {
                    CausalKey::Chunk { .. } => prefill_causal_col_valid(col, at),
                    CausalKey::Batch { .. } => decode_batch_causal_col_valid(col, at),
                };
                if valid {
                    cmask[row * mq_pad + col] = 0.0;
                }
            }
        }
        cmask
    }
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
pub const fn selector_lastrow_col(m: usize) -> usize {
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
}

impl ChunkRows {
    /// A prefill chunk `n` consecutive positions wide.
    pub const fn chunk(n: NonZeroU32) -> ChunkRows {
        QueryRows::new(n)
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
    /// worker's rope-position staging AND its causal-mask fill ([`CausalKey::mask`] composes it
    /// with [`prefill_causal_col_valid`]), pinned by the Kani proof
    /// `prefill_pad_row_holds_last_real_token`.
    pub fn row_logical_pos(self, row: usize, real: usize) -> usize {
        row.min(real - 1)
    }
}

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
///            Kᵀ   : [kvh][hd][POOL_PAGE_SLOTS]         stick-major on slots — the score kernel
///            V    : [kvh][POOL_PAGE_SLOTS][hd]         sticked on hd       — the value kernel
///            Knat : [kvh][POOL_PAGE_SLOTS][hd]         natural K           — the Kᵀ re-transpose source
/// ```
///
/// ⛔ **THE REQUEST IS NOT A DIMENSION HERE AT ALL — IT IS A CHOICE OF PAGES.**
///
/// This layout used to put `ROWS = 32` requests side by side inside every page, immediately inside `kvh`,
/// so that "the distance between two requests" was bakeable geometry (`hd * POOL_PAGE_SLOTS`) and one additive
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
/// * **The page is [`POOL_PAGE_SLOTS`](Self::POOL_PAGE_SLOTS) = the pre-paged `cap`.** A fold covers one page
///   per launch, so the page IS the swept extent: making it wider than the old cache would sweep
///   more KV for the same context — a straight performance regression, which is exactly what a
///   512-slot page cost. Equal to the old cap, a single page's fold is bit-for-bit the work the
///   baseline did, and longer contexts cost additional launches rather than a wider sweep.
/// * **GQA-DEDUPED.** The score and value kernels already read only the group representative, so the
///   `nqh`-sized reservation was `gqa`x dead space.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PagedKvPool {
    /// DISTINCT kv heads — each group of query heads reads one. Nonzero by construction: a pool
    /// with no heads is unrepresentable.
    nkvh: NonZeroU32,
    /// ⭐ THE HEAD WIDTH, AS A TYPE — minted once, in this pool's own constructor: whole sticks
    /// by construction, so no reader re-mints (or re-checks) it, and no `as` crosses it.
    hd: HeadWidth,
}

/// ⛔ THE LARGEST SINGLE dxp GROUP A BUNDLE MAY STAGE (descriptors) — the ONE ceiling the emitter's
/// gathered-fold chunk split and the bake-side oversized-group guard BOTH read.
///
/// Lives HERE — in the leaf crate that owns the emission it bounds — so the split decision and the
/// guard cannot each grow their own constant; the spyre target crate re-exports it under the same
/// name its own paths already used, which is the whole enforcement of "one const": a second
/// divergent value would have to be a NEW name in a file that already resolves this one.
///
/// Distinct from the spyre crate's `GroupSize::CEILING`: that is the largest value the per-kind cap
/// may be SET to; this is the largest group the whole partition may produce once the uncapped kinds
/// (gathered PageFold, `run_may_be_chunked == false`) and time-trip fan-out have had their say. A
/// group past this ceiling is a build error (the oversized-group guard in
/// `lower_subtile_tape_to_superdsc::ktir_groups_via_superdsc`), not a multi-hour dxp that looks
/// like a hang — and the emitter pre-empts it by splitting an over-ceiling gathered fold into
/// self-contained window chunks (see `attn.rs`'s chunk sweep).
pub struct DxGroupCeiling;

impl DxGroupCeiling {
    /// 4204: the granite-8b fp8 baseline's LARGEST MEASURED GROUP — the accepted 30-minute bake's
    /// own maximum (bundle `4f460f8b68a6c85a` group_1, an mq=32 gathered fold: 4 windows ×
    /// (2·32·8·2 legs + 11 ladder) + 64 gather copies = 4204, verified against the plan-only dump
    /// three ways: file count, `bundle_id.json` (mq=32, attn=4722), and the fan-out formula).
    /// ⛔ NOT 512/`GroupSize::CEILING`: the ORIGINAL guard doc claimed the baseline "never exceeds
    /// ~512" and 512 would refuse granite-8b's own accepted bake — a doc comment is not evidence,
    /// and the dump is. Gemma-4-12b fp8's ladder is lawful under 4204 except ONE group: the
    /// 8,300-descriptor mq=32 class-0 fold, which the window-chunk split cuts into k self-contained
    /// chunks each at or under this value; the next-largest gemma group is 4,183 and stays unsplit.
    /// Moving this constant is a MEASURED decision that names the bake, never a silent one.
    pub const MAX_DESCRIPTORS: usize = 4204;
}

/// ⭐ THE SCHEDULER'S KV BLOCK — the number of token slots in one block the serving step's
/// scheduler allocates: the quantity the splice's page walk tiles (`BLOCK_SIZE`) and the KV
/// write's rows step by. A TYPED REFERENCE INPUT, NOT A DERIVATION: the value is the block
/// size of the serving step we port, minted ONCE at [`SchedulerBlockTokens::REFERENCE`]. The
/// reference's own law for the value — a multiple of 64 (`_BLOCK_SIZE_MULTIPLE`, the
/// 128-byte fp16 stick), measured-throughput argmax — is cited, not re-derived: we port the
/// step, we do not redesign its block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedulerBlockTokens(std::num::NonZeroU32);

impl SchedulerBlockTokens {
    /// spyre-inference's own block size, 128: `_DEFAULT_BLOCK_SIZE`
    /// (`spyre_inference/platform.py:146`), which `_align_block_size` only ever aligns a
    /// user's size up to, and which every torch-spyre serve runs (its compiled decode walk
    /// steps the page table and mask by 128 a trip). Half the trips of a 64-slot block for
    /// the same context. The ONE statement of the value.
    pub const REFERENCE: Self = Self(match std::num::NonZeroU32::new(128) {
        Some(v) => v,
        None => panic!("128 is nonzero"),
    });

    /// The block's token count — the accessor every integer boundary calls.
    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// The block's token count as its `NonZeroU32` — for the checked nonzero
    /// arithmetic ([`PoolPageSlots::of_scheduler_blocks`]) that keeps the law in std's
    /// own types.
    pub const fn nz(self) -> std::num::NonZeroU32 {
        self.0
    }

    /// The block's token count as a `usize` — the accessor for the host-side
    /// page/stride arithmetic that counts in `usize`.
    pub const fn as_usize(self) -> usize {
        self.0.get() as usize
    }
}

/// ⭐ THE POOL'S ADDRESSABLE PAGE — how many slots one page of the KV pool holds: masks are
/// blocked by it, the fold sweeps it, a `KvSlot` never exceeds it. A MEASURED POLICY, NOT A
/// DERIVATION (see [`PagedKvPool::POOL_PAGE_SLOTS`]'s own rationale for the k it picks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolPageSlots(std::num::NonZeroU32);

impl PoolPageSlots {
    /// ⭐ THE ONLY CONSTRUCTOR: `k` whole scheduler blocks. This function IS the law "a pool
    /// page is a whole number of scheduler blocks" — a page built any other way is
    /// unrepresentable, so the block→page mapping the write and the walk rely on stays whole
    /// by construction and no separate assert exists. `k` is a MEASURED POLICY choice, and
    /// NONZERO BY TYPE: "a pool page spans at least one scheduler block" is `k`'s type, not
    /// a check.
    pub const fn of_scheduler_blocks(
        blocks: SchedulerBlockTokens,
        k: std::num::NonZeroU32,
    ) -> Self {
        match blocks.nz().checked_mul(k) {
            Some(slots) => Self(slots),
            None => panic!(
                "a pool page of k whole scheduler blocks overflows u32: \
                 the blocks-per-page policy must fit a u32 slot count"
            ),
        }
    }

    /// The page's slot count — the accessor every integer boundary calls.
    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// The page's slot count as a `usize` — the accessor for the host-side
    /// page/stride arithmetic that counts in `usize`.
    pub const fn as_usize(self) -> usize {
        self.0.get() as usize
    }
}

impl PagedKvPool {
    ///
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
    /// the positions for the same bytes — TurboQuant is already in-tree for the GPU target). Then this goes back
    /// up and the win comes with it.
    ///
    /// ⭐ A MEASURED POLICY, NOT A DERIVATION: 256 is chosen on the measurements above
    /// (64/128/256 slots cost 39.5/39.0/42.0 ms at eight requests; the 1024-slot page
    /// measured a genuine 3.25× on device compute, but its DEFAULT pool became 3.84 GB —
    /// larger than the weights — and was reverted). A DISTINCT QUANTITY from the
    /// scheduler's block ([`SchedulerBlockTokens::REFERENCE`], the reference's own 128):
    /// a pool page is a WHOLE NUMBER of scheduler blocks, and k=2 is the policy this
    /// const states through [`PoolPageSlots::of_scheduler_blocks`] — the type's ONLY
    /// constructor, so the relation is law, not assert.
    pub const POOL_PAGE_SLOTS: PoolPageSlots = PoolPageSlots::of_scheduler_blocks(
        SchedulerBlockTokens::REFERENCE,
        match std::num::NonZeroU32::new(2) {
            Some(v) => v,
            None => panic!("2 is nonzero"),
        },
    );

    /// ⭐ THE WIDEST PREFILL CHUNK — the baked row count of the prefix-capable prefill bundle, and
    /// therefore the number of slots ONE continuation chunk's cache write actually touches.
    ///
    /// ⛔ IT IS NOT THE CHUNK'S REAL LENGTH. Every continuation chunk runs on that ONE bundle
    /// (`use_prefix = chunk_start > 0`), and its cache write is a single `[mq, hd]` copy per kv head —
    /// so a chunk with 12 real rows still writes 128 CONSECUTIVE SLOTS, the last 116 of them padding.
    /// That is why this number, and not the chunk length, is what the page has to accommodate.
    ///
    /// Kept here rather than beside the prefill rung ladder because the POOL is what must make room for
    /// it; `codegen`'s `PREFILL_RUNGS` ceiling is asserted equal to it, so the two cannot drift.
    ///
    /// ⭐ 512 — TWO WHOLE PAGES ([`PREFILL_WINDOW_PAGES`](Self::PREFILL_WINDOW_PAGES)). A window wider
    /// than a page is cut, at emission, into one cache-write part per page, each resolving its own page
    /// through the request's block table — so the window may span whole pages, never part of one. A
    /// prompt chunked from a page boundary starts every chunk at offset 0, so no padded write needs
    /// [`chunk_write_start`](Self::chunk_write_start) to step back. It is the ceiling of the ladder, not
    /// every model's: a model whose causal block stays sub-blocked bakes to 128 (see `codegen`'s
    /// `PREFILL_RUNGS`), and the runtime writes at the bundle's own baked width either way. (At 96 the
    /// third chunk of every page stepped back and re-processed the overlap; at 256, one page, a 10k
    /// prompt ran 40 chunks, each re-reading every weight once.)
    pub const PREFILL_CHUNK_SLOTS: usize = 512;

    /// ⭐ PAGES ONE PROMPT WINDOW MAY SPAN — the widest prefill chunk in whole pages (1 when it fits in
    /// one). A window starting at a page boundary may fill all of them; one starting mid-page stays
    /// inside that page (see [`chunk_room`](Self::chunk_room)).
    pub const PREFILL_WINDOW_PAGES: usize =
        Self::PREFILL_CHUNK_SLOTS.div_ceil(Self::POOL_PAGE_SLOTS.as_usize());

    /// ⛔⛔⛔ A CHUNK-FORM MODEL'S **BAKED** PREFILL LADDER MUST REACH THE POOL'S WINDOW — the bake's proof,
    /// panicked on by the macro.
    ///
    /// `codegen`'s const assert ties [`PREFILL_CHUNK_SLOTS`](Self::PREFILL_CHUNK_SLOTS) to the ceiling of
    /// the CONST ladder, and the bake then lowers what it can of it: a rung that does not lower is
    /// skipped, and the prefix-capable bundle is baked at the widest one that did. The host follows
    /// whatever was baked, so a short ladder is never wrong — only silently slower. MEASURED: the
    /// 512-row ladder baked to 256 behind a leftover resident-cache cap, loaded as `PREFILL ladder =
    /// [..., 192, 256]`, and every TTFT cell matched the 256-row build to the millisecond.
    ///
    /// So the check is on what LOWERED. `chunk_form` says the model's causal block is chunk-formed at
    /// the window's width (`NewBlockForm::of(PREFILL_CHUNK_SLOTS, hd, false) == Chunk`) — a model whose
    /// block stays sub-blocked keeps its own shorter ladder by design. `baked` is every rung that
    /// lowered, `refused` every rung that did not and why, and `attempted` the ceiling the ladder was
    /// asked to reach. `Err` names the shortfall and each refusing rung's reason.
    pub fn prefill_ladder_reaches_the_window(
        chunk_form: bool,
        attempted: u32,
        baked: &[u32],
        refused: &[(u32, String)],
    ) -> Result<(), String> {
        let window = Self::PREFILL_CHUNK_SLOTS as u32;
        let top = baked.iter().copied().max().unwrap_or(0);
        if !chunk_form || top >= window {
            return Ok(());
        }
        let why = if refused.is_empty() {
            format!(
                "no rung above {top} was attempted: the ladder's ceiling was {attempted}, below the window"
            )
        } else {
            refused
                .iter()
                .map(|(m, e)| format!("rung m={m} did not lower: {e}"))
                .collect::<Vec<_>>()
                .join("; ")
        };
        Err(format!(
            "the baked prefill ladder tops out at {top} rows, below the pool's {window}-row prompt window \
             (PagedKvPool::PREFILL_CHUNK_SLOTS) for a model whose causal block is chunk-formed at that \
             width — {why}. The host would run {top}-row chunks with no error and none of the window's \
             speed; lower the missing rungs, or narrow the window."
        ))
    }

    /// ⭐ THE WIDTHS A PREFIX-CAPABLE PREFILL BUNDLE IS BAKED AT, out of the prefix-free widths that baked
    /// (`baked`): the top, and one page when the top is wider than a page. Ascending.
    ///
    /// A continuation chunk is capped at the top width, and [`chunk_write_start`](Self::chunk_write_start)
    /// leaves every mid-prompt chunk room for a whole window, so only a prompt's LAST chunk is ever
    /// narrower — and a page is the narrowest a padded write can take without stepping back into slots an
    /// earlier chunk owns. So a tail of at most one page runs one page wide, and anything longer runs a
    /// full window. Nothing in between: a width that is neither would step its write back across the page
    /// it starts in.
    ///
    /// ⛔ ONE WIDTH WAS THE MEASURED COST. Only the top was baked, so a 52-row tail ran a full 512-row
    /// chunk: granite-3.1-2b at ~2k tokens took 460 ms against 384 ms on the 256-row build.
    ///
    /// Empty when nothing baked; just the top when the top fits one page (every model whose causal block
    /// stays sub-blocked, so its bake is byte-for-byte unchanged).
    pub fn prefix_rungs(baked: &[u32]) -> Vec<u32> {
        let page = Self::POOL_PAGE_SLOTS.get();
        let Some(top) = baked.iter().copied().max() else {
            return Vec::new();
        };
        if top > page && baked.contains(&page) {
            vec![page, top]
        } else {
            vec![top]
        }
    }

    /// ⭐ THE FLOOR OF A PACKED WINDOW'S WIDTH — the narrowest per-request width a packed
    /// `(width, seqs)` prefix rung is baked at.
    ///
    /// A window narrower than this is a bundle whose descriptors dominate its bytes: the packed
    /// rung exists to batch k cache-hit tails into ONE chunk, and at a handful of rows per window
    /// the fixed per-op cost the packing shares is larger than the rows it carries — a single
    /// narrow tail runs cheaper on the SOLO rung of the same width. 8 rows is one eighth of a
    /// scheduler block, the point where a wave of 8 windows still sweeps a whole block of real
    /// tokens.
    pub const PACKED_WINDOW_MIN_WIDTH: u32 = 8;

    /// ⛔ THE CEILING OF A PACKED WINDOW'S WIDTH — MEASURED, NOT DERIVED. Widths past one
    /// scheduler block refused at the bake when the packed table was first cut (the nuked line's
    /// measurement, 2026-11-11: `(192, 2)` and `(256, 2)` refused to lower; every width ≤ 128
    /// baked at every count), so the table stops at the block width and never asks for a rung the
    /// bake would refuse — the rung set is exactly what the bake ships.
    pub const PACKED_WINDOW_MAX_WIDTH: u32 = SchedulerBlockTokens::REFERENCE.get();

    /// ⭐ THE PACKED PREFIX RUNGS `(width, seqs)` — at each width the prefix ladder's own points
    /// name, every request count the reserved window-tid space holds whose product lands on a
    /// point the ladder already bakes. Ascending by product.
    ///
    /// A wave of k cache-hit continuation tails (the concurrent-prefill bench's k ~59-row tails)
    /// rode the SOLO prefix ladder one serialized chunk at a time — k fixed costs, k weight
    /// streams. A packed rung runs them as ONE prefix-capable chunk: the attention's one compiled
    /// module spliced once per window, every window binding its own three step tables at its own
    /// reserved tids (`reserved_tids::paged_window_tid`), so the batch is the SPLICE's, not the
    /// scheduler's and not a second lowering.
    ///
    /// The WIDTHS are the prefix points' own — each `m` a point names at a count `s` the tid
    /// space holds (`m/s` in the band) — so the prefix ladder's geometry picks the window widths.
    /// The COUNTS are graded, every value `2..=PAGED_MAX_WINDOWS` and not only powers of two: a
    /// wave's live count is whatever the scheduler hands the step, and a rung exists only at the
    /// powers of two when a 3-tail leftover staged the `(128, 4)` rung's whole 512 baked rows for
    /// 384 real (MEASURED, the warm conc-8 cachetest cell: the tails=3 launch's device exec was
    /// byte-for-byte the tails=4 launch's). A count whose product is no point the ladder bakes
    /// contributes nothing — the product constraint keeps a packed rung's graph rows, its KV sweep
    /// and its pool a point's own, and its body dedups with the solo rung at the same rows (the
    /// seam-fp law: equal row products share seam graphs), so grading adds pairs, never geometry.
    ///
    /// ⛔ THE COUNTS ARE CAPPED BY THE RESERVED TID SPACE — never past
    /// [`PAGED_MAX_WINDOWS`](crate::reserved_tids::PAGED_MAX_WINDOWS): a pair asking for more
    /// windows than the region holds is a bake bug the tid mint's own assert names. The widths
    /// are held to
    /// [`PACKED_WINDOW_MIN_WIDTH`](Self::PACKED_WINDOW_MIN_WIDTH)..=[`PACKED_WINDOW_MAX_WIDTH`](Self::PACKED_WINDOW_MAX_WIDTH)
    /// — the floor because a narrower window is descriptors-not-bytes (the solo rung serves one
    /// narrow tail better), the ceiling because wider refused at the bake (measured, see the
    /// const). A point that names no width in that band for a count contributes nothing: the
    /// table asks only for rungs that bake.
    ///
    /// EMPTY exactly when `baked` is (nothing prefix-capable bakes at all), and otherwise never
    /// empty — every point ≥ one page names at least `(page/2, 2)`, whose product is the page
    /// itself, a point `baked` holds by [`prefix_rungs`](Self::prefix_rungs)'s own admission.
    pub fn packed_prefix_rungs(baked: &[u32]) -> Vec<(u32, u32)> {
        let band = Self::PACKED_WINDOW_MIN_WIDTH..=Self::PACKED_WINDOW_MAX_WIDTH;
        // The widths a prefix point names at a count the tid space holds.
        let mut widths: Vec<u32> = Self::prefix_rungs(baked)
            .iter()
            .copied()
            .flat_map(|m| {
                (2..=crate::reserved_tids::PAGED_MAX_WINDOWS).filter_map(move |s| {
                    if m.is_multiple_of(s) {
                        Some(m / s)
                    } else {
                        None
                    }
                })
            })
            .filter(|w| band.contains(w))
            .collect();
        widths.sort_unstable();
        widths.dedup();
        let mut v = Vec::new();
        for &w in &widths {
            for s in 2..=crate::reserved_tids::PAGED_MAX_WINDOWS {
                if baked.contains(&(w * s)) {
                    v.push((w, s));
                }
            }
        }
        v.sort_by_key(|&(w, s)| (w * s, std::cmp::Reverse(w)));
        v
    }

    /// Which prefix rung a continuation chunk runs on: the index into `widths` (ascending, the rungs
    /// that built) of the smallest width holding the `rows_left` rows of the prompt, else the widest
    /// (the chunk is then capped at it and the prompt continues). `None` only for an empty ladder.
    pub fn prefix_rung_for(widths: &[u32], rows_left: u32) -> Option<usize> {
        widths
            .iter()
            .position(|&m| rows_left <= m)
            .or_else(|| widths.len().checked_sub(1))
    }

    /// ⛔⛔⛔ THE PREFIX-CAPABLE LADDER THAT BAKED MUST TOP OUT AT THE PREFIX-FREE LADDER's TOP — the bake's
    /// proof, panicked on by the macro.
    ///
    /// Every mid-prompt continuation chunk runs at the prefix-free ladder's top width (the cap, and the room
    /// the step-back leaves). A prefix ladder that stops below it hands such a chunk a bundle narrower than its
    /// rows, which the host refuses on the card. A prefix ladder that baked NOTHING while a prefix-free ladder
    /// did is the same shortfall in full: every continuation chunk refused. `Err` names the gap.
    pub fn prefix_ladder_tops_the_ladder(baked: &[u32], prefix: &[u32]) -> Result<(), String> {
        let Some(top) = baked.iter().copied().max() else {
            return Ok(());
        };
        match prefix.iter().copied().max() {
            Some(p) if p == top => Ok(()),
            Some(p) => Err(format!(
                "the prefix-capable prefill ladder tops out at {p} rows, below the prefix-free ladder's {top} \
                 — a mid-prompt continuation chunk runs {top} rows wide and would have no bundle to run on"
            )),
            None => Err(format!(
                "no prefix-capable prefill bundle baked, though the prefix-free ladder reaches {top} rows — \
                 every continuation chunk (start > 0) would be refused"
            )),
        }
    }

    /// ⛔⛔⛔ EVERY PREFIX RUNG THE LADDER ASKED FOR MUST HAVE BAKED. This is the bake's proof, and the macro
    /// panics on it.
    ///
    /// A prefix rung that does not lower used to be logged and skipped. The host then runs the chunk on the
    /// next wider rung that did bake (`prefix_rung_for` over what was built), so the build stays green and
    /// every tail that rung was for pays the wider width. Nothing says so: the log line is build-script
    /// stderr, which cargo swallows on a successful build. MEASURED, the cost of the one rung this ladder
    /// adds: without the page rung, granite-3.1-2b's ~2k-token prompt ran its 52-row tail on the window,
    /// 460 ms against 384.
    ///
    /// [`prefix_ladder_tops_the_ladder`](Self::prefix_ladder_tops_the_ladder) proves the TOP baked, which
    /// mid-prompt chunks need to run at all. This proves the rungs BELOW it, which only cost speed when
    /// missing, so their absence is otherwise silent. `asked` is [`prefix_rungs`](Self::prefix_rungs) of
    /// the prefix-free ladder, `baked` the prefix rungs that lowered, `refused` each that did not and why.
    pub fn prefix_ladder_bakes_every_rung(
        asked: &[u32],
        baked: &[u32],
        refused: &[(u32, String)],
    ) -> Result<(), String> {
        let missing: Vec<u32> = asked
            .iter()
            .copied()
            .filter(|m| !baked.contains(m))
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        let why = missing
            .iter()
            .map(|m| match refused.iter().find(|(r, _)| r == m) {
                Some((_, e)) => format!("rung m={m} did not lower: {e}"),
                None => format!("rung m={m} was asked for and never baked"),
            })
            .collect::<Vec<_>>()
            .join("; ");
        Err(format!(
            "the prefix-capable prefill ladder asked for {asked:?} rows and baked {baked:?} — {why}. A \
             continuation chunk the missing rung was for would run, with no error, on the next wider rung \
             and pay its width; lower the missing rung, or take it out of the ladder."
        ))
    }

    /// ⭐ SLOTS A PADDED CHUNK WRITE MAY OVERRUN ITS PAGE BY — the physical slack every plane carries
    /// past its addressable [`POOL_PAGE_SLOTS`](Self::POOL_PAGE_SLOTS) so that overrun lands in DEAD SPACE instead
    /// of the next kv head's keys.
    ///
    /// ⛔ THIS IS THE LONG-CONTEXT BUG, MADE UNREACHABLE. `PREFILL_CHUNK_SLOTS` was 96, which does not
    /// divide `POOL_PAGE_SLOTS` (256), so a chunk eventually started with less than a full chunk of room left: with
    /// chunks clipped to the room remaining, starts land on page offsets 0, 96, 192, and the one at 192
    /// has 64 slots of room for a 96-slot write. The 32-slot tail used to land at
    /// `block + POOL_PAGE_SLOTS*hd`, which IS the next kv head's block — so kv heads 1..nkvh lost the first
    /// 32 slots of their context. On card that read as a correct FIRST token (the last chunk scores from
    /// its own freshly computed K/V) and then fluent garbage from the second on, for every prompt long
    /// enough to need a third chunk. Measured boundary: coherent at a 192-token prompt, garbage at 204.
    ///
    /// `POOL_PAGE_SLOTS % PREFILL_CHUNK_SLOTS` is the room the last chunk of a page gets; the write needs a
    /// whole `PREFILL_CHUNK_SLOTS`, so the shortfall is the difference. Zero when the chunk divides the
    /// page, which is the only configuration that needs no slack at all.
    /// ⛔ ZERO, AND IT MUST STAY ZERO — the plane is EXACTLY the addressable page.
    ///
    /// This was `PREFILL_CHUNK_SLOTS - POOL_PAGE_SLOTS % PREFILL_CHUNK_SLOTS` (= 32), widening every plane so a
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
    /// DISTINCT FROM [`POOL_PAGE_SLOTS`](Self::POOL_PAGE_SLOTS), which is how many slots are ADDRESSABLE: masks are
    /// blocked by it, the fold sweeps it, and a [`KvSlot`] never exceeds it. The extra
    /// [`WRITE_SLACK`](Self::WRITE_SLACK) slots exist only so a padded chunk write has somewhere harmless
    /// to land, and nothing ever reads them — keeping the two numbers separate is what lets the overrun
    /// be absorbed without widening anything the attention actually sweeps.
    pub const PLANE_SLOTS: usize = Self::POOL_PAGE_SLOTS.as_usize() + Self::WRITE_SLACK;

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

    /// ⛔ THERE IS NO `MAX_PAGES_PER_ROW`, AND ITS ABSENCE IS THE LAW.
    ///
    /// It was a hand-picked const (6, then 16, then 32) that bound a row's context before the
    /// pool's memory ever got a vote — 8192 tokens reachable inside a pool that could hold ~70k,
    /// on a card with 100 Gi. The prefix mask's block count is now DERIVED at bake —
    /// `compute_bundle_layout` reserves `MAX_SEGMENT_BYTES / page_bytes` blocks, the most pages
    /// ONE region of KV can hold, which is ≥ the pages any runtime pool can be budgeted for
    /// (`max_single_allocation()` is one region) — so the reservation covers every pool any card
    /// can express and MEMORY is the binding constraint at runtime, exactly as it is on a GPU.
    /// The number itself lives in each bundle's own pmask placement, read back as
    /// [`BakedMaskBlocks`]; the per-step `MaskBlocks::fits` check and `RowPages::within_mask_reach`
    /// enforce it from the artifact, never from a const.
    ///
    /// The `rows_are_requests == false` arm (solo decode, prompt chunks) reserves the SAME
    /// derived count — one expression feeds both arms, so the old guard tying a separate
    /// `MAX_PAGES_PER_REQUEST` to this const is gone with the pair of constants it related.
    ///
    /// THE DECODE BATCH LADDER, so the widths that get baked and the requests a page holds cannot
    /// disagree. A rung wider than [`ROWS`](Self::ROWS) would bake a launch whose request axis walks
    /// off the end of the page into the next kv head's keys — silently, and only for the highest rows.
    /// ⛔ ONLY RUNGS THE PREFIX MASK CAN SERVE AT THE DERIVED PAGE CEILING.
    ///
    /// A gathered fold takes ONE mask block per page, and every rung reserves the same derived
    /// block count (see the note above), so EVERY rung serves the full depth at full width. An
    /// UNGATHERED rung body — per-(row, page) fold — is still bounded by `blocks / width`, and the
    /// launch-side `MaskBlocks::fits` check refuses such a step loudly rather than letting it read
    /// unstaged mask bytes.
    pub const BATCH_RUNGS: [u32; 5] = [2, 4, 8, 16, 32];

    /// THE WIDEST BAKED RUNG, as a const — how many ROWS a pool must be able to give a page each.
    ///
    /// A const rather than a runtime max over the ladder, because the pool has to be sized before any
    /// batch exists and the answer is known at bake. It is the factor the pool's page count scales by:
    /// every row owns its own pages now, so `pool_pages = pages_per_row * WIDEST_BATCH_RUNG`.
    pub(crate) const WIDEST_BATCH_RUNG: u32 = Self::BATCH_RUNGS[Self::BATCH_RUNGS.len() - 1];

    // ⛔ NO `TARGET_POSITIONS_PER_ROW`, AND NO LADDER-FITS ASSERT. The target const was the depth
    // of a deleted depth-vs-width trade (`widest_reaching` — see `PoolRows::for_admission`'s doc for
    // how the free list dissolved it), and it tracked the mask ceiling by reading the const that
    // no longer exists. The ladder-fits assert tied the widest decode rung to the requests a page
    // holds; a page holds no requests now, so a batch rung is bounded by the pool's PAGES, which is
    // a host allocation question and not a baked one — and the depth a row may reach is the
    // artifact-derived pmask ceiling, enforced per launch.

    pub const fn new(nkvh: NonZeroU32, hd: HeadWidth) -> PagedKvPool {
        PagedKvPool { nkvh, hd }
    }

    /// DISTINCT kv heads, as the nonzero count the pool's own field already is — the reader that
    /// needs the integer calls `.get()` at its boundary and never re-proves it nonzero.
    pub const fn nkvh(&self) -> NonZeroU32 {
        self.nkvh
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
    /// ELEMENTS IN ONE KV HEAD'S BLOCK of a plane — `hd * POOL_PAGE_SLOTS`, the whole of one head's page.
    ///
    /// This replaced `block_index(kvh) = kvh * ROWS` and `request_stride`. The head stride used to be
    /// `ROWS` request strides; it is now simply a head's own footprint, so there is no second quantity to
    /// drift against.
    /// ⭐ HOW MANY SLOTS A PREFILL CHUNK STARTING AT `start` MAY CLAIM — the room left in its window.
    ///
    /// A chunk's cache write is cut into one part per PAGE, each part resolving its own page through the
    /// request's block table (`EmittedOp::kv_slot_offset`) — so a chunk starting at a page boundary may
    /// fill all [`PREFILL_WINDOW_PAGES`](Self::PREFILL_WINDOW_PAGES) pages of its window, and part `p`
    /// lands at the start of the window's page `p`.
    ///
    /// ⛔ A CHUNK STARTING MID-PAGE STAYS INSIDE THAT PAGE. A part addresses the page its FIRST row's slot
    /// falls in and writes its rows consecutively from there, so a part that begins mid-page cannot cross
    /// the page end: the slots past it belong to a page that part does not address, and writing real rows
    /// there would put them somewhere nothing reads them back from. Hence the room is the page remainder
    /// whenever the offset is not zero. (A padded window wider than a page never starts mid-page —
    /// [`chunk_write_start`](Self::chunk_write_start) steps it back to the page start.)
    ///
    /// Total, not fallible: every slot is inside some page, so there is always at least one slot of room.
    pub fn chunk_room(start: KvSlot) -> SlotCount {
        let page = Self::POOL_PAGE_SLOTS.get();
        match start.get() % page {
            0 => SlotCount::new(Self::PREFILL_WINDOW_PAGES as u32 * page),
            off => SlotCount::new(page - off),
        }
    }

    /// ⭐ THE PHYSICAL EXTENT, TYPED — the only way the address law can obtain it, so the ADDRESSABLE count cannot
    /// be substituted for it in a stride.
    pub(crate) const fn plane_extent() -> PlaneExtent {
        PlaneExtent(Self::PLANE_SLOTS as u32)
    }

    pub fn plane_block_elems(&self) -> u64 {
        // `PLANE_SLOTS`, NOT `POOL_PAGE_SLOTS`: this is the distance to the NEXT KV HEAD, so it is exactly the
        // quantity a padded chunk write overruns. Sizing it by the addressable slot count is what put
        // that overrun on top of the next head's keys.
        // Physical, for the same reason: this is the distance to the NEXT KV HEAD, which is exactly what a padded
        // chunk write overruns.
        u64::from(self.hd.get()) * u64::from(Self::plane_extent().get())
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
    pub fn stick_block_elems(&self) -> u64 {
        u64::from(self.hd.get()) * STK as u64
    }

    /// ⭐ THE ENTRY IN **BYTES** — what one index step advances the address by. `stick_block_elems`
    /// times the pool's word length; stated here so the `* 2` is not respelled at every caller.
    pub fn stick_block_bytes(&self) -> u64 {
        self.stick_block_elems() * 2
    }

    /// Elements in one K-or-V plane of a page+layer — `nkvh` kv heads, each holding one page.
    ///
    /// ⭐ `nkvh` BLOCKS, with the block size stated ONCE in
    /// [`plane_block_elems`](Self::plane_block_elems). It used to spell `hd * POOL_PAGE_SLOTS` again here,
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
    pub(crate) fn plane_stride(&self) -> u64 {
        u64::from(self.nkvh.get()) * self.plane_block_elems()
    }

    /// Elements in one page+layer: Kᵀ, then V, then natural K.
    pub fn layer_stride(&self) -> u64 {
        3 * self.plane_stride()
    }

    /// Head width this pool laid itself out for, AS THE TYPE — the one value every reader
    /// takes; no use site re-mints it.
    pub const fn head_width(&self) -> HeadWidth {
        self.hd
    }
}

/// ⭐⭐⭐ HOW A LAUNCH'S CAUSAL NEW-TOKEN BLOCK IS EMITTED — one law, read by the emitter (`assemble_attn`)
/// and by the host that tiles the causal mask (`forward_tape::causal_tiled`), so the mask's layout and the
/// ops that read it cannot disagree.
///
/// ⭐ WHY THERE ARE TWO. The block is `mq_pad` columns wide. Folded as `mq_pad / 64` stick-wide sub-blocks
/// per QUERY HEAD, its op count grows with the chunk's width: granite-2b at 256 prompt rows spends about
/// as much on it as halving the chunk count saves in weight streaming (MEASURED, one card: 10k TTFT 13.6 s
/// at 128 rows against 13.0 s at 256, and ~1k got SLOWER, 0.90 → 0.94 s). As ONE block per KV HEAD over
/// all `mq_pad` columns — the split-KV chunk's own form — it is a fixed handful of ops at any width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NewBlockForm {
    /// Stick-wide sub-blocks folded per query head: decode, every batch, every prompt rung that is not a
    /// whole number of sticks, and every head dim past [`NewBlockForm::CHUNK_MAX_SLABS`].
    SubBlocks,
    /// One block per chunk ROW BLOCK ([`ChunkRowBlocks`]: a kv head at one feature slab, a query head
    /// above it) over all `mq_pad` columns: a prompt chunk whose width is a whole number of sticks (so
    /// its new-V block is exactly `mq_pad` rows and PV can contract it whole) — the split-KV chunk's own
    /// `[rows, slots]` blocks, see [`ChunkScratch`].
    Chunk,
}

impl NewBlockForm {
    /// ⭐ THE WIDEST HEAD DIM, IN FEATURE SLABS, THE CHUNK FORM IS BUILT AT: one slab (granite-2b) and two
    /// (granite-8b) — the two whose per-row-block score and per-slab value forms the split-KV chunk runs
    /// on card. A wider head dim (the gemma bakes' 256 and 512) keeps the sub-blocks byte for byte.
    pub(crate) const CHUNK_MAX_SLABS: u32 = 2;

    /// The form a launch of `mq` rows at head width `hd` takes; `rows_are_requests` is a decode batch.
    /// Takes the typed [`HeadWidth`], so the whole-stick law is by construction and the slab count
    /// is [`HeadWidth::sticks`], never re-divided here.
    pub const fn of(mq: u32, hd: HeadWidth, rows_are_requests: bool) -> NewBlockForm {
        if !rows_are_requests
            && mq >= POOL_STICK
            && mq.is_multiple_of(POOL_STICK)
            && hd.sticks() >= 1
            && hd.sticks() <= Self::CHUNK_MAX_SLABS
        {
            NewBlockForm::Chunk
        } else {
            NewBlockForm::SubBlocks
        }
    }
}

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

/// ⭐⭐⭐⭐ A PLANE'S **PHYSICAL** SLOT EXTENT — the distance to the next kv head, not the addressable count.
///
/// ⛔ `POOL_PAGE_SLOTS` AND `PLANE_SLOTS` ARE BOTH 256 AND MEAN DIFFERENT THINGS. Addressable: masks are blocked by it,
/// the fold sweeps it, a `KvSlot` never exceeds it. Physical: what separates one kv head's block from the next,
/// and therefore the quantity a padded chunk write overruns. They differ by `WRITE_SLACK`, which is 0 today —
/// **and was 32 once, which BROKE head_dim 128** (` Rome. Q` became ` Romes,`), because the extra slots move the
/// feature-stick GROUP stride that every reader re-derives and a head spans two sticks above hd=64.
///
/// So this is a coincidence with a proven failure history: the two were equal, a commit made them differ, and the
/// difference reached the addresses through expressions that could not say which extent they meant. A type at the
/// one site where both appear is what makes the substitution refuse to compile.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) struct PlaneExtent(u32);

impl PlaneExtent {
    /// The physical extent, for the stride arithmetic that is the only consumer.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// WHICH PLANE OF A PAGE+LAYER — the three copies of the same keys the attention needs in different
/// orders. Named, because "2" meaning Kᵀ is the sort of thing a comment has to say.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvPlane {
    /// Natural K, `[POOL_PAGE_SLOTS, hd]` — what the cache write produces.
    Knat,
    /// V, `[POOL_PAGE_SLOTS, hd]` — the value kernel.
    V,
    /// Transposed K, `[hd, POOL_PAGE_SLOTS]` — the score kernel.
    Kt,
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
}

/// A request of a packed rung's windows — minted only by [`AnswerRows::rows`], never spelled
/// by a caller, so an index past the rung's capacity cannot reach a row law.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RequestIndex(usize);

impl RequestIndex {
    /// The request's slot in the rung's window order — the readback's pick order, by which the
    /// launch's slot list seats its requests.
    pub const fn get(self) -> usize {
        self.0
    }
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
