// SPDX-License-Identifier: Apache-2.0
//! The card's nouns: the session over the build's cards, its rung ladders and launch types.

/// ⭐ THE WORLD THIS BUILD SERVES — `Build`, the one `tp/N` the build names (one card when it names
/// none): the same type the bake baked every bundle for, so a session and its bundles agree by type.
pub(crate) type World = scratchy_target_spyre::bundle_code::collective::Build;

/// A session over the build's world: one executor per rank of [`World`], each BORROWING its
/// card's runtime for `'static` (and that runtime borrowing its CARD for `'static` — both
/// process-lifetimes, because the loaded model lives for the process and so do the cards it runs
/// on; the loader opens the cards once, leaks them, and hands out `&'static Cards`; see
/// `load_inner`).
pub(crate) type SuperDscSession =
    scratchy_target_spyre_card::sdsc_runner::SuperDscSession<'static, 'static, World>;

/// The world's OPENED CARDS — phase 1's product, leaked once for the process at the load site, so
/// every runtime phase 2 builds over them borrows a card for `'static`.
pub(crate) type OpenedCards = scratchy_target_spyre::bundle_code::collective::PerRank<
    World,
    scratchy_target_spyre_card::runtime::Card,
>;

/// The build's card RUNTIMES, one per rank of [`World`] — OWNED here, once, for the process; every
/// session borrows them, and each of them borrows its opened card.
pub(crate) type Cards = scratchy_target_spyre::bundle_code::collective::PerRank<
    World,
    scratchy_target_spyre_card::runtime::CardRuntime<'static, World>,
>;

/// The BAKED logits tensor of a bundle: `[rows, width]`, STICK-BLOCKED on the vocab axis.
///
/// Two things about it are not what a reader assumes, and both are silent when `rows == 1`:
///
///  * `width` is the PADDED placement width, not the model's vocab. The emitter pads the vocab axis
///    to a whole number of sticks per core — granite's 49155 is placed as 51200 — so a row spans
///    `width` elements of which only the leading `vocab` are real.
///  * A row is NOT contiguous. [`ktir_superdsc::sdsc_abstract::StickLayout::dev_off`] places
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
        rung: ktir_superdsc::sdsc_abstract::RungWidth,
        vocab: usize,
    ) -> Option<LogitsGeom> {
        let elems = (size_bytes / 2) as usize;
        let rows = rung.count();
        if !size_bytes.is_multiple_of(2) || !elems.is_multiple_of(rows) {
            return None;
        }
        let width = elems / rows;
        (width >= vocab).then_some(LogitsGeom { rows, width })
    }
}

/// One rung of the batched-decode ladder: a bundle baked for exactly `seqs` requests, aliasing the
/// one-request decode session's weights and page pool.
pub(crate) struct DecodeRung {
    /// The row count this rung's bundle was BAKED at — every launch on it binds exactly this many
    /// rows, live or padding, and every mask/fold/logits extent below derives from it.
    pub(crate) seqs: ktir_superdsc::sdsc_abstract::RungWidth,
    /// The rung's own session, aliasing the owner's weights and KV pool. Which body of its walk
    /// ladder a step runs is the session's own answer (`SuperDscSession::step_body`), and the step
    /// the host stages is that body's, at the walk its rung records (`RungBody::walk`).
    pub(crate) sess: SuperDscSession,
}

/// WHICH SESSION A DECODE STEP RUNS ON — the one-request OWNER, or one rung of the ladder.
///
/// The loader's `rn <= 1 → continue` means the W=1 rung IS the owner session (`decode`),
/// not a ladder entry, so a single request is a batch of ONE on the owner — selected by the
/// same `decode_rung_for` call every wider batch uses, never by a second branch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RungId {
    /// The m=1 `decode` session every ladder rung aliases — the W=1 rung.
    Owner,
    /// Ladder rung `i` of `decode_rungs`.
    Ladder(usize),
}

/// ONE DECODE SEAT: the launch-width facts plus the session they describe, whichever rung
/// was chosen. The fields are exactly `DecodeRung`'s, so the launch reads one shape and
/// never re-derives an owner fact it could get wrong — `seqs` is `1` on the owner.
pub(crate) struct DecodeSeat<'a> {
    pub(crate) seqs: ktir_superdsc::sdsc_abstract::RungWidth,
    pub(crate) sess: &'a mut SuperDscSession,
}

/// The sendnn session type. ONE variant: the dxp-compiled SuperDSC bundle is the only thing that
/// runs on silicon. The sengraph families (offline_decoder `GroupedSession`, the static-paged and
/// default-mode single-graph sessions) are GONE — they were the pre-SuperDSC bring-up path and the
/// router sent nothing to them.
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
    /// AND an optional m=N `prefill` session (a SEPARATE SuperDscSession baked at the
    /// ladder's top width, lm_head skipped). The prompt runs through `prefill` in ONE forward
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
pub(crate) struct SuperDscBundle {
    /// The one-request decode session — the OWNER of the resident weights and the KV pool (with
    /// the pool's statics, `paged_step::card_statics`), which every other session aliases.
    pub(crate) decode: SuperDscSession,
    /// ⭐ THE PROMPT-CHUNK LADDER — every rung a prompt chunk can run on, each its session and the
    /// meta of the wiring cell that session was built from ([`ChunkLadder`]).
    pub(crate) chunks: ChunkLadder,
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
    /// ⭐ THE PACKED CONTINUATION RUNGS — a wave of continuation tails batched into one launch
    /// ([`PrefixLadder`]). A lone chunk never runs here: its rung is [`Self::chunks`]'s.
    pub(crate) packed: PrefixLadder,
}

/// ⭐ WHAT A CHUNK'S ATTENTION MUST REACH — which family of rungs can run it, decided by the
/// chunk's own first position, the one fact that separates them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ChunkReach {
    /// The prompt's first chunk (`start == 0`): no key precedes it, so a rung whose walk covers
    /// only its own pages runs it.
    First,
    /// A chunk past the first (`start > 0`): the keys before it are resident, so it runs on a
    /// prefix-capable rung, whose walk reaches the whole pool.
    Continuation,
}

impl ChunkReach {
    /// The reach of a chunk whose first position is `start`.
    pub(crate) const fn of_start(start: usize) -> Self {
        if start == 0 {
            Self::First
        } else {
            Self::Continuation
        }
    }
}

/// ⭐ ONE PROMPT-CHUNK RUNG — its baked row count, its session, and the meta of the wiring cell the
/// session was built from, held together, so a launch's session and its wiring cannot come from two
/// rungs. Built only by the loader ([`ChunkFamily::of_built`]).
pub(crate) struct ChunkRung {
    pub(crate) width: ktir_superdsc::sdsc_abstract::RungWidth,
    pub(crate) meta: crate::spyre_types::BundleMeta,
    pub(crate) sess: SuperDscSession,
}

/// ⭐ ONE FAMILY OF CHUNK RUNGS, NEVER EMPTY — its widest rung, and the narrower ones ascending.
/// Holding the widest apart is what makes the selection total: every chunk has a rung, so no
/// chunk can be refused for want of one.
pub(crate) struct ChunkFamily {
    narrower: Vec<ChunkRung>,
    widest: ChunkRung,
}

impl ChunkFamily {
    /// The family of the rungs the loader built, or `None` when it built none — the loader's to
    /// answer for, at load.
    pub(crate) fn of_built(mut rungs: Vec<ChunkRung>) -> Option<Self> {
        rungs.sort_by_key(|r| r.width.count());
        let widest = rungs.pop()?;
        Some(Self {
            narrower: rungs,
            widest,
        })
    }

    /// ⭐ THE RUNG A CHUNK RUNS ON — the narrowest that holds the `left` rows the prompt still has,
    /// else the widest, which then takes as many rows as it holds and leaves the rest to the next
    /// chunk. The ONE statement of a chunk's width, for both families.
    fn rung_for(&mut self, left: usize) -> &mut ChunkRung {
        match self.narrower.iter().position(|r| r.width.count() >= left) {
            Some(i) => &mut self.narrower[i],
            None => &mut self.widest,
        }
    }

    /// The baked widths, ascending.
    pub(crate) fn widths(&self) -> Vec<usize> {
        self.narrower
            .iter()
            .chain(std::iter::once(&self.widest))
            .map(|r| r.width.count())
            .collect()
    }
}

/// ⭐ THE PROMPT-CHUNK LADDER — both families, each never empty: the rungs a prompt's first chunk
/// runs on, and the prefix-capable rungs every later chunk runs on. [`Self::rung_for`] is the ONE
/// place a chunk's rung is chosen: by its reach, then its rows.
pub(crate) struct ChunkLadder {
    pub(crate) first: ChunkFamily,
    pub(crate) continuation: ChunkFamily,
}

impl ChunkLadder {
    /// The rung a chunk starting at `start` with `left` rows still to run takes.
    pub(crate) fn rung_for(&mut self, start: usize, left: usize) -> &mut ChunkRung {
        match ChunkReach::of_start(start) {
            ChunkReach::First => self.first.rung_for(left),
            ChunkReach::Continuation => self.continuation.rung_for(left),
        }
    }
}

/// ⭐ THE PACKED CONTINUATION LADDER — the rungs a WAVE of continuation tails runs on, ascending by
/// (width, seqs), every one packing at least two tails into one launch. A lone chunk is never one
/// of these: its rung is [`ChunkLadder`]'s.
///
/// The field is private and [`Self::of_built`] is its only mint, so the entries are exactly the
/// sessions that were BUILT from baked bundles; [`Self::packed_fill`] returns a [`PrefixRung`] naming
/// one of them or nothing, so a wave cannot be handed a pair no bundle was baked at.
pub(crate) struct PrefixLadder(Vec<PrefixRungBuilt>);

/// One BUILT rung of [`PrefixLadder`] — its per-request width, the count of tails its wave batches
/// into one launch, its own meta, and its session. The loader mints these from the bundles the bake
/// shipped; the ladder is their only home.
pub(crate) struct PrefixRungBuilt {
    pub(crate) width: usize,
    pub(crate) seqs: usize,
    /// ⭐ THE RUNG'S OWN WIRING'S META — its OWN pair's cell (its extent pair, its `s`-row seat,
    /// its own consts are the pair's, no representative's). The wave front launches through it
    /// beside the session, so a launch's wiring and its session cannot come from two rungs.
    pub(crate) meta: crate::spyre_types::BundleMeta,
    pub(crate) sess: SuperDscSession,
}

/// ONE CHOICE OF [`PrefixLadder`] RUNG, made once per wave group and carried — its width, its
/// request count and its session all read THIS value, so they cannot come from two choices. No
/// public constructor: only [`PrefixLadder::packed_fill`] mints one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefixRung {
    index: usize,
    width: usize,
    seqs: usize,
}

impl PrefixRung {
    pub(crate) fn width(self) -> usize {
        self.width
    }

    /// The rung's request count — its wave size.
    pub(crate) fn seqs(self) -> usize {
        self.seqs
    }
}

impl PrefixLadder {
    /// The ladder of the sessions that built, sorted ascending by (width, seqs) (the manifest's
    /// order already, stated again here so a caller cannot hand it an unsorted list that the
    /// selections below would misread).
    pub(crate) fn of_built(mut rungs: Vec<PrefixRungBuilt>) -> PrefixLadder {
        rungs.sort_by_key(|r| (r.width, r.seqs));
        PrefixLadder(rungs)
    }

    /// The baked (width, seqs) pairs, ascending.
    pub(crate) fn pairs(&self) -> Vec<(usize, usize)> {
        self.0.iter().map(|r| (r.width, r.seqs)).collect()
    }

    /// ⭐ THE RUNG AND GROUP SIZE A WAVE'S NEXT GROUP RUNS ON — the packed rung that seats the
    /// MOST tails of the `live` the wave still holds (capped at the rung's own `seqs`), at
    /// `width` at least the widest tail's rows (every window pads its own tail to its rung's
    /// width). Product ties break toward the smaller rung, then the smaller count — the fewest
    /// padded windows — so when ONE rung holds the whole wave this is that rung exactly (the
    /// smallest product holding both halves). A wave larger than every rung FILLS RUNGS: the
    /// caller launches group after group, widest tails first, and no tail waits on the whole
    /// wave fitting one rung. `None` when no packed rung seats even two tails at the widest
    /// tail's width (a tail wider than every window); the caller then runs the tails as solo
    /// chunks (a selection, never a refusal).
    pub(crate) fn packed_fill(
        &self,
        max_tail_width: usize,
        live: usize,
    ) -> Option<(PrefixRung, usize)> {
        let (index, rung) = self
            .0
            .iter()
            .enumerate()
            .filter(|(_, r)| r.seqs > 1 && r.width >= max_tail_width)
            .max_by_key(|(_, r)| {
                (
                    r.seqs.min(live),
                    std::cmp::Reverse(r.width * r.seqs),
                    std::cmp::Reverse(r.seqs),
                )
            })?;
        Some((
            PrefixRung {
                index,
                width: rung.width,
                seqs: rung.seqs,
            },
            rung.seqs.min(live),
        ))
    }

    /// ⭐ THE WAVE FRONT'S ONE DOOR — the rung's own meta and its own session, borrowed
    /// together, so a packed launch's wiring and the session that runs it cannot come from
    /// two rungs.
    pub(crate) fn packed_parts(
        &mut self,
        rung: PrefixRung,
    ) -> (&crate::spyre_types::BundleMeta, &mut SuperDscSession) {
        let cell = &mut self.0[rung.index];
        (&cell.meta, &mut cell.sess)
    }
}

impl SuperDscBundle {
    /// THE RUNG A DECODE STEP RUNS ON — [`RungId::Owner`] when one request is live (the W=1
    /// rung IS the owner session, by the loader's `rn <= 1 → continue`), else the index of the
    /// SMALLEST ladder rung that holds `live` requests. `None` when `live` exceeds the widest
    /// rung — the caller then refuses: there is no one-forward-per-request decode arm any more.
    ///
    /// The live count is the only input: live request `i` runs in slot `i`, whatever its pages.
    pub(crate) fn decode_rung_for(
        &self,
        live: ktir_superdsc::sdsc_abstract::LiveRows,
    ) -> Option<RungId> {
        if live.get() <= 1 {
            // The owner ALWAYS exists — it is the session every rung aliases — so the W=1
            // seat is never absent and a single request never falls to a second path.
            return Some(RungId::Owner);
        }
        self.decode_rungs
            .iter()
            .position(|r| r.seqs.count() >= live.get())
            .map(RungId::Ladder)
    }

    /// THE SEAT A DECODE STEP LAUNCHES THROUGH — the owner's facts and session for
    /// [`RungId::Owner`], the rung's own for a ladder entry. ONE accessor, so a launch
    /// cannot read one rung's width off another rung's session.
    pub(crate) fn decode_seat(&mut self, id: RungId) -> DecodeSeat<'_> {
        match id {
            RungId::Owner => DecodeSeat {
                // The owner is the m=1 bake; 1 is a baked width by `RungWidth`'s own law.
                seqs: ktir_superdsc::sdsc_abstract::RungWidth::of_baked_rows(1)
                    .expect("1 is a baked width"),
                sess: &mut self.decode,
            },
            RungId::Ladder(i) => {
                let r = &mut self.decode_rungs[i];
                DecodeSeat {
                    seqs: r.seqs,
                    sess: &mut r.sess,
                }
            }
        }
    }
}
