// SPDX-License-Identifier: Apache-2.0
//! The card's nouns: the session over the build's cards, its rung ladders and launch types.

/// ⭐ THE WORLD THIS BUILD SERVES — `Build`, the one `tp/N` the build names (one card when it names
/// none): the same type the bake baked every bundle for, so a session and its bundles agree by type.
pub(crate) type World = scratchy_target_spyre::bundle_code::collective::Build;

/// A session over the build's world: one executor per rank of [`World`], each BORROWING its card
/// for `'static` — the loaded model lives for the process, and so do the cards it runs on (the
/// loader opens them once and hands out `&'static Cards`; see `load_inner`).
pub(crate) type SuperDscSession =
    scratchy_target_spyre_card::sdsc_runner::SuperDscSession<'static, World>;

/// The build's cards, one per rank of [`World`] — OWNED here, once, for the process; every session
/// borrows them.
pub(crate) type Cards = scratchy_target_spyre::bundle_code::collective::PerRank<
    World,
    scratchy_target_spyre_card::runtime::CardRuntime<World>,
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
    // ⛔ THE HOST READBACK HALF IS GONE — `len` (the whole-placement element count for
    // `read_tensor`) and `rows_contiguous` (the de-interleave into per-request rows) served
    // the per-request logits readback the submit/collect split deleted: the sampler answers
    // on the card now and the idx stick is the only readback. Both deleted with their
    // readers, at the auditor's order — the struct stays for the load-time placement
    // refusal and the rung's own log line.
}

/// What a forward should hand back from its lm-head tail. An enum rather than a `bool` because the
/// per-request case CANNOT be served without the tensor's geometry — see [`LogitsGeom`] — and a bare
/// `want_logits: true` gave no place to put it, which is exactly how the batched path came to slice
/// one row's worth and hand it to eight requests.
/// One rung of the batched-decode ladder: a bundle baked for exactly `seqs` requests, aliasing the
/// one-request decode session's weights and page pool.
// ⛔ `sendnn`-ONLY, restored from 8bd5c755b. My branch was cut BEFORE that commit added
// these gates, and resolving the rebase conflict in favour of the split took the whole
// file — which discarded them. They are what makes the KTIR-only build compile.
pub(crate) struct DecodeRung {
    /// The row count this rung's bundle was BAKED at — every launch on it binds exactly this many
    /// rows, live or padding, and every mask/fold/logits extent below derives from it.
    pub(crate) seqs: ktir_superdsc::sdsc_abstract::RungWidth,
    /// The rung's own session, aliasing the owner's weights and KV pool. Which body of its walk
    /// ladder a step runs is the session's own answer (`SuperDscSession::step_body`), and the step
    /// the host stages is that body's (`PagedStep::on_body`).
    pub(crate) sess: SuperDscSession,
    // ⛔ NO `logits` FIELD ANY MORE — the rung's LogitsGeom served the host per-request
    // readback the submit/collect split deleted (the sampler answers on the card; the idx
    // stick is the only readback). The LOAD-TIME refusal stays in spyre_load: a rung whose
    // baked logits placement is missing or not whole rows still panics at load, off
    // `LogitsGeom::from_placement` directly.
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

pub(crate) enum LogitsWanted {
    /// Nothing — every prompt chunk but the final one, which launches no suffix and reads
    /// nothing back.
    None,
    /// The FINAL prompt chunk: its suffix runs, the rung's own minted sampler picks the
    /// first generated token on the card from the folded lm-head row, and the idx stick
    /// is read back exactly like a decode launch's.
    LastRow,
    /// A DECODE BATCH: one row per request, every row a different request. The
    /// rung's sampler picks every token on the card; the idx stick's raw read is
    /// deferred and comes back through the one collect door
    /// (`collect_decode_tokens`) — in-call when the schedule cannot defer, from
    /// the drain when it can. There is no synchronous decode arm.
    PerRequestDeferred,
}

/// The sendnn run backend selected from the baked bundle shape. DEFAULT-mode
/// (`SCRATCHY_SENDNN_MODE=default`) bakes ONE group whose `prefill` and `decode`
/// slots hold the SAME single static-SDPA graph (host-grown KV, no resident paged
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
    /// AND an optional m=N `prefill` session (a SEPARATE SuperDscSession baked at
    /// KTIR_PREFILL_LEN, lm_head skipped). The prompt runs through `prefill` in ONE forward
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
    /// PREFILL WIDTH LADDER — ASCENDING by baked query-row count `mq`; the last entry is the widest.
    /// EMPTY ⇒ no batched-prefill bundle at all (multi-token prompts fall back to the sequential
    /// mq=1 decode path).
    ///
    /// Each rung is baked for EXACTLY its `mq` query rows (the tensor shapes are concrete), so
    /// `run_step` must be called with exactly that m: the real prompt tokens fill rows `[0..real)`
    /// and rows `[real..mq)` are PAD — future positions the causal mask never lets a real row attend,
    /// whose KV slots sit beyond `seq_pos` so decode never reads them. Picking the SMALLEST rung that
    /// still holds the chunk is what keeps that padding cheap: per-chunk cost is `fixed + slope·mq`
    /// with the fixed term dominant, so one width is wrong both ways (too narrow re-pays the fixed
    /// term per extra chunk, too wide computes empty rows).
    ///
    /// A rung needs its OWN session because `mq` sizes the prefix (embed rows), the suffix and every
    /// seg0/seg3 activation. Rungs DO share the resident weights (seg1) and KV (seg2) by alias, so the
    /// ladder costs program + activation memory, not another copy of the model. All rungs carry the
    /// SAME tensor ids (verified: identical 1467-key placement set across widths 15/31/47/63 — only
    /// the 611 activation SIZES scale with `mq`), so ONE prefill `BundleMeta` binds any of them, and
    /// `run_card_launch` rebinds every const on each call — so switching rungs between chunks is free.
    pub(crate) prefill_rungs: Vec<(usize, SuperDscSession)>,
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
    /// The prefill bundles whose walk ladder tops at the whole pool — what a CONTINUATION chunk
    /// (`start > 0`) must run on, because it has resident KV to attend; each launch walks the
    /// narrowest of its rungs that reaches the chunk's last position. See [`PrefixLadder`].
    ///
    /// Every entry in [`prefill_rungs`](Self::prefill_rungs) walks only its own chunk's pages, so it
    /// serves a chunk at `start == 0` (`PagedStep::reach`).
    ///
    /// Empty ⇒ no prefix-capable bundle baked, so a continuation chunk cannot run and is refused
    /// rather than attending a cache its walk does not reach.
    pub(crate) prefill_prefix: PrefixLadder,
}

/// ⭐ THE PREFIX-CAPABLE PREFILL LADDER — the sessions a continuation chunk can run on, ascending by width,
/// and the ONE place a continuation chunk's width is chosen.
///
/// The field is private and [`Self::of_built`] is its only mint, so the entries are exactly the sessions
/// that were BUILT from baked bundles; [`Self::rung_for`] returns a [`PrefixRung`] naming one of them or
/// nothing, so a chunk cannot be handed a width no bundle was baked at.
///
/// ⛔ IT WAS ONE BUNDLE AT THE WIDEST RUNG, and the tail paid for it: granite-3.1-2b's ~2k-token prompt
/// ran its 52-row tail as a full 512-row chunk, 460 ms against 384 on the 256-row build.
pub(crate) struct PrefixLadder(Vec<(usize, SuperDscSession)>);

/// ONE CHOICE OF [`PrefixLadder`] RUNG, made once per chunk and carried — the chunk's cap, its padded
/// write, its session and the baked row count its K/V extent is proven against all read THIS value, so
/// they cannot come from two choices. No public constructor: only [`PrefixLadder::rung_for`] mints one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefixRung {
    index: usize,
    width: usize,
}

impl PrefixRung {
    pub(crate) fn width(self) -> usize {
        self.width
    }
}

impl PrefixLadder {
    /// The ladder of the sessions that built, sorted ascending by width (the manifest's order already,
    /// stated again here so a caller cannot hand it an unsorted list that `rung_for` would misread).
    pub(crate) fn of_built(mut rungs: Vec<(usize, SuperDscSession)>) -> PrefixLadder {
        rungs.sort_by_key(|(m, _)| *m);
        PrefixLadder(rungs)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The baked widths, ascending.
    pub(crate) fn widths(&self) -> Vec<usize> {
        self.0.iter().map(|(m, _)| *m).collect()
    }

    /// ⭐ THE RUNG A CONTINUATION CHUNK RUNS ON — the narrowest that holds the `rows_left` rows the
    /// prompt still has, else the widest (the chunk is then capped at its width). `None` only for an
    /// empty ladder.
    pub(crate) fn rung_for(&self, rows_left: usize) -> Option<PrefixRung> {
        let index = self
            .0
            .iter()
            .position(|(m, _)| *m >= rows_left)
            .or_else(|| self.0.len().checked_sub(1))?;
        Some(PrefixRung {
            index,
            width: self.0[index].0,
        })
    }

    /// The session of a rung THIS ladder chose.
    pub(crate) fn session(&mut self, rung: PrefixRung) -> &mut SuperDscSession {
        &mut self.0[rung.index].1
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
