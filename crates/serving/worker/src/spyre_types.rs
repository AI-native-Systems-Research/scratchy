// SPDX-License-Identifier: Apache-2.0
//! THE NOUNS — what a loaded Spyre model IS.
//!
//! `BundleMeta` (one baked bundle: its wiring, its `BakeFacts`, its capacity), `Loaded` (the
//! sessions and the rung ladders), `ReqState` (one request's draw), and the smaller value types the
//! launch path passes around.
//!
//! ⛔ EVERY MODEL FACT IN HERE ARRIVED GENERATED. `BundleMeta` carries the `&'static Wiring` the
//! `#[forward]` macro emitted; nothing in this file inspects a checkpoint, a config, or a manifest.

use std::path::PathBuf;

use scratchy_core_model::weight::HfModelConfig;
// `--target sendnn` swaps the KTIR emulator runner for the on-silicon sendnn
// runner; the bundle type + session type are cfg-selected, everything else
// (weight load, dynamic sources, KV loop, sampling) is shared.
#[cfg(not(feature = "spyre-hw"))]
use scratchy_target_spyre::runner::SpyreSession;
#[cfg(feature = "spyre-hw")]
use scratchy_target_spyre::sdsc_runner::SuperDscSession;

/// One layer's KV pools: the K and V cache tensors its paged attention gathers from and its KV
/// write scatters into. The emulator never stages them — the session's own zeroed allocation is
/// their starting state; the device writes them every step and they persist — nothing re-stages or
/// reads them back.
pub(crate) struct LayerWiring {
    // ⛔ EMULATOR-ONLY (cfg'd): the card keeps its pool through its own placement, so nothing on
    // the spyre-hw arm reads a per-layer pool id. The struct itself stays on both tiers (its len is
    // the layer count the load log prints).
    /// Tensor id of this layer's K pool.
    #[cfg(not(feature = "spyre-hw"))]
    pub(crate) prefix_k_src: usize,
    /// Tensor id of this layer's V pool.
    #[cfg(not(feature = "spyre-hw"))]
    pub(crate) prefix_v_src: usize,
}

/// Per-phase metadata for one program (decode m=1 or prefill m=M) inside the
/// shared [`SpyreSession`]. Decode and prefill share weight tensor ids (same
/// graph, different m), so the weights live ONCE in the session; this carries
/// only the per-phase wiring + the program index to address `run_step`.
pub(crate) struct BundleMeta {
    /// Program index in the shared session (see [`SpyreSession::run_step`]).
    #[cfg(not(feature = "spyre-hw"))]
    pub(crate) prog: usize,
    /// The paged step this bundle's attention reads — its page table, mask stack and write rows,
    /// staged every step from the request's KV blocks (`scratchy_target_spyre::paged_step`), and
    /// the pool they index.
    ///
    /// ⛔ EMULATOR-ONLY (cfg'd): the card's own staging rides its forward tape.
    #[cfg(not(feature = "spyre-hw"))]
    pub(crate) paged: Option<scratchy_target_spyre::wiring::PagedStep>,
    /// Prefix capacity (mask cols) — max context length this bundle serves.
    pub(crate) capacity: usize,
    /// Baked query-row count: 1 (decode) or M (batched prefill).
    pub(crate) m_cap: usize,
    // ⛔ THE FOUR BELOW ARE EMULATOR-ONLY (cfg'd): their one spyre-hw reader was the host
    // readback/staging the forward tape replaced — the tape reads sources, rotary tables
    // and the embed row straight off `wiring`, so no card launch consults these copies.
    /// Runtime source ids (shared id space across phases).
    #[cfg(not(feature = "spyre-hw"))]
    pub(crate) embed_src: usize,
    /// cos / sin sources as `(id, full width, rotary class)` — GQA gives >1
    /// width, and a hybrid-attention arch gives classes different tables.
    #[cfg(not(feature = "spyre-hw"))]
    pub(crate) cos_srcs: Vec<(usize, usize, scratchy_target_spyre::wiring::RotaryKind)>,
    #[cfg(not(feature = "spyre-hw"))]
    pub(crate) sin_srcs: Vec<(usize, usize, scratchy_target_spyre::wiring::RotaryKind)>,
    pub(crate) result_id: usize,
    /// One per layer, in layer order.
    pub(crate) layers: Vec<LayerWiring>,
    /// Tensors read back from each forward: the result. The KV never comes back — the device
    /// writes it into the pools and reads it there.
    #[cfg(not(feature = "spyre-hw"))]
    pub(crate) output_ids: Vec<usize>,
    /// Every compile-time scalar this bundle's KTIR reads, in registry order — entry `i` is bound at
    /// `scalarmul_scale_tid(i)` as a `[1,1]` tile.
    ///
    /// ⛔ THE EMULATOR MUST BIND THESE OR THE OUTPUT IS SILENTLY WRONG. The programs read
    /// a model constant (a scale, an RMSNorm epsilon or divisor) and the algebraic identities
    /// `0`/`1` off reserved tids rather than baking them as KTIR immediates, because
    /// `dxp_standalone` has no immediate operand and the card path needs a real address. Those tids
    /// are ordinary `func.arguments` of the program the emulator runs too, so an unbound one is a
    /// zero — which turns `x/(1+e)` into `x/e` and every scaled residual into nothing. The card
    /// binds the same list through `wiring::constant_steps`; this is the emulator's copy of it.
    #[cfg(not(feature = "spyre-hw"))]
    pub(crate) scalarmul_scales: &'static [f32],
    /// ⭐ THE WIRING'S OWN PHASE — decode's chained head vs prefill — projected
    /// from `Wiring::phase` at load (`phase: wiring.phase`, never a second
    /// derivation), so the emulator forward's chain-seed staging decides off
    /// the same fact the tape's `ForwardShape` carries. On `Decode` the
    /// per-step embed-row push is REPLACED by the sticks' seed; on `Prefill`
    /// the host pushes the row. Emulator-only: the card path reads the phase
    /// straight off its own `wiring` field below.
    #[cfg(not(feature = "spyre-hw"))]
    pub(crate) phase: scratchy_target_spyre::forward_tape::ForwardPhase,
    /// ⭐ THE MINTED SAMPLER'S SEAT — the wiring's own (`Wiring::sample_seat`), projected at load
    /// like `phase` above (never a second derivation). `Some` on every rung the bake minted the
    /// draw into: the emulator forward stages this seat's law every step and reads the pick back
    /// through the re-pointed result. Emulator-only: the card path reads it straight off its own
    /// `wiring` field.
    #[cfg(not(feature = "spyre-hw"))]
    pub(crate) sample_seat: Option<scratchy_target_spyre::wiring::SampleSeat>,
    /// What the bake placed — asked ONCE, in the target crate, off the generated layout.
    /// Read only by the card launch path, so it carries that cfg like `prog`/`output_ids` above.
    #[cfg(feature = "spyre-hw")]
    pub(crate) facts: scratchy_target_spyre::wiring::BakeFacts,
    /// ⭐ THE GENERATED scratchy_target_spyre::wiring::IRING THIS BUNDLE scratchy_target_spyre::wiring::AS BAKED scratchy_target_spyre::wiring::ITH. Everything above it is a copy taken
    /// out of this static on the way through `Parsed`; the forward tape is read straight off it
    /// (`Wiring::forward_shape`) so the shape a forward plays cannot be a second opinion about
    /// what the bake decided.
    #[cfg(feature = "spyre-hw")]
    pub(crate) wiring: &'static scratchy_target_spyre::wiring::Wiring,
}

impl BundleMeta {
    /// The positions this bundle's attention reaches: on the emulator its page walk's
    /// ([`scratchy_target_spyre::wiring::PagedStep::reach`]), so a step whose positions end at or
    /// below it can run here from any start; on the card, its prefix capacity.
    pub(crate) fn reach(&self) -> usize {
        #[cfg(not(feature = "spyre-hw"))]
        if let Some(p) = self.paged {
            return p.reach();
        }
        self.capacity
    }
}

/// The BAKED logits tensor of a bundle: `[rows, width]`, STICK-BLOCKED on the vocab axis.
///
/// Two things about it are not what a reader assumes, and both are silent when `rows == 1`:
///
///  * `width` is the PADDED placement width, not the model's vocab. The emitter pads the vocab axis
///    to a whole number of sticks per core — granite's 49155 is placed as 51200 — so a row spans
///    `width` elements of which only the leading `vocab` are real.
///  * A row is NOT contiguous. [`scratchy_subtile::sdsc_abstract::StickLayout::dev_off`] places
///    element `(r, c)` at `(c/64)·(rows·64) + r·64 + (c%64)`: the vocab axis is cut into 64-wide
///    blocks and all `rows` rows of a block sit together.
///
/// At `rows == 1` that formula collapses to `c`, which is why the one-request path can slice
/// `..vocab` straight off the buffer — and why the same slice at `r * vocab` for `r > 0` reads
/// mostly request 0's logits instead of request `r`'s. Fluent, wrong, and per-request.
#[cfg(feature = "spyre-hw")]
pub(crate) struct LogitsGeom {
    pub(crate) rows: usize,
    pub(crate) width: usize,
}

#[cfg(feature = "spyre-hw")]
impl LogitsGeom {
    /// From the result tensor's baked `bundle_layout.json` placement (BYTES) and the rung's row
    /// count — the BAKED width, because the tail runs unfolded at the rung's `m` whatever is live.
    /// `None` when the placement is not a whole number of fp16 rows, or is narrower than the
    /// vocab it must hold — either means this is not the tensor we think it is, and gathering from
    /// it would read neighbouring placements.
    pub(crate) fn from_placement(
        size_bytes: u64,
        rung: scratchy_subtile::sdsc_abstract::RungWidth,
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
#[cfg(feature = "spyre-hw")]
pub(crate) struct DecodeRung {
    /// The row count this rung's bundle was BAKED at — every launch on it binds exactly this many
    /// rows, live or padding, and every mask/fold/logits extent below derives from it.
    pub(crate) seqs: scratchy_subtile::sdsc_abstract::RungWidth,
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
#[cfg(feature = "spyre-hw")]
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
#[cfg(feature = "spyre-hw")]
pub(crate) struct DecodeSeat<'a> {
    pub(crate) seqs: scratchy_subtile::sdsc_abstract::RungWidth,
    pub(crate) sess: &'a mut SuperDscSession,
}

#[cfg(feature = "spyre-hw")]
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
#[cfg(feature = "spyre-hw")]
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
#[cfg(feature = "spyre-hw")]
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
#[cfg(feature = "spyre-hw")]
pub(crate) struct PrefixLadder(Vec<(usize, SuperDscSession)>);

/// ONE CHOICE OF [`PrefixLadder`] RUNG, made once per chunk and carried — the chunk's cap, its padded
/// write, its session and the baked row count its K/V extent is proven against all read THIS value, so
/// they cannot come from two choices. No public constructor: only [`PrefixLadder::rung_for`] mints one.
#[cfg(feature = "spyre-hw")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefixRung {
    index: usize,
    width: usize,
}

#[cfg(feature = "spyre-hw")]
impl PrefixRung {
    pub(crate) fn width(self) -> usize {
        self.width
    }
}

#[cfg(feature = "spyre-hw")]
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

    /// Every rung's baked width and session, ascending — for load-time staging, which visits each
    /// session once and chooses none.
    pub(crate) fn rungs_mut(&mut self) -> impl Iterator<Item = (usize, &mut SuperDscSession)> {
        self.0.iter_mut().map(|(m, s)| (*m, s))
    }
}

#[cfg(feature = "spyre-hw")]
impl SuperDscBundle {
    /// THE RUNG A DECODE STEP RUNS ON — [`RungId::Owner`] when one request is live (the W=1
    /// rung IS the owner session, by the loader's `rn <= 1 → continue`), else the index of the
    /// SMALLEST ladder rung that holds `live` requests. `None` when `live` exceeds the widest
    /// rung — the caller then refuses: there is no one-forward-per-request decode arm any more.
    ///
    /// The live count is the only input: live request `i` runs in slot `i`, whatever its pages.
    pub(crate) fn decode_rung_for(
        &self,
        live: scratchy_subtile::sdsc_abstract::LiveRows,
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
                seqs: scratchy_subtile::sdsc_abstract::RungWidth::of_baked_rows(1)
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

/// Model-level state shared by both bundles, borrowed alongside a `&mut Bundle`.
///
/// ⛔ NO `kv_dim` HERE, AND ITS ABSENCE IS THE LOCK. A hybrid-attention arch (gemma-4)
/// gives layers DIFFERENT kv widths (sliding `kv_heads*head_dim` vs global
/// `g_kv*g_hd`), so a model-wide kv width is a number no per-layer question may
/// read — and the KTIR path asks none: each layer's K and V pools are resident
/// tensors at their own baked shapes ([`LayerWiring`]), written and read on the
/// device by the layer's paged step.
pub(crate) struct Shared<'a> {
    pub(crate) embed_tokens: &'a [f32],
    pub(crate) hidden: usize,
    pub(crate) vocab: usize,
    // ⛔ NO `rope_theta` HERE — same law as `kv_dim` below on `Loaded`. A hybrid-
    // attention arch has no single θ: each cos/sin source carries its own
    // `RotaryKind` (θ as bits) in the wiring, and the rotary tables are built
    // from that. A model-wide θ here built the WRONG class's table silently.
}

/// Everything [`Worker::load_model`] resolves: model-level shared state + the
/// per-phase bundles. Generation (one token/step) runs through the cheap m=1
/// `decode` bundle; the prompt runs through the batched m=M `prefill` bundle.
pub(crate) struct Loaded {
    /// On-disk model dir (weights + `config.json`); surfaced to `create_worker`.
    pub(crate) model_dir: PathBuf,
    /// Parsed HF `config.json` — the engine needs it for scheduling/cache sizing.
    pub(crate) hf_config: HfModelConfig,
    /// `embed_tokens.weight` `[vocab, hidden]` kept for the per-token gather.
    pub(crate) embed_tokens: Vec<f32>,
    pub(crate) hidden: usize,
    pub(crate) vocab: usize,
    // ⛔ NO `rope_theta` FIELD. The KTIR path builds every rotary table from the
    // per-source `RotaryKind` the wiring carries; the info! at load time prints
    // θ from the wiring's own geometry, where the single answer still lives.
    // A field here invites a reader the classes can disagree with.
    /// KTIR path: ONE resident session holding both programs (weights once).
    #[cfg(not(feature = "spyre-hw"))]
    pub(crate) session: SpyreSession,
    /// sendnn path: the SuperDSC dxp-bundle session or the offline_decoder
    /// layer-group split — selected from the baked bundle shape.
    #[cfg(feature = "spyre-hw")]
    pub(crate) session: SendnnSession,
    /// m=1 decode program metas, one per rung, ASCENDING by reach ([`BundleMeta::reach`]). The
    /// generation loop runs the smallest rung that reaches the current position — on the emulator the
    /// page walk's rungs, so a step walks the fewest pages its context fills; never empty. The
    /// sendnn path bakes a single bucket, so its ladder is the full-cap entry.
    pub(crate) decode: Vec<BundleMeta>,
    /// m=M prompt-chunk program metas, ASCENDING by reach — on the emulator the page walk's rungs,
    /// each chunk run on the smallest that reaches its context. Empty when none is baked (prefill then
    /// walks the decode program, one token per forward).
    pub(crate) prefill: Vec<BundleMeta>,
    /// The prefill program meta PRESERVED for the batched-prefill path. `model.prefill` is nulled for
    /// SuperDsc so the PER-FORWARD binding uses the decode graph (the prefill manifest has different
    /// tensor ids); but `run_card_launch` binds this meta to the DISTINCT prefill SESSION (where the ids
    /// match), so it is kept here. Some only for SuperDsc with a baked prefill bundle; None otherwise.
    pub(crate) batched_prefill: Option<BundleMeta>,
    /// THE KV POOL'S BYTES — on the card, the KV segment the bake placed (every layer's pools and the
    /// pool's statics): the number the ENGINE sizes the scheduler's block allocator from
    /// ([`Worker::determine_available_memory`]), beside [`Self::pool_stride_bytes`].
    ///
    /// `None` on the emulator, which sizes no device pool.
    pub(crate) kv_budget_bytes: Option<u64>,
    /// ONE PAGE OF THE POOL, in bytes — the KV segment over its pages, so the engine's
    /// `budget / kv_block_bytes` arithmetic lands on the pool's own page count, which
    /// `kv_cache_num_blocks_override` then holds it to.
    ///
    /// `None` on the emulator.
    pub(crate) pool_stride_bytes: Option<u64>,
    /// ⭐ THE CONTEXT THIS RUN SERVES, in tokens — the `--max-model-len` declared (else the model's
    /// own limit), which the pool the bake placed must hold: a declaration past the pool is refused at
    /// load, never clamped. `usize::MAX` on the emulator (no cap).
    pub(crate) served_context: usize,
}

/// Per-request generation state the worker keeps beside `InputBatch`.
pub(crate) struct ReqState {
    // ⛔ THE TOKENS, THE PROMPT LENGTH AND THE TOKEN COUNT ARE NOT HERE: they live in `InputBatch`,
    // the engine's backend-neutral per-request store, beside the block table. A request's keys sit at
    // its own positions, in the pages its block table names, so nothing about its KV is this
    // struct's either.
    /// ⭐ THE REQUEST'S DRAW — the seed and inverse temperature every launch this request's rows
    /// run under, decided ONCE here at admission (nothing re-derives it mid-generation: the seat's
    /// per-step staging is a pure function of this value plus the row's absolute position, the
    /// step salt). `None` is the greedy law — temperature ≤ 0, or a request that arrived with no
    /// sampling params at all — and stages the exact bytes the pre-sampler path did
    /// (quadruple `(0, 0, 0, 0)`, `inv_t = 1.0`).
    ///
    /// ⛔ WHAT IT DOES NOT CARRY IS REFUSED AT THE DERIVATION ([`SampledDraw::of`]), not silently
    /// dropped here: this path's draw is temperature alone, so a request asking for `top_p`,
    /// `top_k`, `min_p`, penalties or `n > 1` is turned away at admission rather than served a
    /// draw that ignores half of what it asked for.
    ///
    /// BOTH TIERS: every reader stages the minted sampler's seat laws, and the bake mints the
    /// draw into the emulator's bundles exactly as into the card's — the emulator's forward
    /// stages the same law per step (packed, at its views' own dtypes) and reads the same idx
    /// stick back.
    pub(crate) draw: Option<SampledDraw>,
}

/// ⭐ ONE REQUEST'S DRAW, in the two numbers the card's noise kernel wants: a
/// 64-bit seed (hashed on-card with the row's step salt and lane) and the
/// inverse temperature the folded lm-head row is scaled by before the noise is
/// added. Carried on [`ReqState::draw`] — one value per request, staged per
/// step by the seat's binders.
///
/// Both tiers stage it: the bake mints the draw into every rung it bakes, the
/// emulator's included (see [`ReqState::draw`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SampledDraw {
    /// The seed of the row's quadruple `(seed_lo, seed_hi, step, 1)` — the
    /// request's own when it arrived seeded, else the FNV-1a of its id (the
    /// `fnv_seed` law: deterministic within a run, no process RNG, so an
    /// unseeded draw is still a reproducible function of the request). The
    /// key is request-owned alone (the P3 localization ruling): no row word,
    /// so a request draws the same noise in any baked row.
    pub(crate) seed: u64,
    /// `1 / T` as f32 — the sampler's own scaling (`score = x·inv_t + g·gate`),
    /// so temperature 1.0 stages the same `1.0` the greedy law does and only
    /// the gate and seed words differ.
    pub(crate) inv_t: f32,
}

impl SampledDraw {
    /// ⭐ THE DRAW A REQUEST'S SAMPLING PARAMS ASK FOR — the ONE derivation, run
    /// at admission. `Ok(None)` is the greedy law (no params, or temperature
    /// ≤ 0 — the exact bytes the pre-sampler path staged). `Ok(Some)` is a draw:
    /// the seeded-or-derived 64-bit seed and `1/T`. `Err` is a request this
    /// path cannot serve honestly.
    ///
    /// ⛔ THE ERRORS ARE THE POINT. This path's on-card draw is TEMPERATURE
    /// ALONE: `top_p`, `top_k`, `min_p`, the three penalties and `n > 1` have
    /// no card leg here. Serving such a request anyway would answer a nucleus
    /// ask with an unfiltered softmax sample — silently wrong in the one
    /// direction nobody can see from the output — so the request is refused at
    /// admission, naming every knob it asked for. The day a knob gains a card
    /// leg, its arm moves from this refusal to the draw.
    pub(crate) fn of(
        req_id: &str,
        sp: Option<&scratchy_core_common::SamplingParams>,
    ) -> Result<Option<SampledDraw>, String> {
        let Some(sp) = sp else {
            // No params at all: nothing asked for a draw, and the greedy law is
            // what the seat stages.
            return Ok(None);
        };
        if sp.temperature <= 0.0 {
            // Temperature 0 (and anything below it) is greedy by definition —
            // the gate word's 0, `inv_t` irrelevant.
            return Ok(None);
        }
        let refused: Vec<String> = [
            (sp.top_p != 1.0, format!("top_p={}", sp.top_p)),
            (sp.top_k > 0, format!("top_k={}", sp.top_k)),
            (sp.min_p > 0.0, format!("min_p={}", sp.min_p)),
            (
                sp.presence_penalty != 0.0,
                format!("presence_penalty={}", sp.presence_penalty),
            ),
            (
                sp.frequency_penalty != 0.0,
                format!("frequency_penalty={}", sp.frequency_penalty),
            ),
            (
                sp.repetition_penalty != 1.0,
                format!("repetition_penalty={}", sp.repetition_penalty),
            ),
            (sp.n > 1, format!("n={}", sp.n)),
        ]
        .into_iter()
        .filter(|(on, _)| *on)
        .map(|(_, what)| what)
        .collect();
        if !refused.is_empty() {
            return Err(format!(
                "this Spyre path samples with temperature alone — request {req_id} also asked for \
                 {}, which has no card leg here; refuse rather than serve a draw that ignores it",
                refused.join(", ")
            ));
        }
        Ok(Some(SampledDraw {
            seed: sp
                .seed
                .unwrap_or_else(|| scratchy_core_common::fnv_seed64(req_id)),
            inv_t: (1.0 / sp.temperature) as f32,
        }))
    }
}
