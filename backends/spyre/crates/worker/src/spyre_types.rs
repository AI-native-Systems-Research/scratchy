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
    /// ⭐ WHERE THIS WORKER'S SHARED HOST THREADS RUN — the load thread (pinned when the cards
    /// join) and the decode-loop thread (pinned as its first act, `Worker::pin_driving_thread`):
    /// the node every card sits on, or rank 0's when they span nodes.
    #[cfg(feature = "spyre-hw")]
    pub(crate) world_cpus: scratchy_target_spyre_card::runtime::WorldCpus,
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

#[cfg(feature = "spyre-hw")]
pub(crate) use crate::card::types::*;
