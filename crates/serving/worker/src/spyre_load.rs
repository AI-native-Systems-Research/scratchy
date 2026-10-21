// SPDX-License-Identifier: Apache-2.0
//! LOADING — resolve the generated wiring, stage the generated weights, build the sessions.
//!
//! ⛔ NOTHING HERE ANALYSES A MODEL. The wiring is a `static` the `#[forward]` macro emitted and
//! `wiring_to_parsed` reads it; the weights arrive already resolved as `BoundWeight`s from the
//! generated loader. What is left is device-side: segment sizing, the KV pool cut, and the
//! sessions the launches go through.

use scratchy_core_model::weight::HfModelConfig;
use scratchy_layers::weights::GpuWeights;
// The pool/slot vocabulary is CARD-ONLY, so its imports carry the cfg of the blocks that use it.
#[cfg(feature = "spyre-hw")]
use scratchy_subtile::sdsc_abstract::{PagedKvPool, PoolPages, PoolPartition, PoolRows};
use scratchy_target_spyre::SpyreAllocator;
use scratchy_target_spyre::manifest::bytes_to_f32;
// `--target sendnn` swaps the KTIR emulator runner for the on-silicon sendnn
// runner; the bundle type + session type are cfg-selected, everything else
// (weight load, dynamic sources, KV loop, sampling) is shared.
#[cfg(not(feature = "spyre-hw"))]
use scratchy_target_spyre::manifest::KtirBundle;
#[cfg(feature = "spyre-hw")]
use scratchy_target_spyre::manifest::SengraphBundle;
#[cfg(not(feature = "spyre-hw"))]
use scratchy_target_spyre::runner::SpyreSession;
#[cfg(feature = "spyre-hw")]
use scratchy_target_spyre::sdsc_runner::SuperDscSession;
use scratchy_tensors::DType as SDType;
use tracing::info;
// `debug`/`warn` are only reached from the card blocks below.
#[cfg(feature = "spyre-hw")]
use tracing::{debug, warn};

/// A weight staged for binding: its tensor id, bytes, dtype and shape.
pub(crate) type StagedWeight = (usize, Vec<u8>, SDType, Vec<usize>);

/// ⭐ THE RESIDENT SEGMENTS EVERY BORROWER ADOPTS, AS ONE LIST. Four sites alias these (the
/// prefix-capable prefill, the prefill ladder rungs, the decode batch rungs, and the top prefill
/// session), and a segment missing from ONE of them leaves that session pointing at a 128 B
/// placeholder where the model's weights should be — which is not a crash, it is garbage output from
/// one rung only. Mirrors `superdsc_exec::SEG_RESIDENT`, which is what the executor's own
/// placeholder/H2D skips read; seg5 is empty for every model whose weights fit one segment, and
/// aliasing an empty segment is a no-op the size/placement checks accept.
///
/// Card-only, like [`adopt_from_owner`], its only reader.
#[cfg(feature = "spyre-hw")]
const RESIDENT_SEGS: [(i64, &str); 2] = [(1, "resident WEIGHTS"), (2, "resident KV")];

/// ⭐ EVERYTHING A BORROWING SESSION NEEDS FROM THE OWNER, AS ONE CALL: the resident segments it
/// ADOPTS by alias, and the spilled weight tail it must be handed its own COPY of.
///
/// ⛔ THE TWO ARE NOT INTERCHANGEABLE, which is the whole reason this exists. A resident segment
/// (weights, KV) is placed identically in every rung's bundle, so the borrower can share the owner's
/// region outright and pay nothing. The weight SPILL slot cannot be shared: it is also an
/// intermediate COLOR segment (`WEIGHT_SPILL_SEGS`), so its extent depends on the rung's query width
/// and `alias_seg_from`'s placement-equality check would — correctly — refuse it. The borrower needs
/// its own region holding its own copy, and `write_seg` from offset 0 is what puts it there:
/// `spill_weight_tail` places the tail at offset 0 of the slot precisely so ONE device image is
/// portable to a bundle whose extent for that segment differs.
///
/// ⛔ AND IT IS NOT OPTIONAL FOR ANY SESSION THAT PRODUCES LOGITS. The spilled tail IS the lm_head /
/// tied embedding, and every prefill rung and decode rung runs the suffix. A session that skips this
/// reads a zero-filled weight where the embedding should be: zero logits, so argmax returns token 0.
/// Not a crash — one wrong token from one rung, exactly the failure shape [`RESIDENT_SEGS`] warns of.
/// The IMAGE is taken from the owner AFTER its prepare, so it is the converted, re-tiled device bytes
/// and not the host weight layout, which is why this copies a segment rather than re-staging a
/// tensor.
///
/// Carries the cfg of the `SuperDscSession` it takes: that type — and every session-borrowing site
/// below — is CARD-ONLY, so the emulator build has no owner for anything to adopt from.
#[cfg(feature = "spyre-hw")]
fn adopt_from_owner(
    borrower: &mut SuperDscSession,
    owner: &SuperDscSession,
    spill: &[(i64, Vec<u8>)],
) -> Result<(), String> {
    for (seg, what) in RESIDENT_SEGS {
        borrower
            .alias_seg_from(owner, seg)
            .map_err(|e| format!("seg{seg} ({what}) alias failed: {e}"))?;
    }
    for (seg, image) in spill {
        borrower.write_seg_prefix(*seg, image).map_err(|e| {
            format!(
                "seg{seg} (spilled weight tail, {} B) copy failed: {e}",
                image.len()
            )
        })?;
    }
    Ok(())
}

/// ⭐ STAGE A SESSION'S FULL-CHUNK CAUSAL MASK AT LOAD, so its first full launch finds the key held
/// and builds, converts and uploads nothing for it.
///
/// The mask is a function of its key alone, and a `rows`-row session's full key — `Chunk { rows, rows }`
/// — is a compile-time fact of the rung: `rows` is the ladder entry the launch selects this session by
/// and runs it at, and the bytes are [`causal_full_chunk`]'s, i.e. [`causal_bind`] under the wiring's
/// own head law, the same function the launch's `CausalMask` step calls. MEASURED before this, on
/// granite-3.1-2b fp16 at ~1k tokens: the first request paid each 512-row session's mask build as
/// +49.7 ms (prefix-free) and +40.0 ms (prefix) over the warm request.
///
/// A launch under any OTHER key (a prompt's partial tail) re-binds over the staged one, as it would
/// over a key a previous launch left; the next full chunk then re-binds the full key once.
///
/// [`causal_full_chunk`]: scratchy_target_spyre::forward_tape::causal_full_chunk
/// [`causal_bind`]: scratchy_target_spyre::forward_tape::causal_bind
#[cfg(feature = "spyre-hw")]
fn stage_full_chunk_causal(
    sess: &mut SuperDscSession,
    wiring: &scratchy_target_spyre::wiring::Wiring,
    rows: usize,
) -> Result<u64, String> {
    let width = u32::try_from(rows)
        .ok()
        .and_then(scratchy_subtile::sdsc_abstract::RungWidth::of_baked_rows)
        .ok_or_else(|| format!("a {rows}-row session has no rung width"))?;
    let (id, key, vals) = scratchy_target_spyre::forward_tape::causal_full_chunk(wiring, width);
    sess.stage_keyed(id, key, &vals)
        .map_err(|e| format!("the {rows}-row session's full-chunk causal mask: {e}"))
}

/// The segments this bundle spilled weights into and HOW MANY BYTES of each the spill occupies —
/// [`adopt_from_owner`]'s work list, empty for every model whose weights fit one segment.
///
/// Read off the bake's own verdict, `layout.weight_tail`, which the emitter proved equal to its weight
/// placements before baking it (`bake_layout`). The match has no wildcard: a new kind of tail is a
/// compile error here, not a session that silently skips its copy.
///
/// ⛔ THE LENGTH IS NOT A DETAIL — it is what stops this from corrupting the borrower's scratch. The
/// spill slot is ALSO an intermediate color segment, and a prefill rung's intermediates are WIDER
/// than the decode owner's (they scale with the query rows), so the borrower's segment is larger than
/// the owner's. Copying the owner's WHOLE segment image would therefore land the owner's intermediate
/// bytes on top of the borrower's zero-initialised scratch — turning a zeroed slot into arbitrary
/// non-zero data for any tensor read before it is written. Bounding the copy to the tail's own
/// `[0, len)` leaves every borrower's scratch exactly as `prepare` zeroed it.
///
/// Card-only: the emulator path builds no borrowing session, so nothing reads this work list.
#[cfg(feature = "spyre-hw")]
fn spilled_weight_spans(
    layout: &scratchy_target_spyre::bundle_code::BundleLayout<'static>,
) -> Vec<(i64, usize)> {
    use scratchy_target_spyre::bundle_code::WeightTail;
    match layout.weight_tail {
        WeightTail::InPlace => Vec::new(),
        WeightTail::Spilled { segment, bytes } => vec![(segment as i64, bytes as usize)],
    }
}

/// ⭐ WHAT A KV BUDGET FUNDS PER REQUEST, IN TOKENS — the pool veto's own advice arithmetic,
/// hoisted to ONE spelling because the default-context law reads the same number
/// ([`scratchy_serving_engine::gpu_budget::resolve_served_context`]).
///
/// The reserve (one shared scratch page — [`PoolPartition::reserve`]) comes off the top,
/// the rest splits evenly across the launch rows, and a row spends its pages whole
/// (`PagedKvPool::PAGE_SLOTS` slots each). ⛔ IT IS THE EXACT INVERSE OF THE POOL DEMAND: a
/// declaration of exactly this many tokens re-passes the budget veto with zero slack, and one
/// token more does not — which is what makes an unset flag's resolution veto-proof by
/// construction instead of by re-checking. Pinned as such by `served_context_tests` below.
///
/// Gated to the builds with a caller: the pool veto (`spyre-hw`) and the wiring test
/// (`test`, so it runs on the emulator build, where the card-only blocks are cfg'd out —
/// the same cfg artifact the `decode_facts` pair above documents). Pure arithmetic over
/// the shared pool vocabulary either way.
#[cfg(any(feature = "spyre-hw", test))]
pub(crate) fn affordable_context_tokens(
    budget_bytes: u64,
    stride_bytes: u64,
    rows: scratchy_subtile::sdsc_abstract::PoolRows,
) -> u64 {
    use scratchy_subtile::sdsc_abstract::{PagedKvPool, PoolPartition};
    (budget_bytes / stride_bytes).saturating_sub(PoolPartition::reserve() as u64)
        / rows.get().get() as u64
        * PagedKvPool::PAGE_SLOTS as u64
}

use crate::error::ExecutorResult;
#[cfg(feature = "spyre-hw")]
use crate::spyre_pool::*;
use crate::spyre_types::*;
use crate::spyre_worker::*;
use crate::worker_factory::resolve_model_path;

/// Parsed per-program metadata + the owned manifest (kept alive for the shared
/// session) + model-level dims. No weights, no session — the caller builds those
/// ONCE across both programs (decode + prefill share weight tensor ids).
pub(crate) struct Parsed {
    pub(crate) embed_src: usize,
    /// The attention length-mask source, when the wiring names one.
    pub(crate) attn_mask_src: Option<usize>,
    /// cos / sin runtime sources as `(source id, full column width, rotary class)`.
    /// GQA gives more than one width (Q vs K); a hybrid-attention arch (gemma-4)
    /// gives classes different tables. Each is filled by tiling the class's own
    /// per-position rotary row across heads.
    pub(crate) cos_srcs: Vec<(usize, usize, scratchy_target_spyre::wiring::RotaryKind)>,
    pub(crate) sin_srcs: Vec<(usize, usize, scratchy_target_spyre::wiring::RotaryKind)>,
    pub(crate) result_id: usize,
    /// See [`crate::spyre_types::BundleMeta::result_is_index`] — the bake's
    /// own statement of whether this rung's result is the idx stick.
    pub(crate) result_is_index: bool,
    /// See [`crate::spyre_types::BundleMeta::result_want`] — the result's
    /// readback extent in elements.
    pub(crate) result_want: usize,
    pub(crate) capacity: usize,
    pub(crate) m_cap: usize,
    pub(crate) hidden: usize,
    pub(crate) kv_dim: usize,
    pub(crate) vocab: usize,
    pub(crate) layers: Vec<LayerWiring>,
    #[cfg(not(feature = "spyre-hw"))]
    pub(crate) output_ids: Vec<usize>,
    /// See [`crate::spyre_types::BundleMeta::scalarmul_scales`].
    #[cfg(not(feature = "spyre-hw"))]
    pub(crate) scalarmul_scales: &'static [f32],
}

/// Parse one embedded [`KtirBundleData`]: manifest + runtime source ids +
/// per-layer AttnDecode wiring + dims. No disk read, no weights, no session.
/// Build the per-program wiring from the GENERATED static — no parsing.
///
/// ⛔ THIS IS `parse_bundle` WITH THE ANALYSIS REMOVED, NOT REWRITTEN. Every
/// value below was already a typed constant in the macro; it was flattened to
/// JSON, baked as a string, and rebuilt here by `serde_json` plus
/// `match role.as_str()` plus a positional scan of the node list. All three
/// steps are gone: roles are an ENUM (so a new one is a compile error, not an
/// unmatched `_`), and per-layer AttnDecode ids are read rather than inferred
/// from "the argument after the prefix-K argument".
pub(crate) fn wiring_to_parsed(
    w: &'static scratchy_target_spyre::wiring::Wiring,
) -> ExecutorResult<Parsed> {
    // (The kernel-table check moved to where the LAYOUT is in hand: `BakeFacts::of` is followed by
    // `Wiring::verify_class_placements(&code.layout)` once per baked bundle, which cross-checks the
    // class sets against the placements — the fact that actually failed on gemma-4, where a fresh
    // re-staging at the same base head dim agreed with a wrong table.)
    let capacity = w
        .capacity()
        .ok_or_else(|| werr("wiring has no attn_mask (re-bake with the length mask)"))?;
    let layers: Vec<LayerWiring> = w
        .layers
        .iter()
        .map(|l| LayerWiring {
            prefix_k_src: l.prefix_k as usize,
            // Emulator-only, like the field itself: the card keeps the K/V cache RESIDENT and writes
            // this step's V device-side, so no host binding names prefix-V there.
            #[cfg(not(feature = "spyre-hw"))]
            prefix_v_src: l.prefix_v as usize,
            new_k_id: l.new_k as usize,
            new_v_id: l.new_v as usize,
            kv_width: l.kv_width as usize,
        })
        .collect();
    if layers.is_empty() {
        return Err(werr("wiring carries no per-layer AttnDecode wiring"));
    }
    let triples = |v: &'static [(u32, u32, scratchy_target_spyre::wiring::RotaryKind)]| -> Vec<
        (usize, usize, scratchy_target_spyre::wiring::RotaryKind),
    > {
        v.iter()
            .map(|&(i, w, k)| (i as usize, w as usize, k))
            .collect()
    };
    let embed_src = w.embed_src as usize;
    // The result's own answer shape, off the wiring's one derivation — the
    // stick's element count on a minted rung, the vocab row otherwise. Both
    // flow into the BundleMeta as plain copies (the emulator path carries no
    // `wiring` reference to re-ask).
    let result_is_index = w.result_is_index();
    let result_want = if result_is_index {
        let (rows, cols) = w.tensor_shapes[w.result as usize];
        (rows * cols) as usize
    } else {
        w.geometry.vocab as usize
    };
    Ok(Parsed {
        embed_src,
        cos_srcs: triples(w.cos_srcs),
        sin_srcs: triples(w.sin_srcs),
        result_id: w.result as usize,
        result_is_index,
        result_want,
        capacity,
        m_cap: (w.tensor_shapes[embed_src].0 as usize).max(1),
        hidden: w.geometry.hidden as usize,
        kv_dim: w.geometry.kv_dim as usize,
        vocab: w.geometry.vocab as usize,
        layers,
        #[cfg(not(feature = "spyre-hw"))]
        scalarmul_scales: w.scalarmul_scales,
        // The tensors the host reads back after a pass: the result, plus each layer's new K/V,
        // which is how the KV is threaded when the device holds no cache of its own. Carries the
        // field's own `cfg` for exactly that reason — the card DOES hold a cache of its own, so there
        // is no per-layer readback to name there.
        #[cfg(not(feature = "spyre-hw"))]
        output_ids: std::iter::once(w.result as usize)
            .chain(
                w.layers
                    .iter()
                    .flat_map(|l| [l.new_k as usize, l.new_v as usize]),
            )
            .collect(),
        attn_mask_src: w.attn_mask.map(|m| m as usize),
    })
}

/// The host bytes behind a tensor. Spyre's allocator is a HOST allocator, so a
/// `GpuTensor`'s pointer is host memory and this is a plain view — no copy, no
/// device transfer.
///
/// ⛔ NOT `cfg(not(spyre-hw))`, AND IT NEVER COULD BE. This carried an emulator-only gate while its
/// sole caller, [`stage_bound_weights`], is ungated and needs the bytes on BOTH tiers — the card
/// branch immediately below the call re-binds `bytes` to physically transpose `[n,k]` → `[k,n]` before
/// staging. So the gate made the card build reference a function that did not exist there
/// (`cannot find function gpu_tensor_bytes`, twice), which is why no `spyre-hw` binary has ever linked
/// on this branch. Host bytes are correct at both tiers: the allocator is a host allocator either way,
/// and the card path uploads FROM this buffer rather than reading device memory through it.
pub(crate) fn gpu_tensor_bytes(t: &scratchy_tensors::GpuTensor) -> &[u8] {
    // SAFETY: `t` was produced by the generated binding from a `Weights` field,
    // whose buffer the `GpuWeights` allocator owns and keeps alive for the
    // caller's scope; `size_bytes()` is that buffer's own extent.
    unsafe { std::slice::from_raw_parts(t.as_ptr::<u8>(), t.size_bytes()) }
}

/// Stage every bound weight as TYPED BYTES (no f32 round-trip): f16 verbatim,
/// bf16 narrowed on ingest. Bound VERBATIM (zero-copy, no transpose) — GEMM
/// weights stay on-disk `[out, in] = [n, k]` and the emitter reads them with a
/// transpose-B `indexing_maps`.
///
/// ⭐ THE INPUT IS THE GENERATED BINDING, NOT A MANIFEST. Every entry already
/// names a `Weights` field, so there is no on-disk key to rebuild here. That
/// deleted three things this function used to carry:
///   - the `"{disk}.weight"` / `"{disk}.weight_scale"` naming rule, which could
///     not express a bare `nn.Parameter` (gemma-4's `layer_scalar`) — the
///     reason gemma-4 could not load;
///   - the `raw_cache` + `share_count` machinery, which existed only so a TIED
///     `lm_head` could re-read the embedding's disk tensor. Tying is resolved
///     in the generated loader now, and both fields view ONE buffer, so there
///     is nothing to re-read and nothing to duplicate;
///   - the tie-redirect special case and its "the manifest `disk` is
///     AUTHORITATIVE" comment.
///
/// What remains is staging: orientation, device-width padding, K-splitting and
/// the bf16→f16 narrow — h2d work, which is spyre's to do.
#[allow(clippy::type_complexity)]
/// The prefix-KV source ids of a wiring — placed, inside `0..num_sources`, and DEVICE-RESIDENT.
///
/// ⛔ THEY ARE NOT BOUND PER STEP. The cache lives in seg2 and is filled by the pool and by the
/// device's own cachewr, so `Executor::require_sources_filled` must know to skip them or it
/// refuses every launch. Derived from the wiring rather than listed by hand, so a model with a
/// different layer count cannot desync it.
#[cfg(feature = "spyre-hw")]
fn resident_source_ids(
    w: &scratchy_target_spyre::wiring::Wiring,
    // ⛔ THE SPILLED WEIGHT TAIL BELONGS HERE, and leaving it out is a REFUSED LAUNCH, not a silent
    // wrong answer — `require_sources_filled` classifies it `SourceFiller::Nobody` and stops.
    //
    // 🛑 MEASURED: `superdsc prefill run_step (mq=21, start=0): predict: 1 caller-filled source(s)
    // were never written this step — t446`. A borrowing session binds NOTHING (that is the point:
    // one stage+H2D for the whole ladder), so a weight normally classifies as
    // `SourceFiller::SegmentOwner` — its segment is ALIASED and the owner filled it. A spilled tail
    // cannot be aliased (its slot also holds an intermediate color, whose extent is m-dependent, so
    // no two bundles agree on the segment's size), so it is handed over as a COPY into this session's
    // own region instead — `adopt_from_owner`. Nothing binds it and nothing aliases it, so the only
    // accurate answer left is that the LOADER declared it resident, which is what this list says.
    spilled_weight_tail: &[u32],
) -> Vec<u32> {
    w.layers
        .iter()
        .flat_map(|l| [l.prefix_k, l.prefix_v])
        .chain(spilled_weight_tail.iter().copied())
        // ⭐ THE LOAD-CONST TIDS — the staged tables are bound ONCE at load on the
        // owner session (`bind_load_consts`, before prepare, resident like a
        // weight). A BORROWING rung never binds them itself, so a table whose
        // placement sits outside the aliased segments classifies `Nobody` and
        // every launch on that rung is refused; the loader's declaration is the
        // accurate answer for them exactly as it is for the spilled tail. Keyed
        // on the wiring's own list (op-blind: a second kernel with staged
        // tables rides the same union), and it can only excuse ids at or past
        // the graph's own source count — a real source can never collide with
        // a minted table id.
        .chain(w.load_consts.iter().map(|&(tid, _)| tid))
        .collect()
}

/// The tensor ids of the spilled weight tail — the companion of [`spilled_weight_spans`], which
/// carries the same decision as byte ranges. Both key on the bake's verdict, `layout.weight_tail`:
/// a bound weight is in the tail exactly when it is placed in the slot that verdict names.
/// Card-only, for the same reason [`spilled_weight_spans`] is.
#[cfg(feature = "spyre-hw")]
fn spilled_weight_tail_tids(
    layout: &scratchy_target_spyre::bundle_code::BundleLayout<'static>,
    weight_ids: impl IntoIterator<Item = usize>,
) -> Vec<u32> {
    use scratchy_target_spyre::bundle_code::WeightTail;
    let slot = match layout.weight_tail {
        WeightTail::InPlace => return Vec::new(),
        WeightTail::Spilled { segment, .. } => segment,
    };
    weight_ids
        .into_iter()
        .filter(|id| {
            layout
                .place_of_tid(*id as u32)
                .is_some_and(|p| p.segment == slot)
        })
        .map(|id| id as u32)
        .collect()
}

/// ⭐ THE RUNTIME HALF OF THE SCALARMUL WEIGHT FOLD — scale the fold targets' staged bytes.
///
/// `scalar_mul_weight_folds` recognized, at BUILD time, every multiplier that commutes into a
/// matmul's weight (`(x·W)·s == x·(W·s)`); the wiring carries the `(tensor id, multiplier)` list
/// and this is the whole of the fold at runtime: one elementwise multiply, at load, over the staged
/// bytes this model is about to bind — the card's post-narrowing f16, the emulator's source dtype.
/// The folded `ScalarMul` never becomes a program, so nothing in the run touches the scale again.
///
/// ⛔ ONE ROUNDING, IN f32: each element is widened, scaled, and narrowed once — the same discipline
/// the bf16→f16 narrowing pass follows, so the staged bytes carry `fp16(x)·s` and not a compound of
/// two roundings. The fold is NOT byte-identical to the un-folded tape (which rounded `x·W` before
/// scaling); that is the algebra trading rounding ORDER, and it is why the fold's acceptance gate is
/// token-level agreement, not byte equality.
///
/// ⛔ REFUSES WHAT THE RECOGNITION DID NOT PREVENT: a fold naming a tensor the binding did not
/// stage, or a payload a multiply cannot scale in place (fp8 — that is a requantize; integers).
/// Either is a wiring↔recognition desync, not an input to interpret.
fn apply_weight_scale_folds(
    weights: &mut [StagedWeight],
    folds: &[(u32, f32)],
) -> ExecutorResult<()> {
    use rayon::prelude::*;
    for (tid, scale) in folds {
        let (id, bytes, dt, _) = weights
            .iter_mut()
            .find(|(id, _, _, _)| *id == *tid as usize)
            .ok_or_else(|| {
                werr(format!(
                    "ScalarMul weight fold names tensor t{tid}, which the generated binding did \
                     not stage — the wiring and the weight binding disagree about the sources"
                ))
            })?;
        // SAFETY-free elementwise scale: every branch below reinterprets the Vec's own bytes as
        // the dtype it already declares, so the length check is the whole of the guard.
        match *dt {
            SDType::F16 => {
                if bytes.len() % 2 != 0 {
                    return Err(werr(format!(
                        "fold target t{id} holds {} bytes — not a whole number of f16 elements",
                        bytes.len()
                    )));
                }
                let n = bytes.len() / 2;
                let elems = unsafe {
                    std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut half::f16, n)
                };
                elems
                    .par_iter_mut()
                    .for_each(|e| *e = half::f16::from_f32(e.to_f32() * scale));
            }
            SDType::BF16 => {
                if bytes.len() % 2 != 0 {
                    return Err(werr(format!(
                        "fold target t{id} holds {} bytes — not a whole number of bf16 elements",
                        bytes.len()
                    )));
                }
                let n = bytes.len() / 2;
                let elems = unsafe {
                    std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut half::bf16, n)
                };
                elems
                    .par_iter_mut()
                    .for_each(|e| *e = half::bf16::from_f32(e.to_f32() * scale));
            }
            SDType::F32 => {
                if bytes.len() % 4 != 0 {
                    return Err(werr(format!(
                        "fold target t{id} holds {} bytes — not a whole number of f32 elements",
                        bytes.len()
                    )));
                }
                let n = bytes.len() / 4;
                let elems =
                    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, n) };
                elems.par_iter_mut().for_each(|e| *e *= scale);
            }
            other => {
                return Err(werr(format!(
                    "ScalarMul weight fold names t{id}, staged as {other:?} — a fold target must \
                     be a float payload the multiply can scale in place. The recognition refuses \
                     fp8 weights at build time, so reaching here means the wiring's fold list and \
                     the lowering's recognition have desynced"
                )));
            }
        }
    }
    Ok(())
}

/// ⭐ THE RUNTIME HALF OF THE HEAD-MAJOR o_proj RESTRUCTURE — materialize the permuted copies.
///
/// The layout minted, at BUILD time, a second, k-shuffled o_proj copy for every layer
/// (`oproj_headmajor_tid`) wherever the geometry gate applies (multi-slab head dim + fp8 W8A8 o
/// matmul). The wiring carries the `(companion id, copy id)` list and this is the whole of the
/// copy at runtime: each listed `to` row is the SAME bytes, dtype and shape as its `from` row.
/// The PERMUTATION itself is not here — it happens at staging, through the copy's own
/// `kernel_weights` retile descriptor (the 6-axis walk that enumerates the head-major
/// contraction order), so this is a source re-read, not a transform.
///
/// ⛔ RUNS AFTER THE FOLD PASS, so a folded o_proj's permuted copy carries the same post-fold
/// bytes its companion stages — the two must agree element-for-element, and the fold is the last
/// pass that touches the companion.
/// ⛔ REFUSES WHAT THE MINT DID NOT PREVENT: a copy naming a `from` row the binding did not
/// stage, or a `to` id the binding already owns — either is a wiring↔mint desync, not an input
/// to interpret.
fn materialize_weight_copies(
    weights: &mut Vec<StagedWeight>,
    copies: &[(u32, u32)],
) -> ExecutorResult<()> {
    for (from, to) in copies {
        if weights.iter().any(|(id, ..)| *id == *to as usize) {
            return Err(werr(format!(
                "head-major weight copy names t{to}, which the generated binding already staged — \
                 the layout minted a permuted copy for an id the binding owns, so the two \
                 disagree about the sources"
            )));
        }
        // Cloned out of the borrow before the push — the row is the copy's whole content.
        let (bytes, dt, shape) = {
            let (_, bytes, dt, shape) = weights
                .iter()
                .find(|(id, ..)| *id == *from as usize)
                .ok_or_else(|| {
                    werr(format!(
                        "head-major weight copy names companion t{from}, which the generated \
                         binding did not stage — the wiring and the weight binding disagree about \
                         the sources"
                    ))
                })?;
            (bytes.clone(), *dt, shape.clone())
        };
        weights.push((*to as usize, bytes, dt, shape));
    }
    Ok(())
}

pub(crate) fn stage_bound_weights(
    bound: &[scratchy_forward_compiler::BoundWeight],
    // ⛔ FOR PLACEMENT EXISTENCE ONLY — never for names. The K-split branches
    // below must ask whether the BAKED bundle actually placed its block tids
    // (on a quantized model the emitter returns before the split, leaving the
    // tid with a zero-byte placement). That is a fact about the TAPE, not about
    // a weight, and reading it from the environment instead is what silently
    // emptied every completion in `22a8f469`. Asked of the GENERATED wiring
    // now, which carries the same per-id extents the manifest did.
    wiring: &scratchy_target_spyre::wiring::Wiring,
    // The allocator that owns the bound tensors' buffers: staging TAKES each buffer
    // (no copy) at its last binding instead of copying it while the original stays alive.
    alloc: &mut SpyreAllocator,
) -> ExecutorResult<Vec<StagedWeight>> {
    let _t_load = std::time::Instant::now();
    let mut weights: Vec<StagedWeight> = Vec::new();
    let _take_s = 0f64;
    // The LAST binding of each buffer — a buffer two bindings share (a tied head, k=v) is
    // copied for every binding but its last, which takes it.
    let last_binding: std::collections::HashMap<*const u8, usize> = bound
        .iter()
        .enumerate()
        .map(|(i, bw)| (gpu_tensor_bytes(&bw.tensor).as_ptr(), i))
        .collect();
    for (bi, bw) in bound.iter().enumerate() {
        let is_gemm = bw.is_gemm;
        let s_id = bw.id as usize;
        // ⭐⭐⭐ `[rows, cols]` COMES FROM THE BAKE, NOT FROM THE TENSOR'S OWN DIMS.
        //
        // Everything below — the orientation guard, the device-width pad, the K-split gate, the
        // emitted shape — is written against the BUNDLE's convention: the LOGICAL `[rows = k,
        // cols = n]`. That pair used to come from `manifest.tensors[s.id]`, and
        // `wiring.tensor_shapes` is the same table, generated instead of parsed.
        //
        // ⛔ IT IS NOT `bw.staged_shape()`. That reads the tensor's ON-DISK dims, which for a GEMM
        // weight are `[n, k]` — the same two numbers in the other order, under the same names. The
        // padding call then became `for_output(1, k, n)`, which is not an error but a PLAUSIBLE
        // answer: granite's k (2048 = 32 sticks) already fills the machine so the bump is a no-op,
        // while its n (49155 → 769 PRIME sticks) needs the full-occupancy pad to 51200. The host
        // staged 49155 unpadded columns into a placement sized for 51200:
        //   `stage: kernel weight 't726' host size 201338880 B != ... (209715200 B)`
        // Every other weight in the model was unaffected, because their n is already splittable
        // and padding either axis is the same no-op. Asking the bake removes the question.
        let (rows, cols) = match wiring.tensor_shapes.get(s_id) {
            Some(&(r, c)) => (r as usize, c as usize),
            None => {
                return Err(werr(format!(
                    "sendnn stage: weight source {s_id} has no baked shape in the wiring                      ({} entries) — the generated loader and the generated wiring disagree about                      which sources exist; refusing rather than guessing from the tensor's dims.",
                    wiring.tensor_shapes.len(),
                )));
            }
        };
        // The tensor's own bytes and dtype — the generated binding already
        // resolved WHICH tensor, so this is a copy out of the field's buffer
        // and nothing else. `raw_name` survives only for the card path's size diagnostic below.
        #[cfg(feature = "spyre-hw")]
        let raw_name = scratchy_target_spyre::bundle_code::PlaceId::Act((s_id) as u32);
        let bytes = {
            let src = gpu_tensor_bytes(&bw.tensor);
            let (ptr, len) = (src.as_ptr(), src.len());
            if last_binding.get(&ptr) == Some(&bi) {
                // SAFETY: this is the buffer's last binding and staging is its last reader
                // in the load — nothing reads through `bw.tensor` (or any tensor viewing
                // the same buffer) after this point.
                unsafe { alloc.take(ptr, len) }.unwrap_or_else(|| src.to_vec())
            } else {
                src.to_vec()
            }
        };
        let dt = bw.tensor.dtype();
        // The sengraph emits every GEMM `MatMul` with `transpose_b=false`, so the
        // bound `ModelInput` must be the logical `[in, out] = [k, n]` (= on-disk
        // `Wᵀ`). The on-disk safetensors buffer is `[out, in] = [n, k]`, recorded
        // in the manifest as `[rows=k, cols=n]` (the LOGICAL `[in,out]`). The KTIR
        // emulator path keeps the buffer VERBATIM as `[n, k]` and reads it with a
        // transpose-B `indexing_maps` (Arg shape `[cols=n, rows=k]`). The sendnn
        // path has NO transpose-B (the graph optimizer's static SDPA needs plain
        // MatMul), so we must PHYSICALLY transpose `[n, k]` → `[k, n]` here to match
        // the dd2-PROVEN M5 graph (which binds each weight `.T`). Norms (`[1, hidden]`,
        // `is_gemm=false`) stay verbatim. lm_head reuses the embed buffer and is a
        // GEMM weight, so it is transposed to `embed.T = [hidden, vocab]` here too.
        // ⭐ EMU-BLINDNESS (W29, on-card 2026-10-13): that verbatim + transpose-B
        // read is layout-preserving, so the emulator CANNOT see a staging-layout
        // defect — the router W's missing stick-tile (staged verbatim into a
        // stick-tiled slot → `A @ perm(W)` → junk) was emu-green at every suite
        // while the card was garbage. A weight's DEVICE layout is a card-side
        // contract (the kernel0 manifest + the N-pad below); pin it there, never
        // expect the emulator to catch it.
        #[cfg(feature = "spyre-hw")]
        let (bytes, shape) = if is_gemm {
            // on-disk `[n, k]` = `[cols, rows]`; transpose to `[k, n]` = `[rows, cols]`.
            let elem = dt.size_bytes();
            // ⛔ LOAD-TIME GUARD: the on-disk GEMM buffer MUST be exactly
            // `[n,k] = [cols,rows]` elements, or the re-tile's disk-order stride map indexes
            // outside it and produces silent corruption. Catches a manifest↔disk dim mismatch (wrong
            // file / quant mismatch / a lowering dim error) before it becomes garbage
            // tokens. (A pure byte-ORDER flip preserves the count `k*n==n*k` and is NOT
            // caught here — that orientation is locked lowering-side by
            // `validate_default_mode_matmul_orientation`.)
            if bytes.len() != rows * cols * elem {
                return Err(werr(format!(
                    "sendnn load_weights: GEMM weight '{raw_name}' is {} bytes but the manifest \
                     declares rows*cols*elem = {rows}*{cols}*{elem} = {}; cannot transpose \
                     [n,k]->[k,n] (orientation contract) — manifest/disk shape mismatch.",
                    bytes.len(),
                    rows * cols * elem
                )));
            }
            // NO TRANSPOSE. The buffer stays in its on-disk `[n, k] = [cols, rows]` orientation and
            // the re-tile reads it there, via `stride_map_disk_order`. That map resolves the same
            // logical element as the transposed one (proven in `sdsc_df_width`), so this deletes a
            // whole pass over the model -- 4.4s of the 6.94s `load_weights` on granite-3.1-8b, plus
            // the full second copy it wrote.
            let t = bytes;
            {
                // DEVICE stick padding — the shared `bump_sticks_to_splittable` rule: round to a
                // whole 64-stick, then bump a prime/awkward stick count (granite lm_head
                // 49216/64=769 ⇒ <8 cores) to a multiple of 8 (→ 49664). Zero-pad this GEMM
                // weight's N columns to the SAME width so the staged host buffer equals the
                // kernel RetileDescriptor's device_size; the shim's `prod(device_size)` staging
                // guard then matches. The pad columns are ZERO logits the host never reads (the
                // read uses the LOGICAL `vocab = n`). No-op for every already-splittable weight.
                // TYPE-SAFE shared device width: `DeviceWidth` is the SOLE padding rule (the same
                // law as the emitter's kernel0 manifest and the slot reservation), so the staged
                // buffer width == the emitted device width by construction — no shim mismatch
                // possible.
                //
                // ⭐ `for_pointwise`, NOT `for_output(1,·)`: this buffer is ONE per model — it must
                // satisfy the OWNING session's manifest, and the owner is the prefill (m = the
                // chunk width), never m=1. For every weight crossing the util floor at m=1 (all
                // dense weights) the two laws are IDENTICAL — both bump. The pointwise floor
                // (512) is the max any bucket reads; each bucket's gemm takes
                // its own stick-prefix of the one buffer.
                let n_dev =
                    scratchy_target_spyre::lower_subtile_tape_to_superdsc::DeviceWidth::for_pointwise(
                        cols as u32,
                    )
                    .get() as usize;
                let n_padded = if n_dev != cols {
                    // In `[n, k]` the padded axis `n` is the OUTER one, so widening it is appending
                    // zero rows at the end -- `resize`, not a strided rebuild. (Transposed it was the
                    // inner axis and every row had to be copied to a wider stride.) Same padded bytes,
                    // and the only weight that needs any is lm_head.
                    let mut padded = t;
                    padded.resize(n_dev * rows * elem, 0);
                    padded
                } else {
                    t
                };
                // ⭐ THE DENSE fp8 K-PAD (gemma-4-26b's GeGLU DOWN, k=2112 → 2176): the quantize
                // chain and the packed fp8 retile both run at `fp8_k_pad(k)` — a whole 128-elem
                // SEN143_FP8 stick, the convert's own law — so the staged buffer's INNER axis (the
                // contraction k, `rows` in the manifest's `[k, n]` pair) must widen to match
                // `prod(device_size)` or the retile's staging guard refuses. Unlike the N-pad above,
                // the K axis is INNER in the on-disk `[n, k]` order: a `resize` would only append
                // zeros after the LAST row, so this is a STRIDED REBUILD — each of the `n_dev` rows
                // copies its `k` real elements and zero-fills its `k_pad − k` tail. The pad rows are
                // ZERO, which the contraction makes inert (0·w contributes nothing); the amax never
                // sees them (it reads the activation's logical width). Baked in the wiring because
                // WHICH axis is k is a bake fact.
                match wiring
                    .fp8_k_pads
                    .iter()
                    .find(|&&(id, _)| id as usize == s_id)
                {
                    Some(&(_, k_pad)) => {
                        let k = rows;
                        if k_pad as usize <= k {
                            return Err(werr(format!(
                                "sendnn load_weights: fp8 K-pad for source {s_id} says k_pad={k_pad} \
                                     but the manifest declares k={rows} — the pad must widen, never narrow \
                                     (wiring/manifest desync)."
                            )));
                        }
                        // Post-N-pad the buffer is `[n_dev, k]` row-major (k inner, `rows` = k).
                        let mut padded = vec![0u8; n_dev * k_pad as usize * elem];
                        for n_row in 0..n_dev {
                            let src_off = n_row * k * elem;
                            let dst_off = n_row * k_pad as usize * elem;
                            padded[dst_off..dst_off + k * elem]
                                .copy_from_slice(&n_padded[src_off..src_off + k * elem]);
                        }
                        (padded, vec![k_pad as usize, n_dev])
                    }
                    None => (n_padded, vec![rows, n_dev]),
                }
            }
        } else {
            (bytes, vec![rows, cols])
        };
        #[cfg(not(feature = "spyre-hw"))]
        let shape = if is_gemm {
            vec![cols, rows]
        } else {
            vec![rows, cols]
        };
        weights.push((s_id, bytes, dt, shape));
    }
    // ── NARROW bf16/f32 → f16 IN PARALLEL (all cores). This is the SAME conversion
    //    `SuperDscSession::new`'s bind loop did — but SERIALLY, on the startup critical
    //    path (~15s for a 2B bf16 model). Fanning it here makes the bind a pure memcpy
    //    (its F16 branch), so the conversion cost is amortized across cores instead of
    //    single-threaded. Byte-identical: the exact `f32_to_f16_le(bytes_to_f32(..))`
    //    the bind used, just relocated + parallelized. The host-weight DBG dump above
    //    ran on the pre-narrowing bytes, so its numbers are unchanged. ──
    #[cfg(feature = "spyre-hw")]
    {
        use rayon::prelude::*;
        let _t_cvt = std::time::Instant::now();
        // DIRECT bf16/f32 → f16, no intermediate f32 Vec (per /tmp/zoo). Cast the raw
        // bytes to &[bf16]/&mut [f16] and convert with a SIMD-friendly parallel zip
        // ACROSS ELEMENTS — so the two huge embed/lm_head tensors get all cores too,
        // not one core each. Every `bytes` here is a freshly-allocated Vec<u8> (take_to
        // _cpu_bytes / transpose output), hence ≥2-byte aligned; the host is LE, so the
        // cast equals `from_le_bytes`. Byte-identical to half::f16::from_f32(x.to_f32()).
        weights
            .par_iter_mut()
            .for_each(|(_, bytes, dt, _)| match *dt {
                SDType::F16 => {}
                SDType::BF16 => {
                    let n = bytes.len() / 2;
                    let mut out = vec![0u8; n * 2];
                    let src = unsafe {
                        std::slice::from_raw_parts(bytes.as_ptr() as *const half::bf16, n)
                    };
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut half::f16, n)
                    };
                    dst.par_iter_mut()
                        .zip(src.par_iter())
                        .for_each(|(o, &i)| *o = half::f16::from_f32(i.to_f32()));
                    *bytes = out;
                    *dt = SDType::F16;
                }
                // fp8 weight STAYS 1 byte — do NOT narrow to f16. The `other` arm's bytes_to_f32→f16
                // would DEQUANT the fp8 weight on load (the reward-hack: fp16 inference mislabeled as
                // fp8, zero bandwidth win). The 1-byte fp8 weight is staged verbatim at the 128-lane
                // stick and read 1-byte by matmulfp8 — the real W8A8 win. (byte-identical for dense.)
                SDType::Fp8E4m3 => {}
                other => {
                    let f32v = bytes_to_f32(bytes.as_slice(), other);
                    let mut f16 = Vec::with_capacity(f32v.len() * 2);
                    for &v in &f32v {
                        f16.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
                    }
                    *bytes = f16;
                    *dt = SDType::F16;
                }
            });
        tracing::debug!(
            "[timing]   ↳ bf16→f16 narrow took {:.2}s",
            _t_cvt.elapsed().as_secs_f64()
        );
    }
    // ── THE SCALARMUL WEIGHT FOLD'S RUNTIME HALF. Runs AFTER the narrowing on the card (so the
    //    multiply lands on the final f16 bytes — one rounding) and on the source dtype on the
    //    emulator; the folded ScalarMul has no program on either tier, so the scale enters the run
    //    HERE and nowhere else. ──
    apply_weight_scale_folds(&mut weights, wiring.weight_scale_folds)?;
    // ── THE HEAD-MAJOR o_proj COPIES' RUNTIME HALF — after the folds, so each permuted copy
    //    carries the same post-fold bytes its companion stages. ──
    materialize_weight_copies(&mut weights, wiring.weight_copies)?;
    tracing::debug!(
        "[timing] load_weights ({} tensors) took {:.2}s (of which take/disk {:.2}s; transpose = remainder)",
        weights.len(),
        _t_load.elapsed().as_secs_f64(),
        _take_s,
    );
    Ok(weights)
}

impl SpyreWorker {
    /// Resolve the bundle + load all static weights + derive dims + wiring.
    pub(crate) fn load_inner(&self) -> ExecutorResult<Loaded> {
        // Overlap the ~5.7s flex device bring-up (runtime-init inside prepare) with the
        // host-side weight load below by kicking it off on a background thread NOW.
        // std::call_once makes prepare's own init block only on the remainder — a pure
        // latency win with no change to the bytes that reach the device.
        #[cfg(feature = "spyre-hw")]
        SuperDscSession::prewarm_runtime();

        let model_dir = resolve_model_path(
            &self.model_path,
            self.config.hf_token.as_deref(),
            self.config.gguf_file.as_deref(),
            None,
        )?;
        let hf_config = HfModelConfig::from_dir(&model_dir)
            .map_err(|e| werr(format!("config.json in {}: {e}", model_dir.display())))?;

        // Load on-disk weights into a host GpuWeights, then resolve the compiled
        // arch through scratchy's registry (fingerprint) — exactly like the
        // cuda/metal workers. NO stem-glob, NO ~/.cache read: the matched
        // ScratchyWeights carries the macro-embedded KTIR bundle.
        let _t_gw = std::time::Instant::now();
        let mut gw = GpuWeights::from_dir(&model_dir, SpyreAllocator::new())
            .map_err(|e| werr(format!("GpuWeights::from_dir: {e}")))?;
        // Startup is dominated by how many times the checkpoint's bytes are moved, so each
        // move is timed separately: this one is disk/page-cache -> host.
        tracing::debug!(
            "[timing] GpuWeights::from_dir took {:.2}s",
            _t_gw.elapsed().as_secs_f64()
        );

        let arch = hf_config.architectures.first().cloned().unwrap_or_default();
        let hf_fp = scratchy_forward_compiler::HfFingerprint {
            rope_scaling_type: hf_config
                .extra
                .get("rope_scaling")
                .and_then(|rs| rs.get("rope_type").or_else(|| rs.get("type")))
                .and_then(|v| v.as_str()),
            rope_scaling_hash: hf_config
                .extra
                .get("rope_scaling")
                .map(scratchy_forward_compiler::hash_json_value),
            rope_theta: hf_config.rope_theta,
        };
        let max_model_len = hf_config.max_position_embeddings.unwrap_or(4096);

        // Fetch the embedded KTIR bundle (spyre's forward representation) for THIS
        // checkpoint — READ-ONLY (fingerprint sniff), leaving every tensor in `gw`
        // for the destructive byte extraction below. spyre's typed `Weights::load`
        // is real (it builds the same fielded struct cuda/metal do), but the KTIR
        // runner streams raw weights, so the worker takes spyre's forward path
        // here instead. The fingerprint (`gw` shapes + `hf_fp`) discriminates
        // variants: an arch registers many distinct base models (llama-2-13b …
        // llama-3.2-3b), each with its own bundle + layer count, so the bundle
        // MUST be matched to the loaded checkpoint, not picked by arch name alone.
        // ⭐ ONE MATCHED VARIANT ANSWERS EVERYTHING.
        //
        // This used to be TWO independent registry walks — `resolve_*_bundle`
        // for the bundle, then a hand-rolled disk load for the weights — with
        // a comment claiming it ran "AFTER try_load" while nothing called
        // `try_load` at all. An arch registers many base models, so two walks
        // means two variant matches with nothing checking they agree: the
        // bundle could come from one and the weights from another.
        //
        // `try_load` runs the GENERATED `Weights::load` — the same loader
        // cuda/metal use — and the bundle, the weight binding and the embed
        // table are all read off THAT model.
        let _t_try = std::time::Instant::now();
        let model = scratchy_forward_compiler::try_load(
            &mut gw,
            (),
            arch.as_str(),
            1,
            0,
            max_model_len,
            hf_fp,
        )
        .map_err(|e| werr(format!("try_load {arch:?}: {e}")))?;
        let model = match model {
            scratchy_forward_compiler::ArchLoad::Loaded(m) => m,
            // The arch is compiled; the checkpoint just doesn't match any of its
            // variants, so widening the model feature won't help — the
            // quantization layout or shapes are the mismatch.
            scratchy_forward_compiler::ArchLoad::NoVariantMatched => {
                return Err(werr(format!(
                    "the {arch:?} arch is compiled for spyre, but no compiled model variant \
                     matches this checkpoint — its shapes or quantization layout differ from \
                     every (model stem, quant preset) pair in this build"
                )));
            }
            scratchy_forward_compiler::ArchLoad::ArchNotCompiled => {
                return Err(werr(format!(
                    "no compiled spyre model matched the {arch:?} checkpoint — \
                     build the CLI with this model's feature \
                     (e.g. --features spyre,model/llama-3.2-1b)"
                )));
            }
        };
        // `try_load` runs the GENERATED loader over every tensor, so it is a whole pass
        // across the checkpoint in its own right — timed apart from the raw byte extraction
        // that follows it.
        tracing::debug!(
            "[timing] try_load ({arch}) took {:.2}s",
            _t_try.elapsed().as_secs_f64()
        );
        #[cfg(not(feature = "spyre-hw"))]
        let bundle: &'static KtirBundle = model
            .ktir_bundle()
            .and_then(|b| b.downcast_ref::<KtirBundle>())
            .ok_or_else(|| werr("the matched model carries no KTIR bundle"))?;
        #[cfg(feature = "spyre-hw")]
        let bundle: &'static SengraphBundle = model
            .sengraph_bundle()
            .and_then(|b| b.downcast_ref::<SengraphBundle>())
            .ok_or_else(|| werr("the matched model carries no sendnn bundle"))?;

        // The embed table for the host-side token gather. Taken FROM THE MODEL:
        // the generated loader already removed it from `gw`, and the key this
        // used to hardcode (`"model.embed_tokens.weight"`) is itself the
        // gemma-4 bug — its decoder root is `model.language_model`.
        let embed_t = model
            .superdsc_embed()
            .ok_or_else(|| werr("the matched model exposes no embedding table"))?
            .map_err(|e| werr(format!("embed table: {e}")))?;
        let _t_embed = std::time::Instant::now();
        let embed_tokens = bytes_to_f32(gpu_tensor_bytes(&embed_t), embed_t.dtype());
        // The embed table is `[vocab, hidden]` and this widens ALL of it to f32 up front,
        // so it is sized by the vocabulary, not by the prompt.
        tracing::debug!(
            "[timing] embed table -> f32 ({} elems) took {:.2}s",
            embed_tokens.len(),
            _t_embed.elapsed().as_secs_f64()
        );

        // Parse the decode + prefill manifests (sources/wiring/dims). For KTIR the
        // bundle carries decode + optional prefill directly; for sendnn the
        // LAYER-GROUP bundle carries N groups sharing the full manifest (baked
        // per-group), so parse group 0's decode/prefill manifest (== the full one).
        // KTIR bakes one decode program PER CAP BUCKET (all share weight ids — same graph,
        // different prefix cap), so parse every one. sendnn bakes a single group-0 program.
        // ⭐ NO PARSING. The wiring is a GENERATED static on the loaded model —
        // the same facts the `manifest_json` string carried, as typed values.
        // ⛔ NOT `sendnn`-GATED. The wiring is GENERATED data, emitted for every spyre build, and
        // `stage_bound_weights` reads its `tensor_shapes` on both paths — that is where the staged
        // `[rows, cols]` comes from now that the manifest no longer supplies it. Gating it left the
        // KTIR-only build referencing a binding that did not exist.
        let wirings: &'static scratchy_target_spyre::wiring::Wirings = model
            .superdsc_wiring()
            .and_then(|w| w.downcast_ref::<scratchy_target_spyre::wiring::Wirings>())
            .ok_or_else(|| werr("the matched model exposes no launch wiring"))?;
        let (dparsed_all, pparsed) = {
            let d = vec![wiring_to_parsed(&wirings.decode)?];
            // A model with no batched-prefill program simply has none; the
            // chunked path below already handles `None`.
            let p = match wirings.prefill.as_ref() {
                Some(p) => Some(wiring_to_parsed(p)?),
                None => None,
            };
            (d, p)
        };
        // Model-level dims are identical across cap buckets, so the first is the reference.
        let dparsed = &dparsed_all[0];

        // NOT overlapped, and the measurement is why. Starting the decode bundle load before
        // `load_weights` gained nothing on startup (still 20s) and cost decode 11.6 -> 11.3 tok/s.
        // Building a session ALLOCATES, so beginning it ahead of the prefill session's seg1
        // reservation reorders the allocation the reservation exists to control -- the same effect
        // that is worth 1.8 tok/s between orderings here, showing up smaller.
        // Load weights ONCE (all programs/groups share ids) as typed bytes.
        //
        // ⭐ FROM THE GENERATED BINDING, NOT FROM DISK NAMES. Each entry names
        // a `Weights` FIELD (`w.self_attn_q_proj[3]`), so nothing here rebuilds
        // an on-disk key. That deletes the `"{disk}.weight"` rule, the
        // `raw_cache`/share-count machinery it needed, and the tie-redirect
        // special case — the generated loader already resolved all three.
        let bound = model
            .superdsc_weights()
            .ok_or_else(|| werr("the matched model exposes no weight binding"))?
            .map_err(|e| werr(format!("weight binding: {e}")))?;
        tracing::info!(
            "[weights] superdsc_weights returned {} bound weight(s); wiring has {} tensor shape(s)",
            bound.len(),
            wirings.decode.tensor_shapes.len()
        );
        let weights = stage_bound_weights(&bound, &wirings.decode, gw.allocator_mut())?;
        tracing::info!(
            "[weights] staged {} weight(s) for the session",
            weights.len()
        );
        // NOTE: the RoPE rotate-half permutation P (ROPE_P_TID) is NOT a load-time weight.
        // It is a synthetic seg0 ACTIVATION re-bound per step in `superdsc_forward_chunk`
        // (see the `acts` block there) — a seg1 weight would only be H2D'd at PrepareModel,
        // which binds ONLY the manifest's model weights, leaving a synthetic P at ZERO.

        // KTIR: ONE resident session over [prefill?, decode]. sendnn: N layer-group
        // [prefill, decode] sessions (the host threads the hidden state group→group).
        // Program order: prefill first (index 0) when present, then the decode cap buckets in
        // ascending-cap order — so a bucket's program index is `decode_prog_base + i`.
        #[cfg(not(feature = "spyre-hw"))]
        let (session, decode_prog_base) = {
            // ⭐ THE PROGRAMS COME OUT OF THE REGISTRY, BY FINGERPRINT. `#[forward]` submitted them
            // as const data; the bundle names which set is this model's. That is the same resolver
            // the ladder rungs and the re-rolled siblings already use, so a program is found in one
            // place.
            let groups = |fp: &str| -> ExecutorResult<
                &'static [scratchy_target_spyre::bundle_code::LaunchGroup<'static>],
            > {
                let code = scratchy_target_spyre::bundle_code::bundle(fp).ok_or_else(|| {
                    werr(format!(
                        "no KTIR programs compiled into this binary for fp={fp} — the emit ran for \
                         a different model, or without the lowering. Registered: {:?}",
                        scratchy_target_spyre::bundle_code::registered_fps(),
                    ))
                })?;
                Ok(&code.groups)
            };
            let mut programs: Vec<(
                &[scratchy_target_spyre::bundle_code::LaunchGroup<'static>],
                Vec<u64>,
            )> = Vec::new();
            if let (Some(pp), Some(pb)) = (pparsed.as_ref(), bundle.prefill.as_ref()) {
                programs.push((
                    groups(pb.fp)?,
                    pp.output_ids.iter().map(|i| *i as u64).collect(),
                ));
            }
            let base = programs.len();
            for dp in dparsed_all.iter() {
                programs.push((
                    groups(bundle.decode[0].fp)?,
                    dp.output_ids.iter().map(|i| *i as u64).collect(),
                ));
            }
            let refs: Vec<(
                &[scratchy_target_spyre::bundle_code::LaunchGroup<'static>],
                &[u64],
            )> = programs.iter().map(|(g, o)| (*g, o.as_slice())).collect();
            let _t_sess = std::time::Instant::now();
            let s = SpyreSession::new_multi(&refs, weights)
                .map_err(|e| werr(format!("build resident session: {e}")))?;
            // The SECOND move of the checkpoint: host bytes -> resident HBM sticks.
            tracing::debug!(
                "[timing] SpyreSession::new_multi took {:.2}s",
                _t_sess.elapsed().as_secs_f64()
            );
            (s, base)
        };
        // ⭐ scratchy_target_spyre::wiring::HAT THE BAKE PLACED, per bundle — one value each, not six locals.
        //
        // These were `scalarmul_scales_v`, `prefill_uses_ones`, `prefill_ones_len`,
        // `prefill_uses_identity`, `decode_uses_identity` and `baked_lm_head_ksplit`: declared
        // here, assigned ~350 lines below inside a `#[cfg(feature = "spyre-hw")]` branch, and read
        // ~600 lines below that. Each needed `#[allow(unused_mut, unused_assignments)]` to
        // compile, and that allow is what let one of them go on being derived for a bind that had
        // stopped existing. `BakeFacts::of` asks the layout once.
        //
        // ⚠️ ONE `allow` SURVIVES, DOWN FROM SIX, and it is a cfg artifact rather than a silenced
        // problem: a value computed inside `cfg(sendnn)` and consumed outside it must be declared
        // before the branch, so its initialiser is dead in exactly the build that assigns it.
        // `None` IS read on the non-sendnn path, and what it feeds is one call away.
        // ⛔ ON THE CARD, `decode_facts` HAS NO VALUE UNTIL THE DECODE BAKE GIVES IT ONE: the
        // branch below that bakes the decode bundle assigns it, and every other card path returns
        // an error first, so the compiler proves it is set before it is read (the
        // `served_context_v` pattern below). `BakeFacts::NONE` is not a default for the card.
        // `prefill_facts` is assigned only when a prefill bundle is baked, and is
        // `BakeFacts::NONE` (a bundle that placed nothing) when there is none. On the emulator
        // both are `NONE`, read once.
        #[cfg(feature = "spyre-hw")]
        let decode_facts: scratchy_target_spyre::wiring::BakeFacts;
        #[cfg(not(feature = "spyre-hw"))]
        let decode_facts = scratchy_target_spyre::wiring::BakeFacts::NONE;
        #[cfg(feature = "spyre-hw")]
        let mut prefill_facts = scratchy_target_spyre::wiring::BakeFacts::NONE;
        #[cfg(not(feature = "spyre-hw"))]
        let prefill_facts = scratchy_target_spyre::wiring::BakeFacts::NONE;
        // The KV byte budget the paged pool gets sized against — the card's real capacity ×
        // `--gpu-memory-utilization` − the bundle's non-KV segments. Declared out here for the same reason
        // `scalarmul_scales_v` is: it is computed in the SuperDsc branch below (sendnn only) but has to
        // reach the `Loaded` this function returns, where `determine_available_memory` reads it so the
        // ENGINE sizes the scheduler's block allocator off the SAME number the pool was cut from.
        #[allow(unused_mut, unused_assignments)]
        let mut kv_budget_bytes_v: Option<u64> = None;
        // The context the sized pool will actually serve (tokens) — the ONE context law's answer
        // (`resolve_served_context`) over the declaration and what the budget funds, decided in
        // the SuperDsc branch below where budget, page stride and launch rows first meet, and
        // ALWAYS decided there on this tier: every path through the pool sizing either resolves
        // or refuses, so the binding is left UNINITIALIZED on the card tier and the compiler
        // holds that invariant (no dead `usize::MAX` initializer to warn about, no `#[allow]`).
        // `usize::MAX` on the paths that never size a paged pool (the emulator tier), so they
        // report no cap and `kv_max_addressable_tokens` falls back to the baked mask reach alone.
        #[cfg(feature = "spyre-hw")]
        let served_context_v: usize;
        #[cfg(not(feature = "spyre-hw"))]
        let served_context_v: usize = usize::MAX;
        // The paged pool's page stride — filled beside `kv_budget_bytes_v` (the SuperDsc branch
        // below is the one site both are derived) so `Worker::kv_block_bytes` can hand the engine
        // the same stride the pool was sized by. Card tier only: the emulator never sizes a pool,
        // and the construction below carries `None` there.
        #[cfg(feature = "spyre-hw")]
        let mut pool_stride_bytes_v: Option<u64> = None;
        #[cfg(feature = "spyre-hw")]
        let session = {
            // ⛔ GUARD (mirror of the build-time flit-cap guard): an EMPTY group
            // graph_json means the lowering REFUSED to emit it (the split still
            // overflowed, or a group failed a guard). Surface it clearly rather
            // than letting DeserializeFromString crash on empty input.
            for grp in bundle.groups {
                if grp.prefill.graph_json.is_empty() || grp.decode.graph_json.is_empty() {
                    return Err(werr(format!(
                        "sendnn layer-group {} sengraph is EMPTY — the build-time lowering refused \
                         to emit it (per-job flit-cap or a per-group guard). See the \
                         `[spyre-sendnn] … layer-group split` build log.",
                        grp.group
                    )));
                }
            }
            if is_superdsc_bundle(bundle) {
                // ── SUPERDSC dxp bundle (the typed-Rust SuperDSC OpSpec emitter). The
                //    lowering owns the 32-core work-division and emits a DIRECTORY of
                //    per-supernode dxp blobs + plan (NOT a DeepTools sengraph JSON); the
                //    codegen stamps the decode-slot `graph_json` with `SUPERDSC_BUNDLE:<fp>`.
                //    Recover the baked dxp dir by `fp` and construct the resident session
                //    (DeepRT/libdxp run-path): KV is DEVICE-RESIDENT, single-stream, one
                //    token/step for the MVP (behaves like Default in every match below). ──
                let g0 = bundle
                    .groups
                    .first()
                    .expect("superdsc bundle has one group");
                let fp = g0
                    .decode
                    .graph_json
                    .strip_prefix(SUPERDSC_SENTINEL)
                    .ok_or_else(|| {
                        werr("superdsc bundle decode-slot missing SUPERDSC_BUNDLE: sentinel")
                    })?;
                // The dxp-compiled bundle for this fingerprint, carried IN THIS BINARY —
                // `#[forward]` compiled it at build time and `inventory::submit!`ed the value.
                let code = scratchy_target_spyre::bundle_code::bundle(fp).ok_or_else(|| {
                    werr(format!(
                        "no SuperDSC bundle compiled into this binary for fp={fp} — the emit either \
                         had no dxp_standalone to run (a cardless build) or ran for a different model. \
                         Registered: {:?}",
                        scratchy_target_spyre::bundle_code::registered_fps(),
                    ))
                })?;
                // ⭐ THE SEGMENT BUDGET's runtime work list, read BEFORE any session is built (the
                // first of them is constructed below, and `weights` moves into the DECODE session
                // later on): which weights the emitter spilled out of the weight segment, as byte
                // spans to copy and as ids to declare resident. Empty for every model whose weights
                // fit one segment, which makes every use of it below a no-op.
                let spill_spans = spilled_weight_spans(&code.layout);
                let spill_tids =
                    spilled_weight_tail_tids(&code.layout, weights.iter().map(|w| w.0));
                // Model dims from the SAME decode parse the other sessions read: vocab +
                // kv_dim. The resident KV pool is `num_blocks` blocks of 64 slots covering
                // the prefix capacity (mask cols) — same block_size the paged path uses.
                let vocab = dparsed.vocab;
                let kv_dim = dparsed.kv_dim;
                // THE DEFAULT POOL IS FLAT — 8 pages, whatever the batch — AND A REQUEST'S CONTEXT DOES
                // NOT SHRINK WHEN THE BATCH WIDENS.
                //
                // ⭐ SO THE DEFAULT SCALES WITH THE LADDER AGAIN: `pages_per_row * WIDEST_BATCH_RUNG`.
                //
                // This scaling was removed when the request axis moved INSIDE the page — a page then held
                // `ROWS` requests' KV per kv head, so every row owned the whole page list and 8 pages was
                // 2048 positions for EVERY row. That is the design that has just been undone: a page holds
                // SLOTS now and a row owns its own PAGES, so a pool of 8 cannot seat a 32-wide rung at all
                // (`split_pool(8, 32)` is zero pages per row, which is the error this fixes).
                //
                // And it costs nothing, because it is the same factor the page just SHRANK by: dropping the
                // request dimension made a page 32x smaller (`page_stride` was ~1 GB), so 32x as many pages
                // is the memory the pool already had — while giving every row its own pages, which is what
                // makes one launch able to stride across the batch.
                //
                // An explicit SUPERDSC_POOL_PAGES still wins, including downward: a smaller pool is a
                // legitimate memory-vs-context call. It is read as PAGES PER ROW, so it keeps meaning the
                // same thing to a caller — the context one sequence gets.
                // ⛔ AND THE DEFAULT IS A BYTE BUDGET, NOT A PAGE COUNT.
                //
                // `pages_per_row * WIDEST_BATCH_RUNG` is a COUNT, and a page's SIZE is the model's: 31.5 MB
                // for granite-2b (hd=64) but **70.8 MB at hd=128** (double the head dim, and `PLANE_SLOTS`
                // carries the padded-write slack). So one default asked 7.7 GB of the 2b and **17.3 GB of
                // granite-3.1-8b**, which the allocator refuses — `RAS::FLEXALLOCATOR::OutOfMemory`,
                // "all existing regions lack sufficient free space", with 94 GB of the card FREE: no single
                // REGION serves an 18 GB contiguous request. The 8b loaded before the request axis moved
                // into the page, when the same default meant 8 pages (~566 MB); giving every batch row its
                // own stripe multiplied it by `WIDEST_BATCH_RUNG` = 32x, and nothing capped the product.
                //
                // A budget in BYTES is the quantity that is actually scarce, so derive the count from it and
                // the model's own page stride. Every model then gets as much context as its pages allow and
                // no model is handed a pool that cannot be allocated. `SUPERDSC_POOL_PAGES` still overrides,
                // in pages PER ROW, so an explicit request is honoured exactly as before.
                // ⛔⛔ AND THE WIDTH IS A DECISION, NOT `WIDEST_BATCH_RUNG`.
                // THE ADMISSION WIDTH — how many requests may decode together. Not a capacity decision:
                // pages come from a free list, so one request can reach the whole pool whatever this is, and
                // `PoolRows` bounds only how many LAUNCH SLOTS exist.
                //
                // ⛔⛔⛔ THERE IS NOTHING TO READ IT FROM, AND THAT IS THE INVARIANT. The width must BE a
                // ladder rung: `decode_rung_for` picks the smallest rung holding the live count and that
                // launch strides its slots arithmetically, so any other width has a live count whose rung
                // addresses slots the pool never seated. A value that must be one of a fixed set is a
                // decision, not a preference, and the ladder is the only thing entitled to make it — which
                // is why `SUPERDSC_POOL_ROWS` is deleted rather than clamped or refused.
                //
                // ⭐ THE SMALLEST RUNG THAT HOLDS WHAT THIS RUN WILL ADMIT. It used to be the WIDEST rung
                // unconditionally, which is free for capacity (a free list lets one request reach the whole
                // pool at any width) and expensive for the RESERVE: `PoolPartition` holds back
                // `rows * HOLE_PAGES_PER_ROW + 1` pages for the holes a launch's rows write into, so 32 rows
                // reserved 65 of 136 pages against launches a `--max-num-seqs 4` run cannot perform.
                // `AdmittedRequests` is the decided value; `for_admission` rounds it UP to a rung, because a
                // pool cut for fewer rows than a launch binds addresses slots it never seated.
                // 🛑 AND IT DOES NOT REFUSE AN OVER-WIDE `--max-num-seqs`. It used to, and the reason it
                // gave had stopped being true: *"a request holds its KV row for its whole life ... the
                // (WIDEST+1)-th concurrent request would be refused mid-run with `every one of the N KV
                // row(s) is held by a live request`"*. That refusal, and the `free_row`/`rows_held` scarcity
                // it came from, were DELETED when the request axis left the pool — see the comment on
                // `ensure_pages`: *"a request needs no identity in the pool at all ... and so is the refusal
                // that fired when every row was held — a concurrency limit that existed only because rows
                // were a scarce per-request resource."* Nothing fails mid-run at WIDEST+1 any more.
                //
                // ⭐ WHAT ACTUALLY HAPPENS PAST THE LADDER IS A THROUGHPUT CLIFF, NOT A FAILURE.
                // `decode_rung_for` returns `None` when no rung holds the live count, `batched` stays empty,
                // and every request takes the per-request path (~93 µs of launch each) — correct output,
                // much slower.
                //
                // ⭐⭐ SO IT IS REPORTED AS A CAP AND THE SCHEDULER IS HELD TO IT — the cuda rule.
                // cuda refuses no `max_num_seqs`: it derives its captured-graph ladder FROM it
                // (`auto_capture_sizes(max_num_seqs)`), so its fast path always covers the configured
                // concurrency. A BAKED ladder cannot grow to meet the flag, so this backend holds the same
                // invariant from the other end via `Worker::max_num_seqs_override`. A bare `scr serve` and
                // an explicit `--max-num-seqs 256` then behave IDENTICALLY (both 32-wide, batched), which
                // the refusal made incoherent: it rejected the explicit spelling of the very default it
                // was silently substituting.
                //
                let pool_rows = PoolRows::for_admission(self.admitted.unwrap_or_else(|| {
                    // No usable cap ⇒ assume the widest a launch can express. The unsafe direction is
                    // too FEW rows, never too many.
                    scratchy_subtile::sdsc_abstract::AdmittedRequests::new(
                        PoolRows::WIDEST.get().get() as usize,
                    )
                    .expect("WIDEST is nonzero")
                }));
                //
                // ⭐⭐⭐ THE POOL IS SIZED BY WHAT THE RUN DECLARED, AND THE BUDGET ONLY GETS TO REFUSE.
                //
                // 🛑 IT USED TO BE SIZED BY A CONSTANT — `DEFAULT_POOL_BUDGET_BYTES = 8 GiB`, divided by the
                // model's page stride and clamped. That made every capacity question circular: the constant
                // decided how deep a request could go, so when granite-3.1-8b died mid-generation at
                // `request needs 769 positions but its share is 3 page(s)`, the only available answer was to
                // re-cut the same 8 GiB differently (WIDTH or DEPTH, never both) and hope. Nothing in the
                // program knew what the caller had actually asked for. The user's objection was exactly
                // this: *"we size the pools based on the command line max-seq-len and other such
                // parameters."*
                //
                // So: `PoolDemand::of_declaration(context, rows)` says what `--max-model-len` ×
                // `--max-num-seqs` costs in pages, reserve included, and the byte budget is checked
                // AGAINST that number rather than producing it.
                //
                // ⛔ AND IT REFUSES AT LOAD, IT DOES NOT CLAMP. A budget too small for the declaration is a
                // run that will kill some request mid-generation, hundreds of correct tokens in, at a depth
                // that depends on which requests happened to share the pool — the worst failure shape there
                // is. Serving a quietly smaller context leaves the caller's belief in place, which is
                // `capping-is-not-validating` verbatim. The refusal names both flags and both page counts,
                // so the fix is arithmetic the caller can do.
                //
                // ⛔⛔⛔ AND THE BUDGET IS THE CARD'S, NOT A CONSTANT'S. This was
                // `const POOL_BUDGET_BYTES: u64 = 8 GiB` — a number with no relationship to the device it
                // sizes an allocation on, which is how a bare `scr serve` came to refuse itself: 4096 ×
                // 32 rows needs ~15 GB, the constant said 8, and the arithmetic in the refusal never
                // mentioned the card at all. On a 128 GB part that refusal is pure fiction.
                //
                // 🛑 THE SAME GUESS HAS ALREADY BEEN PAID FOR ONE LAYER DOWN. `runtime::build_runtime`
                // sized the allocator's domain off a 32 GiB placeholder until a real `FlexAllocator` OOM on
                // hardware found it; the fix was to ask the card
                // (`flex_senlib_device_memory_size` → `DeviceHandle::GetDmpaSize`). The KV pool kept its own
                // placeholder and repeated the bug.
                //
                // ⭐ AND THE FORMULA IS THE SHARED ONE, NOT A THIRD SPELLING OF IT.
                // `scratchy_serving_engine::gpu_budget::compute_available_kv_bytes` is documented
                // "backend-neutral ... shared by the CUDA and Metal Worker implementations" and lives in a
                // crate this file already depends on. cuda calls it; metal calls it; spyre had a constant.
                // Three backends, one law, and the only one that opted out is the one that then could not
                // start.
                //
                // ⭐⭐⭐ THE POOL'S OWN DEMAND, DERIVED ONCE, HERE — above the budget on purpose: the
                // census below needs it (the prefill anchor's spare includes a SECOND pool exactly
                // this size, and whether the anchor reserves at all is decided against it), so
                // deriving it inside `num_blocks` would force a second spelling of the pool law.
                // `num_blocks` keeps only the budget's veto. A PAGE'S BYTES, from the pool's own
                // law: a page holds every LAYER, each layer three planes (Knat, V, Kᵀ), each plane
                // `kv_dim` features across the plane's PHYSICAL slot extent, at 2 bytes. Confirmed
                // against the runtime's own report — it logs `kvstride=1769472` per layer for
                // granite-8b, and `3 * 1024 * 288 * 2 = 1769472`, `* 40 layers = 70778880 B` =
                // the `page_stride` it then prints. Derived here rather than read back because
                // the pool must be SIZED before the session that would report it.
                // ⛔ FROM THE LOCAL WIRING, NOT `self.model` — `self.model` is still None here
                // (this IS `load_model`, it is what fills it), so reading it silently gave
                // `layers = 1`. That made the stride ONE LAYER's worth and the clamp handed back
                // the very default the budget was meant to replace: the fix compiled, ran, logged
                // nothing wrong, and changed nothing. A fallback that is *plausible* (1 layer) is
                // worse than no fallback.
                // ⭐ FROM THE BAKE, NOT config.json. The layer count is a fact the tape was
                // LOWERED at, so reading it from the wiring cannot disagree with the bundle and
                // has no default to fire.
                let pool_stride_bytes = {
                    let layers = (wirings.decode.geometry.layers as usize).max(1);
                    let plane_slots =
                        scratchy_subtile::sdsc_abstract::PagedKvPool::PLANE_SLOTS as u64;
                    (3 * kv_dim as u64 * plane_slots * 2 * layers as u64).max(1)
                };
                // THE DECLARED DEPTH, AS DECLARED — `--max-model-len` when the CLI gave one, else
                // the model's own limit. ⛔ NO MASK CAP, AND ITS ABSENCE IS THE LAW. This was
                // `min(declared, PagedKvPool::MAX_PAGES_PER_ROW * PAGE_SLOTS)` — a const main
                // deleted with the derived ceiling (bfe42f56d), and the rebase met its corpse
                // here as E0599. The prefix mask's block count is DERIVED at bake from the device
                // (`MAX_SEGMENT_BYTES / page_bytes`, one region of KV), so it covers every pool
                // any card can budget for, and the pool is ONE allocation capped at that same
                // region — a budget-funded declaration is therefore always within the mask's
                // reach, and the binding law for a given run is the budget veto below, which
                // REFUSES rather than clamps (`capping-is-not-validating`, this block's own law).
                // The artifact still guards every launch (`MaskBlocks::fits`), every page grant
                // (`within_mask_reach`), and admission (`kv_max_addressable_tokens` = min(served,
                // the mask reach)) — a mask cap HERE could only fire on worker arithmetic that
                // disagrees with the bake's, which is exactly what those per-launch guards exist
                // to catch.
                // ⛔ AND IT STAYS THE DECLARED DEPTH THROUGH THE CENSUS, BY DESIGN: the anchor's
                // reserve policy below needs a pool size to decide against, and what an UNSET
                // flag should resolve to is only answerable once the budget exists — the
                // resolution (and the explicit ask's refusal) happen at the budget, in the
                // context-law block below.
                let mut declared_ctx = self.declared_context.unwrap_or(max_model_len).max(1);
                let mut pool_demand = scratchy_subtile::sdsc_abstract::PoolDemand::of_declaration(
                    scratchy_subtile::sdsc_abstract::SlotCount::new(declared_ctx as u32),
                    pool_rows,
                );
                let mut declared_pages = pool_demand.pages().ok_or_else(|| {
                    werr(format!(
                        "--max-model-len {declared_ctx} overflows a page count"
                    ))
                })?;
                let mut pool_need_bytes = declared_pages.get() as u64 * pool_stride_bytes;
                //
                // ── THE BATCHED-PREFILL ANCHOR BUNDLE, resolved here because the census below
                // reads its layout — the anchor session's residency is a budget input now, and
                // whether it RESERVES or BORROWS is a capacity decision the budget itself makes. ──
                // Did the prefill reroll bake at all? (fp == the decode fp ⇒ C4 fell back.)
                let top_prefill = g0
                    .prefill
                    .graph_json
                    .strip_prefix(SUPERDSC_SENTINEL)
                    .filter(|pfp| *pfp != fp)
                    .and_then(scratchy_target_spyre::bundle_code::bundle);
                // ── PLACEMENT ANCHOR = the NARROWEST rung, not the widest ────────────────────────
                // This is the one session built BEFORE the decode session, so its footprint is what
                // positions decode's weight region (see the ownership note below). It used to be the
                // WIDEST rung, which made that footprint track the ladder CEILING: every segment that
                // scales with `mq` roughly doubles when `mq_pad` goes 64→128, measured on the baked
                // layouts as 2798.9 MiB at a 47-row ceiling vs 2993.7 MiB at 96 — with seg1 (weights,
                // 2522.1 MiB) and seg2 (KV) identical in both. So raising the ceiling silently shifted
                // decode's weights ~195 MiB and cost a few percent of BOTH TTFT and decode, on a
                // bundle a short prompt never even runs. Anchoring on the narrowest rung pins the
                // pre-decode footprint at the `mq_pad=64` value for every ceiling — the configuration
                // that measured faster — and the widest rung is then just another borrowing rung
                // built after decode, like all the others.
                // NOT "build decode first": the measurement below says decode's weights landing
                // EARLIER is the slow case (34.7 vs 36.5 tok/s), and decode-first is the earliest of
                // all. The fix is to hold the reservation CONSTANT, not to remove it.
                // ⛔ THE ANCHOR'S BUNDLE AND ITS WIDTH LABEL COME FROM THE SAME ENTRY, AS ONE PAIR —
                // and a manifest rung whose sentinel names no compiled bundle is an INVENTORY DESYNC
                // that REFUSES, never a fallback. The old `.or(top_prefill)` silently ran the TOP
                // prefill bundle under the NARROWEST rung's width label when the narrowest entry
                // failed to resolve (the mislabel class the anchor-width note below records), and
                // the ladder then built the widest rung too — the same bundle as two real sessions.
                // `top_prefill` remains exactly what it was gated to be: the anchor ONLY when the
                // ladder has no multi-token rung at all, so its width is the prefill wiring's own
                // `m_cap` in that case and never disagrees with the bundle it runs.
                let narrowest_rung = g0
                    .prefill_rungs
                    .iter()
                    .filter(|(rm, _)| *rm > 1)
                    .min_by_key(|(rm, _)| *rm);
                let prefill_code = match (top_prefill, narrowest_rung) {
                    (Some(_tp), Some((rm, sent))) => Some(sent
                        .strip_prefix(SUPERDSC_SENTINEL)
                        .and_then(scratchy_target_spyre::bundle_code::bundle)
                        .ok_or_else(|| {
                            werr(format!(
                                "the manifest's narrowest prefill rung (m={rm}) names no bundle \
                                 compiled into this binary — an inventory desync between the bake \
                                 and the manifest, refusing rather than running the top prefill \
                                 bundle under the narrow rung's width"
                            ))
                        })?),
                    (Some(tp), None) => Some(tp),
                    (None, _) => None,
                };
                //
                // ⛔ `None` REFUSES, IT DOES NOT FALL BACK. A capacity query that fails and quietly returns
                // to 8 GiB is the defect restored under a log line, and a *plausible* default is worse than
                // none — it sizes a real allocation the card may not serve.
                let card_bytes = scratchy_target_spyre::runtime::device_memory_bytes()
                    .ok_or_else(|| {
                        werr(
                            "could not read the Spyre card's memory capacity \
                             (flex_senlib_device_memory_size returned 0, i.e. no live device handle). The \
                             KV pool is sized from the real card, so there is no safe default to assume \
                             here. Check that the device is present and the flex runtime can start.",
                        )
                    })?;
                // WHAT THE BUNDLE ALREADY OWNS ON THE CARD, charged as `weights_and_overhead` — every
                // segment except the KV one, whose bytes are what we are about to size. The segments are
                // typed (`SEG_INTERMEDIATE`/`SEG_WEIGHT`/`SEG_KV`/`SEG_MASK`), so this reads the KV segment
                // out by name rather than by index arithmetic.
                // ⛔⛔⛔ AND THE WEIGHT SEGMENT'S EXTRA BANKS, WHICH `segment_bytes` DOES NOT COUNT.
                // `segment_bytes[SEG_WEIGHT]` is BANK 0's bytes; a banked bundle's remaining banks are
                // separate device regions of exactly the same kind — resident weights, allocated once
                // and aliased by every borrower. Leaving them out authorises a KV pool against memory
                // the weights already hold: 419 MB for granite-3.1-8b-fp16, and the size of half a
                // model for anything that needs banking to fit at all.
                //
                // Charged ONCE, not per session, precisely because a bank IS aliasable — that is the
                // difference between this and the spilled tail below.
                let bank_bytes: u64 = code.layout.weight_bank_bytes.iter().sum();
                let seg_bytes = code.layout.segment_bytes;
                let non_kv_bytes: u64 = bank_bytes
                    + seg_bytes
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != scratchy_target_spyre::superdsc_exec::SEG_KV.get())
                        .map(|(_, b)| *b)
                        .sum::<u64>();
                // ⛔⛔⛔ AND THE SPILLED WEIGHT TAIL IS CHARGED ONCE PER SESSION, NOT ONCE.
                //
                // `segment_bytes` describes ONE bundle, so summing it counts every segment a single
                // time. That is right for the segments a borrower ADOPTS ([`RESIDENT_SEGS`]: it holds a
                // 128 B placeholder and aliases the owner's region) and WRONG for the spill slot, which
                // cannot be aliased — its co-tenant intermediate colour is m-dependent, so every
                // borrowing session allocates the slot and takes its own COPY of the tail.
                //
                // 🛑 MEASURED on granite-3.1-8b fp16: 27 sessions × 419,430,400 B = 10.55 GiB of device
                // memory that this budget could not see, so it authorised a KV pool against ~10 GiB that
                // was already spoken for. Harmless at `--max-model-len 4096` (the pool wanted 1,980 MB
                // of a far larger budget) and a real OOM the moment the context grows. The tail bytes,
                // not the whole slot: the slot's intermediate colour is allocated per session with or
                // without a spill, so only the tail is new — and only for the sessions BEYOND the owner,
                // which `non_kv_bytes` has already counted once above.
                //
                // The session count is an UPPER BOUND from the declared rung lists rather than the
                // sessions that will actually be built (a rung may fail to build and be skipped, and the
                // ladder is constructed further down, after this budget). Over-charging shrinks the KV
                // pool; under-charging hands out memory that does not exist.
                let tail_bytes: u64 = spill_spans.iter().map(|(_, len)| *len as u64).sum();
                let borrowers = (g0.prefill_rungs.len() + g0.decode_rungs.len() + 1) as u64;
                let non_kv_bytes = non_kv_bytes + tail_bytes * borrowers;
                // ⛔⛔⛔ AND THE BATCHED-PREFILL ANCHOR'S OWN RESIDENCY — the one session the decode
                // bundle's `segment_bytes` cannot describe. The anchor is a DIFFERENT bundle (the
                // narrowest prefill rung): its intermediate/mask/logits segments are sized for m query
                // rows, it holds its own KV pool — an exact twin of the pool this budget is sizing —
                // and, when it RESERVES, a full spare copy of the weight home (SEG_WEIGHT + every
                // extra bank). Leaving all of that uncounted is what put gemma-4-26b under: the census
                // said 62,601 MB, the allocator was holding ~92 GB when decode's first extra bank asked
                // for 17.06 GB against 10.62 GB free, and the run died at `prepare: weight bank 1`
                // (2026-10-09, wall 9). The anchor's LIVE segments are charged unconditionally — its
                // build is fatal, so they are mandatory — and its SPARE only under the policy below.
                // The optional ladders (prefill rungs past the anchor, decode rungs, the prefix-capable
                // prefill) stay UNCHARGED on purpose: each is built soft — a failed build is a WARN and
                // a narrower rung, never a dead run — so charging them would refuse configs that run
                // fine degraded.
                let (anchor_live_bytes, anchor_spare_bytes) = match prefill_code {
                    Some(pc) => {
                        let weight = scratchy_target_spyre::superdsc_exec::SEG_WEIGHT.get();
                        let kv = scratchy_target_spyre::superdsc_exec::SEG_KV.get();
                        let live: u64 = pc
                            .layout
                            .segment_bytes
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != weight && *i != kv)
                            .map(|(_, b)| *b)
                            .sum();
                        let spare: u64 = pc.layout.segment_bytes[weight]
                            + pc.layout.weight_bank_bytes.iter().sum::<u64>();
                        (live, spare)
                    }
                    None => (0, 0),
                };
                // ⛔⛔⛔ AND THE LADDER THE SPARE HAS TO SURVIVE. Every borrowing session still
                // allocates its OWN regions for every segment EXCEPT the two it aliases (seg1/seg2
                // hold 128 B placeholders, and the banks follow seg1), and the prefill ladder
                // builds those on a background thread WHILE the decode session stages — so rung
                // segments land interleaved with decode's own banks. The pool budget above charges
                // none of this ON PURPOSE: a rung that cannot allocate is a WARN and a narrower
                // ladder, never a dead run. The RESERVE decision has no such license. The spare is
                // permanent placement, and granting it against headroom the ladder is about to
                // consume does not degrade the rungs — it kills the OWNER: the rung allocations
                // succeed, and decode's mandatory bank is the allocation that finds the card full.
                //
                // 🛑 MEASURED on gemma-4-31b fp8 (2026-10-14, 4096 ctx × 2 rows): the census said
                // 86,633 MB ≤ the 88,473 MB that --gpu-memory-utilization 0.90 funds, so the anchor
                // RESERVED; the ladder's rungs then took ~24,500 MB of real non-resident segments
                // while decode staged, and the run died at `prepare: weight bank 1 alloc of
                // 15622566400 B failed` with 2,196,278,528 B free — the same `weight bank 1` wall
                // gemma-4-26b hit at wall 9, one census layer further down. Charged here, the 31b
                // borrows and mandatory + ladder + pool fits the tensor domain with room to spare.
                //
                // Charged as the build will allocate it, so the count stays honest in both
                // directions: the entries the ladder loops themselves SKIP are not charged (`rm`
                // and `rn` ≤ 1 name sessions already built, and the narrowest `rm > 1` is the
                // anchor's own bundle — `prefill_code` above — whose live segments
                // `anchor_live_bytes` already charges; the skip is read off the entry list, not a
                // second derivation of `prefill_m`). The prefix-capable prefill and the decode
                // batch rungs build soft after the join and are charged for the same reason: their
                // allocations are real whenever they succeed.
                let seg_weight = scratchy_target_spyre::superdsc_exec::SEG_WEIGHT.get();
                let seg_kv = scratchy_target_spyre::superdsc_exec::SEG_KV.get();
                let non_resident =
                    |lay: &scratchy_target_spyre::bundle_code::BundleLayout<'static>| {
                        lay.segment_bytes
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != seg_weight && *i != seg_kv)
                            .map(|(_, b)| *b)
                            .sum::<u64>()
                    };
                let anchor_rung_m = g0
                    .prefill_rungs
                    .iter()
                    .filter(|(rm, _)| *rm > 1)
                    .min_by_key(|(rm, _)| *rm)
                    .map(|(rm, _)| *rm);
                let ladder_rung_bytes: u64 = g0
                    .prefill_rungs
                    .iter()
                    .filter(|(rm, _)| *rm > 1 && Some(*rm) != anchor_rung_m)
                    .filter_map(|(_, sent)| sent.strip_prefix(SUPERDSC_SENTINEL))
                    .filter_map(scratchy_target_spyre::bundle_code::bundle)
                    .map(|rc| non_resident(&rc.layout))
                    .sum::<u64>()
                    + g0.decode_rungs
                        .iter()
                        .filter(|(rn, _, _)| rn.get() > 1)
                        .filter_map(|(_, _, sent)| sent.as_str().strip_prefix(SUPERDSC_SENTINEL))
                        .filter_map(scratchy_target_spyre::bundle_code::bundle)
                        .map(|rc| non_resident(&rc.layout))
                        .sum::<u64>()
                    + g0.prefill_prefix
                        .iter()
                        .filter(|(pmq, _)| *pmq > 1)
                        .filter_map(|(_, psent)| psent.strip_prefix(SUPERDSC_SENTINEL))
                        .filter_map(scratchy_target_spyre::bundle_code::bundle)
                        .map(|rc| non_resident(&rc.layout))
                        .sum::<u64>();
                // ⛔ DEVICE REGIONS ARE NEVER FREED — `DevAddr`'s own doc: "device memory lives until
                // process exit". The anchor's spare is not a loan the alias later returns:
                // `alias_seg_from` repoints the borrower's descriptors and the borrower's own regions
                // stay on the allocator's books forever. So the spare is PLACEMENT LUXURY — its
                // allocation is what places decode's weight region, worth 1.8 tok/s on
                // granite-3.1-2b, measured — never a load requirement, and it is taken ONLY when the
                // card funds it: mandatory residency plus BOTH KV pools (decode's and the anchor's
                // twin) plus the spare plus the ladder's real rung segments must fit inside what
                // `--gpu-memory-utilization` allots. A model that fails that check BORROWS exactly
                // like every ladder rung — 128 B placeholders now, `adopt_from_owner` onto decode's
                // regions once the owner is built — costing the placement effect, not the run.
                let anchor_mandatory_bytes = non_kv_bytes + anchor_live_bytes;
                let anchor_reserves = prefill_code.is_some()
                    && (anchor_mandatory_bytes
                        + anchor_spare_bytes
                        + 2 * pool_need_bytes
                        + ladder_rung_bytes)
                        <= (card_bytes.tensor_capacity() as f64 * self.gpu_memory_utilization)
                            as u64;
                let non_kv_bytes = non_kv_bytes
                    + anchor_live_bytes
                    + if anchor_reserves {
                        anchor_spare_bytes + pool_need_bytes
                    } else {
                        0
                    };
                if prefill_code.is_some() {
                    tracing::info!(
                        "superdsc paged: prefill anchor {} — its {} MB spare (a twin of the weight \
                         home + a second {} MB KV pool) {} inside the {} MB that \
                         --gpu-memory-utilization {:.2} funds beyond the {} MB of mandatory \
                         residency plus the {} MB the ladder's rungs allocate",
                        if anchor_reserves {
                            "RESERVES"
                        } else {
                            "BORROWS"
                        },
                        anchor_spare_bytes / (1024 * 1024),
                        pool_need_bytes / (1024 * 1024),
                        if anchor_reserves {
                            "fits"
                        } else {
                            "does NOT fit"
                        },
                        (card_bytes.tensor_capacity() as f64 * self.gpu_memory_utilization) as u64
                            / (1024 * 1024),
                        self.gpu_memory_utilization,
                        anchor_mandatory_bytes / (1024 * 1024),
                        ladder_rung_bytes / (1024 * 1024),
                    );
                }
                // ⛔⛔⛔ THE ALLOCATABLE TENSOR CAPACITY, NOT THE CARD. `FlexAllocator` reserves one
                // of its seven equal regions for PROGRAM memory, so a tensor allocation is served
                // from six — and the budget below sizes exactly such an allocation. Spending
                // `card_bytes.bytes()` here over-stated the pool by a full region: the log said
                // `KV budget 86491 MB` while the allocator's own OOM for the next allocation
                // reported `total_capacity_bytes=103079215104` (96 GiB = 6/7 of 112 GiB). See
                // `CardMemory::tensor_capacity`.
                let pool_budget_bytes =
                    scratchy_serving_engine::gpu_budget::compute_available_kv_bytes(
                        card_bytes.tensor_capacity() as usize,
                        non_kv_bytes as usize,
                        // No separate activation profile on this backend: the intermediate segment IS the
                        // activation footprint and it is already counted in `non_kv_bytes` above. Passing it
                        // again here would charge it twice.
                        0,
                        self.gpu_memory_utilization,
                    ) as u64;
                // ⛔ AND THE POOL IS **ONE** ALLOCATION, so it cannot exceed one region however much
                // tensor capacity is left over. Without this cap the budget can authorise a pool the
                // allocator will refuse with a self-contradictory OOM (requested < free) — the same
                // failure mode as an over-sized weight segment, one layer up.
                let pool_budget_bytes = pool_budget_bytes.min(card_bytes.max_single_allocation());
                tracing::info!(
                    "superdsc paged: KV budget {} MB = {} MB allocatable tensor memory (card {} MB \
                     less the flex program region) × --gpu-memory-utilization {:.2} − {} MB already \
                     reserved by the sessions (the decode bundle's segments but KV + banks, plus the \
                     prefill anchor's {} MB live{}{}, incl. {} MB of spilled weight tail copied into \
                     {} borrowing session(s)) − 150 MB redundancy, capped at the {} MB a single \
                     allocation can be",
                    pool_budget_bytes / (1024 * 1024),
                    card_bytes.tensor_capacity() / (1024 * 1024),
                    card_bytes.bytes() / (1024 * 1024),
                    self.gpu_memory_utilization,
                    non_kv_bytes / (1024 * 1024),
                    anchor_live_bytes / (1024 * 1024),
                    if anchor_reserves {
                        format!(
                            " + {} MB spare (RESERVED: it places decode's weight region)",
                            (anchor_spare_bytes + pool_need_bytes) / (1024 * 1024)
                        )
                    } else if prefill_code.is_some() {
                        " (BORROWING: the spare did not fit)".to_string()
                    } else {
                        String::new()
                    },
                    if prefill_code.is_some() {
                        ""
                    } else {
                        " (no batched-prefill bundle)"
                    },
                    (tail_bytes * borrowers) / (1024 * 1024),
                    borrowers,
                    card_bytes.max_single_allocation() / (1024 * 1024),
                );
                if pool_budget_bytes == 0 {
                    return Err(werr(format!(
                        "the Spyre card reports {} MB of memory, but --gpu-memory-utilization {:.2} leaves \
                         nothing for the KV cache after the {} MB this model's sessions already reserve \
                         (the decode bundle's segments and banks, the prefill anchor's live segments{}, \
                         and the spilled weight tail's copies). Raise --gpu-memory-utilization, or use a \
                         smaller model.",
                        card_bytes.bytes() / (1024 * 1024),
                        self.gpu_memory_utilization,
                        non_kv_bytes / (1024 * 1024),
                        if anchor_reserves {
                            format!(
                                " + its {} MB reserved spare",
                                (anchor_spare_bytes + pool_need_bytes) / (1024 * 1024)
                            )
                        } else {
                            String::new()
                        },
                    )));
                }
                // Carry it to `Loaded` so the engine's own KV sizing reads this number and not a constant.
                kv_budget_bytes_v = Some(pool_budget_bytes);
                // And the stride beside it, so the engine's block-count arithmetic
                // (`kv_block_bytes` → `budget/stride`, clamped to the host range) divides by the
                // same page the pool was cut from — the two counts are then one count.
                pool_stride_bytes_v = Some(pool_stride_bytes);
                let num_blocks = {
                    // ⭐ THE POOL'S LAW LIVES ABOVE, ONCE — `pool_need_bytes` / `declared_pages` /
                    // `pool_stride_bytes` / `pool_demand` are the single derivation (hoisted there
                    // because the census and the anchor's reserve policy need them first). This
                    // block holds the CONTEXT LAW's one application to the pool:
                    // [`resolve_served_context`] over (the CLI's `--max-model-len`, the model's
                    // own limit, what this budget funds per request), the pool re-derived for
                    // whatever it served, and the log that states what was decided. The law's
                    // boundary is the old veto's boundary exactly — an explicit `--max-model-len`
                    // over what the budget funds is REFUSED with the advice below — and the one
                    // behavior it adds is the unset flag: the model's own limit resolves DOWN to
                    // what the budget funds instead of refusing, at info, because a default sized
                    // by the card is a decision, not a broken ask.
                    let affordable_ctx =
                        affordable_context_tokens(pool_budget_bytes, pool_stride_bytes, pool_rows);
                    match scratchy_serving_engine::gpu_budget::resolve_served_context(
                        self.declared_context,
                        max_model_len,
                        affordable_ctx as usize,
                    ) {
                        scratchy_serving_engine::gpu_budget::ServedContext::Refused {
                            declared,
                            affordable,
                        } => {
                            return Err(werr(format!(
                                "--max-model-len {declared} × --max-num-seqs {} needs {} KV page(s) \
                                 ({} MB at {} MB per page), but the KV budget is {} MB ({} page(s)). This run \
                                 would fail a request mid-generation once the pool ran out, so it is refused \
                                 here instead. Lower --max-model-len to about {affordable} or --max-num-seqs \
                                 to {}.",
                                pool_rows.get(),
                                declared_pages.get(),
                                pool_need_bytes / (1024 * 1024),
                                pool_stride_bytes / (1024 * 1024),
                                pool_budget_bytes / (1024 * 1024),
                                pool_budget_bytes / pool_stride_bytes,
                                // What the budget funds as a WIDTH, stated as the flag the caller
                                // typed: one page per row past the reserve, at the declaration's
                                // own per-row page count.
                                (pool_budget_bytes / pool_stride_bytes).saturating_sub(1)
                                    / pool_demand.per_row().get() as u64,
                            )));
                        }
                        scratchy_serving_engine::gpu_budget::ServedContext::Tokens(served) => {
                            // ⭐ AN UNSET DEFAULT RESOLVES HERE, NEVER ABOVE: the census and the
                            // anchor's reserve policy ran against the DECLARED depth (the model's
                            // own limit when no flag was given), because the budget that funds the
                            // resolution is only known once they have. The resolved pool is
                            // veto-proof by construction — `affordable_context_tokens` is the
                            // exact inverse of the demand arithmetic, so the re-derived
                            // `pool_need_bytes` fits the budget it was resolved from — and the
                            // anchor keeps the BORROW decision the bigger declared pool forced,
                            // which is the conservative direction (borrowing costs the placement
                            // effect, never the run).
                            if served < declared_ctx {
                                info!(
                                    "max_model_len unset: serving {served} of the model's \
                                     {declared_ctx} — what the {} MB KV budget funds across {} \
                                     launch row(s) at {} MB per page",
                                    pool_budget_bytes / (1024 * 1024),
                                    pool_rows.get(),
                                    pool_stride_bytes / (1024 * 1024),
                                );
                                declared_ctx = served;
                                pool_demand =
                                    scratchy_subtile::sdsc_abstract::PoolDemand::of_declaration(
                                        scratchy_subtile::sdsc_abstract::SlotCount::new(
                                            served as u32,
                                        ),
                                        pool_rows,
                                    );
                                declared_pages = pool_demand.pages().ok_or_else(|| {
                                    werr(format!("--max-model-len {served} overflows a page count"))
                                })?;
                                pool_need_bytes = declared_pages.get() as u64 * pool_stride_bytes;
                            }
                            // What the run SERVES, never `usize::MAX` on a sized pool, so
                            // `kv_max_addressable_tokens` reports this run's own number and the
                            // engine's admission cap agrees with the pool by construction.
                            served_context_v = served;
                            info!(
                                "superdsc paged: KV pool {} page(s) = --max-model-len {declared_ctx} ({} page(s) per \
                                 request) × {} launch row(s) + reserve — {} MB of a {} MB budget at {} MB per page",
                                declared_pages.get(),
                                pool_demand.per_row().get(),
                                pool_rows.get(),
                                pool_need_bytes / (1024 * 1024),
                                pool_budget_bytes / (1024 * 1024),
                                pool_stride_bytes / (1024 * 1024),
                            );
                            declared_pages.get() as usize
                        }
                    }
                };
                if let Some(sp) = PagedKvPool::split_pool(num_blocks as u32, pool_rows) {
                    // ⛔ THIS LINE HAS NOW BEEN WRONG TWICE, AND BOTH TIMES BECAUSE IT DESCRIBED A
                    // MECHANISM INSTEAD OF PRINTING THE NUMBERS. First it advertised an
                    // `SUPERDSC_POOL_ROWS` override that had been deleted; then, an hour later, it said the
                    // width was "fixed by the widest baked decode rung" — which I had just stopped being
                    // true in the commit above. A log that explains its own derivation goes stale every time
                    // the derivation moves. So it states the three FACTS a reader needs, and each is read
                    // from the value that decided it.
                    let host = PoolPartition::of_pool(
                        PoolPages::of_pool(num_blocks).expect("a nonzero pool"),
                        pool_rows,
                    )
                    .map_or(0, |p| p.host_blocks());
                    tracing::info!(
                        "superdsc paged: KV pool {} page(s) — {} offered to the SCHEDULER's block \
                         allocator, {} reserved for the masked holes of {} launch row(s). One request may \
                         use all of the offered pages and two may share one (prefix caching).",
                        num_blocks,
                        host,
                        num_blocks as u32 - host,
                        sp.rows().get(),
                    );
                }
                // What the DECODE bake placed — one call, and its own refusal.
                let df = scratchy_target_spyre::wiring::BakeFacts::of(&code.layout);
                df.require_identity(&code.fp).map_err(werr)?;
                // ⭐ THE CLASS SETS vs THE PLACEMENTS — the load-time cross-check that closes the
                // "emu green, card garbage" gap for the kernel-table class: the emulator executes
                // the KTIR dialect ops and never reads these tids, so nothing on its path can see
                // a wiring/layout desync. Here, every rope-P and identity class tid must be placed
                // at exactly its `hd²·2` footprint, and every placed class tid must be in the
                // wiring's set — the gemma-4 defect (a base-hd table in a max-hd placement) is
                // unrepresentable once this runs.
                wirings
                    .decode
                    .verify_class_placements(&code.layout)
                    .map_err(werr)?;
                // ⭐ THE CAUSAL MASK'S ROW EXTENT vs ITS PLACEMENT — the load-time cross-check for
                // the `nqh` bake. The host stages `causal_tiled(cmask, mq, mq_pad, nqh)` rows into
                // `ATTN_CAUSAL_TID`'s `nqh·mq·mq_pad`-element placement, and the head count those
                // two halves share is `Geometry::nqh`. The host used to derive it as
                // `hidden / head_dim` — right only for a square Q projection, wrong on gemma-4,
                // and the shortfall was SILENT (the bind guard refuses over-binds only) while the
                // unstaged tail of the additive mask read as "no mask". This makes any head count
                // other than the bake's a load-time refusal. At the decode row count `mq = 1`,
                // where the pad law gives every baked rung the one-stick score width 64.
                wirings
                    .decode
                    .verify_causal_placement(&code.layout, 1, 64)
                    .map_err(werr)?;
                // ⭐ AND THE TAPE IS CHECKED AGAINST THE ARTIFACT, ONCE, HERE. Every tensor the
                // forward would bind must be one this bundle placed. They come from the same bake
                // and so agree "by construction" — which is exactly what was also true of the
                // K-split merge-init zero: a bind for a tid the emitter names nowhere, matching no
                // placement, reported by nothing for as long as it existed.
                let n_consts =
                    scratchy_target_spyre::wiring::synthetic_constants(&df.constant_env(
                        &wirings.decode,
                        64,
                        scratchy_target_spyre::wiring::StagedRows::ONE,
                    ))
                    .len();
                let unplaced = wirings
                    .decode
                    // One row: the load-time check only asks WHICH tensors the tape names, and
                    // that set does not vary with the row count.
                    //
                    // ⭐ `true` IS DELIBERATE AND IS NOT THE LAUNCH'S ANSWER. Every other caller
                    // derives this from `prefix_src.is_some()` — the step exists iff the source
                    // does. Here the question is the opposite one: what is the WIDEST set of
                    // tensors any launch of this bundle could bind, so that a bake missing one is
                    // caught now rather than on the card. Asking with `false` would let an
                    // unplaced prefix mask through.
                    .forward_shape(
                        scratchy_target_spyre::wiring::StagedRows::ONE,
                        true,
                        // ⭐ AND THE GATHER'S ANSWER IS THIS BUNDLE'S OWN, from the layout being checked
                        // — not `true` like the mask above. The mask's `true` asks for the WIDEST tensor
                        // set because a launch may or may not stage a mask; a gather is not optional
                        // that way. The index step exists iff this bake PLACED the index tensor, so
                        // asking `true` here would report `KV_BLOCK_INDEX_TID` unplaced for every
                        // non-gathering bundle and refuse to serve every model.
                        scratchy_target_spyre::wiring::GathersKv::of_layout(&code.layout),
                        n_consts,
                    )
                    .unplaced(&code.layout);
                if !unplaced.is_empty() {
                    return Err(werr(format!(
                        "superdsc decode bundle {}: the forward tape binds {} tensor(s) this bake \
                         never placed ({}). The wiring and the layout come from ONE bake and have \
                         desynced — refusing to serve rather than skipping the binds.",
                        code.fp,
                        unplaced.len(),
                        unplaced
                            .iter()
                            .map(|id| id.to_string())
                            .collect::<Vec<_>>()
                            .join(", "),
                    )));
                }
                decode_facts = df;
                // ── PREFILL-ALL-BUT-LAST: the DISTINCT m=N prefill dxp bundle (C4 stamps its own fp
                //    in the prefill graph slot; lm_head skipped). Build a SECOND SuperDscSession from
                //    it (binding the SAME weights again — 2× resident, the spike cost) so the prompt
                //    runs batched through it, then its seg2 KV is copied to the decode session. If the
                //    prefill slot's fp == the decode fp (the prefill reroll did NOT bake → C4 fell
                //    back), prefill=None ⇒ the worker uses the sequential mq=1 path. ──
                // `prefill_code` (the anchor bundle), `anchor_reserves` (its capacity policy) and the
                // KV pool's own demand were all derived UP AT THE CENSUS — the budget reads them, so
                // they exist before this point. The session build below is their first consumer.
                let (mut prefill_ss, prefill_m) = match prefill_code {
                    Some(pcode) => {
                        // The baked prefill m (query rows). run_step MUST use exactly this m.
                        //
                        // ⭐⭐⭐⭐⭐ FROM THE MANIFEST'S OWN RUNG ENTRY, NOT from the embed
                        // activation's placement bytes. The old derivation was
                        // `place_of_tid(embed).size / (hidden·2)`, and BOTH of its halves were
                        // wrong on their own:
                        //
                        // 1. THE TID. It read `dparsed.embed_src` — the DECODE program's embed
                        //    tid — against the PREFILL bundle's layout, but tensor ids are PER
                        //    PROGRAM (codegen: "the prefill graph numbers its own"), so on
                        //    gemma-4's allglobal tiny fixture it named the attention Q activation
                        //    `[mq, nqh·hd]` and the division answered `mq·nqh·hd/hidden` = 4× the
                        //    real width (7 read as 28).
                        // 2. THE DIVISOR. Even at the right tid, an activation's placement
                        //    reserves the DEVICE width, `bump_sticks_to_splittable(hidden)` —
                        //    the ≥8-core occupancy pad (`lower_subtile_tape_to_superdsc.rs`'s
                        //    `nbytes`) — so `size / (hidden·2)` is right only while hidden's own
                        //    stick count is already core-splittable. granite's 2048 (32 sticks)
                        //    is; gemma-4's 3840 (60 sticks → padded to 4096) is NOT, and the
                        //    tiny fixture's 128 (2 sticks → padded to 512) is not — the quotient
                        //    overstates the row count by `bump/hidden` (16/15 at production
                        //    scale, 4× on the fixture).
                        //
                        // The mismeasured width went straight into the prefill LADDER as the
                        // anchor rung's label: on the tiny fixture the 7-row bundle registered as
                        // "28", sorted above the 11/15/… rungs, and an 8-token prompt — the
                        // fixture's own request — selected it by `prefill_rung_for`'s
                        // smallest-m≥real rule, then ran its attention ops past placements sized
                        // for 7 rows. The manifest's `prefill_rungs` rows already state each
                        // bundle's width at the bake (the same list this anchor was selected
                        // FROM), so the label is read back off the entry the anchor came from —
                        // no second derivation to disagree with the first.
                        let pl = &pcode.layout;
                        // The anchor is `top_prefill`-gated: either the manifest's NARROWEST rung
                        // (the normal path — its width is that entry's own `rm`, the same value
                        // the selection just keyed on) or the top prefill bundle itself (no
                        // ladder — its width is the prefill wiring's own `m_cap`, the shape the
                        // embed activation was staged at, from the wiring and not from the
                        // placement bytes).
                        let prefill_m = g0
                            .prefill_rungs
                            .iter()
                            .filter(|(rm, _)| *rm > 1)
                            .min_by_key(|(rm, _)| *rm)
                            .map_or_else(
                                || {
                                    wirings.prefill.as_ref().map_or(0, |p| {
                                        p.tensor_shapes[p.embed_src as usize].0 as usize
                                    })
                                },
                                |(rm, _)| *rm as usize,
                            );
                        // What the PREFILL bake placed. From the bundle, never from
                        // `SCRATCHY_RMS_MATMUL_REDUCE`, so a bundle-vs-env mismatch cannot leave a
                        // matmul reading an unbound (zero) seg0.
                        prefill_facts = scratchy_target_spyre::wiring::BakeFacts::of(pl);
                        // The prefill bake's class sets against ITS placements — the same
                        // cross-check the decode bake just passed, for the same reason.
                        wirings
                            .prefill
                            .as_ref()
                            .unwrap_or(&wirings.decode)
                            .verify_class_placements(pl)
                            .map_err(werr)?;
                        // ⭐ AND THE CAUSAL MASK against the PREFILL bake, at its own row count:
                        // the placement reserves `nqh·prefill_m·pad` elements and the host stages
                        // that many, so a head count other than the bake's (`Geometry::nqh`) is a
                        // refusal here. The pad is the same `div_ceil(64)·64` law the emitter and
                        // `run_prefill_batch` both apply — `PaddedMq::of_bundle` at a prompt
                        // chunk's width.
                        wirings
                            .prefill
                            .as_ref()
                            .unwrap_or(&wirings.decode)
                            .verify_causal_placement(
                                pl,
                                prefill_m as u32,
                                (prefill_m as u32).div_ceil(64) * 64,
                            )
                            .map_err(werr)?;
                        // OWNS ITS WEIGHTS ON PURPOSE — do NOT switch this to `new_borrowing`.
                        // This session is allocated BEFORE the decode session, so its 2.6 GB seg1
                        // reservation is what places decode's own weight region. Making it borrow (a
                        // 128 B placeholder) moved decode's weights ~2.6 GB earlier in HBM and cost
                        // ~1.8 tok/s of DECODE throughput, measured: 36.5 tok/s with this session
                        // owning its weights vs 34.7 with it borrowing. Decode is weight-BANDWIDTH-
                        // bound (2.43 GB per token at ~84 GB/s), so where that region lands is
                        // load-bearing. The effect is NON-LINEAR in session count — 1→2 sessions cost
                        // 1.8 tok/s, 2→7 cost 0.1 — which is the signature of the FIRST allocation
                        // ahead of decode mattering, not of per-session overhead.
                        // The ladder rungs below still borrow, so the ladder stays affordable: this
                        // costs ONE spare weight copy (exactly what the worker held before the ladder
                        // existed), not one per rung.
                        // `pdir` is the NARROWEST rung (see the anchor note above), so this
                        // reservation is the same size at every ladder ceiling.
                        let _t = std::time::Instant::now();
                        // RESERVES seg1 at full size without populating it. The comment above is
                        // still the reason this session must not BORROW: its 2.6 GB reservation places
                        // decode's weight region, worth 1.8 tok/s. But that is the ALLOCATION talking,
                        // not the bytes -- this session is aliased onto decode's copy a few lines
                        // below, so everything staged here was overwritten by a pointer swap. Keeping
                        // the reservation and dropping the write saves the bind + stage (6.97s
                        // measured) and the `weights.clone()` of a ~2.6 GB host buffer with it.
                        //
                        // ⛔ BUT THE RESERVATION IS LUXURY, AND LUXURY IS BUDGETED (wall 9,
                        // 2026-10-09). Device regions are NEVER freed, so the anchor's spare — a
                        // twin of SEG_WEIGHT, every extra bank, and a second KV pool — is permanent
                        // dead placement, and at gemma-4-26b scale (~26 GB of spare on a 96 GiB
                        // tensor domain) it starved decode's own banks: `prepare: weight bank 1
                        // alloc of 17057081344 B failed` with 10.6 GB free. `anchor_reserves`
                        // (decided at the census, against the card's funded capacity) keeps the
                        // measured granite behavior exactly where the card pays for it, and takes
                        // the BORROWING form — identical to every ladder rung: placeholders here,
                        // `adopt_from_owner` onto decode's regions after the owner is built — when
                        // it does not. What that costs is the placement effect, never the run.
                        let pss = if anchor_reserves {
                            SuperDscSession::new_reserving(
                                pcode,
                                vocab,
                                kv_dim,
                                num_blocks,
                                wirings
                                    .prefill
                                    .as_ref()
                                    .unwrap_or(&wirings.decode)
                                    .num_sources,
                                &resident_source_ids(
                                    wirings.prefill.as_ref().unwrap_or(&wirings.decode),
                                    &spill_tids,
                                ),
                            )
                        } else {
                            SuperDscSession::new_borrowing(
                                pcode,
                                vocab,
                                kv_dim,
                                num_blocks,
                                wirings
                                    .prefill
                                    .as_ref()
                                    .unwrap_or(&wirings.decode)
                                    .num_sources,
                                &resident_source_ids(
                                    wirings.prefill.as_ref().unwrap_or(&wirings.decode),
                                    &spill_tids,
                                ),
                                &RESIDENT_SEGS.map(|(seg, _)| seg),
                            )
                        }
                        .map_err(|e| werr(format!("build SuperDSC PREFILL dxp session: {e}")))?;
                        info!(
                            "[spyre-worker] sendnn: SUPERDSC PREFILL session ready (bundle {}, \
                             m={prefill_m}, cap {} slots, paged={} page_slots={}, pmask reserves \
                             {:?} column(s), {:.2}s) — batched prompt prefill enabled",
                            pcode.fp,
                            pss.cap,
                            pss.paged,
                            pss.page_slots,
                            pss.pmask_slots,
                            _t.elapsed().as_secs_f64()
                        );
                        (Some(pss).filter(|_| prefill_m > 1), prefill_m)
                    }
                    None => {
                        info!(
                            "[spyre-worker] sendnn: no batched SUPERDSC prefill bundle — multi-token \
                             prompts use SEQUENTIAL prefill (token-by-token through the decode bundle's \
                             resident KV)"
                        );
                        (None, 0)
                    }
                };
                // LADDER, OVERLAPPED. Building the 20 narrow rungs costs 6.05s (302 ms each,
                // measured) and needs no weights — each one only creates and prepares, then borrows
                // seg1/seg2 by alias. So it runs on a background thread while THIS thread stages the
                // decode session's 8.9 GB, and the cost disappears into work that was happening anyway.
                //
                // Deferring these to the request path instead moved the same 6.05s into TTFT
                // (120ms -> 312ms measured): the first token comes OUT of prefill, so anything built
                // from the prefill path is inside TTFT wherever in the chunk loop it sits. Overlapping
                // at startup is what makes it free rather than merely moved.
                //
                // The thread only CREATES sessions (`SuperDscSession` is `Send`); the `alias_seg_from`
                // calls stay on this thread after the join, because they need `&ss` and are a pointer
                // swap, not work.
                let ladder: Vec<(
                    usize,
                    &'static scratchy_target_spyre::bundle_code::BundleCode<'static>,
                )> = g0
                    .prefill_rungs
                    .iter()
                    .filter_map(|(rung_m, sentinel)| {
                        let rm = *rung_m as usize;
                        if rm == prefill_m || rm <= 1 {
                            return None;
                        }
                        sentinel
                            .strip_prefix(SUPERDSC_SENTINEL)
                            .and_then(scratchy_target_spyre::bundle_code::bundle)
                            .map(|c| (rm, c))
                    })
                    .collect();
                let _t_ladder = std::time::Instant::now();
                // FANNED OUT. One thread building 20 rungs serially is ~6s, and that cost is per
                // BUNDLE (re-roll + ~1558 placements), not per weight byte — so it is the SAME for
                // every model while the staging it hides scales with the weights. granite-3.1-8b
                // stages ~6s and hides it completely; granite-3.1-2b stages ~1.7s and does not, which
                // is why 2b's startup barely moved while 8b's fell by two thirds. Splitting the rungs
                // across threads removes the cost instead of hiding it.
                //
                // The rungs are independent: each is a `new_borrowing` on its own bundle dir with no
                // shared state, and every `alias_seg_from` still happens later on the owning thread.
                // Capped at the smaller of the rung count and the core count, so a short ladder does
                // not spawn threads it cannot use.
                // ⛔⛔⛔ THE REGION seg1 LANDS IN DECIDES THE DECODE MODE, AND THAT ORDER IS
                // WORTH 1 ms PER TOKEN.
                //
                // MEASURED 2026-08-14, granite-3.1-2b fp8, 16 runs of one prompt: mean ITL is BIMODAL —
                // 25.4-25.7 ms or 26.4-26.7 ms, a 1.0 ms gap with NOTHING between, and the whole per-token
                // histogram shifts rather than a few tokens being slow. `SCRATCHY_SDSC_SEGADDR_DIAG` correlated
                // it 16 for 16 with one thing: which device region seg1 — the WEIGHTS, read by every layer on
                // every token — landed in.
                //     seg1 in region 0x04…  ->  7 runs, ALL 25.4-25.7 ms
                //     seg1 in 0x10…/0x14…   ->  9 runs, ALL 26.4-26.7 ms
                // seg2 (the KV pool) showed NO correlation — region 0x14 appears in both modes. Decode is
                // bandwidth-bound on the weights, so where they land is the whole difference.
                //
                // The ladder used to be spawned FIRST, so this session's ~2.6 GB seg1 raced twenty rung
                // allocations across ~18 threads for whichever region the allocator hands out first — and the
                // session that serves EVERY TOKEN lost that race 9 times in 16. Allocating it first makes the
                // fast placement the deterministic one. The rungs ALIAS seg1 from it anyway
                // (`alias_seg_from`, after the join), so they never wanted a region of their own.
                //
                // ⛔ THE LADDER STILL BUILDS IN PARALLEL. This reorders WHO ALLOCATES FIRST; it does not
                // serialize startup, and the aliasing after the join is unchanged.
                let ladder_join = {
                    let nthreads = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4)
                        .min(ladder.len().max(1));
                    let chunk = ladder.len().div_ceil(nthreads.max(1));
                    let chunks: Vec<
                        Vec<(
                            usize,
                            &'static scratchy_target_spyre::bundle_code::BundleCode<'static>,
                        )>,
                    > = if chunk == 0 {
                        Vec::new()
                    } else {
                        ladder.chunks(chunk).map(|c| c.to_vec()).collect()
                    };
                    // Owned copy for the background build: the spilled tail's ids are declared
                    // resident by every session, and these rungs are built off this thread.
                    let ladder_spill = spill_tids.clone();
                    std::thread::spawn(move || {
                        let workers: Vec<_> = chunks
                            .into_iter()
                            .map(|part| {
                                let rung_spill = ladder_spill.clone();
                                std::thread::spawn(move || {
                                    part.into_iter()
                                        .map(|(rm, rcode)| {
                                            let r = SuperDscSession::new_borrowing(
                                                rcode,
                                                vocab,
                                                kv_dim,
                                                num_blocks,
                                                wirings.decode.num_sources,
                                                &resident_source_ids(&wirings.decode, &rung_spill),
                                                &[1, 2],
                                            );
                                            (rm, r)
                                        })
                                        .collect::<Vec<_>>()
                                })
                            })
                            .collect();
                        workers
                            .into_iter()
                            .flat_map(|w| w.join().unwrap_or_default())
                            .collect::<Vec<_>>()
                    })
                };

                let _t_sess = std::time::Instant::now();
                // The DECODE session is the sole OWNER of the resident weights — it is the only one that
                // binds and stages them; the batched-prefill session and every ladder rung borrow seg1
                // from it by alias. So this takes `weights` by MOVE and no copy of that ~2.6 GB host
                // buffer is ever made.
                let mut ss = SuperDscSession::new(
                    code,
                    vocab,
                    kv_dim,
                    num_blocks,
                    wirings.decode.num_sources,
                    &resident_source_ids(&wirings.decode, &spill_tids),
                    weights,
                )
                .map_err(|e| werr(format!("build SuperDSC dxp session: {e}")))?;
                // ⭐ THE SPILLED WEIGHT TAIL'S DEVICE IMAGE, read back from the OWNER once its
                // prepare has staged and uploaded it. Every borrowing session gets a copy of this
                // (see `adopt_from_owner`) because the slot it lives in cannot be aliased. Empty —
                // and every line below a no-op — for every model whose weights fit one segment.
                let mut spill_images: Vec<(i64, Vec<u8>)> = Vec::new();
                for (seg, len) in spill_spans {
                    let mut image = ss.read_seg(seg).map_err(|e| {
                        werr(format!(
                            "reading the DECODE session's seg{seg} (the spilled weight tail, which \
                             every borrowing session needs its own copy of): {e}"
                        ))
                    })?;
                    // A short read is a wiring bug, not something to copy a truncated weight from:
                    // the borrower would then hold a HALF tail and produce plausible-looking wrong
                    // logits. The tail spans [0, len) by construction (`spill_weight_tail`).
                    if image.len() < len {
                        return Err(werr(format!(
                            "the DECODE session's seg{seg} read back only {} B, but its spilled \
                             weight tail spans {len} B — refusing to hand the borrowing sessions a \
                             truncated lm_head",
                            image.len()
                        )));
                    }
                    image.truncate(len);
                    info!(
                        "[spyre-worker] sendnn: SEGMENT BUDGET: seg{seg} holds a {len} B spilled \
                         weight tail; captured its device image for the borrowing sessions, which \
                         cannot alias that segment"
                    );
                    spill_images.push((seg, image));
                }

                // ZERO THE KV POOL ONCE. A page handed to a request holds whatever was in pool
                // memory, and decode attention computes q·k over the WHOLE page and applies the
                // validity mask AFTERWARDS — so every uninitialized slot still goes through the
                // arithmetic. Uninitialized bytes decode as denormals/NaN, which take the slow paths.
                //
                // At a 128-token context ~70 of a 256-slot page are real and ~186 are garbage, every
                // page, every layer, every step. Prefill never sees this (it attends only its own
                // real tokens), which is why prefill puts 8 rows through the weights in 30.4 ms where
                // decode takes 47.0 for the same 8.
                //
                // Priced accidentally: two changes that left the KV wrong — one writing to bad
                // addresses, one not writing it at all — BOTH landed at ~4.55 s/it against 3.65
                // correct. That is ~0.9 s/it for garbage floats, on top of whatever the normally-
                // uninitialised tail of each page already costs.
                //
                // Zeros are the right fill, not merely a safe one: 0.0 is a normal float, and a
                // masked-off slot contributes exp(-inf)=0 either way, so this changes no result.
                ss.zero_seg(2).map_err(|e| {
                    werr(format!(
                        "zeroing the KV pool (seg2) on-device: {e} — refusing to serve over \
                         uninitialized KV pages"
                    ))
                })?;
                info!("[spyre-worker] sendnn: KV pool zeroed on-device (no host mirror)");
                debug!(
                    "[timing] SuperDscSession::new (bind weights + prepare) took {:.2}s",
                    _t_sess.elapsed().as_secs_f64()
                );
                info!(
                    "[spyre-worker] sendnn: SUPERDSC dxp session ready (fp={fp}, vocab {vocab}, \
                     kv_dim {kv_dim}, num_blocks {num_blocks}, cap {} slots, paged={} \
                     page_slots={}, pmask reserves {:?} column(s))",
                    ss.cap, ss.paged, ss.page_slots, ss.pmask_slots,
                );
                // ── SHARE seg2 (the resident KV) between the PREFILL and DECODE sessions ──
                // Both bundles place seg2 identically by construction (kc/vc at nqh·hd·cap·2, the
                // resident kct at nkvh·hd·cap·2 — no term depends on the query-row count m), so
                // batched prefill can write the prompt's KV DIRECTLY into the region decode reads,
                // instead of handing it over with a whole-segment read_seg→write_seg round-trip.
                // That round-trip was 188.7 MB of DMA plus two 90 MiB host memcpys and a 90 MiB
                // zeroed allocation, all serialized around a synchronize() — the single largest
                // item in TTFT (~125–133 ms) and independent of prompt length.
                // Only the PREFILL session's descriptor is repointed; the decode session is not
                // touched, so its addresses, transfer lengths and launched binaries are unchanged.
                // Done AFTER both sessions' prepare (SuperDscSession::new runs it), since prepare
                // zero-inits and H2Ds every segment and would otherwise wipe the shared region.
                // The shim hard-checks full placement equality and refuses loudly on any mismatch.
                // ── ALSO SHARE seg1 (the resident WEIGHTS) ──
                // Each session's `prepare` H2Ds its OWN full copy of seg1, so prefill+decode held the
                // model's weights TWICE on-card: 2,644,627,456 B (642 tensors) duplicated. Nothing in a
                // weight's placement depends on the query-row count `m`, so the two seg1 layouts are
                // byte-identical — VERIFIED against the baked `bundle_layout.json` of the decode body and
                // of every prefill width (63/47/31/15): same 642 tensors, same (offset,size) for all of
                // them, zero differing entries. Weights are read-only in both phases, so the borrower can
                // never write through the alias. The shim re-checks placement equality per tensor in BOTH
                // directions and refuses loudly rather than aliasing a mismatched layout, so a future
                // divergence surfaces at startup instead of as silent garbage.
                // This is also what makes the prefill WIDTH LADDER affordable: one session per rung would
                // otherwise cost another 2.6 GB each.
                // Ordering: both sessions' `prepare` has already run (decode's LAST), so the region the
                // borrower adopts holds freshly-written weights. Aliasing does NOT save prefill's initial
                // H2D — that already happened during its own `prepare` — and it does NOT free anything
                // either: device regions live until process exit (`DevAddr`'s doc), so when the anchor
                // RESERVED, its own regions stay on the allocator's books as the dead placement the
                // census charged; when it BORROWED it only ever held 128 B placeholders, and there was
                // never a second live copy to avoid.
                if let Some(pf) = prefill_ss.as_mut() {
                    adopt_from_owner(pf, &ss, &spill_images).map_err(|e| {
                        werr(format!(
                            "the PREFILL session could not take the DECODE session's resident \
                             state: {e}"
                        ))
                    })?;
                    info!(
                        "[spyre-worker] sendnn: PREFILL/DECODE now SHARE seg1 (weights) + seg2 \
                         (resident KV) — per-prompt KV handoff eliminated"
                    );
                }
                // Rung builds are the last unmeasured block of startup: each rung's `prepare` is timed
                // but its `create` (re-roll + ~1558 placements) is not, so the ladder's cost has only
                // ever been a residual. Declared HERE, in the same scope as the report below — inside
                // the ladder block they are out of scope by the time it runs.
                // Every rung that ends up serving, ascending by width. The top rung is appended
                // after the join; the narrow ones arrive from the background thread.
                let mut prefill_ladder: Vec<(usize, SuperDscSession)> = Vec::new();
                let mut _n_ladder = 0usize;
                // JOIN: the background creates are done (or finishing); alias each onto the decode
                // session's regions here, on the thread that owns `ss`.
                for (rm, built) in ladder_join.join().unwrap_or_default() {
                    match built {
                        Ok(mut rs) => {
                            let mut ok = true;
                            if let Err(e) = adopt_from_owner(&mut rs, &ss, &spill_images) {
                                warn!(
                                    "[spyre-worker] sendnn: prefill ladder rung m={rm}: {e} — \
                                     SKIPPED (a wider rung covers this width)"
                                );
                                ok = false;
                            }
                            if ok {
                                _n_ladder += 1;
                                prefill_ladder.push((rm, rs));
                            }
                        }
                        Err(e) => warn!(
                            "[spyre-worker] sendnn: prefill ladder rung m={rm}: create failed: {e} \
                             — SKIPPED (a wider rung covers this width)"
                        ),
                    }
                }
                // The TOP rung is `prefill_ss` itself — built and aliased above, never on the
                // background thread, because its allocation is what places decode's weight region.
                if let Some(top) = prefill_ss {
                    prefill_ladder.push((prefill_m, top));
                }
                prefill_ladder.sort_by_key(|(m, _)| *m);
                debug!(
                    "[timing] prefill ladder: {} rung(s) ready in {:.2}s wall — built in parallel on \
                     background threads while the decode session staged",
                    _n_ladder,
                    _t_ladder.elapsed().as_secs_f64(),
                );
                info!(
                    "[spyre-worker] sendnn: PREFILL ladder = {:?} (chunk runs on the smallest rung \
                     that holds it; ceiling {} rows/chunk)",
                    prefill_ladder.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
                    prefill_ladder.last().map(|(m, _)| *m).unwrap_or(0),
                );
                // ── PREFIX-CAPABLE PREFILL LADDER ── the bundles a continuation chunk (start>0) runs
                // on. Every ladder rung above is prefix-FREE, so these are the ONLY prefill bundles
                // that can attend resident KV. Borrows seg1+seg2 like the rungs, and is built AFTER
                // the decode session so it cannot affect decode's weight-region placement. A rung that
                // fails to build is logged and skipped: its chunks run on a wider rung, more padded.
                let mut prefix_built: Vec<(usize, SuperDscSession)> = Vec::new();
                for &(pmq, psent) in g0.prefill_prefix {
                    if pmq <= 1 || psent.is_empty() {
                        continue;
                    }
                    match psent
                        .strip_prefix(SUPERDSC_SENTINEL)
                        .and_then(scratchy_target_spyre::bundle_code::bundle)
                    {
                        Some(pcode2) => match SuperDscSession::new_borrowing(
                            pcode2,
                            vocab,
                            kv_dim,
                            num_blocks,
                            wirings
                                .prefill
                                .as_ref()
                                .unwrap_or(&wirings.decode)
                                .num_sources,
                            &resident_source_ids(
                                wirings.prefill.as_ref().unwrap_or(&wirings.decode),
                                &spill_tids,
                            ),
                            &[1, 2],
                        ) {
                            Ok(mut ps) => match adopt_from_owner(&mut ps, &ss, &spill_images) {
                                Ok(()) => {
                                    info!(
                                        "[spyre-worker] sendnn: prefix-capable prefill session \
                                         ready (m={pmq}, bundle {}) — used for start>0 chunks",
                                        pcode2.fp
                                    );
                                    prefix_built.push((pmq as usize, ps));
                                }
                                Err(e) => warn!(
                                    "[spyre-worker] sendnn: prefix-capable prefill m={pmq}: {e} — \
                                     SKIPPED"
                                ),
                            },
                            Err(e) => warn!(
                                "[spyre-worker] sendnn: prefix-capable prefill m={pmq} session \
                                 build failed: {e} — SKIPPED"
                            ),
                        },
                        None => warn!(
                            "[spyre-worker] sendnn: prefix-capable prefill m={pmq} sentinel did \
                             not resolve — SKIPPED"
                        ),
                    }
                }
                let mut prefill_prefix = PrefixLadder::of_built(prefix_built);
                if prefill_prefix.is_empty() {
                    warn!(
                        "[spyre-worker] sendnn: no prefix-capable prefill session built — continuation \
                         chunks will be REFUSED"
                    );
                } else {
                    info!(
                        "[spyre-worker] sendnn: PREFIX ladder = {:?} (a continuation chunk runs on the \
                         smallest rung that holds the rows it has left)",
                        prefill_prefix.widths(),
                    );
                }
                // ── DECODE BATCH LADDER: one session per baked request count ──
                // Same construction and the same failure rule as the prefill ladder: each rung
                // aliases the decode session's weights and page pool, so a rung costs its programs
                // and its activations, never another copy of the model. A rung that fails to build
                // is logged and SKIPPED — that batch runs on a narrower rung, in more than one
                // forward, which is slower and never wrong.
                let mut decode_ladder: Vec<DecodeRung> = Vec::new();
                // ⛔ THE MANIFEST'S `SweptCols` IS DELIBERATELY NOT BOUND. It is the CEILING body's sweep —
                // one number per BATCH WIDTH — and the two host decisions that used to read it are
                // per-BODY (see `DecodeRung`). Binding it to `_` rather than to a name is the point: a
                // named ceiling sitting in this loop is what got copied onto the rung three times.
                for (rung_n, _ceiling_swept, sentinel) in g0.decode_rungs {
                    // ⛔ `RungSeqs::get`, so this reads a BATCH WIDTH and says so. The sibling ladder
                    // (`sk_bucket_rungs`) is keyed by the SWEEP EXTENT `active_cap`; selecting over it with
                    // `>= live` would bind a body baked for a 64-column sweep because four requests are
                    // live, and the fold would cover `active_cap * pages` columns while the deepest row
                    // needs `write_slot + 1`. Both keys were bare `u32` until now.
                    let rn = rung_n.get() as usize;
                    // The one-request rung IS the `decode` session already built and aliased above.
                    if rn <= 1 {
                        continue;
                    }
                    // The manifest's `RungSeqs` IS a baked row count — this is the one boundary where
                    // it becomes the launch-width vocabulary everything below runs on.
                    let Some(rung_width) =
                        scratchy_subtile::sdsc_abstract::RungWidth::of_baked_rows(rung_n.get())
                    else {
                        continue;
                    };
                    let Some(rcode) = sentinel
                        .as_str()
                        .strip_prefix(SUPERDSC_SENTINEL)
                        .and_then(scratchy_target_spyre::bundle_code::bundle)
                    else {
                        warn!(
                            "[spyre-worker] sendnn: decode batch rung seqs={rn}: sentinel names no \
                             bundle compiled into this binary — SKIPPED"
                        );
                        continue;
                    };
                    // The rung's own logits placement, read from the bundle it actually runs. Derived,
                    // never assumed: the width is stick-PADDED above the vocab, and the row count is
                    // this rung's `seqs`, so both differ per rung. A rung whose placement does not
                    // describe `rn` whole rows is SKIPPED rather than gathered from — that batch runs
                    // on a narrower rung, which is slower and never wrong.
                    // ⭐ THE PREFIX MASK'S REAL CAPACITY, FROM THIS RUNG'S OWN BUNDLE.
                    //
                    // `pmask = [nqh*mq, cap]` (`lower_subtile_tape_to_superdsc.rs:2567`), so the fold's
                    // total swept columns must fit `cap`: one mask block per pass, `passes = pages*width`,
                    // each block `COLS` wide ⇒ **`pages * width <= cap / PAGE_SLOTS`**. That bound is what
                    // caps a batched request's context, and it was guessed twice before being read once.
                    //
                    // 🛑 BY TID, NOT BY SIZE. `ATTN_MASK_TID = u32::MAX - 3` names pmask. Picking "the
                    // largest placement in the mask segment" is how I read the PREFILL bundle's pmask and
                    // spent four theories on a refutation that was about the wrong bundle — and the cache
                    // holds hundreds of per-group dirs whose seg-3 sizes span 421 KB..17 MB, so size
                    // identifies nothing. `rdir` here is THIS WIDTH's decode bundle, and `rn` IS its mq.
                    // ⭐ AND THE KV (seg-2) RESERVATION FROM THE SAME BUNDLE.
                    //
                    // What to compare it against: `70,778,880 B` is EXACTLY one 8b page
                    // (3 planes * kv_dim * PLANE_SLOTS * 2 * layers), so a bundle appears to reserve ONE
                    // page while the runtime pool is `num_blocks` of them. If seg-2 here is one page, the
                    // pool must be allocated at prepare instead — and the open question is what then bounds
                    // the fold at 8 pages x 4 rows when 6 x 4 works.
                    let seg2 = rcode.layout.segment_bytes[2];
                    debug!(
                        "[kv-seg] rung seqs={rn}: seg2={seg2} B ({} MB) — one page is \
                         3*kv_dim*PLANE_SLOTS*2*layers; compare against num_blocks*page_stride",
                        seg2 / (1024 * 1024),
                    );
                    // ⭐ KEEP THE CAPACITY, do not just print it: it is the bound on a batched request's
                    // context, so a step needing more blocks than the bundle baked overran the placement in
                    // silence — and the bytes past it read as ZERO in an additive mask, which means VALID.
                    // Checked at the launch now.
                    let mut rung_mask_cap: Option<
                        scratchy_subtile::sdsc_abstract::BakedMaskBlocks,
                    > = None;
                    if let Some(pmask) = rcode.layout.place_of_tid(
                        scratchy_target_spyre::lower_subtile_tape_to_superdsc::ATTN_MASK_TID,
                    ) {
                        // ⭐ QUERY HEADS FROM THE BAKED GEOMETRY — THE COUNT ITSELF, NOT THE OLD
                        // `hidden / head_dim` QUOTIENT. The quotient is the head count only for a
                        // SQUARE Q projection; gemma-4's is not (hidden 3840 against nqh·hd
                        // 4096/8192), and a mask-capacity derived from the quotient overstates how
                        // many blocks a rung's placement holds — the tail blocks read unstaged
                        // zeros, which in an ADDITIVE mask means VALID. `Geometry::nqh` is baked
                        // from `num_attention_heads` at the same bounds the tape was lowered at.
                        let g = &wirings.decode.geometry;
                        let nqh = g.nqh.max(1) as u64;
                        let bytes = pmask.size;
                        let per = nqh * rn as u64 * 2;
                        let cap = if per > 0 { bytes / per } else { 0 };
                        let page = scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u64;
                        // The block count is the ONE fact; which law binds it is per-BODY (the
                        // launch's own `gathers` picks the fold form), so state both readings
                        // instead of printing one of them as if the bundle had picked it.
                        debug!(
                            "[mask-cap] rung seqs={rn}: pmask={bytes} B, nqh={nqh} => {} mask \
                             block(s) — a per-page (gathered) fold bounds pages by it, a \
                             per-(row,page) fold by blocks/width",
                            cap / page,
                        );
                        rung_mask_cap =
                            scratchy_subtile::sdsc_abstract::BakedMaskBlocks::of_placement(
                                bytes,
                                nqh as u32,
                                rung_width,
                                scratchy_subtile::sdsc_abstract::SlotCount::new(
                                    scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32,
                                ),
                            );
                    }
                    // THE FOLD-ROW REGIME, from the bundle it will actually run: the launch derives its
                    // intermediate-segment stride from it (`set_int_stride`). A rung whose fold groups
                    // DISAGREE is SKIPPED like any other rung that fails to load — that batch runs on a
                    // narrower rung, slower and never wrong, whereas guessing a stride is the
                    // silent-wrong-history arm this field exists to close.
                    let fold_rows = match rcode.fold_rows() {
                        Ok(scratchy_target_spyre::bundle_code::FoldRows::PerRequest) => {
                            scratchy_subtile::sdsc_abstract::FoldRowRegime::PerRequest
                        }
                        Ok(scratchy_target_spyre::bundle_code::FoldRows::WholeBatch) => {
                            scratchy_subtile::sdsc_abstract::FoldRowRegime::WholeBatch
                        }
                        Err(e) => {
                            warn!(
                                "[spyre-worker] sendnn: decode batch rung seqs={rn}: {e} — SKIPPED; \
                                 that batch runs on a narrower rung"
                            );
                            continue;
                        }
                    };
                    let logits = rcode
                        .layout
                        .place_of_tid(dparsed.result_id as u32)
                        .and_then(|p| {
                            LogitsGeom::from_placement(p.size, rung_width, vocab as usize)
                        });
                    let Some(logits) = logits else {
                        warn!(
                            "[spyre-worker] sendnn: decode batch rung seqs={rn}: its baked logits \
                             placement (t{}) is missing or is not {rn} whole rows of >= {vocab} — \
                             SKIPPED",
                            dparsed.result_id
                        );
                        continue;
                    };
                    match SuperDscSession::new_borrowing(
                        rcode,
                        vocab,
                        kv_dim,
                        num_blocks,
                        wirings.decode.num_sources,
                        &resident_source_ids(&wirings.decode, &spill_tids),
                        &[1, 2],
                    ) {
                        Ok(mut rs) => {
                            let mut ok = true;
                            if let Err(e) = adopt_from_owner(&mut rs, &ss, &spill_images) {
                                warn!(
                                    "[spyre-worker] sendnn: decode batch rung seqs={rn}: {e} — \
                                     SKIPPED"
                                );
                                ok = false;
                            }
                            if ok {
                                info!(
                                    "[spyre-worker] sendnn: decode batch rung seqs={rn} ready \
                                     (logits [{}, {}])",
                                    logits.rows, logits.width
                                );
                                decode_ladder.push(DecodeRung {
                                    seqs: rung_width,
                                    mask_cap: rung_mask_cap,
                                    fold_rows,
                                    // ⛔ NO `swept` AND NO `gathers_kv` HERE ANY MORE — both are
                                    // per-BODY and this is per-bundle; see `DecodeRung`. The launch
                                    // asks `sess.step_body(start)`, which is the selector itself.
                                    sess: rs,
                                    logits,
                                });
                            }
                        }
                        Err(e) => warn!(
                            "[spyre-worker] sendnn: decode batch rung seqs={rn} FAILED to build \
                             ({e}) — SKIPPED; that batch runs on a narrower rung"
                        ),
                    }
                }
                decode_ladder.sort_by_key(|r| r.seqs);
                // ── STAGE EVERY CHUNK SESSION'S FULL-CHUNK CAUSAL MASK, ONCE, HERE ── so no request's
                // first full chunk pays the build (see `stage_full_chunk_causal`). Each session under
                // the wiring its launches run under: the prefill meta's for both prompt ladders, the
                // decode meta's for the solo decode session, whose every step is the one-row key. The
                // decode BATCH rungs key by how many requests are live, which no load-time fact knows,
                // so they still bind at first use.
                //
                // ⭐ AND FAULT THE SAME SESSIONS' ACTIVATION SHADOWS IN FIRST, so a request's first
                // binds write into pages already mapped (see `materialize_activation_shadow`). The
                // same set and for the same reason: these are what a first request runs on. A decode
                // batch rung's activation segment reserves `nqh · rows` prefix-mask rows for every page
                // a pool can hold (the `pmbytes` law), of which a step writes only its live blocks, so
                // it keeps faulting in only what it writes rather than committing the reservation.
                {
                    let _t = std::time::Instant::now();
                    let pw = wirings.prefill.as_ref().unwrap_or(&wirings.decode);
                    let mut staged = (0usize, 0u64);
                    let (mut shadow_bytes, mut shadow_s) = (0u64, 0f64);
                    let mut stage = |sess: &mut SuperDscSession,
                                     w: &scratchy_target_spyre::wiring::Wiring,
                                     rows: usize|
                     -> ExecutorResult<()> {
                        let t = std::time::Instant::now();
                        shadow_bytes += sess.materialize_activation_shadow().map_err(|e| {
                            werr(format!("the {rows}-row session's activation shadow: {e}"))
                        })?;
                        shadow_s += t.elapsed().as_secs_f64();
                        staged.1 += stage_full_chunk_causal(sess, w, rows).map_err(werr)?;
                        staged.0 += 1;
                        Ok(())
                    };
                    for (rows, sess) in prefill_ladder.iter_mut() {
                        stage(sess, pw, *rows)?;
                    }
                    for (rows, sess) in prefill_prefix.rungs_mut() {
                        stage(sess, pw, rows)?;
                    }
                    stage(&mut ss, &wirings.decode, 1)?;
                    info!(
                        "[spyre-worker] sendnn: staged {} session(s)' full-chunk causal mask at load \
                         ({} KB sent, {:.0} ms in all) — a full chunk's first launch binds nothing \
                         for it; their activation shadows faulted in first ({} KB, {:.0} ms)",
                        staged.0,
                        staged.1 >> 10,
                        _t.elapsed().as_secs_f64() * 1e3,
                        shadow_bytes >> 10,
                        shadow_s * 1e3,
                    );
                }
                let sb = SuperDscBundle {
                    // Every page starts free: no request holds one yet.
                    pool_pages: num_blocks,
                    pool_rows,
                    decode: ss,
                    decode_rungs: decode_ladder,
                    prefill_rungs: prefill_ladder,
                    prefill_prefix,
                };
                // THE SPLIT, SAID OUT LOUD. It decides the servable context per request AND the
                // concurrency ceiling, so it is the first thing to look at when a request is refused —
                // for length or for admission. Printed from `kv_split()`, which reads the ONE stored
                // stripe count (`pool_rows`) that the pool was also SIZED by.
                //
                // 🛑 "row(s) SHARING ALL OF THEM" is what this line used to say, and it was the exact
                // inverse of the truth — the same rot that survived in `PoolSplit`'s doc. Rows do not
                // share pages: `physical(lp) = row * pages_per_row + lp` gives each row an EXCLUSIVE
                // stripe, which is the whole mechanism behind the uniform page stride. A reader
                // debugging "why did this request run out at N positions" would have been told the pool
                // was N times bigger than the share it actually gets.
                if let Some(sp) = sb.kv_split() {
                    // 🛑 THIS LINE HAS NOW BEEN WRONG TWICE, IN OPPOSITE DIRECTIONS. First it said rows
                    // "share all of them" while the pool was STRIPED (fixed in 201ab4f7); then it said
                    // "striped into N exclusive rows of M pages = X positions PER REQUEST" after the free
                    // list made pages an arbitrary BLOCK TABLE again (f5e26a3e), where a request can draw
                    // the WHOLE pool and `pages_per_row` bounds nothing. Both times the number it printed
                    // was the first thing a reader would consult about a length refusal, and both times it
                    // was off by a factor of `rows`. State only what is still true: the pool's SIZE, the
                    // context ONE request can reach (all of it), and what `rows` still bounds — ADMISSION,
                    // because a row is now purely the identity of a launch slot.
                    let per_page = sb.decode.page_slots.max(1);
                    // ⛔ NO `.expect()` IN A LOG LINE. This read `PoolPages::of_pool(..).expect("a pool
                    // with pages")`, so a pool that failed its own construction would abort the process from
                    // inside an eprintln — killing the run at the exact moment the summary would have said
                    // why. An instrument must not be able to end the thing it reports on.
                    let host_pages = PoolPages::of_pool(sb.pool_pages)
                        .and_then(|pp| PoolPartition::of_pool(pp, sb.pool_rows))
                        .map_or(0, |p| p.host_blocks() as usize);
                    info!(
                        "[spyre-worker] sendnn: KV pool {} page(s) of {per_page} slot(s) = {} positions. \
                         Pages come from the SCHEDULER's block table (two requests may share one — that is \
                         what a prefix-cache hit is); the top {} page(s) are reserved for the batched \
                         write's hole and are never offered to the scheduler. At most {} concurrent \
                         request(s) (launch slots, not a capacity share)",
                        sb.pool_pages,
                        sb.pool_pages * per_page,
                        sb.pool_pages.saturating_sub(host_pages),
                        sp.rows().get(),
                    );
                }
                SendnnSession::SuperDsc(sb)
            } else {
                // ⛔ NOT A SUPERDSC BUNDLE. The sengraph offline_decoder path that used to run these
                // is deleted, so a bundle without the `SUPERDSC_BUNDLE:` sentinel has nothing to
                // run it — say so at LOAD, where the fingerprint and the bake log are still in
                // hand, rather than at the first forward.
                return Err(werr(
                    "sendnn: this bundle is not a SuperDSC bundle (no SUPERDSC_BUNDLE: sentinel on \
                     the decode slot). The sengraph offline_decoder runtime has been removed; \
                     re-bake with SCRATCHY_SENDNN_MODE=superdsc.",
                ));
            }
        };
        // KTIR program routing (prefill=0, decode=1). sendnn routes per group inside
        // forward; keep these for the BundleMeta (decode/prefill phase index).
        #[cfg(feature = "spyre-hw")]
        let (decode_prog_base, prefill_prog) = (1usize, Some(0usize));
        #[cfg(not(feature = "spyre-hw"))]
        let prefill_prog = if pparsed.is_some() {
            Some(0usize)
        } else {
            None
        };

        #[allow(unused_variables)]
        // ⛔ THE FACTS ARE AN ARGUMENT, NOT A PATCH. This built every meta with `false`/`0`
        // placeholders and the callers then assigned the real values field-by-field — so a phase
        // whose caller forgot one served a bundle claiming it placed nothing.
        let mk_meta =
            |p: Parsed,
             prog: usize,
             facts: scratchy_target_spyre::wiring::BakeFacts,
             wiring: &'static scratchy_target_spyre::wiring::Wiring| BundleMeta {
                #[cfg(not(feature = "spyre-hw"))]
                prog,
                attn_mask_src: p.attn_mask_src,
                capacity: p.capacity,
                m_cap: p.m_cap,
                embed_src: p.embed_src,
                cos_srcs: p.cos_srcs,
                sin_srcs: p.sin_srcs,
                result_id: p.result_id,
                result_is_index: p.result_is_index,
                result_want: p.result_want,
                layers: p.layers,
                #[cfg(not(feature = "spyre-hw"))]
                output_ids: p.output_ids,
                #[cfg(not(feature = "spyre-hw"))]
                scalarmul_scales: p.scalarmul_scales,
                #[cfg(feature = "spyre-hw")]
                facts,
                #[cfg(feature = "spyre-hw")]
                wiring,
            };

        // Model-level dims (identical across cap buckets) from the first decode parse.
        let hidden = dparsed.hidden;
        // head_dim from the HF config (cos sources are now full rope width).
        let head_dim = hf_config
            .head_dim()
            .ok_or_else(|| werr("config missing head_dim / num_attention_heads"))?;
        let kv_dim = dparsed.kv_dim; // the info! below prints it on BOTH paths; the Loaded field is spyre-hw-only
        let vocab = dparsed.vocab;
        // ⭐ ROPE θ FROM THE BAKE.
        //
        // The old comment argued the `unwrap_or(1e4)` was safe because it is
        // HF's spec default and "never fires for any config that carries
        // rope_theta". True, and beside the point: the risk was never a missing
        // field, it was TWO SOURCES. The rotary tables the bundle was lowered
        // with came from the compiler's bounds; re-reading config.json here
        // makes a second answer that can differ (a re-quantized checkpoint, an
        // edited config) with nothing comparing them. Baked, there is one.
        //
        // I hit this exact bug from the other side while baking the wiring: a
        // `bounds`-only lookup missed θ (it is a FLOAT, so it lives in
        // `scalars`) and the `unwrap_or(1e4)` silently emitted θ=10000 for
        // llama-3.2, whose real θ is 500000. Fluent, subtly-wrong output.
        let rope_theta = wirings.decode.geometry.rope_theta();

        let mut decode: Vec<BundleMeta> = dparsed_all
            .into_iter()
            .enumerate()
            .map(|(i, p)| mk_meta(p, decode_prog_base + i, decode_facts, &wirings.decode))
            .collect();
        // Ascending by capacity so `select_decode`'s first-fit picks the smallest.
        decode.sort_by_key(|b| b.capacity);
        let prefill = pparsed.map(|p| {
            mk_meta(
                p,
                prefill_prog.unwrap(),
                prefill_facts,
                wirings.prefill.as_ref().unwrap_or(&wirings.decode),
            )
        });
        // ⛔ DEFAULT-MODE: the single static-SDPA graph IS the decode graph (prefill slot holds the same
        // graph). Its PrimaryInput ids (`t0`/`t5..t8`/prefix-KV) come from the DECODE manifest. The
        // separately-emitted m=prefill prefill manifest has DIFFERENT tensor ids, so using its BundleMeta for
        // the PER-FORWARD binding would bind the wrong names to the default graph (silent wrong output). So
        // `model.prefill` is nulled for Default/Paged/SuperDsc (every forward drives the decode meta). BUT
        // the batched-prefill path binds the prefill meta to the DISTINCT prefill SESSION (via
        // run_prefill_batch), where the ids DO match — so preserve it in `batched_prefill` (SuperDsc only).
        #[cfg(feature = "spyre-hw")]
        let (prefill, batched_prefill) = if matches!(session, SendnnSession::SuperDsc(_)) {
            (None, prefill)
        } else {
            (prefill, None)
        };
        #[cfg(not(feature = "spyre-hw"))]
        let batched_prefill: Option<BundleMeta> = None;

        let caps: Vec<usize> = decode.iter().map(|b| b.capacity).collect();
        info!(
            "[spyre-worker] arch {arch} via try_load (decode m={}) prefill={} | {} layers, \
             hidden {hidden}, head_dim {head_dim}, kv_dim {kv_dim}, vocab {vocab}, \
             decode caps {caps:?}, theta {rope_theta}",
            decode[0].m_cap,
            prefill
                .as_ref()
                .map_or_else(|| "none".to_string(), |p| format!("m={}", p.m_cap)),
            decode[0].layers.len(),
        );

        Ok(Loaded {
            model_dir,
            hf_config,
            embed_tokens,
            hidden,
            #[cfg(feature = "spyre-hw")]
            head_dim,
            #[cfg(feature = "spyre-hw")]
            kv_dim,
            vocab,
            session,
            decode,
            prefill,
            batched_prefill,
            kv_budget_bytes: kv_budget_bytes_v,
            pool_stride_bytes: {
                #[cfg(feature = "spyre-hw")]
                let stride = pool_stride_bytes_v;
                #[cfg(not(feature = "spyre-hw"))]
                let stride = None;
                stride
            },
            served_context: served_context_v,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests — the ScalarMul weight fold's runtime half
// ---------------------------------------------------------------------------

#[cfg(test)]
mod weight_scale_fold_tests {
    use super::*;

    /// One staged weight as `stage_bound_weights` shapes them: `(id, bytes, dtype, dims)`.
    /// The element values go through the SAME one-rounding encode the load path produces
    /// (`f32 → dtype` once), so the test asserts the multiply's effect, not encoding drift.
    fn staged(id: usize, dt: SDType, vals: &[f32]) -> StagedWeight {
        let bytes: Vec<u8> = vals
            .iter()
            .flat_map(|v| match dt {
                SDType::F16 => half::f16::from_f32(*v).to_le_bytes().to_vec(),
                SDType::BF16 => half::bf16::from_f32(*v).to_le_bytes().to_vec(),
                SDType::F32 => v.to_le_bytes().to_vec(),
                _ => panic!("test helper stages float payloads only"),
            })
            .collect();
        let dims = vec![1, vals.len()];
        (id, bytes, dt, dims)
    }

    /// Decode staged bytes back to f32 (the payload already narrowed; widen for comparison).
    fn decoded(dt: SDType, bytes: &[u8]) -> Vec<f32> {
        match dt {
            SDType::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c).to_f32())
                .collect(),
            SDType::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::bf16::from_le_bytes(*c).to_f32())
                .collect(),
            SDType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            _ => panic!("test helper decodes float payloads only"),
        }
    }

    /// The multiply is one f32 round-trip per element, on EVERY float payload the load can
    /// stage, and touches NOTHING but the fold targets: granite's 0.22 over f16 (the card's
    /// post-narrowing bytes), bf16, and f32, with an un-named neighbor keeping its bytes.
    #[test]
    fn scales_every_float_payload_with_one_rounding_and_spares_the_unnamed() {
        let vals = [2.0f32, -0.5, 3.25];
        let mut weights = vec![
            staged(0, SDType::F16, &[7.0, 8.0]), // not a fold target
            staged(1, SDType::F16, &vals),
            staged(2, SDType::BF16, &vals),
            staged(3, SDType::F32, &vals),
        ];
        let folds: Vec<(u32, f32)> = [(1u32, 0.22f32), (2, 0.22), (3, 0.22)].to_vec();
        apply_weight_scale_folds(&mut weights, &folds).expect("float payloads fold");
        for (id, dt) in [(1usize, SDType::F16), (2, SDType::BF16), (3, SDType::F32)] {
            let (_, bytes, sdt, _) = &weights[id];
            assert_eq!(*sdt, dt);
            // The pinned discipline: `from_f32(widen(x) * s)` — ONE narrowing, in f32. A
            // fold that instead scaled in the NARROW dtype (or double-rounded) differs in
            // the last bit somewhere across these values and fails here.
            let want: Vec<f32> = vals
                .iter()
                .map(|v| match dt {
                    SDType::F16 => half::f16::from_f32(v * 0.22).to_f32(),
                    SDType::BF16 => half::bf16::from_f32(v * 0.22).to_f32(),
                    _ => v * 0.22,
                })
                .collect();
            assert_eq!(decoded(dt, bytes), want, "t{id} ({dt:?}) kept one rounding");
        }
        let (_, bytes, _, _) = &weights[0];
        assert_eq!(
            decoded(SDType::F16, bytes),
            vec![7.0, 8.0],
            "a tensor no fold names keeps its staged bytes"
        );
    }

    /// A fold naming a tensor the binding did not stage is a wiring↔recognition desync —
    /// refused with the tid in the message, never silently skipped.
    #[test]
    fn refuses_a_fold_naming_an_unstaged_tensor() {
        let mut weights = vec![staged(1, SDType::F16, &[1.0])];
        let err = apply_weight_scale_folds(&mut weights, &[(7u32, 2.0)])
            .expect_err("unstaged fold target");
        assert!(
            err.to_string().contains("t7"),
            "the refusal names the tid: {err}"
        );
        // The staged tensor is untouched — a refusal leaves the bytes alone.
        assert_eq!(decoded(SDType::F16, &weights[0].1), vec![1.0]);
    }

    /// An fp8 payload is a requantize, not a scale — the recognition refuses fp8 fold targets
    /// at BUILD time, so an fp8 fold reaching the runtime half can only be a desync. Refused.
    #[test]
    fn refuses_an_fp8_fold_target_as_a_desync() {
        let mut weights = vec![(1usize, vec![0u8; 4], SDType::Fp8E4m3, vec![1, 4])];
        let err =
            apply_weight_scale_folds(&mut weights, &[(1u32, 2.0)]).expect_err("fp8 fold target");
        assert!(
            err.to_string().contains("desynced"),
            "the refusal says what reaching here means: {err}"
        );
    }
}

#[cfg(test)]
mod weight_copy_tests {
    use super::*;
    use scratchy_target_spyre::lower_subtile_tape_to_superdsc::oproj_headmajor_tid;

    /// One staged fp8 o_proj-shaped weight as `stage_bound_weights` shapes it:
    /// `(id, bytes, dtype, dims)`. The values are arbitrary — the copy is byte-for-byte
    /// and dims-agnostic, which is the law under test.
    fn oproj_row(id: usize, bytes: &[u8]) -> StagedWeight {
        (id, bytes.to_vec(), SDType::Fp8E4m3, vec![4096, 4096])
    }

    /// Each listed copy is the COMPANION'S OWN ROW under the reserved tid — same bytes,
    /// same dtype, same dims — because the permutation is not here: it happens at staging,
    /// through the copy's own retile descriptor, so this is a source re-read, not a
    /// transform. The companions and any neighbor keep their rows; the binding grows by
    /// exactly the copy count.
    #[test]
    fn stages_each_copy_as_the_companions_own_bytes() {
        let mut weights = vec![
            oproj_row(9, &[1, 2, 3, 4, 5, 6, 7, 8]),  // layer 0's o_w
            oproj_row(11, &[0, 0, 0, 0, 0, 0, 0, 0]), // the gate — no copy names it
            oproj_row(21, &[9, 9, 9, 9]),             // layer 1's o_w (different bytes AND length)
        ];
        let copies = vec![
            (9u32, oproj_headmajor_tid(9)),
            (21u32, oproj_headmajor_tid(21)),
        ];
        materialize_weight_copies(&mut weights, &copies).expect("both copies stage");
        assert_eq!(
            weights.len(),
            5,
            "the binding grows by exactly the copy count"
        );
        for (from, to) in &copies {
            let row = weights
                .iter()
                .find(|(id, ..)| *id == *to as usize)
                .unwrap_or_else(|| panic!("the copy t{to} is staged"));
            let companion = weights
                .iter()
                .find(|(id, ..)| *id == *from as usize)
                .unwrap_or_else(|| panic!("the companion t{from} is staged"));
            assert_eq!(
                row.1, companion.1,
                "t{to}: byte-for-byte — the shuffle is at staging, not here"
            );
            assert_eq!(row.2, companion.2, "t{to}: same dtype");
            assert_eq!(row.3, companion.3, "t{to}: same dims");
        }
        // Append-only: the three staged rows are exactly what they were.
        assert_eq!(weights[0], oproj_row(9, &[1, 2, 3, 4, 5, 6, 7, 8]));
        assert_eq!(weights[1], oproj_row(11, &[0, 0, 0, 0, 0, 0, 0, 0]));
        assert_eq!(weights[2], oproj_row(21, &[9, 9, 9, 9]));
    }

    /// A copy naming a companion the binding did not stage is a wiring↔mint desync —
    /// refused with the tid in the message, never silently skipped, and the staged rows
    /// are left alone.
    #[test]
    fn refuses_a_copy_naming_an_unstaged_companion() {
        let mut weights = vec![oproj_row(9, &[1, 2, 3, 4])];
        let err = materialize_weight_copies(&mut weights, &[(7u32, oproj_headmajor_tid(7))])
            .expect_err("unstaged companion");
        assert!(
            err.to_string().contains("t7"),
            "the refusal names the companion: {err}"
        );
        assert_eq!(
            weights,
            vec![oproj_row(9, &[1, 2, 3, 4])],
            "a refusal leaves the staged rows alone"
        );
    }

    /// A copy whose id the binding already staged is the mirror desync — the layout minted
    /// a permuted copy for an id the binding owns. Refused rather than overwriting the
    /// binding's own row.
    #[test]
    fn refuses_a_copy_whose_id_the_binding_already_staged() {
        let to = oproj_headmajor_tid(9);
        let mut weights = vec![oproj_row(9, &[1, 2, 3, 4]), oproj_row(to as usize, &[5, 6])];
        let err = materialize_weight_copies(&mut weights, &[(9u32, to)])
            .expect_err("copy id the binding owns");
        assert!(
            err.to_string().contains("already staged"),
            "the refusal says what the collision is: {err}"
        );
        assert_eq!(
            weights,
            vec![oproj_row(9, &[1, 2, 3, 4]), oproj_row(to as usize, &[5, 6])],
            "a refusal leaves the staged rows alone"
        );
    }
}

/// ⭐ THE DEFAULT-CONTEXT LAW, WIRED AT THE ARITHMETIC THAT DECIDES IT — the granite-3.1-2b
/// case the law was decided on. The pool veto's site is card-only, but the number it refuses
/// against and the number an unset flag resolves to are pure arithmetic over the shared pool
/// vocabulary, so the wiring is pinned HERE: the affordable expression, the ONE predicate, and
/// the demand the resolution produces, checked against the budget it was resolved from.
#[cfg(test)]
mod served_context_tests {
    use super::*;
    use scratchy_serving_engine::gpu_budget::{ServedContext, resolve_served_context};
    use scratchy_subtile::sdsc_abstract::{AdmittedRequests, PoolDemand, PoolRows, SlotCount};

    /// granite-3.1-2b's `max_position_embeddings`, from the checked-in config.
    const GRANITE_2B_MAX_POS: usize = 131_072;
    /// One page of granite-3.1-2b KV — the same stride the load derives from the bake's
    /// `kv_dim`: three planes (Knat, V, Kᵀ) × 8 kv heads × head dim 64 (= kv_dim 512
    /// features) × 256 plane slots × 2 B × 40 layers. The runtime's own report agrees:
    /// `kvstride=786432` per layer, `page_stride=31457280`.
    const GRANITE_2B_PAGE_BYTES: u64 = 3 * (8 * 64) * 256 * 2 * 40;
    /// The pool is ONE allocation, so the budget that decides the default is a single flex
    /// region (`bundle::MAX_SEGMENT_BYTES`) — whichever card is under it. This is why the
    /// 2b's default over-runs: 131,072 tokens at two rows is 1,025 pages ≈ 32.2 GB, two
    /// regions' worth.
    const ONE_REGION_BYTES: u64 = 16 * 1024 * 1024 * 1024;

    /// `scr chat` declares `--max-num-seqs 1`; the width rounds UP to the ladder's smallest
    /// rung, so a chat run's pool is cut for two launch rows.
    fn chat_rows() -> PoolRows {
        PoolRows::for_admission(AdmittedRequests::new(1).expect("1 is nonzero"))
    }

    #[test]
    fn the_granite_2b_default_resolves_to_the_affordable_number_and_loads() {
        let rows = chat_rows();
        let affordable = affordable_context_tokens(ONE_REGION_BYTES, GRANITE_2B_PAGE_BYTES, rows);
        // (16 GiB / 31,457,280 B − 1 reserve page) / 2 rows × 256 slots = 69,632 — the number
        // the card prints from the same arithmetic. NOT the model's 131,072.
        assert_eq!(affordable, 69_632);
        assert!(
            affordable < GRANITE_2B_MAX_POS as u64,
            "the premise: one region cannot buy the 2b's default"
        );
        // THE LAW: the unset default resolves to the affordable number…
        assert_eq!(
            resolve_served_context(None, GRANITE_2B_MAX_POS, affordable as usize),
            ServedContext::Tokens(affordable as usize),
            "granite-2b's default 131072 resolves to what one region funds"
        );
        // …and an explicit over-budget ask refuses WITH that number as the computed advice.
        assert_eq!(
            resolve_served_context(
                Some(GRANITE_2B_MAX_POS),
                GRANITE_2B_MAX_POS,
                affordable as usize
            ),
            ServedContext::Refused {
                declared: GRANITE_2B_MAX_POS,
                affordable: affordable as usize
            }
        );
        // ⭐ THE RESOLUTION IS VETO-PROOF BY CONSTRUCTION: the pool demanded for exactly the
        // affordable number fits the budget it was resolved from…
        let need = |tokens: u64| {
            PoolDemand::of_declaration(SlotCount::new(tokens as u32), rows)
                .pages()
                .expect("a region-scale declaration overflows no page count")
                .get() as u64
                * GRANITE_2B_PAGE_BYTES
        };
        assert!(
            need(affordable) <= ONE_REGION_BYTES,
            "the resolved pool fits: {} ≤ {}",
            need(affordable),
            ONE_REGION_BYTES
        );
        // …and one token MORE does not — the boundary is tight, so the resolution is the
        // LARGEST context that re-passes the veto, never a rounded guess under it.
        assert!(
            need(affordable + 1) > ONE_REGION_BYTES,
            "one token past the resolution busts the budget: {}",
            need(affordable + 1)
        );
    }

    /// The law's boundary IS the old veto's boundary: for every explicit declaration, the
    /// predicate refuses exactly the declarations whose pool demand exceeds the budget — the
    /// rewrite changed the unset branch only, never the refusal's edge.
    #[test]
    fn the_laws_boundary_is_the_old_vetos_boundary() {
        let rows = chat_rows();
        let affordable =
            affordable_context_tokens(ONE_REGION_BYTES, GRANITE_2B_PAGE_BYTES, rows) as usize;
        // 256 = one page; 257 crosses into a second; 65,792 = a deep in-budget ask;
        // 69,632/69,633 straddle the resolution; 131,072 is the two-region ask.
        for declared in [256_usize, 257, 4096, 65_792, 69_632, 69_633, 131_072] {
            let demand = PoolDemand::of_declaration(SlotCount::new(declared as u32), rows)
                .pages()
                .expect("a region-scale declaration overflows no page count");
            let need = demand.get() as u64 * GRANITE_2B_PAGE_BYTES;
            let old_veto_fires = need > ONE_REGION_BYTES;
            match resolve_served_context(Some(declared), GRANITE_2B_MAX_POS, affordable) {
                ServedContext::Refused { .. } => {
                    assert!(old_veto_fires, "{declared}: refused inside the budget");
                }
                ServedContext::Tokens(served) => {
                    assert!(!old_veto_fires, "{declared}: served past the budget");
                    assert_eq!(served, declared, "{declared}: verbatim inside the budget");
                }
            }
        }
    }
}

/// ⭐ THE WEIGHT SETS THIS BINARY CARRIES FIT ONE DEVICE REGION — read off the bundles it baked.
///
/// The segment budget's verdict is a field of every baked layout (`WeightTail`), so whether a model's
/// weights spilled is a property of the binary, not of a serve log. MEASURED on granite-3.1-8b fp16:
/// padding its gate/up to 224 sticks pushed the weight set past 16 GiB, spilled the 400 MiB tied
/// embedding into seg6, and put 12,000 MB of per-session copies on the card — memory and load time,
/// found in a serve log (the decode slowdown that build also had was the pad's own trip count, not
/// the spill; see `WeightTail`). That model is the one this crate's CI scope bakes
/// (`scratchy-models/granite-3.1-8b-instruct`), so a width or layout change that pushes it back over
/// the region fails here, at `cargo test`, instead of on a card.
#[cfg(test)]
mod weight_tail_tests {
    use scratchy_target_spyre::bundle_code::{WeightTail, bundle, registered_fps};

    #[test]
    fn every_baked_bundle_keeps_its_weight_tail_in_the_weight_segment() {
        let fps = registered_fps();
        assert!(
            !fps.is_empty(),
            "the build scope names a model, so this binary carries its bundles"
        );
        for fp in fps {
            let code = bundle(fp).expect("a registered fingerprint resolves");
            assert_eq!(
                code.layout.weight_tail,
                WeightTail::InPlace,
                "bundle {fp} spilled its weight tail: its weight set no longer fits one device \
                 region (segment bytes {:?})",
                code.layout.segment_bytes,
            );
        }
    }
}
