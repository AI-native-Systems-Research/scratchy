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

/// The segments this bundle spilled weights into and HOW MANY BYTES of each the spill occupies —
/// [`adopt_from_owner`]'s work list, empty for every model whose weights fit one segment.
///
/// Derived from the LAYOUT and the bound weight ids rather than from a baked flag: a spill slot is a
/// declared [`WEIGHT_SPILL_SEGS`] segment that some bound WEIGHT is actually placed in. Asking the
/// two sources that already exist keeps this from drifting out of step with the emitter's decision.
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
    weight_ids: impl IntoIterator<Item = usize>,
) -> Vec<(i64, usize)> {
    use scratchy_target_spyre::lower_subtile_tape_to_superdsc::WEIGHT_SPILL_SEGS;
    let mut spans: std::collections::BTreeMap<i64, usize> = Default::default();
    for id in weight_ids {
        let Some(p) = layout.place_of_tid(id as u32) else {
            continue;
        };
        if !WEIGHT_SPILL_SEGS.contains(&(p.segment as usize)) {
            continue;
        }
        let end = (p.offset + p.size) as usize;
        let e = spans.entry(p.segment as i64).or_insert(0);
        *e = (*e).max(end);
    }
    spans.into_iter().collect()
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
    /// cos / sin runtime sources as `(source id, full column width)`. GQA gives
    /// more than one width (Q vs K); each is filled by tiling the per-position
    /// rotary row across heads (the sendnn rope consumes them full-width).
    pub(crate) cos_srcs: Vec<(usize, usize)>,
    pub(crate) sin_srcs: Vec<(usize, usize)>,
    pub(crate) result_id: usize,
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
    // The emitted identity / rope-P kernels against a fresh `stage_2d` — once per program, not
    // per token. See `Wiring::verify_kernel_tables` for why "the emitter used the same helper"
    // is not on its own sufficient.
    w.verify_kernel_tables().map_err(werr)?;
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
        })
        .collect();
    if layers.is_empty() {
        return Err(werr("wiring carries no per-layer AttnDecode wiring"));
    }
    let pairs = |v: &'static [(u32, u32)]| -> Vec<(usize, usize)> {
        v.iter().map(|&(i, w)| (i as usize, w as usize)).collect()
    };
    let embed_src = w.embed_src as usize;
    Ok(Parsed {
        embed_src,
        cos_srcs: pairs(w.cos_srcs),
        sin_srcs: pairs(w.sin_srcs),
        result_id: w.result as usize,
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
        .collect()
}

/// The tensor ids of the spilled weight tail — the companion of [`spilled_weight_spans`], which
/// carries the same decision as byte ranges. Both read it off the layout so neither can drift from
/// the emitter. Card-only, for the same reason [`spilled_weight_spans`] is.
#[cfg(feature = "spyre-hw")]
fn spilled_weight_tail_tids(
    layout: &scratchy_target_spyre::bundle_code::BundleLayout<'static>,
    weight_ids: impl IntoIterator<Item = usize>,
) -> Vec<u32> {
    use scratchy_target_spyre::lower_subtile_tape_to_superdsc::WEIGHT_SPILL_SEGS;
    weight_ids
        .into_iter()
        .filter(|id| {
            layout
                .place_of_tid(*id as u32)
                .is_some_and(|p| WEIGHT_SPILL_SEGS.contains(&(p.segment as usize)))
        })
        .map(|id| id as u32)
        .collect()
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
) -> ExecutorResult<Vec<(usize, Vec<u8>, SDType, Vec<usize>)>> {
    let _t_load = std::time::Instant::now();
    let mut weights: Vec<(usize, Vec<u8>, SDType, Vec<usize>)> = Vec::new();
    let _take_s = 0f64;
    for bw in bound {
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
        let bytes = gpu_tensor_bytes(&bw.tensor).to_vec();
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
            // DEVICE stick padding — the SHARED `bump_sticks_to_splittable` rule the emitter uses for a
            // FLOP-heavy matmul OUTPUT (`lower_matmul_node`'s `n_dev`): round to a whole 64-stick, then if
            // the stick count is prime/awkward (granite lm_head 49216/64=769 ⇒ <8 cores) bump it to a
            // multiple of 8 (→ 49664). Zero-pad this GEMM weight's N columns to the SAME width so the staged
            // host buffer equals the kernel RetileDescriptor's device_size; the shim's `prod(device_size)`
            // staging guard then matches. All weight matmuls cross the util floor, so this blind bump equals
            // the emitter's macs-conditional one. The pad columns are ZERO logits the host never reads (the
            // read uses the LOGICAL `vocab = n`). No-op for every already-splittable weight (all but lm_head).
            // TYPE-SAFE shared device width: the weight `[rows=k, cols=n]` staged at m=1. `DeviceWidth`
            // is the SOLE padding rule (same as the emitter's `n_dev`/kernel0), so the staged buffer width
            // == the emitted device width by construction — no runtime shim mismatch possible.
            let n_dev =
                scratchy_target_spyre::lower_subtile_tape_to_superdsc::DeviceWidth::for_output(
                    1,
                    cols as u32,
                    rows as u32,
                )
                .get() as usize;
            if n_dev != cols {
                // In `[n, k]` the padded axis `n` is the OUTER one, so widening it is appending
                // zero rows at the end -- `resize`, not a strided rebuild. (Transposed it was the
                // inner axis and every row had to be copied to a wider stride.) Same padded bytes,
                // and the only weight that needs any is lm_head.
                let mut padded = t;
                padded.resize(n_dev * rows * elem, 0);
                (padded, vec![rows, n_dev])
            } else {
                (t, vec![rows, cols])
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
        .map_err(|e| werr(format!("try_load {arch:?}: {e}")))?
        .ok_or_else(|| {
            werr(format!(
                "no compiled spyre model matched the {arch:?} checkpoint — \
                 build the CLI with this model's feature \
                 (e.g. --features spyre,model/llama-3.2-1b)"
            ))
        })?;
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
        let weights = stage_bound_weights(&bound, &wirings.decode)?;
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
        // ⛔ THE MUTABILITY IS THE CARD PATH'S, so it carries that cfg rather than an
        // `#[allow(unused_mut)]`: only the branch below assigns these, and on the emulator the
        // binding is a plain `None` that is read once.
        #[cfg(feature = "spyre-hw")]
        let mut decode_facts: Option<scratchy_target_spyre::wiring::BakeFacts> = None;
        #[cfg(not(feature = "spyre-hw"))]
        let decode_facts: Option<scratchy_target_spyre::wiring::BakeFacts> = None;
        #[cfg(feature = "spyre-hw")]
        let mut prefill_facts: Option<scratchy_target_spyre::wiring::BakeFacts> = None;
        #[cfg(not(feature = "spyre-hw"))]
        let prefill_facts: Option<scratchy_target_spyre::wiring::BakeFacts> = None;
        // The KV byte budget the paged pool gets sized against — the card's real capacity ×
        // `--gpu-memory-utilization` − the bundle's non-KV segments. Declared out here for the same reason
        // `scalarmul_scales_v` is: it is computed in the SuperDsc branch below (sendnn only) but has to
        // reach the `Loaded` this function returns, where `determine_available_memory` reads it so the
        // ENGINE sizes the scheduler's block allocator off the SAME number the pool was cut from.
        #[allow(unused_mut, unused_assignments)]
        let mut kv_budget_bytes_v: Option<u64> = None;
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
                let spill_spans = spilled_weight_spans(&code.layout, weights.iter().map(|w| w.0));
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
                // 🛑 THE SAME GUESS HAS ALREADY BEEN PAID FOR ONE LAYER DOWN. `fxa_rust_abi::build_runtime`
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
                // ⛔ `None` REFUSES, IT DOES NOT FALL BACK. A capacity query that fails and quietly returns
                // to 8 GiB is the defect restored under a log line, and a *plausible* default is worse than
                // none — it sizes a real allocation the card may not serve.
                let card_bytes = scratchy_target_spyre::fxa_rust_abi::device_memory_bytes()
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
                     reserved by the bundle (all segments but KV, incl. {} MB of spilled weight tail \
                     copied into {} borrowing session(s)) − 150 MB redundancy, capped at the \
                     {} MB a single allocation can be",
                    pool_budget_bytes / (1024 * 1024),
                    card_bytes.tensor_capacity() / (1024 * 1024),
                    card_bytes.bytes() / (1024 * 1024),
                    self.gpu_memory_utilization,
                    non_kv_bytes / (1024 * 1024),
                    (tail_bytes * borrowers) / (1024 * 1024),
                    borrowers,
                    card_bytes.max_single_allocation() / (1024 * 1024),
                );
                if pool_budget_bytes == 0 {
                    return Err(werr(format!(
                        "the Spyre card reports {} MB of memory, but --gpu-memory-utilization {:.2} leaves \
                         nothing for the KV cache after the {} MB this model's bundle already reserves. \
                         Raise --gpu-memory-utilization, or use a smaller model.",
                        card_bytes.bytes() / (1024 * 1024),
                        self.gpu_memory_utilization,
                        non_kv_bytes / (1024 * 1024),
                    )));
                }
                // Carry it to `Loaded` so the engine's own KV sizing reads this number and not a constant.
                kv_budget_bytes_v = Some(pool_budget_bytes);
                let num_blocks = {
                    // A PAGE'S BYTES, from the pool's own law: a page holds every LAYER, each layer three
                    // planes (Knat, V, Kᵀ), each plane `kv_dim` features across the plane's PHYSICAL slot
                    // extent, at 2 bytes. Confirmed against the runtime's own report — it logs
                    // `kvstride=1769472` per layer for granite-8b, and `3 * 1024 * 288 * 2 = 1769472`,
                    // `* 40 layers = 70778880 B` = the `page_stride` it then prints. Derived here rather
                    // than read back because the pool must be SIZED before the session that would report it.
                    // ⛔ FROM THE LOCAL `hf_config`, NOT `self.model` — `self.model` is still None here
                    // (this IS `load_model`, it is what fills it), so reading it silently gave `layers = 1`.
                    // That made the stride ONE LAYER's worth and the clamp handed back the very default the
                    // budget was meant to replace: the fix compiled, ran, logged nothing wrong, and changed
                    // nothing. A fallback that is *plausible* (1 layer) is worse than no fallback.
                    // ⭐ FROM THE BAKE, NOT config.json. The comment above
                    // records what the `unwrap_or(1)` fallback cost: a
                    // plausible-but-wrong 1 that compiled, ran, logged nothing
                    // and silently undid the fix. The layer count is a fact the
                    // tape was LOWERED at, so reading it from the wiring cannot
                    // disagree with the bundle and has no default to fire.
                    let layers = (wirings.decode.geometry.layers as usize).max(1);
                    let plane_slots =
                        scratchy_subtile::sdsc_abstract::PagedKvPool::PLANE_SLOTS as u64;
                    let stride = (3 * kv_dim as u64 * plane_slots * 2 * layers as u64).max(1);

                    // THE DECLARED DEPTH, capped by what the prefix mask can address. `--max-model-len`
                    // when the CLI gave one, else the model's own limit — and `kv_max_addressable_tokens`
                    // is the same bound the engine applies to admission, so quoting it here cannot invent a
                    // depth the fold could not serve.
                    let reach = PagedKvPool::MAX_PAGES_PER_ROW as usize * PagedKvPool::PAGE_SLOTS;
                    let declared = self
                        .declared_context
                        .unwrap_or(max_model_len)
                        .min(reach)
                        .max(1);
                    let demand = scratchy_subtile::sdsc_abstract::PoolDemand::of_declaration(
                        scratchy_subtile::sdsc_abstract::SlotCount::new(declared as u32),
                        pool_rows,
                    );
                    let pages = demand.pages().ok_or_else(|| {
                        werr(format!("--max-model-len {declared} overflows a page count"))
                    })?;
                    let need_bytes = pages.get() as u64 * stride;
                    if need_bytes > pool_budget_bytes {
                        return Err(werr(format!(
                            "--max-model-len {declared} × --max-num-seqs {} needs {} KV page(s) \
                             ({} MB at {} MB per page), but the KV budget is {} MB ({} page(s)). This run \
                             would fail a request mid-generation once the pool ran out, so it is refused \
                             here instead. Lower --max-model-len to about {} or --max-num-seqs to {}.",
                            pool_rows.get(),
                            pages.get(),
                            need_bytes / (1024 * 1024),
                            stride / (1024 * 1024),
                            pool_budget_bytes / (1024 * 1024),
                            pool_budget_bytes / stride,
                            // What the budget does fund, stated as the flags the caller typed. The reserve is
                            // ONE page (the shared scratch page) since the per-row hole run was deleted.
                            (pool_budget_bytes / stride).saturating_sub(
                                scratchy_subtile::sdsc_abstract::PoolPartition::reserve() as u64,
                            ) / pool_rows.get().get() as u64
                                * PagedKvPool::PAGE_SLOTS as u64,
                            (pool_budget_bytes / stride).saturating_sub(1)
                                / demand.per_row().get() as u64,
                        )));
                    }
                    tracing::info!(
                        "superdsc paged: KV pool {} page(s) = --max-model-len {declared} ({} page(s) per \
                         request) × {} launch row(s) + reserve — {} MB of a {} MB budget at {} MB per page",
                        pages.get(),
                        demand.per_row().get(),
                        pool_rows.get(),
                        need_bytes / (1024 * 1024),
                        pool_budget_bytes / (1024 * 1024),
                        stride / (1024 * 1024),
                    );
                    pages.get() as usize
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
                decode_facts = Some(df);
                // ── PREFILL-ALL-BUT-LAST: the DISTINCT m=N prefill dxp bundle (C4 stamps its own fp
                //    in the prefill graph slot; lm_head skipped). Build a SECOND SuperDscSession from
                //    it (binding the SAME weights again — 2× resident, the spike cost) so the prompt
                //    runs batched through it, then its seg2 KV is copied to the decode session. If the
                //    prefill slot's fp == the decode fp (the prefill reroll did NOT bake → C4 fell
                //    back), prefill=None ⇒ the worker uses the sequential mq=1 path. ──
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
                let prefill_code = top_prefill
                    .and_then(|_| {
                        g0.prefill_rungs
                            .iter()
                            .filter(|(rm, _)| *rm > 1)
                            .min_by_key(|(rm, _)| *rm)
                            .and_then(|(_, sent)| sent.strip_prefix(SUPERDSC_SENTINEL))
                            .and_then(scratchy_target_spyre::bundle_code::bundle)
                    })
                    .or(top_prefill);
                let (mut prefill_ss, prefill_m) = match prefill_code {
                    Some(pcode) => {
                        // The baked prefill m (query rows) = the embed activation's placement rows:
                        // size / (hidden·2). run_step MUST use exactly this m.
                        let pl = &pcode.layout;
                        let prefill_m = pl
                            .place_of_tid(dparsed.embed_src as u32)
                            .map_or(0, |p| (p.size as usize) / (dparsed.hidden * 2));
                        // What the PREFILL bake placed. From the bundle, never from
                        // `SCRATCHY_RMS_MATMUL_REDUCE`, so a bundle-vs-env mismatch cannot leave a
                        // matmul reading an unbound (zero) seg0.
                        prefill_facts = Some(scratchy_target_spyre::wiring::BakeFacts::of(pl));
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
                        let pss = SuperDscSession::new_reserving(
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
                if let Err(e) = ss.zero_seg(2) {
                    warn!("[spyre-worker] sendnn: KV pool zero-fill failed: {e}");
                } else {
                    info!("[spyre-worker] sendnn: KV pool zeroed on-device (no host mirror)");
                }
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
                // H2D — that already happened during its own `prepare`; it reclaims the duplicate region.
                if let Some(pf) = prefill_ss.as_mut() {
                    adopt_from_owner(pf, &ss, &spill_images).map_err(|e| {
                        werr(format!(
                            "the PREFILL session could not take the DECODE session's resident \
                             state: {e}"
                        ))
                    })?;
                    info!(
                        "[spyre-worker] sendnn: PREFILL/DECODE now SHARE seg1 (weights, ~2.6 GB \
                         reclaimed) + seg2 (resident KV) — per-prompt KV handoff eliminated"
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
                // ── PREFIX-CAPABLE PREFILL SESSION ── the bundle a continuation chunk (start>0)
                // runs on. Every ladder rung above is prefix-FREE, so this is the ONLY prefill bundle
                // that can attend resident KV. Borrows seg1+seg2 like the rungs, and is built AFTER
                // the decode session so it cannot affect decode's weight-region placement.
                let mut prefill_prefix: Option<(usize, SuperDscSession)> = None;
                {
                    let (pmq, psent) = g0.prefill_prefix;
                    if pmq > 1 && !psent.is_empty() {
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
                                Ok(mut ps) => {
                                    let mut ok = true;
                                    if let Err(e) = adopt_from_owner(&mut ps, &ss, &spill_images) {
                                        warn!(
                                            "[spyre-worker] sendnn: prefix-capable prefill \
                                             m={pmq}: {e} — continuation chunks will be REFUSED"
                                        );
                                        ok = false;
                                    }
                                    if ok {
                                        info!(
                                            "[spyre-worker] sendnn: prefix-capable prefill session \
                                             ready (m={pmq}, bundle {}) — used for start>0 chunks",
                                            pcode2.fp
                                        );
                                        prefill_prefix = Some((pmq as usize, ps));
                                    }
                                }
                                Err(e) => warn!(
                                    "[spyre-worker] sendnn: prefix-capable prefill m={pmq} session \
                                     build failed: {e} — continuation chunks will be REFUSED"
                                ),
                            },
                            None => warn!(
                                "[spyre-worker] sendnn: prefix-capable prefill sentinel did not \
                                 resolve — continuation chunks will be REFUSED"
                            ),
                        }
                    }
                }
                // ── DECODE BATCH LADDER: one session per baked request count ──
                // Same construction and the same failure rule as the prefill ladder: each rung
                // aliases the decode session's weights and page pool, so a rung costs its programs
                // and its activations, never another copy of the model. A rung that fails to build
                // is logged and SKIPPED — that batch runs on a narrower rung, in more than one
                // forward, which is slower and never wrong.
                let mut decode_ladder: Vec<DecodeRung> = Vec::new();
                for (rung_n, rung_swept, sentinel) in g0.decode_rungs {
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
                        // ⭐ QUERY HEADS FROM THE BAKED GEOMETRY. The head count
                        // IS a model fact — which is precisely why it must come
                        // from the artifact the tape was lowered at rather than
                        // from `config.json`, whose `unwrap_or(1)` could hand
                        // back a plausible wrong answer the bundle disagrees
                        // with. `hidden / head_dim` is the same derivation the
                        // emitter uses (`nqh = hidden_size / head_dim`).
                        let g = &wirings.decode.geometry;
                        let nqh = (g.hidden as u64 / (g.head_dim as u64).max(1)).max(1);
                        let bytes = pmask.size;
                        let per = nqh * rn as u64 * 2;
                        let cap = if per > 0 { bytes / per } else { 0 };
                        let page = scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u64;
                        debug!(
                            "[mask-cap] rung seqs={rn}: pmask={bytes} B, nqh={nqh} => cap={cap} \
                             slots => pages*width <= {} (this rung allows {} page(s) per row)",
                            cap / page,
                            (cap / page) / rn.max(1) as u64,
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
                                    swept: *rung_swept,
                                    mask_cap: rung_mask_cap,
                                    fold_rows,
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
        let kv_dim = dparsed.kv_dim;
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
            .map(|(i, p)| {
                mk_meta(
                    p,
                    decode_prog_base + i,
                    decode_facts.unwrap_or(scratchy_target_spyre::wiring::BakeFacts::NONE),
                    &wirings.decode,
                )
            })
            .collect();
        // Ascending by capacity so `select_decode`'s first-fit picks the smallest.
        decode.sort_by_key(|b| b.capacity);
        let prefill = pparsed.map(|p| {
            mk_meta(
                p,
                prefill_prog.unwrap(),
                prefill_facts.unwrap_or(scratchy_target_spyre::wiring::BakeFacts::NONE),
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
            head_dim,
            kv_dim,
            vocab,
            rope_theta,
            session,
            decode,
            prefill,
            batched_prefill,
            kv_budget_bytes: kv_budget_bytes_v,
        })
    }
}
