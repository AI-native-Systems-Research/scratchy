// SPDX-License-Identifier: Apache-2.0
//! The emulator's forward: one chunk through the KTIR session.

#[cfg(feature = "spyre-hw")]
use crate::card::pool::*;
use crate::spyre_forward::*;
#[cfg(feature = "spyre-hw")]
use crate::spyre_types::SuperDscSession;
use crate::spyre_types::*;
use crate::spyre_worker::*;
use scratchy_serving_worker::error::ExecutorResult;
#[cfg(feature = "spyre-hw")]
use scratchy_target_spyre::paged_step::{BatchSeq, StepRows};
#[cfg(not(feature = "spyre-hw"))]
use scratchy_target_spyre::runner::SpyreSession;
#[cfg(feature = "spyre-hw")]
use scratchy_target_spyre_card::sdsc_runner::StepLogits;

/// The KTIR-emulator twin of [`forward_chunk`]: one fused forward through the resident session,
/// with the prefix KV threaded from the host cache.
pub(crate) fn forward_chunk(
    sh: &Shared<'_>,
    session: &mut SpyreSession,
    b: &BundleMeta,
    prefill_meta: Option<&BundleMeta>,
    Chunk {
        state: req,
        id,
        tokens,
        start,
        toks,
    }: Chunk<'_>,
) -> ExecutorResult<ChunkOut> {
    let _ = prefill_meta; // the batched-prefill session pair is a sendnn-only shape
    let n = toks.len();
    // KTIR caps a forward at the baked m_cap. The sendnn paged prefill runs the
    // WHOLE prompt in one forward (s0 = padded bucket up to capacity), so its only
    // cap is the prefix capacity, checked below.
    {
        let mcap = b.m_cap;
        if n == 0 || n > mcap {
            return Err(werr(format!("forward_chunk: n={n} not in 1..={mcap}")));
        }
    }
    if start + n > b.capacity {
        return Err(werr(format!(
            "positions {start}..{} exceed the bundle's KV capacity {} — the model's own context",
            start + n,
            b.capacity,
        )));
    }
    // ⭐ NO `hd` HERE EITHER. The rotary tables are built per class (each cos/sin
    // source's own `RotaryKind`), so this function no longer reads a model-wide
    // head dim.
    let h = sh.hidden;
    // ⭐ THE REQUEST'S KV BLOCKS — the scheduler's grant, read out of the one store it was installed
    // in. Its pages already hold the keys of positions `0..start` (this request's earlier steps, or
    // the request whose prefix a cache hit shares), and this forward's write scatters positions
    // `start..start + n` into them; the paged step below checks the grant reaches that far.
    // Block `b` is pool page `b + 1`: page 0 is the null page a padded row writes
    // (`paged_step::page_of_block`).
    let blocks: Vec<u32> = tokens
        .block_table(id)
        .map(|t| {
            t.iter()
                .map(|&b| scratchy_target_spyre::paged_step::page_of_block(b as u32))
                .collect()
        })
        .unwrap_or_default();
    // The number of activation ROWS the bound buffers carry: the baked `m_cap`. Each forward fills
    // n <= m_cap real rows; the remaining pad rows REPLICATE the last real token (see the staging
    // loop below — the lm-head tail reads the last BAKED row, which must carry the last real token).
    let rows = b.m_cap;

    // Build this pass's DYNAMIC sources (the static weights stay resident in the
    // session — no per-forward weight marshal). Embedding gather + per-position
    // RoPE tables, every row (real rows their own token/position, pad rows the
    // last real token's — see the staging loop).
    let mut emb = vec![0.0f32; rows * h];
    // ⭐ PER-CLASS rotary rows, computed once per position per CLASS. A hybrid-
    // attention arch (gemma-4) gives sliding and global layers DIFFERENT tables
    // (head dim, θ and partial-rotary all differ), and every cos/sin source
    // names its own class in the wiring — one table per class, then each source
    // tiles the one it carries. The wrong table is silent garbage (fluent,
    // wrong output), never a fault.
    let classes: Vec<scratchy_target_spyre::wiring::RotaryKind> = b
        .cos_srcs
        .iter()
        .map(|&(_, _, k)| k)
        .chain(b.sin_srcs.iter().map(|&(_, _, k)| k))
        .collect();
    // Per-class, per-position `[class_head_dim]` rows.
    //
    // ⭐ THE STAGING LOOP COVERS ALL `rows` (NOT JUST `toks.len()`), AND A PAD ROW
    // (i >= n) REPLICATES THE LAST REAL TOKEN — the same law the card path's
    // `ChunkRows::row_logical_pos` pins (Kani: `prefill_pad_row_holds_last_real_token`).
    // The emitter folds the vocab-wide lm-head tail to m=1 over the LAST BAKED row
    // (`selector_lastrow_col(m_cap)`), which is a PAD row whenever `n < m_cap`; if that
    // row is not numerically the last real token's forward, the first sampled token is
    // whatever the lm_head computes over a row the model never fed. The card path has
    // replicated since its own pad-row law landed; the emulator left pad rows ZERO.
    // MEASURED on gemma-4-26b-a4b-it fp8 (23-token prompt on the m=31 rung): first
    // sampled token EOS (id 106), `LAST_HIDDEN` ≈ final-norm row 0, pad-row logits 0.
    let mut cos_tab: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(rows); classes.len()];
    let mut sin_tab: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(rows); classes.len()];
    for i in 0..rows {
        // A pad row's TOKEN and ROTARY POSITION are the last real token's —
        // `min(i, n-1)` over tokens, `start + min(i, n-1)` over positions.
        let r = i.min(n - 1);
        let t = toks[r];
        if t >= sh.vocab {
            return Err(werr(format!("token {t} >= vocab {}", sh.vocab)));
        }
        emb[i * h..(i + 1) * h].copy_from_slice(&sh.embed_tokens[t * h..(t + 1) * h]);
        for (ci, &k) in classes.iter().enumerate() {
            let (c, s) = scratchy_target_spyre::manifest::rope_cos_sin_kind((start + r) as u32, k);
            cos_tab[ci].push(c);
            sin_tab[ci].push(s);
        }
    }
    // Tile a class's head_dim-wide rotary rows across heads to fill a
    // `[rows, width]` source. Every row carries its staged position (pad rows the
    // last real token's clamped one). `width` is a multiple of the class's own head dim.
    let tile =
        |rrows: &[Vec<f32>], width: usize, kind: scratchy_target_spyre::wiring::RotaryKind| {
            let chd = kind.head_dim() as usize;
            let mut buf = vec![0.0f32; rows * width];
            let reps = width / chd;
            for (i, row) in rrows.iter().enumerate() {
                for r in 0..reps {
                    let off = i * width + r * chd;
                    buf[off..off + chd].copy_from_slice(row);
                }
            }
            buf
        };
    // KTIR builds ONE dynamic-source list (embed + cos/sin + per-layer prefix-KV)
    // for the single fused forward. sendnn builds per-GROUP lists below (it threads
    // the hidden state group→group), reusing `emb`/`cos_tab`/`sin_tab` directly.
    let mut dynamic: Vec<(u64, Vec<f32>, Vec<usize>)> = {
        let mut dynamic =
            Vec::with_capacity(1 + b.cos_srcs.len() + b.sin_srcs.len() + 2 * b.layers.len());
        // The embedding is this chunk's rows of the hidden state.
        dynamic.push((b.embed_src as u64, emb, vec![rows, sh.hidden]));
        // Each rope table is tiled to the width its consumer reads, from THE
        // SOURCE'S OWN CLASS's rows. `classes` is cos-then-sin, so a sin
        // source's class index is offset past the cos sources.
        let n_cos = b.cos_srcs.len();
        for (ci, &(cid, w, kind)) in b.cos_srcs.iter().enumerate() {
            dynamic.push((cid as u64, tile(&cos_tab[ci], w, kind), vec![rows, w]));
        }
        for (ci, &(sid, w, kind)) in b.sin_srcs.iter().enumerate() {
            dynamic.push((
                sid as u64,
                tile(&sin_tab[n_cos + ci], w, kind),
                vec![rows, w],
            ));
        }
        dynamic
    };
    let vocab = sh.vocab;

    // ⭐ THE PAGED STEP — the page table, the mask stack and the write rows, staged from the
    // request's blocks by `scratchy_target_spyre::paged_step` (the law the bake's numeric gate stages
    // by too), then one fused forward through the resident session. The pools themselves are not
    // staged: every layer's write scatters this step's K/V into them and its walk gathers from them,
    // on the device, step after step.
    {
        let step = b
            .paged
            .ok_or_else(|| werr("this bundle states no paged step — its attention has no pools"))?;
        let rows = scratchy_target_spyre::paged_step::StepRows::new(start, n).ok_or_else(|| {
            werr(format!(
                "request {id}: a decode step of zero rows — no token to run"
            ))
        })?;
        let srcs = scratchy_target_spyre::paged_step::request_sources(&step, &blocks, &rows)
            .map_err(|e| werr(format!("request {id}: paged step: {e}")))?;
        session
            .set_typed_sources(
                srcs.int32
                    .into_iter()
                    .map(|(tid, bytes, shape)| {
                        (tid as usize, bytes, scratchy_tensors::DType::I32, shape)
                    })
                    .collect(),
            )
            .map_err(|e| werr(format!("paged step: {e}")))?;
        let (mask_tid, mask, mask_shape) = srcs.mask;
        dynamic.push((u64::from(mask_tid), mask, mask_shape));
        // The result is the one readback, whole: the answer rows' logits, the host draws from.
        let outputs: Vec<(u64, usize)> = b.output_ids.iter().map(|id| (*id as u64, 0)).collect();
        let out = session
            .run_step(b.prog, dynamic, &outputs)
            .map_err(|e| werr(format!("run_step: {e}")))?;
        tokens.set_tokens_in_pool(id, start + n);
        // ⭐ THE ANSWER ROWS ARE THE RESULT'S FIRST ROWS: a prompt chunk's lm-head tail is folded to
        // its requests' last rows (request `r`'s logits at row `r`), a decode batch's rows are its
        // sequences'. The host draws each row's token under the request's own law; the caller
        // consumes the last row's, the last real token's forward.
        let logits = out.get(&(b.result_id as u64)).ok_or_else(|| {
            werr(format!("the logits t{} came back unread", b.result_id))
        })?;
        let answers = match b.paged.map(|p| p.paged_rows) {
            Some(scratchy_target_spyre::paged_step::PagedRows::Batch) => n,
            _ => b.paged.map_or(1, |p| p.extent.requests().get() as usize),
        };
        let picks = logits
            .chunks(vocab)
            .take(answers)
            .enumerate()
            .map(|(r, row)| {
                draw_token(
                    r,
                    row,
                    &RowLaw {
                        draw: req.draw.as_ref(),
                        step: (start + n - 1) as u32,
                    },
                )
            })
            .collect::<ExecutorResult<Vec<u32>>>()?;
        Ok(ChunkOut::Tokens(picks))
    }
}

/// The smallest prompt-chunk rung that reaches `need` positions, if any does. `rungs` is
/// ascending by reach.
pub(crate) fn select_prefill(rungs: &[BundleMeta], need: usize) -> Option<&BundleMeta> {
    rungs.iter().find(|b| b.reach() >= need)
}
