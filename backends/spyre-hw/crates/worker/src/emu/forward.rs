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
        // ⭐ THE CHAIN REPLACES THE EMBED ROW PUSH here too — a chained wiring's
        // gather reads the two persistent sticks, so the host's contribution is
        // their seed: the LAST REAL token's decompose (the token this forward
        // feeds — the same value the card path's ChainSeed steps state, and
        // idempotent with the device's own ChainStage writes for the same
        // reason: step t feeds the token step t-1 picked). ⭐ THE WORDS RIDE AS
        // THEIR VALUES, not their bits: the gather's eval arm reads them as
        // numbers (`s_stick[..] as usize / k`, `l_stick[..] as usize`) and so do
        // the emulator's doors, which write the same sticks numerically at the
        // model dtype. The words the gather reads are f16-exact: the s stick's
        // lane 0 is `k·s` (granite-3.1-2b: k = 4, s ≤ 768 — a multiple of 4
        // below 4096) and the l stick's word 0 is `l < 64`. A `f32::from_bits`
        // word is a denormal the f16 channel flushes to zero, which seeded every
        // decode step with token 0's row.
        match b.phase {
            scratchy_target_spyre::forward_tape::ForwardPhase::Decode { chain } => {
                // ⭐ THE SEEDS ARE PER-REQUEST-ROW AT THE FORWARD'S OWN WIDTH —
                // the ONE per-row spelling (`chain_s_seed` / `chain_l_seed`,
                // the same fns the card path's ChainSeed tape arm calls): row i
                // carries ITS OWN token's decompose, pad rows repeat the last
                // real token's (the same clamp the embed fill above uses,
                // idempotent with the device's own ChainStage writes). The s
                // seed is `[rows, lanes]` — one whole pattern stick per row —
                // and the l seed is `[rows·l_sticks, lanes]`, row i's ramp
                // sticks then row i+1's, the placement shape the mint states.
                // At `rows == 1` both pushes are the solo spelling
                // byte-for-byte.
                let lanes = scratchy_subtile::subtile_ir::CHAIN_IDX_BLOCK as usize;
                let l_sticks = scratchy_subtile::subtile_ir::chain_l_sticks(chain.hidden) as usize;
                let toks32: Vec<u32> = toks.iter().map(|&t| t as u32).collect();
                // The emulator plays a world of one: the whole table.
                let block = scratchy_subtile::subtile_ir::ChainBlock::whole(chain.block);
                let s_words =
                    scratchy_subtile::subtile_ir::chain_s_seed(chain.k, &toks32, rows, block);
                let l_words =
                    scratchy_subtile::subtile_ir::chain_l_seed(&toks32, chain.hidden, rows, block);
                dynamic.push((
                    chain.s_src as u64,
                    s_words.iter().map(|&w| w as f32).collect(),
                    vec![rows, lanes],
                ));
                dynamic.push((
                    chain.l_src as u64,
                    l_words.iter().map(|&w| w as f32).collect(),
                    vec![rows * l_sticks, lanes],
                ));
            }
            scratchy_target_spyre::forward_tape::ForwardPhase::Prefill => {
                // The embedding is this chunk's rows of the hidden state.
                dynamic.push((b.embed_src as u64, emb, vec![rows, sh.hidden]));
            }
        }
        // ⭐ THE COMPILE-TIME SCALARS, AT THEIR RESERVED TIDS. The programs read each
        // model constant (a ScalarMul multiplier, an RMSNorm epsilon or divisor) and the algebraic
        // identities `0`/`1` from a bound `[1,1]` tile rather than a KTIR immediate, so that ONE
        // program serves both consumers: `dxp_standalone` has no immediate operand, and an
        // `arith.constant` the card cannot read is not a constant the card has. They are ordinary
        // `func.arguments` here, so leaving one unbound is a zero, and a zero in this position is
        // silent — `x/(1+e)` becomes `x/e` and every scaled residual vanishes.
        for (i, v) in b.scalarmul_scales.iter().enumerate() {
            let tid = scratchy_target_spyre::lower_subtile_tape_to_superdsc::scalarmul_scale_tid(i);
            dynamic.push((u64::from(tid), vec![*v], vec![1, 1]));
        }
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
        // ⭐ ASK FOR THE ROWS THIS FORWARD WROTE, NOT THE WHOLE RESIDENT TENSOR. Decode and prefill
        // share one session, so every tensor is resident at the WIDEST program's row count: a
        // one-row decode step had its outputs decoded at the prefill's `m`. The result is the one
        // readback — `vocab` of the logits row, or, for a rung that draws (its result re-pointed at
        // the idx stick, `[rows, W]` at the seat's minted width), the whole stick (`0`).
        let result_elems = if b.sample_seat.is_some() { 0 } else { vocab };
        let wanted: std::collections::HashMap<u64, usize> =
            std::iter::once((b.result_id as u64, result_elems)).collect();
        let outputs: Vec<(u64, usize)> = b
            .output_ids
            .iter()
            .map(|id| (*id as u64, wanted.get(&(*id as u64)).copied().unwrap_or(0)))
            .collect();
        // ⭐ THE MINTED DRAW'S SEAT, STAGED FRESH EVERY STEP. The bake mints the token draw into
        // this rung (`Wiring::sample_seat`), and its five operands go stale the moment a step
        // completes — the step word IS the step's freshness, a held one is a held token. One law
        // per seat row, by the card's own prompt law: row `r` runs this request's draw at
        // `start + min(r, n - 1)`, a pad row clamped to the last real position (at decode `n = 1`
        // and the one row sits at `start`). The words are `seat_words`' — the one statement of the
        // law both tiers lay out — packed `[rows, 1]` at the views' own dtypes (i32 words, f32
        // inv_t), the memref-faithful layout the bake's numeric gate stages them in; the f16
        // channel `run_step` takes would encode them into half the bytes their views span.
        if let Some(seat) = b.sample_seat {
            let rows = seat.rows as usize;
            let draw = req.draw;
            let words: Vec<([i32; 4], f32)> = (0..rows)
                .map(|r| {
                    seat_words(&RowLaw {
                        draw: draw.as_ref(),
                        step: (start + r.min(n - 1)) as u32,
                    })
                })
                .collect();
            let mut typed: Vec<(usize, Vec<u8>, scratchy_tensors::DType, Vec<usize>)> = seat
                .seed_words
                .iter()
                .enumerate()
                .map(|(k, &tid)| {
                    (
                        tid as usize,
                        words.iter().flat_map(|(w, _)| w[k].to_le_bytes()).collect(),
                        scratchy_tensors::DType::I32,
                        vec![rows, 1],
                    )
                })
                .collect();
            typed.push((
                seat.invt as usize,
                words
                    .iter()
                    .flat_map(|(_, inv_t)| inv_t.to_le_bytes())
                    .collect(),
                scratchy_tensors::DType::F32,
                vec![rows, 1],
            ));
            session
                .set_typed_sources(typed)
                .map_err(|e| werr(format!("sampler seat: {e}")))?;
        }
        let out = session
            .run_step(b.prog, dynamic, &outputs)
            .map_err(|e| werr(format!("run_step: {e}")))?;
        tokens.set_tokens_in_pool(id, start + n);
        // ⭐⭐⭐ ONE LOGITS ROW, AT ROW 0 — because that is what the bundle COMPUTES.
        //
        // A prefill bundle does not run its vocab-wide lm_head at `mq`. `lower_one_node`'s
        // `is_prefill_lm_head_tail` arm folds it to the m=1 tail it really is
        // (main's `lower_prefill_lm_head_at_m1`): the activation is sliced to the LAST prompt row
        // and the output is narrowed to row 0. Only that
        // row's logits are ever read, and computing all `mq` rows of a 128k-column output would be
        // `mq`× the work for one row of answer.
        //
        // ⛔ SO `(0..n)` WAS A CLAIM THE BUNDLE NEVER MADE. Rows 1..n are never written, and the
        // caller takes the LAST of what it gets (`next_back()`), so a prompt chunk with n > 1
        // sampled an unwritten row — zeros — and generation continued from a token the model never
        // chose. It coincided with row 0 only at n == 1, which is why decode looked correct and a
        // prompt that fit ONE chunk looked correct. MEASURED: the result tensor comes back with
        // ~one row of `vocab` nonzeros and every argmax below `vocab`, i.e. in row 0.
        //
        // ⭐ THE PICK IS THE MINTED SAMPLER'S, READ BACK LIKE THE CARD'S. The bake mints the draw
        // into this rung and re-points its result at the idx stick — one word per seat row at lane
        // 0, every lane equal (the store is a whole stick per row) — so no host argmax runs: the
        // emulator's readback is memref-faithful, row `r`'s lane 0 is element `r·width`, and each
        // word passes the same token law the card's decode applies (`token_of_idx_word`). The
        // caller consumes the LAST row's pick, the folded tail's: the last real token's forward.
        match b.sample_seat {
            Some(seat) => {
                let idx = out.get(&(b.result_id as u64)).ok_or_else(|| {
                    werr(format!(
                        "the minted draw's idx stick t{} came back unread",
                        b.result_id
                    ))
                })?;
                let rows = seat.rows as usize;
                let width = idx.len() / rows.max(1);
                tracing::debug!(
                    "EMU_PICK t{} rows={rows} width={width} words={:?} seat={:?}",
                    b.result_id,
                    &idx[..idx.len().min(8)],
                    seat
                );
                let picks = (0..rows)
                    .map(|r| token_of_idx_word(r, idx[r * width] as i32, seat.vocab as usize))
                    .collect::<ExecutorResult<Vec<u32>>>()?;
                Ok(ChunkOut::Tokens(picks))
            }
            None => Ok(ChunkOut::Prompt),
        }
    }
}

/// The smallest prompt-chunk rung that reaches `need` positions, if any does. `rungs` is
/// ascending by reach.
pub(crate) fn select_prefill(rungs: &[BundleMeta], need: usize) -> Option<&BundleMeta> {
    rungs.iter().find(|b| b.reach() >= need)
}
