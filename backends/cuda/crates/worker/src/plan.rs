// SPDX-License-Identifier: Apache-2.0
//! ONE STEP, PLANNED ON THE HOST — everything the CUDA worker decides about a scheduler step
//! before it plays a tape, as plain functions with no toolchain.
//!
//! The worker runs one sequence per step (`max_num_seqs` is 1). Each step it takes the one
//! scheduled request ([`scheduled_request`]), picks the bucket — the compiled tape — that holds
//! the step's tokens ([`choose_bucket`]), checks its tokens end inside the KV cache's length
//! ([`Position`]), pads the token ids and positions to that bucket ([`fill_inputs`]), hands the
//! kernels the scheduler's block table and GDN slot as the worker's pools' own ([`block_table`],
//! [`gdn_slot`]), and when the step reaches the end of the request's history ([`emits_token`])
//! takes the greedy token of its last real row ([`greedy_token`]).
//!
//! Sampling is greedy only in this build: the API refuses anything else at its door, from the
//! factory's `served()`, so every request this plans for decodes greedily.

use std::collections::HashMap;
use std::fmt;

use scratchy_layers::gdn_state::{GdnStart, GdnStep};
use scratchy_target_cuda::pool::{BlockId, GdnPoolConfig, KvPoolConfig, SlotId};
use scratchy_target_cuda::step::{CUDA_KV_BLOCK_SIZE, CUDA_MAX_BLOCKS_PER_SEQ};
use scratchy_target_cuda::tape::KvTape;

/// Why a step could not be planned. Every message names what to change.
#[derive(Clone, Debug, PartialEq)]
pub enum PlanError {
    /// More than one request runs tokens this step; the worker runs one.
    ManyRequests { req_ids: Vec<String> },
    /// No compiled bucket holds `tokens` rows; the largest is `largest`.
    NoBucket { tokens: usize, largest: u32 },
    /// The step's tokens end at `tokens`, past the `max` a sequence's KV cache holds
    /// ([`CUDA_MAX_POSITIONS`]).
    KvTooLong { tokens: u64, max: u64 },
    /// Block-table entry `entry` is block `block`, outside the KV pool's `num_blocks`
    /// ([`KvPoolConfig::blocks`]).
    KvBlockOutsidePool {
        entry: usize,
        block: usize,
        num_blocks: u32,
    },
    /// The GDN slot allocator gave slot `slot`, outside the GDN pool's `num_slots`.
    GdnSlotOutsidePool { slot: usize, num_slots: u32 },
    /// The result holds `elems` bf16 values, not `bucket_m` rows of logits with a row `row`.
    LogitsRow {
        elems: usize,
        bucket_m: u32,
        row: usize,
    },
    /// Token `id`'s logit is NaN, so no argmax exists.
    NanLogit { id: usize },
    /// The GDN slot allocator's step code for a sequence at token `start` is `code`: not a plain
    /// step, or fresh other than exactly at token 0.
    GdnStart { start: u32, code: u32 },
    /// `--kv-cache-dtype` names a cache other than the e4m3 one this worker has.
    KvCacheDtype { dtype: String },
    /// `--calculate-kv-scales` is set; the KV scale is fixed at 1.0.
    KvScales,
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ManyRequests { req_ids } => write!(
                f,
                "the CUDA worker runs one sequence per step; max_num_seqs is 1, and this step \
                 runs {} requests ({})",
                req_ids.len(),
                req_ids.join(", ")
            ),
            Self::NoBucket { tokens, largest } => write!(
                f,
                "the step runs {tokens} tokens; the largest compiled CUDA bucket holds {largest}"
            ),
            Self::KvTooLong { tokens, max } => write!(
                f,
                "the sequence is {tokens} tokens; a block table holds at most {max}"
            ),
            Self::KvBlockOutsidePool {
                entry,
                block,
                num_blocks,
            } => write!(
                f,
                "block-table entry {entry} is block {block}, outside the pool's {num_blocks} \
                 blocks"
            ),
            Self::GdnSlotOutsidePool { slot, num_slots } => write!(
                f,
                "the GDN sequence's slot is {slot}, outside the pool's {num_slots} slots"
            ),
            Self::LogitsRow {
                elems,
                bucket_m,
                row,
            } => write!(
                f,
                "the result holds {elems} logits, not {bucket_m} rows of logits with a row {row}"
            ),
            Self::NanLogit { id } => {
                write!(
                    f,
                    "token {id}'s logit is NaN: the forward produced no argmax"
                )
            }
            Self::GdnStart { start, code } => write!(
                f,
                "the GDN step from token {start} is {:?}: a plain step is fresh exactly at token 0",
                GdnStep::decode(*code)
            ),
            Self::KvCacheDtype { dtype } => write!(
                f,
                "the CUDA backend does not support a KV cache of {dtype}: its KV cache is e4m3 at \
                 a fixed scale of 1.0 (--kv-cache-dtype auto, fp8 or fp8_e4m3)"
            ),
            Self::KvScales => write!(
                f,
                "the CUDA backend does not compute KV scales: its KV scale is fixed at 1.0 (drop \
                 --calculate-kv-scales)"
            ),
        }
    }
}

impl std::error::Error for PlanError {}

/// The one request this step runs tokens for and how many, from the scheduler's
/// `num_scheduled_tokens`: `None` when the step runs none, and a refusal naming them when it runs
/// more than one.
pub fn scheduled_request(
    num_scheduled_tokens: &HashMap<String, usize>,
) -> Result<Option<(&str, usize)>, PlanError> {
    let mut running = (num_scheduled_tokens.iter()).filter(|&(_, &n)| n > 0);
    match (running.next(), running.next()) {
        (None, _) => Ok(None),
        (Some((id, &n)), None) => Ok(Some((id.as_str(), n))),
        (Some(_), Some(_)) => {
            let mut req_ids: Vec<String> = (num_scheduled_tokens.iter())
                .filter(|&(_, &n)| n > 0)
                .map(|(id, _)| id.clone())
                .collect();
            req_ids.sort();
            Err(PlanError::ManyRequests { req_ids })
        }
    }
}

/// The bucket a step of `tokens` rows runs in: of the compiled tapes' `bucket_ms` (in whatever
/// order the tapes are), the index and rows of the smallest that holds them. One token is the
/// one-row decode tape; more is the smallest prefill tape at least that long.
pub fn choose_bucket(bucket_ms: &[u32], tokens: usize) -> Result<(usize, u32), PlanError> {
    (bucket_ms.iter().copied().enumerate())
        .filter(|&(_, m)| m as usize >= tokens)
        .min_by_key(|&(_, m)| m)
        .ok_or(PlanError::NoBucket {
            tokens,
            largest: bucket_ms.iter().copied().max().unwrap_or(0),
        })
}

/// The tokens a sequence's KV cache holds: [`CUDA_MAX_BLOCKS_PER_SEQ`] blocks of
/// [`CUDA_KV_BLOCK_SIZE`].
pub const CUDA_MAX_POSITIONS: u32 = CUDA_MAX_BLOCKS_PER_SEQ * CUDA_KV_BLOCK_SIZE;

/// A step's first position, whose tokens end inside the KV cache's length
/// ([`CUDA_MAX_POSITIONS`]). Only [`Self::of`] makes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Position(u32);

impl Position {
    /// `start` for a step of `tokens`, or `KvTooLong` when they end past the cache's length.
    pub fn of(start: u32, tokens: usize) -> Result<Self, PlanError> {
        let end = u64::from(start) + tokens as u64;
        let max = u64::from(CUDA_MAX_POSITIONS);
        if end > max {
            return Err(PlanError::KvTooLong { tokens: end, max });
        }
        Ok(Self(start))
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The tape's inputs for a step of `chunk` (the real token ids) from position `start` in a bucket
/// of `bucket_m` rows, into the worker's reusable buffers: the ids padded with 0 to `bucket_m`,
/// and the positions `start .. start + bucket_m`, the padding rows' included.
pub fn fill_inputs(
    chunk: &[u32],
    start: Position,
    bucket_m: u32,
    ids: &mut Vec<u32>,
    positions: &mut Vec<u32>,
) {
    debug_assert!(
        chunk.len() <= bucket_m as usize,
        "the bucket holds the chunk"
    );
    ids.clear();
    ids.extend_from_slice(chunk);
    ids.resize(bucket_m as usize, 0);
    positions.clear();
    // The real rows end inside the cache's length (`Position`); a padding row's position is
    // never read past the step's tokens, and saturates only for a bucket of 2³² − 8192 rows.
    positions.extend((0..bucket_m).map(|row| start.get().saturating_add(row)));
}

/// The scheduler's block ids as blocks of the worker's KV pool `pool`, into `out`. A block outside
/// it is refused.
pub fn block_table(
    blocks: &[usize],
    pool: KvPoolConfig,
    out: &mut Vec<BlockId>,
) -> Result<(), PlanError> {
    out.clear();
    for (entry, &block) in blocks.iter().enumerate() {
        let id = BlockId::of(block, pool).ok_or(PlanError::KvBlockOutsidePool {
            entry,
            block,
            num_blocks: pool.blocks(),
        })?;
        out.push(id);
    }
    Ok(())
}

/// The GDN slot allocator's `slot` as a slot of the worker's GDN pool `pool`. A slot outside it is
/// refused.
pub fn gdn_slot(slot: usize, pool: GdnPoolConfig) -> Result<SlotId, PlanError> {
    SlotId::of(slot, pool).ok_or(PlanError::GdnSlotOutsidePool {
        slot,
        num_slots: pool.num_slots,
    })
}

/// Whether a step of `tokens` from `start` samples a token: it reaches the end of the request's
/// history of `history` tokens (the prompt's last chunk, or a decode). A prefill chunk short of
/// the prompt's end samples nothing.
pub fn emits_token(start: usize, tokens: usize, history: usize) -> bool {
    start + tokens >= history
}

/// Row `row` of a `[bucket_m, vocab]` result of bf16 logits.
pub fn logits_row(result: &[u16], bucket_m: u32, row: usize) -> Result<&[u16], PlanError> {
    let refuse = PlanError::LogitsRow {
        elems: result.len(),
        bucket_m,
        row,
    };
    let rows = bucket_m as usize;
    if rows == 0 || row >= rows || !result.len().is_multiple_of(rows) {
        return Err(refuse);
    }
    let vocab = result.len() / rows;
    Ok(&result[row * vocab..(row + 1) * vocab])
}

/// The greedy token of a row of bf16 logits: its argmax, the lowest id on a tie. A NaN logit is
/// refused rather than ranked.
pub fn greedy_token(row: &[u16]) -> Result<u32, PlanError> {
    let mut best: Option<(usize, f32)> = None;
    for (id, &bits) in row.iter().enumerate() {
        // bf16 is the top half of an f32.
        let v = f32::from_bits(u32::from(bits) << 16);
        if v.is_nan() {
            return Err(PlanError::NanLogit { id });
        }
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((id, v));
        }
    }
    let (id, _) = best.ok_or(PlanError::LogitsRow {
        elems: 0,
        bucket_m: 1,
        row: 0,
    })?;
    Ok(id as u32)
}

/// The greedy token the device's argmax wrote (`CudaWorker::forward_greedy`, W7): [`greedy_token`]
/// of the same row, the token id, or its `−1 − i` for a row holding a NaN, `i` its lowest NaN id,
/// refused as [`greedy_token`] refuses it.
pub fn greedy_code(code: i32) -> Result<u32, PlanError> {
    match u32::try_from(code) {
        Ok(id) => Ok(id),
        Err(_) => Err(PlanError::NanLogit {
            id: (-1 - i64::from(code)) as usize,
        }),
    }
}

/// The bytes one KV block of `block_size` tokens costs in a pool of `kv`'s geometry: per KV
/// layer a K and a V block of its own class's `kv_heads · block_size · head_dim` e4m3, one byte
/// each (Gemma's sliding and global layers differ), from the pool's own layout
/// ([`KvTape::pool_layout`]). `None` on overflow.
pub fn kv_block_bytes(kv: &KvTape, block_size: usize) -> Option<usize> {
    let (_, one_block) = kv.pool_layout(1)?;
    let per_token = one_block.checked_div(u64::from(kv.block_size))?;
    usize::try_from(per_token).ok()?.checked_mul(block_size)
}

/// Refuse a KV cache this worker does not have. Its cache is a fixed fact of the target, e4m3 at a
/// per-tensor scale of 1.0 (vLLM's default for a checkpoint without KV scales): `auto`, `fp8` and
/// `fp8_e4m3` name it, every other dtype is refused, and so are computed KV scales.
pub fn check_kv_cache_dtype(dtype: &str, calculate_kv_scales: bool) -> Result<(), PlanError> {
    if !matches!(dtype, "auto" | "fp8" | "fp8_e4m3") {
        return Err(PlanError::KvCacheDtype {
            dtype: dtype.to_string(),
        });
    }
    if calculate_kv_scales {
        return Err(PlanError::KvScales);
    }
    Ok(())
}

/// Whether a sequence's GDN state starts from zero this step, from the slot allocator's step
/// `code` (`GdnSlotAllocator::step`) for a sequence at token `start`: fresh exactly at token 0,
/// whatever the step's bucket (a one-token prompt runs fresh on the one-row decode tape), and a
/// plain step, with no drafts or checkpoints. Anything else is refused.
pub fn gdn_fresh(code: u32, start: u32) -> Result<bool, PlanError> {
    let step = GdnStep::decode(code);
    let fresh = step.start == GdnStart::Fresh;
    if !step.is_plain() || fresh != (start == 0) {
        return Err(PlanError::GdnStart { start, code });
    }
    Ok(fresh)
}

/// Why a checkpoint's `quantization_config` is not one CUDA's FP8 tapes read: per-channel e4m3
/// weights (`weights.strategy: "channel"`, no block structure), dynamic per-token activations
/// (`input_activations.strategy: "token"`, `dynamic: true`), and no KV-cache scheme. `field` is
/// the config's path to the value, `found` the value as the checkpoint states it.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantRefusal {
    pub field: String,
    pub found: String,
    pub want: &'static str,
}

impl fmt::Display for QuantRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { field, found, want } = self;
        write!(
            f,
            "the checkpoint's quantization_config.{field} is {found}; the CUDA backend's FP8 \
             tapes need {want} (per-channel e4m3 weights, dynamic per-token activations, no KV \
             scheme)"
        )
    }
}

impl std::error::Error for QuantRefusal {}

/// Refuse a checkpoint `config.json` (`config`, the whole file) whose `quantization_config` CUDA's
/// FP8 path does not read; one with none (a bf16 checkpoint) passes, its weight table's dtype
/// check deciding the rest. Each refusal names its field.
///
/// Block scaling (`weight_block_size`, or a group's `weights.block_structure`, e.g. 128×128
/// blocks) is refused here and nowhere else; when an `Fp8Block` weight kind and its kernels land
/// (`steps_from_tape::gemm`'s weight-kind match), this is where the refusal lifts.
pub fn check_quantization_config(config: &serde_json::Value) -> Result<(), QuantRefusal> {
    use serde_json::Value;
    let Some(q) = config.get("quantization_config").filter(|q| !q.is_null()) else {
        return Ok(());
    };
    let refuse = |field: String, found: Option<&Value>, want| QuantRefusal {
        field,
        found: found.map_or_else(|| "absent".to_string(), Value::to_string),
        want,
    };
    if let Some(v) = q.get("weight_block_size").filter(|v| !v.is_null()) {
        return Err(refuse("weight_block_size".into(), Some(v), "none"));
    }
    if let Some(v) = q.get("kv_cache_scheme").filter(|v| !v.is_null()) {
        return Err(refuse("kv_cache_scheme".into(), Some(v), "null"));
    }
    let groups = q.get("config_groups").and_then(Value::as_object);
    let Some(groups) = groups.filter(|g| !g.is_empty()) else {
        return Err(refuse(
            "config_groups".into(),
            q.get("config_groups"),
            "a group",
        ));
    };
    for (name, g) in groups {
        let at = |path: &str| format!("config_groups.{name}.{path}");
        let w = g.get("weights");
        if let Some(v) = w
            .and_then(|w| w.get("block_structure"))
            .filter(|v| !v.is_null())
        {
            return Err(refuse(at("weights.block_structure"), Some(v), "null"));
        }
        let strategy = w.and_then(|w| w.get("strategy"));
        if strategy.and_then(Value::as_str) != Some("channel") {
            return Err(refuse(at("weights.strategy"), strategy, "\"channel\""));
        }
        let a = g.get("input_activations");
        let strategy = a.and_then(|a| a.get("strategy"));
        if strategy.and_then(Value::as_str) != Some("token") {
            return Err(refuse(
                at("input_activations.strategy"),
                strategy,
                "\"token\"",
            ));
        }
        let dynamic = a.and_then(|a| a.get("dynamic"));
        if dynamic.and_then(Value::as_bool) != Some(true) {
            return Err(refuse(at("input_activations.dynamic"), dynamic, "true"));
        }
    }
    Ok(())
}
