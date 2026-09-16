// SPDX-License-Identifier: Apache-2.0
//! KTIR bundle manifest + host-side source-fill helpers — the emulator-free
//! half of the Spyre run path. Depends only on `serde`, so it stays available
//! for emit-side / weight-loading code that never links `ktir_emulator` (which is
//! behind the optional `runner` feature). Execution lives in [`crate::runner`].

use anyhow::{Context, Result};
use std::path::Path;

/// ⭐ THE TWO AXES OF THE RUNG GRID, from the one crate that defines them.
///
/// Re-exported rather than redeclared: the BAKED ladder ([`crate::bundle_code::LadderRung`]) is keyed by
/// `SweptCols` and this manifest's `decode_rungs` by `RungSeqs`, and a second definition of either would
/// be a second way to spell the same count — which is exactly how the two axes became interchangeable.
pub use crate::bundle_code::{RungSeqs, SweptCols};

/// The registry sentinel naming a baked sibling bundle — `"SUPERDSC_BUNDLE:<fingerprint>"`.
///
/// ⛔ A NEWTYPE AND NOT A `&str` because every other string in this manifest is also a `&'static str`
/// (graph JSON, manifest JSON, a g2 fingerprint), and a sentinel is the one that must be `strip_prefix`ed
/// and materialised through the registry before it names anything. Passing a graph blob where a sentinel
/// belongs used to typecheck.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BundleSentinel(&'static str);

impl BundleSentinel {
    pub const fn new(s: &'static str) -> BundleSentinel {
        BundleSentinel(s)
    }

    /// The raw sentinel, for the ONE consumer that strips its prefix and materialises it.
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

/// The bundle's `manifest.json`: tensor shapes, the source-fill recipe, and the
/// per-node func/file/arg wiring.
#[derive(serde::Deserialize)]
pub struct Manifest {
    /// Tensor id of the final logits.
    pub result: usize,
    /// Tensor ids `0..num_sources` are inputs the caller fills.
    pub num_sources: u32,
    /// Runtime decode position for the self-check gate (= baked `valid_len − 1`):
    /// the host fills the mask to keep exactly this many prefix columns, so the
    /// bundle reproduces `eval_dag`'s `valid_len-1` prefix slice. A worker
    /// overrides it per generated token. Absent ⇒ 0.
    #[serde(default)]
    pub decode_position: u32,
    /// Tensor id of the AttnDecode runtime length mask, if the model has a
    /// maskable decode. The host fills it `[1, capacity]` (0 on valid columns,
    /// large-negative past `decode_position`) before running; it is NOT one of
    /// `num_sources` and is written by no node.
    #[serde(default)]
    pub attn_mask: Option<usize>,
    pub tensors: Vec<TensorMeta>,
    /// How to fill each source tensor (weights via `disk`/`loc`, else runtime).
    #[serde(default)]
    pub sources: Vec<SourceEntry>,
    pub nodes: Vec<NodeMeta>,
}

#[derive(serde::Deserialize)]
pub struct TensorMeta {
    pub id: usize,
    pub rows: usize,
    pub cols: usize,
    pub is_source: bool,
}

/// One source-fill instruction. `role` is `weight` / `embed` / `cos` / `sin` /
/// `prefix_k` / `prefix_v`.
#[derive(serde::Deserialize, Clone)]
pub struct SourceEntry {
    pub id: usize,
    pub role: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub base: Option<String>,
    /// On-disk safetensors key PREFIX (no `.weight`/`.scales` suffix). The
    /// worker appends the dtype-appropriate suffix; tied `lm_head` redirects to
    /// `model.embed_tokens.weight`.
    #[serde(default)]
    pub disk: Option<String>,
    #[serde(default)]
    pub layer: Option<u64>,
    #[serde(default)]
    pub is_gemm: Option<bool>,
    #[serde(default)]
    pub loc: Option<Loc>,
}

#[derive(serde::Deserialize, Clone)]
pub struct Loc {
    pub bucket: u32,
    pub op_idx: u32,
    pub slot: u32,
}

#[derive(serde::Deserialize)]
pub struct NodeMeta {
    /// KTIR func name (one func per node — `ktir_emulator`'s parser scopes args per
    /// func, so each node is parsed/executed on its own).
    #[serde(rename = "fn")]
    pub func: String,
    /// Relative path to this node's `.mlir` file in the bundle dir.
    pub mlir: String,
    pub args: Vec<ArgMeta>,
}

#[derive(serde::Deserialize)]
pub struct ArgMeta {
    pub name: String,
    pub tensor: usize,
    pub is_output: bool,
}

impl Manifest {
    pub fn load(dir: &Path) -> Result<Self> {
        let txt = std::fs::read_to_string(dir.join("manifest.json"))
            .with_context(|| format!("read manifest in {dir:?}"))?;
        Self::from_json(&txt)
    }

    /// Parse a manifest from in-memory JSON text — the path the macro-embedded
    /// bundle takes (no disk read). `load` reads the file then delegates here.
    pub fn from_json(txt: &str) -> Result<Self> {
        serde_json::from_str(txt).context("parse manifest.json")
    }

    /// Element count of tensor `id` (`rows * cols`).
    pub fn tensor_len(&self, id: usize) -> usize {
        let t = &self.tensors[id];
        t.rows * t.cols
    }

    /// `[rows, cols]` of tensor `id`.
    pub fn shape(&self, id: usize) -> Vec<usize> {
        let t = &self.tensors[id];
        vec![t.rows, t.cols]
    }
}

/// One emitted KTIR bundle, named by the FINGERPRINT its programs are registered under.
///
/// ⭐ A NAME, NOT A COPY. `#[forward]` bakes the programs themselves through `inventory::submit!`
/// as a `bundle_code::BundleCode`; this says WHICH one, and `bundle_code::bundle(fp)` resolves it —
/// the same registry the ladder rungs and re-rolled siblings already resolve through, so there is
/// one set of programs and one resolver.
///
/// ⛔ IT CARRIED THE PROGRAMS AS TEXT: a manifest string plus one `(func_name, mlir)` per node,
/// rendered by the emitter and parsed back by the runner in the same process. Both ends of that
/// round trip are gone — the programs are values, and the IO contract they need is the generated
/// `wiring::Wiring`, not a re-parsed manifest.
pub struct KtirBundleData {
    pub fp: &'static str,
}

/// The per-model embedded bundles: one m=1 decode bundle PER static cap bucket
/// (ascending capacity — the worker runs the smallest cap covering the current
/// position, since emulator decode is O(capacity)), and the optional m=M
/// batched-prefill bundle (baked at the max cap).
pub struct KtirBundle {
    pub decode: &'static [KtirBundleData],
    pub prefill: Option<KtirBundleData>,
}

/// One emitted **sendnn** bundle (the `--target sendnn` silicon path), baked
/// `&'static` by the `#[forward]` macro: the same [`Manifest`] IO contract the
/// KTIR path uses (`manifest_json`) plus the op-graph as a single JSON string
/// (`graph_json`) the DeepTools toolchain compiles. The PrimaryInputs are named
/// `t{id}` matching the manifest tensor ids, so the worker binds weights /
/// dynamic sources by id exactly as for KTIR.
pub struct SengraphBundleData {
    pub manifest_json: &'static str,
    pub graph_json: &'static str,
    /// Fingerprint of `graph_json` (lowercase hex of a stable content hash),
    /// keying the AHEAD-OF-TIME-compiled g2 blob baked by `scratchy-builder-spyre`
    /// (the peer of cuda's cudaforge `.a` cache). The runtime looks the g2 up by
    /// this fingerprint via `scratchy_builder_spyre::baked_g2(fingerprint)`; on a
    /// hit it loads `already_compiled=true` (NO `CompileGraph` at startup), on a
    /// miss it falls back to the in-process `CompileGraph` path. Empty for bundles
    /// emitted before the AoT cache existed (always falls back).
    pub g2_fingerprint: &'static str,
}

/// One LAYER GROUP's `[prefill, decode]` sub-bundle (the layer-group sub-bundle
/// split — see `split_by_layer_group`). The whole 30-layer decoder overflows the
/// card's per-job flit cap as one supernode, so the model is split into N groups
/// of ~`group_size` layers, each its OWN symbolic `[prefill, decode]` offline_decoder
/// bundle with its own paged KV. The host threads the hidden state `[seq, hidden]`
/// group→group: group 0 takes the embedding output; group g>0 takes the previous
/// group's output residual, bound via this group's `input_tensor` id (the worker
/// binds the previous group's `result_tensor` buffer to it — there is NO manifest
/// "source role"; the residual thread is the `input_tensor`/`result_tensor` id
/// linkage below). The final group's output feeds the host final-RMSNorm
/// (in-graph) + lm_head.
pub struct SengraphBundleGroup {
    /// 0-based group index; covers layers `[g*group_size, (g+1)*group_size)`.
    pub group: u32,
    /// Tensor id (`t{id}`) of this group's INCOMING hidden `PrimaryInput`. Group 0
    /// binds it from the embedding gather; group g>0 binds it from the previous
    /// group's output buffer (the host residual thread).
    pub input_tensor: u32,
    /// Tensor id of this group's OUTGOING hidden (the `out0` source). Equals the
    /// next group's `input_tensor`; the final group's == the model result (the
    /// post-final-RMSNorm hidden the host feeds to lm_head).
    pub result_tensor: u32,
    /// PREFILL sub-graph (idx 0 of this group's offline_decoder pair).
    pub prefill: SengraphBundleData,
    /// DECODE sub-graph (idx 1).
    pub decode: SengraphBundleData,
    /// PREFILL WIDTH LADDER `[(mq, sentinel)]`, ascending, top == [`prefill`](Self::prefill).
    ///
    /// A prefill bundle is baked at a FIXED query-row count `mq`, and per-chunk cost is
    /// `fixed + slope·mq` (measured on granite-3.1-2b: ~42 ms + ~0.7 ms/row, so the fixed term
    /// dominates). One width is therefore wrong in both directions: too narrow multiplies the fixed
    /// term across extra chunks, too wide computes padding rows that carry no tokens. The worker
    /// picks the SMALLEST rung ≥ the chunk's real token count, so neither happens.
    ///
    /// Unlike the decode `sk_bucket` ladder — which re-lowers ONE tape at a different attention
    /// sweep extent and so swaps only the body — `mq` also sizes the prefix (embed of `mq` rows) and
    /// the suffix, so every rung is a full 3-bundle set with its own sentinel. All rungs still share
    /// ONE resident weights+KV: no term in the weight or KV placement depends on `mq` (the same
    /// property that licenses the prefill↔decode seg2 alias).
    ///
    /// EMPTY ⇒ no ladder; the worker uses [`prefill`](Self::prefill) at its baked width.
    pub prefill_rungs: &'static [(u32, &'static str)],
    /// DECODE BATCH LADDER `[(seqs, sentinel)]`, ascending — the request counts the decode bundle is
    /// baked at.
    ///
    /// ⛔⛔⛔ KEYED BY [`RungSeqs`], NOT BY A BARE `u32` — because THREE lists have held the name
    /// `decode_rungs` and they are keyed by TWO DIFFERENT QUANTITIES: this one and the worker's by the
    /// BATCH WIDTH (`seqs`), and `bundle_code::RerollMeta::rungs` by the SWEEP EXTENT (`active_cap`,
    /// the 64/128/256 ladder — verified emitted at all three by running the emitter with
    /// `SCRATCHY_SDSC_FOLD_TRACE=1`). Both were `Vec<(u32, _)>`, so assigning one to the other, or
    /// `position(|r| r.0 >= live)` over the wrong one, was a type-correct way to select a body baked for a
    /// 64-column sweep because four requests are live. The previous mitigation was a RENAME plus three
    /// comments begging the reader to keep them apart; this makes the mix-up fail to compile.
    ///
    /// A decode step runs one token of each RUNNING request, and how many that is changes step to
    /// step: the scheduler decides it. A SuperDSC bundle is a static-shape graph and cannot take that
    /// number, so one rung is baked per width and the worker picks the SMALLEST rung >= the live
    /// request count. Without it the worker runs one forward per request and streams the whole weight
    /// set once per token per request, which is the entire cost of a decode step paid again for each.
    ///
    /// Each rung is the SAME tape at a wider row count — decode at mq=B is the prefill path's own
    /// emission, not a second implementation. Rungs share ONE resident weights+KV: no term in either
    /// placement depends on the row count.
    ///
    /// EMPTY ⇒ no ladder; the worker decodes one request per forward, as it always did.
    pub decode_rungs: &'static [(RungSeqs, SweptCols, BundleSentinel)],
    /// `(mq, sentinel)` of the ONE prefill bundle baked with a full resident-prefix sweep, for chunks
    /// with `start > 0`. EMPTY string ⇒ absent.
    ///
    /// Every [`prefill_rungs`](Self::prefill_rungs) entry is baked PREFIX-FREE: a chunk at `start == 0`
    /// has no resident KV to attend, so its 4 prefix blocks would be masked out in full — 308 of the
    /// 710 ops in a prefill layer (43%), about 12,320 launches per forward, launched to compute
    /// nothing. Dropping them is what makes the ladder rungs 402 ops/layer instead of 710.
    ///
    /// A CONTINUATION chunk does have resident KV and must attend it, so it needs this bundle instead.
    /// It is baked at the WIDEST rung so one bundle covers every `start <= cap`; such a chunk pays some
    /// padding, which is the right trade because it is the rarer case.
    pub prefill_prefix: (u32, &'static str),
}

/// The per-model sendnn bundle. With the layer-group split it is a vector of
/// `[prefill, decode]` sub-bundles (one per layer group); the worker compiles
/// each group independently and threads the hidden state on the host. A single-
/// group vector is the degenerate (whole-model, only if it fit) case.
pub struct SengraphBundle {
    /// Per-layer-group `[prefill, decode]` sub-bundles, in group order. The first
    /// group consumes the embedding; the last produces the final hidden.
    pub groups: &'static [SengraphBundleGroup],
}

/// One zeroed host buffer per tensor. The caller fills the source slots
/// (`id < num_sources`) before running the bundle.
pub fn alloc_buffers(manifest: &Manifest) -> Vec<Vec<f32>> {
    manifest
        .tensors
        .iter()
        .map(|t| vec![0.0f32; t.rows * t.cols])
        .collect()
}

/// Decode raw typed bytes to f32 (for host-side use like the embedding gather).
pub fn bytes_to_f32(bytes: &[u8], dt: scratchy_tensors::DType) -> Vec<f32> {
    use scratchy_tensors::DType;
    match dt {
        DType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        DType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_le_bytes(*c).to_f32())
            .collect(),
        DType::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::bf16::from_le_bytes(*c).to_f32())
            .collect(),
        other => panic!("bytes_to_f32: unsupported dtype {other:?}"),
    }
}

/// NeoX rotary `cos`/`sin` tables for one position, as the emitter's
/// `emit_rope` reads them: a `[head_dim]` row whose first half holds
/// `cos/sin(pos · θ^(-2i/head_dim))` for `i in 0..head_dim/2` (the second half is
/// unread). `theta` = rope θ (1e4 for SmolLM2). Returns `(cos, sin)`.
pub fn rope_cos_sin(pos: u32, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; head_dim];
    let mut sin = vec![0.0f32; head_dim];
    for i in 0..half {
        let inv_freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
        let ang = pos as f32 * inv_freq;
        let (s, c) = ang.sin_cos();
        cos[i] = c;
        sin[i] = s;
        cos[i + half] = c;
        sin[i + half] = s;
    }
    (cos, sin)
}

/// The additive length-mask row `[1, capacity]` the AttnDecode prefix scores
/// add: `0` on columns `< decode_position` (the valid prefix), a large negative
/// (narrows to f16 −inf ⇒ exp = 0 ⇒ those cache positions drop out of the
/// softmax) past it. `decode_position` is clamped to `capacity`. The emitter's
/// own softmax `-inf` init is `-1e38`, so the mask matches it.
pub fn attn_mask_fill(capacity: usize, decode_position: usize) -> Vec<f32> {
    let keep = decode_position.min(capacity);
    let mut m = vec![0.0f32; capacity];
    for x in m.iter_mut().skip(keep) {
        *x = -1.0e38;
    }
    m
}

/// THE PREFILL CHUNK'S CAUSAL MASK, `[mq, mq]` row-major: `0` where row `r` may attend column `c`,
/// a large negative elsewhere so it leaves the softmax through `exp(-inf) = 0`.
///
/// ⭐ THE TRIANGLE IS NOT RESTATED HERE. `prefill_causal_col_valid` is the SSOT — a prompt chunk's
/// rows are consecutive positions of ONE sequence, so row `r` legitimately attends every earlier
/// row. (A decode BATCH's rows are independent requests and take the diagonal instead; that is a
/// different predicate, and confusing the two is fluent output that read another request's tokens.)
pub fn attn_causal_mask_fill(mq: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; mq * mq];
    for r in 0..mq {
        for c in 0..mq {
            if !scratchy_subtile::sdsc_abstract::prefill_causal_col_valid(c, r) {
                m[r * mq + c] = -1.0e38;
            }
        }
    }
    m
}

/// argmax of a logits row.
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod rope_tests {
    use super::rope_cos_sin;

    /// The RoPE cos/sin table MUST repeat the half-frequencies (cos[i]==cos[i+half], sin likewise) so each
    /// rotation pair (d, d+half) shares one angle — the property the Kani `rope_forms_2d_rotation_per_pair`
    /// harness assumes (proven there: repetition + the ACTUAL `rope_p_entry` P ⇒ a proper norm-preserving
    /// 2D rotation). This test ties that modeled repetition to the ACTUAL `rope_cos_sin` output.
    /// `rope_cos_sin` uses libm `sin_cos` (the float ALU leaf, un-Kani-able), so the repetition — entangled
    /// with trig in the code — is locked here at the actual-code level (the Kani-analog for trig code, like
    /// the shim's C++ `static_assert`). Fail-first: dropping the `cos[i+half]=cos[i]` assignment breaks it.
    #[test]
    fn rope_cos_sin_repeats_half_frequencies() {
        for &(pos, hd, theta) in &[
            (0u32, 128usize, 1e7f32), // granite pos 0
            (1, 128, 1e7),
            (64, 128, 1e7), // granite decode step
            (255, 128, 1e7),
            (5, 64, 1e4), // a different hd/theta
        ] {
            let (cos, sin) = rope_cos_sin(pos, hd, theta);
            let half = hd / 2;
            for i in 0..half {
                assert_eq!(
                    cos[i].to_bits(),
                    cos[i + half].to_bits(),
                    "cos repeat pos={pos} hd={hd} i={i}"
                );
                assert_eq!(
                    sin[i].to_bits(),
                    sin[i + half].to_bits(),
                    "sin repeat pos={pos} hd={hd} i={i}"
                );
            }
        }
    }
}
