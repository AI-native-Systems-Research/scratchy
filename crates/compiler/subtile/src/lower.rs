// SPDX-License-Identifier: Apache-2.0
//! `LoweringInput` — flat, topologically-ordered, backend-agnostic
//! description of a solved forward.
//!
//! One [`OpDesc`] per op, each naming its kind, row count, and inputs
//! (either a prior op's output or an external source — a
//! weight/activation/extern). It is deliberately a *plain data* mirror
//! of the macro crate's internal `Fuf` + solver `Assignment`: the
//! macro crate (which alone can see those types) does the trivial
//! structural translation and hands the result to
//! [`crate::subtile_ir::lower_region`], so all the decomposition logic
//! lives in this Mac-testable crate.
//!
//! [`fuse_silu_mul`] is a pre-pass that fuses adjacent `Silu` → `Mul`
//! into [`LoweredOp::SiluMul`] before lowering — the GPU has only the
//! fused arm, so fusion must happen before scheduling places the pair
//! on workers.
//!
//! Slated for deletion in plan §4 commit 7 (when `to_wavefront.rs`
//! builds [`crate::subtile_ir::SubtileIR`] directly).

use crate::subtile_ir::SourceShape;
use ktir_superdsc::head_counts::{HeadDim, ModelAttnGeometry};

/// One input edge of an op: a prior op's output, or an external source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputRef {
    /// Output of op `ops[idx]` (must be `< this op's index`).
    Op(usize),
    /// External source `sources[idx]` — a weight, activation, or extern.
    Ext(usize),
}

/// How a [`LoweredOp::Gemm`]'s WEIGHT is stored and matmul'd — the weight's quantization scheme AS A
/// TYPE. Deliberately NOT a bool: a bool can only say fp8-or-not, and the moment int4 weights land
/// ("we will have int4 weights soon") a second `weight_int4: bool` makes `weight_fp8 && weight_int4` a
/// nonsensical-but-constructible state. The `#[forward]` macro maps this at EXPANSION time from the
/// weight's `StorageFormat` (the scheme is already known at compile time — no need to launder it through
/// a runtime bool), and the SDSC emitter matches it EXHAUSTIVELY: adding a scheme is a `cargo build`
/// error at every lowering site until it is handled (guard-driven), never a silent misread-as-dense.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmWeight {
    /// Dense bf16/fp16 weight. Matmul inputs: `[act, weight]` (arity-2). The path every dense model takes.
    Dense,
    /// compressed-tensors fp8-dynamic: per-CHANNEL static fp8 weight + per-TOKEN dynamic fp8 activation.
    /// Matmul inputs: `[act, weight_fp8, weight_scale]` (arity-3). matmulfp8 (fp8×fp8→fp16) + dequant by
    /// `w_scale[n]·a_scale[m]`. Lowered by the fp8 branch in `lower_subtile_tape_to_superdsc`.
    Fp8Dynamic,
    /// MLX-affine packed weight (`bits` codes per element, one scale+bias per
    /// `group_size` elements of K). Matmul inputs stay arity-2 `[act, weight]`: the
    /// scales and biases resolve from the SAME weight source under different tensor
    /// roles, so they are not separate IR operands. Metal realizes it with the qmv
    /// family at decode and qmm_t at prefill. See [`AffineInt4`] for what is checked.
    Affine { affine: AffineInt4 },
}

/// Bits per packed weight element. Minted only for widths a kernel family exists for,
/// so a checkpoint declaring an unsupported width is refused at lowering rather than
/// unpacked with the wrong stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct QuantBits(u32);

impl QuantBits {
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Elements of K sharing one scale/bias pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GroupSize(u32);

impl GroupSize {
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// ⭐ AN MLX-AFFINE INT4 WEIGHT'S QUANTIZATION GEOMETRY — minted only by
/// [`AffineInt4::mint`], the ONE place `bits`, `group_size` and the matmul's `k` are
/// checked against each other.
///
/// ⛔ THE GROUP MUST TILE K, AND NOTHING DOWNSTREAM RE-CHECKS IT. The qmv kernels index
/// `scales[n][k / group_size]`; a `k` that is not a whole number of groups reads one
/// group past the end of the last row — which lands inside the NEXT output channel's
/// scales, so it dequantizes with a plausible wrong scale and produces fluent, wrong
/// output rather than faulting. `bits` is checked here for the same reason: the packing
/// stride is `32 / bits` elements per word, and a width no kernel exists for would
/// silently unpack at the wrong stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineInt4 {
    bits: QuantBits,
    group: GroupSize,
}

impl AffineInt4 {
    /// The packed widths a metal qmv kernel family exists for.
    const SUPPORTED_BITS: [u32; 2] = [4, 8];

    /// `Some` iff `bits` names a kernel family and `group_size` tiles `k` exactly.
    pub const fn mint(bits: u32, group_size: u32, k: u32) -> Option<Self> {
        if group_size == 0 || k == 0 || !k.is_multiple_of(group_size) {
            return None;
        }
        // `const fn` cannot iterate a slice; state the membership directly.
        if bits != Self::SUPPORTED_BITS[0] && bits != Self::SUPPORTED_BITS[1] {
            return None;
        }
        Some(Self {
            bits: QuantBits(bits),
            group: GroupSize(group_size),
        })
    }

    pub const fn bits(self) -> QuantBits {
        self.bits
    }

    pub const fn group(self) -> GroupSize {
        self.group
    }
}

/// The op a node performs, with the shape parameters the decomposition
/// needs. Grows alongside [`SubOp`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoweredOp {
    /// `out[M, n] = act[M, k] @ W[n, k]^T`. Inputs: `[act, weight]`.
    /// `k` is derived from the activation's column count at lowering
    /// time — it is not a separate field. (Carrying `k` separately
    /// would require a runtime `assert_eq!(in0_cols, k, …)` to defend
    /// against producer/consumer drift; that proof lives
    /// either on the producing op's column witness or as a structural
    /// derivation — never as a runtime assert.)
    /// `weight` = the weight's quantization scheme (see [`GemmWeight`]), which fixes the matmul input
    /// arity + the on-card op sequence. [`GemmWeight::Dense`] is the ordinary fp16 gemm (`[act, weight]`).
    Gemm { n: u32, weight: GemmWeight },
    /// `out[M, d] = rmsnorm(x, weight + gain_offset, eps)`. Inputs:
    /// `[x, weight]`. `gain_offset` carries the Gemma-family `(1+w)`
    /// convention as DATA on the tape (0.0 everywhere else) — each
    /// target's rmsnorm applies it; a target without an offset form
    /// refuses a non-zero value.
    RmsNorm { eps: f32, gain_offset: f32 },
    /// `silu(x)`. Input: `[x]`. Shape-preserving.
    Silu,
    /// `gelu(x)` (tanh approximation). Input: `[x]`. Shape-preserving.
    Gelu,
    /// `softcap * tanh(x / softcap)` — the logit/attention soft cap
    /// (the CAP value is a model const). Input: `[x]`. Shape-preserving.
    TanhSoftCap,
    /// Unit-gain RMSNorm (no learnable scale — Gemma4 `v_norm`).
    /// Input: `[x]`. Shape-preserving; per-head when the input routes
    /// through a [`LoweredOp::Reshape`] view.
    RmsNormUnit { eps: f32 },
    /// Multiply by a loaded `[1]`-shaped weight (Gemma4
    /// `layer_scalar[layer]`). Inputs: `[x, weight]`. Shape-preserving.
    ScalarWeightMul,
    /// Qwen3.5 attention output-gate split: deinterleave the doubled
    /// q_proj output per head into `[query | gate]`. Inputs: `[qg]`.
    /// TWO outputs — slot 0 = q (this op's width), slot 1 = gate
    /// (recorded as a second producer entry, like the rope triple).
    GateSplit { half_cols: u32 },
    /// Qwen3.5 attention output gate: `out = attn * sigmoid(gate)`.
    /// Inputs: `[attn, gate]`. Shape-preserving.
    GateApply,
    /// Qwen3.5-MoE shared-expert combine:
    /// `out = routed + shared * sigmoid(g)` (`g` is `[T, 1]`,
    /// row-broadcast). Inputs: `[routed, shared, g]`. Shape-preserving.
    GateScale,
    /// Vision pixel load: the host-staged patchified pixel buffer
    /// lands in the op's slot. No inputs; output `[T, in_features]`.
    LoadPixels { in_features: u32 },
    /// Row permutation by a host-staged index table: Qwen2.5-VL's
    /// window attention gathers tokens into window order on encoder
    /// entry (`indices_kind` 0 = `window_index`) and back to natural
    /// order after the merger (1 = `reverse_indices`). The table is a
    /// runtime input, not an operand, so the op takes ONE input.
    /// Shape-preserving.
    EmbeddingGather { indices_kind: u8 },
    /// Vision position embeddings: the host-staged (interpolated,
    /// per-image-grid) table lands in the op's slot, for the
    /// `add(pos_embeds, hidden)` that follows. No inputs; output
    /// `[T, width]`. Like `LoadPixels` this is SYNTHESIZED by vision
    /// lowering, never written in a DSL body.
    LoadPosEmbeds { width: u32 },
    /// Vision 2D rope pair-rotation over q/k, in place. Inputs:
    /// `[q, k]`; TWO outputs (slots 0/1 = q/k, same buffers).
    VisionRope,
    /// `x * sigmoid(1.702x)`. Input: `[x]`. Shape-preserving.
    QuickGelu,
    /// Exact-erf gelu (the vision patch-merger MLP). Input: `[x]`.
    /// Shape-preserving.
    GeluErf,
    /// Varlen bidirectional vision attention. Inputs: `[q, k, v]`;
    /// `cu_kind` selects the cu_seqlens/max_seqlen extern set
    /// (0 default / 1 full / 2 window). Output = q width.
    VarlenAttention { cu_kind: u8 },
    /// Bidirectional encoder attention (no KV cache): contiguous
    /// Q/K/V from upstream slots, non-causal. Inputs: `[q, k, v]`.
    /// Output `[T, q_width]`.
    EncoderAttn { geom: ModelAttnGeometry, scale: f32 },
    /// Gated-DeltaNet linear attention (Qwen3.5 / Qwen3-Next). Inputs:
    /// `[qkv, z, a, b, weight_bundle]` (the bundle is the opaque
    /// `linear_attn[layer]` struct source). Output `[T, value_dim]`;
    /// the conv/ssm state is runtime-ambient per layer.
    GatedDeltaNet,
    /// Gemma-4 fused MoE block: router (with its own leading rmsnorm +
    /// temperature softmax + per-expert score scale) over `router_in`,
    /// GeGLU experts over `expert_in`. Inputs:
    /// `[router_in, expert_in, router_bundle, experts_bundle]`.
    /// Shape-preserving (out width = hidden).
    GemmaMoe {
        num_experts: u32,
        top_k: u32,
        moe_inter: u32,
        group_size: u32,
        bits: u32,
    },
    /// Fused mixture-of-experts block: router → top-k → gathered
    /// expert MLPs → weighted sum (+ optional shared-expert tail).
    /// Inputs: `[x, moe_weight_bundle]`. Shape-preserving (out width =
    /// hidden). All structural facts ride the op; the weight bundle
    /// resolves as ONE source (router/experts/shared under one base).
    Moe {
        /// Qwen order (`softmax → topk → gather`, optional renorm +
        /// shared expert) vs Mixtral order (`topk → softmax`).
        qwen_shared: bool,
        num_experts: u32,
        top_k: u32,
        moe_inter: u32,
        shared_inter: u32,
        norm_topk: bool,
        /// Per-expert affine quantization (metal requires it).
        group_size: u32,
        bits: u32,
    },
    /// `a * b`. Inputs: `[a, b]`. Shape-preserving.
    Mul,
    /// `x * scale` where `scale` is a compile-time constant. Input: `[x]`.
    /// Shape-preserving. Emitted for architectures that scale an activation
    /// by a scalar hyperparameter — e.g. Granite's `embedding_multiplier`,
    /// `residual_multiplier`, and `recip(logits_scaling)` (see
    /// `arch/granite`). The constant rides in the type (not a runtime
    /// operand), so there is no scalar tensor to resolve and no
    /// producer/consumer drift to defend against. Mirrors the interpreter
    /// IR's `ir::Instruction::ScalarMul(_, _, f32)` that the metal / cuda
    /// backends already lower.
    ScalarMul { scale: f32 },
    /// Fused `silu(gate) * up`. Inputs: `[gate, up]`. Shape-preserving.
    /// Produced by [`fuse_silu_mul`] from an adjacent `Silu` + `Mul`; maps
    /// to the GPU's fused `silu_mul` arm.
    SiluMul,
    /// `a + b` (e.g. the residual). Inputs: `[a, b]`. Shape-preserving.
    Add,
    /// Row mean: `out[T, 1] = mean(x, axis=-1)`. Inputs: `[x]`.
    Mean,
    /// `a - b` with `[T, 1]` broadcast on b (the mean-centering).
    /// Inputs: `[a, b]`. Shape of a.
    Sub,
    /// Row-broadcast bias add: `out[m, n] = in[m, n] + bias[n]`.
    /// Inputs: `[act, bias]` (bias is a `[1, n]` weight source).
    /// Shape-preserving.
    BiasAdd,
    /// A VIEW: re-interpret the input as `[m * rows_mult / rows_div,
    /// cols]` (rows scale with the token count; the vision patch
    /// merger DIVIDES by its merge factor). Inputs: `[x]`. Metadata —
    /// no device dispatch; per-head norms (qwen3 qk-norm) read their
    /// row multiplier from it.
    Reshape {
        rows_mult: u32,
        rows_div: u32,
        cols: u32,
    },
    /// NeoX rotary over `[M, heads * head_dim]`. Inputs: `[x, cos, sin]`,
    /// where cos/sin are the new token's position rows `[1, head_dim]`.
    /// Shape-preserving.
    RopeRotate { head_dim: HeadDim },
    /// The K-side `rope_append` for the GPU megakernel: rotate K + write the
    /// rotated K / un-rotated V into the paged KV cache for `layer`. Inputs:
    /// `[K, cos, sin, V]`. Host eval is rotation only (the cache write is
    /// GPU-only — see [`SubOp::RopeAppend`]); shape-preserving on K. The macro
    /// emits this for `rope_append`'s K slot; the Q slot stays `RopeRotate`.
    RopeAppend {
        head_dim: HeadDim,
        layer: u32,
        /// Selects the attention-geometry class on hybrid
        /// sliding/global arches (mirrors the runtime instruction's
        /// flag); `true` on uniform models.
        is_global: bool,
        /// GPT-NeoX pairing (`false`) vs interleaved pairing (`true`,
        /// commandr). Rides to the rope AND its attention consumer.
        interleaved: bool,
    },
    /// Fused decode attention. Inputs: `[Q, (K_seg, V_seg)...]` —
    /// concatenated along the KV axis. The fused decode passes the prefix
    /// cache as `Ext` (`Source`) segments and the rotated new token as
    /// `Op` (`Sub`) segments, so the new K/V is an internal dataflow edge,
    /// never a cache round-trip. Output: `[M, geom.q_width()]`.
    AttnDecode {
        /// The model's head geometry, minted once from its `config.json` — the head counts and the
        /// head dim as ONE value whose GQA grouping is already proven, never three integers a
        /// struct literal could transpose.
        geom: ModelAttnGeometry,
        scale: f32,
        /// Logical valid sequence length = `decode_position + 1` (the
        /// prefix positions actually written into the KV cache, plus the
        /// new token). DECOUPLED from the cache TENSOR's row capacity: the
        /// cache may be allocated `[MAX_POSITION, kv]` with only the first
        /// `valid_len` rows holding real K/V and the rest uninitialized.
        /// Drives the `SameForwardRopeAppend` prefix-segment slice so
        /// `eval_dag` attends to EXACTLY `valid_len` positions regardless
        /// of cache capacity (the host-verification pin); the GPU length
        /// mask uses the runtime `DecodePosition` kernel arg, which equals
        /// `valid_len - 1` at the same decode step.
        valid_len: u32,
        /// Sliding-window class (the window LENGTH is a model const;
        /// the class is per-layer data). `false` = full attention.
        sliding: bool,
    },
}

/// One op in the forward.
#[derive(Clone, Debug)]
pub struct OpDesc {
    pub op: LoweredOp,
    /// Row count (M); decode is 1.
    pub m: u32,
    pub inputs: Vec<InputRef>,
}

/// A whole forward, ready to lower.
#[derive(Clone, Debug)]
pub struct LoweringInput {
    /// External tensors (weights / activations / externs), indexed by
    /// `InputRef::Ext` and by `SourceId` in the produced graph.
    pub sources: Vec<SourceShape>,
    /// Ops in topological order; op `i` may reference ops `< i`.
    pub ops: Vec<OpDesc>,
    /// Which op's output is the forward's result.
    pub result: usize,
}

// ── Silu + Mul → SiluMul fusion ─────────────────────────────────────

/// Fuse each adjacent `Silu` → `Mul(silu, up)` into one
/// [`LoweredOp::SiluMul`] (SwiGLU). The GPU has only a *fused* `silu_mul`
/// arm (no standalone silu), and fusion must happen **before scheduling**
/// so the pair lands on one worker — the serializer otherwise rejects a
/// standalone `Silu`. A `Silu` is fused only when its output feeds exactly
/// one consumer (that `Mul`) and is not the forward result; everything else
/// is left untouched. Bit-exact: `SiluMul` computes `silu(gate) * up`,
/// identical to the two separate ops, so `eval_dag` is unchanged.
///
/// `Mul` is commutative, so the `Silu`-producing operand becomes `gate` and
/// the other becomes `up` regardless of input order. Op references are
/// re-indexed after the dropped `Silu`s are removed.
pub fn fuse_silu_mul(input: &LoweringInput) -> LoweringInput {
    use std::collections::HashMap;
    let n = input.ops.len();

    // Count consumers of each op output (op inputs across the forward + the
    // result) so a multiply-consumed `Silu` is never duplicated by fusion.
    let mut uses = vec![0u32; n];
    for od in &input.ops {
        for r in &od.inputs {
            if let InputRef::Op(j) = r {
                uses[*j] += 1;
            }
        }
    }
    uses[input.result] += 1;

    // Decide fusions: mul index → (silu index, gate ref, up ref).
    let is_silu = |i: usize| matches!(input.ops[i].op, LoweredOp::Silu);
    let mut fuse_at: HashMap<usize, (usize, InputRef, InputRef)> = HashMap::new();
    let mut dropped = vec![false; n];
    for (j, od) in input.ops.iter().enumerate() {
        if !matches!(od.op, LoweredOp::Mul) || od.inputs.len() != 2 {
            continue;
        }
        // A fusable operand is a single-use, non-result `Silu` output.
        let fusable = |r: InputRef, dropped: &[bool]| -> Option<usize> {
            if let InputRef::Op(i) = r
                && is_silu(i)
                && uses[i] == 1
                && input.result != i
                && !dropped[i]
            {
                return Some(i);
            }
            None
        };
        let (a, b) = (od.inputs[0], od.inputs[1]);
        let pick = fusable(a, &dropped)
            .map(|si| (si, b))
            .or_else(|| fusable(b, &dropped).map(|si| (si, a)));
        if let Some((si, up)) = pick {
            let gate = input.ops[si].inputs[0];
            fuse_at.insert(j, (si, gate, up));
            dropped[si] = true;
        }
    }
    if fuse_at.is_empty() {
        return input.clone();
    }

    // Old op index → new index, with the dropped `Silu`s removed.
    let mut new_idx = vec![usize::MAX; n];
    let mut next = 0usize;
    for (i, d) in dropped.iter().enumerate() {
        if !d {
            new_idx[i] = next;
            next += 1;
        }
    }
    let remap = |r: InputRef| -> InputRef {
        match r {
            InputRef::Op(j) => {
                debug_assert_ne!(
                    new_idx[j],
                    usize::MAX,
                    "ref to a dropped Silu survived fusion"
                );
                InputRef::Op(new_idx[j])
            }
            InputRef::Ext(e) => InputRef::Ext(e),
        }
    };

    let mut ops: Vec<OpDesc> = Vec::with_capacity(next);
    for (i, od) in input.ops.iter().enumerate() {
        if dropped[i] {
            continue;
        }
        if let Some((_, gate, up)) = fuse_at.get(&i) {
            ops.push(OpDesc {
                op: LoweredOp::SiluMul,
                m: od.m,
                inputs: vec![remap(*gate), remap(*up)],
            });
        } else {
            ops.push(OpDesc {
                op: od.op,
                m: od.m,
                inputs: od.inputs.iter().map(|r| remap(*r)).collect(),
            });
        }
    }
    LoweringInput {
        sources: input.sources.clone(),
        ops,
        result: new_idx[input.result],
    }
}
