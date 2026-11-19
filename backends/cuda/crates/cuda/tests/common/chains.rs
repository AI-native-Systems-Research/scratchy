// SPDX-License-Identifier: Apache-2.0
//! GEMM chains: the canonicals the tape tests lower, and the `static` tapes written out by hand
//! from the issue's numbers, which `tape_layout` checks against what the compiler lowers.
//!
//! Every source is declared in the IR as `[K, N]`; the weight's bytes on the device are `[N, K]`.
//! Source 0 is the activation, rebound to the embedded hidden: colour 0, the forward's input.

use scratchy_ir::model_facts::{
    ActDtype, Attention, AttnScale, BlockSize, ConvDim, ConvKernel, FactsParts, Gdn, GdnHeadDim,
    GdnHeads, HeadDim, HeadGeometry, HiddenSize, KvCodec, KvHeads, KvRope, ModelFacts,
    NormWeightOffset, QueryHeads, RmsNormEps, Rope, RopePairing, RotDim, ScaleDtype,
};
use scratchy_subtile::fixtures::{front_end_lowered, rebind};
use scratchy_subtile::handoff::{LoweredDecode, SourceBinding};
use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{SourceShape, SubOp};
use scratchy_target_cuda::step::{BucketM, KDim, NDim, Rows};
use scratchy_target_cuda::step::{CudaStep, GemmBf16};
use scratchy_target_cuda::tape::{
    ArenaLayout, ArenaOffset, ArenaSpan, CudaTape, GdnTape, KvTape, TapeInput, WeightIx,
};
use scratchy_target_cuda_compiler::canonical::{
    CanonicalAt, CanonicalGdn, CanonicalKv, CudaCanonical, LoweringFacts, SharedExpertOwner,
    lower_canonical,
};

/// The facts of a fixture model: its attention and linear attention, `rms_norm_eps` 1e-6, and no
/// vocabulary (the fixtures' input is the embedded hidden). The rest (the dtypes, a hidden of
/// 2048, a dense KV codec) no lowering reads.
pub const fn fixture_facts(attention: Option<Attention>, gdn: Option<Gdn>) -> ModelFacts {
    ModelFacts::new(FactsParts {
        act_dtype: ActDtype::Bf16,
        scale_dtype: ScaleDtype::F16,
        hidden: HiddenSize::new(2048),
        intermediate: None,
        vocab: None,
        eps: RmsNormEps::new(1e-6),
        norm_offset: NormWeightOffset::new(0.0),
        final_softcap: None,
        kv_codec: KvCodec::Dense,
        kv_layers: None,
        attention,
        vision: None,
        gdn,
        drafter: None,
        spec_drafts: None,
        capture_slots: None,
        target: None,
    })
}

/// One attention class: `q_heads` query heads over `kv_heads` kv heads of `head_dim`, `rot_dim` of
/// each rotated, in 64-token pages.
pub const fn class(q_heads: u32, kv_heads: u32, head_dim: u32, rot_dim: u32) -> HeadGeometry {
    HeadGeometry::new(
        QueryHeads::new(q_heads),
        KvHeads::new(kv_heads),
        HeadDim::new(head_dim),
        RotDim::new(rot_dim),
        BlockSize::new(64),
    )
}

/// A uniform roped attention: qwen3.5-0.8b's 8 query heads over 2 kv heads, at `head_dim` with
/// `rot_dim` of each head rotated half-rotated, keys rotated at the write.
pub const fn uniform_attention(head_dim: u32, rot_dim: u32) -> Attention {
    let g = class(8, 2, head_dim, rot_dim);
    let rope = Rope::new(RopePairing::Standard, KvRope::AtWrite, None);
    Attention::new(g, g, AttnScale::new(0.0625), None, None, Some(rope))
}

/// `facts` lowered at rope base frequency `theta`, with no local one and no DSL shared expert.
pub const fn lowering(facts: ModelFacts, theta: Option<f64>) -> LoweringFacts {
    LoweringFacts {
        facts,
        rope_theta: theta,
        rope_local_theta: None,
        shared_expert: SharedExpertOwner::Undeclared,
    }
}

/// `f` with its attention `attention`.
pub const fn with_attention(f: LoweringFacts, attention: Option<Attention>) -> LoweringFacts {
    LoweringFacts {
        facts: ModelFacts::new(FactsParts {
            attention,
            ..f.facts.parts()
        }),
        ..f
    }
}

/// qwen3.5's model facts: `head_dim` 256, a quarter of it rotated, `rope_theta` 1e7, no linear
/// attention, and no embedding table (the fixtures' input is the embedded hidden). Every fixture
/// but the GDN layer lowers with them; one with no rope or gate reads none of them.
pub const QWEN3_5_FACTS: LoweringFacts = lowering(
    fixture_facts(Some(uniform_attention(256, 64)), None),
    Some(1e7),
);

/// A Gated-DeltaNet geometry: `k_heads` key heads of `k_dim`, a `conv_kernel`-tap conv, and
/// qwen3.5-0.8b's 16 value heads of 128 (which the op's operands state, so no lowering reads them).
pub const fn gdn_of(k_heads: u32, k_dim: u32, conv_kernel: u32) -> Gdn {
    Gdn::new(
        GdnHeads::new(k_heads),
        GdnHeads::new(16),
        GdnHeadDim::new(k_dim),
        GdnHeadDim::new(128),
        ConvKernel::new(conv_kernel),
        ConvDim::new(6144),
    )
}

/// [`QWEN3_5_FACTS`] with qwen3.5-0.8b's linear attention: 16 key heads of 128, a 4-tap conv,
/// `rms_norm_eps` 1e-6. The GDN layer's fixture lowers with them.
pub const QWEN3_5_GDN_FACTS: LoweringFacts = lowering(
    fixture_facts(Some(uniform_attention(256, 64)), Some(gdn_of(16, 128, 4))),
    Some(1e7),
);

/// The facts Gemma 4's fixtures lower with: [`QWEN3_5_FACTS`], since a GEMM's column tile is its
/// shape's, not the model's.
pub const GEMMA4_FACTS: LoweringFacts = QWEN3_5_FACTS;

/// Panic unless `tape` is, field by field, what CUDA's compiler lowers `l` to.
///
/// Every struct is destructured with no `..`, so a field added to the tape, its layout or the
/// canonical does not compile here until it is compared.
pub fn assert_tape_is_lowered(name: &str, tape: &CudaTape, l: &LoweredDecode) {
    assert_tape_is_lowered_with(name, tape, l, QWEN3_5_FACTS);
}

/// [`assert_tape_is_lowered`] under `facts`.
pub fn assert_tape_is_lowered_with(
    name: &str,
    tape: &CudaTape,
    l: &LoweredDecode,
    facts: LoweringFacts,
) {
    let at = CanonicalAt {
        stem: name,
        m: u64::from(tape.bucket_m),
        rows: scratchy_subtile::sample_rows::ResultRows::Every,
    };
    let lowered =
        lower_canonical(l, facts, at).unwrap_or_else(|e| panic!("{name}: lower_canonical: {e}"));
    assert_tape_is(name, tape, &lowered);
}

/// Panic unless `tape` is, field by field, `lowered`.
pub fn assert_tape_is(name: &str, tape: &CudaTape, lowered: &CudaCanonical) {
    let CudaTape {
        bucket_m,
        arena:
            ArenaLayout {
                colour_bytes,
                colour_offsets,
                total_bytes,
            },
        input,
        result,
        weight_bytes,
        steps,
        rotary,
        kv,
        gdn,
        moe,
        rotary_local,
        gemv_scratch,
        greedy,
    } = *tape;
    let CudaCanonical {
        bucket_m: lowered_m,
        colour_bytes: lowered_bytes,
        colour_offsets: lowered_offsets,
        total_bytes: lowered_total,
        input: lowered_input,
        result: lowered_result,
        weight_bytes: lowered_weights,
        steps: lowered_steps,
        rotary: lowered_rotary,
        kv: lowered_kv,
        gdn: lowered_gdn,
        moe: lowered_moe,
        rotary_local: lowered_rotary_local,
        gemv_scratch: lowered_gemv_scratch,
        greedy: lowered_greedy,
    } = lowered;
    assert_eq!(bucket_m, BucketM(*lowered_m), "{name}: bucket_m");
    assert_eq!(colour_bytes, lowered_bytes, "{name}: colour bytes");
    assert_eq!(colour_offsets, lowered_offsets, "{name}: offsets");
    assert_eq!(total_bytes, *lowered_total, "{name}: total");
    assert_eq!(input, *lowered_input, "{name}: input");
    assert_eq!(result, *lowered_result, "{name}: result");
    assert_eq!(weight_bytes, lowered_weights, "{name}: weight_bytes");
    assert_eq!(steps, lowered_steps, "{name}: steps");
    assert_eq!(rotary, *lowered_rotary, "{name}: rotary");
    assert_eq!(rotary_local, *lowered_rotary_local, "{name}: rotary_local");
    assert_eq!(gemv_scratch, *lowered_gemv_scratch, "{name}: gemv_scratch");
    assert_eq!(greedy, *lowered_greedy, "{name}: greedy");
    let kv = kv.map(
        |KvTape {
             layers,
             kv_heads,
             head_dim,
             block_size,
             max_blocks_per_seq,
             block_table,
             slot_mapping,
             sequence,
             classes,
             layer_class,
         }| CanonicalKv {
            layers: layers.to_vec(),
            kv_heads: kv_heads.get(),
            head_dim: head_dim.get(),
            block_size: block_size.get(),
            max_blocks_per_seq: max_blocks_per_seq.get(),
            block_table,
            slot_mapping,
            sequence,
            classes: classes.to_vec(),
            layer_class: layer_class.to_vec(),
        },
    );
    assert_eq!(kv.as_ref(), lowered_kv.as_ref(), "{name}: kv");
    let gdn = gdn.map(
        |GdnTape {
             layers,
             num_k_heads,
             num_v_heads,
             head_k,
             head_v,
             conv_kernel,
             sequence,
             scratch,
             scratch_layout,
         }| CanonicalGdn {
            layers: layers.to_vec(),
            num_k_heads: num_k_heads.get(),
            num_v_heads: num_v_heads.get(),
            head_k: head_k.get(),
            head_v: head_v.get(),
            conv_kernel: conv_kernel.get(),
            sequence,
            scratch,
            scratch_layout,
        },
    );
    assert_eq!(gdn.as_ref(), lowered_gdn.as_ref(), "{name}: gdn");
    assert_eq!(moe, *lowered_moe, "{name}: moe");
}

/// The chains' activation width: `x` is `[M, HIDDEN]`.
pub const HIDDEN: u32 = 512;

/// `x [m, k0]` through one dense GEMM per `ns` entry, each reading the one before; the last is
/// the result. Every source bound to a weight, as the shared front end binds them.
pub fn chain_input(m: u32, k0: u32, ns: &[u32]) -> LoweringInput {
    let mut sources = vec![SourceShape { rows: m, cols: k0 }];
    let mut ops = Vec::new();
    let mut k = k0;
    for (i, &n) in ns.iter().enumerate() {
        sources.push(SourceShape { rows: k, cols: n });
        let act = match i {
            0 => InputRef::Ext(0),
            i => InputRef::Op(i - 1),
        };
        ops.push(OpDesc {
            op: SubOp::MatmulTile {
                n,
                weight: GemmWeight::Dense,
            },
            m,
            inputs: vec![act, InputRef::Ext(i + 1)],
        });
        k = n;
    }
    LoweringInput {
        exported: None,
        result: ops.len() - 1,
        sources,
        ops,
    }
}

/// `input` as the shared front end binds it, with source 0 rebound to the embedded hidden.
pub fn with_hidden(input: LoweringInput) -> LoweredDecode {
    let mut l = front_end_lowered(input);
    rebind(&mut l, |b| b[0] = SourceBinding::EmbeddedHidden);
    l
}

/// `x [m, 512] → W1 (N = 1024) → y1 [m, 1024] → W2 (N = 512) → y2 [m, 512]`.
pub fn two_gemm(m: u32) -> LoweredDecode {
    with_hidden(chain_input(m, HIDDEN, &[1024, 512]))
}

/// [`two_gemm`] at `m = 256`, then `y2 → W3 (N = 1024) → y3 [256, 1024]`.
pub fn three_gemm() -> LoweredDecode {
    with_hidden(chain_input(256, HIDDEN, &[1024, 512, 1024]))
}

const fn gemm(a: u64, b: u32, c: u64, m: u32, n: u32, k: u32) -> CudaStep {
    CudaStep::Gemm(GemmBf16 {
        a: ArenaOffset(a),
        b: WeightIx(b),
        c: ArenaOffset(c),
        m: Rows(m),
        n: NDim(n),
        k: KDim(k),
        residual: None,
    })
}

const fn span(offset: u64, bytes: u64) -> ArenaSpan {
    ArenaSpan {
        offset: ArenaOffset(offset),
        bytes,
    }
}

/// [`two_gemm`]`(128)`: colours x, y1, y2.
pub static TWO_GEMM_M128: CudaTape = CudaTape {
    bucket_m: BucketM(128),
    arena: ArenaLayout {
        colour_bytes: &[131072, 262144, 131072],
        colour_offsets: &[0, 131072, 393216],
        total_bytes: 524288,
    },
    input: TapeInput::Hidden(span(0, 131072)),
    result: span(393216, 131072),
    weight_bytes: &[0, 1048576, 1048576],
    steps: &[
        gemm(0, 1, 131072, 128, 1024, 512),
        gemm(131072, 2, 393216, 128, 512, 1024),
    ],
    rotary: None,
    kv: None,
    gdn: None,
    moe: None,
    rotary_local: None,
    gemv_scratch: None,
    greedy: None,
};

/// [`two_gemm`]`(256)`: colours x, y1, y2.
pub static TWO_GEMM_M256: CudaTape = CudaTape {
    bucket_m: BucketM(256),
    arena: ArenaLayout {
        colour_bytes: &[262144, 524288, 262144],
        colour_offsets: &[0, 262144, 786432],
        total_bytes: 1048576,
    },
    input: TapeInput::Hidden(span(0, 262144)),
    result: span(786432, 262144),
    weight_bytes: &[0, 1048576, 1048576],
    steps: &[
        gemm(0, 1, 262144, 256, 1024, 512),
        gemm(262144, 2, 786432, 256, 512, 1024),
    ],
    rotary: None,
    kv: None,
    gdn: None,
    moe: None,
    rotary_local: None,
    gemv_scratch: None,
    greedy: None,
};

/// [`three_gemm`]: `y1` is dead after step 1, so `y3` takes its colour, and the arena is the
/// two-GEMM one. `y3` is the result.
pub static THREE_GEMM_M256: CudaTape = CudaTape {
    bucket_m: BucketM(256),
    arena: ArenaLayout {
        colour_bytes: &[262144, 524288, 262144],
        colour_offsets: &[0, 262144, 786432],
        total_bytes: 1048576,
    },
    input: TapeInput::Hidden(span(0, 262144)),
    result: span(262144, 524288),
    weight_bytes: &[0, 1048576, 1048576, 1048576],
    steps: &[
        gemm(0, 1, 262144, 256, 1024, 512),
        gemm(262144, 2, 786432, 256, 512, 1024),
        gemm(786432, 3, 262144, 256, 1024, 512),
    ],
    rotary: None,
    kv: None,
    gdn: None,
    moe: None,
    rotary_local: None,
    gemv_scratch: None,
    greedy: None,
};
