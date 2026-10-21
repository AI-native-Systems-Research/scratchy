//! Shared fixture plumbing for the golden diff and the status table.
//!
//! Each test binary (golden_diff, divergence, fusion, gather, make_ttir,
//! math_width, slicing, fixture_status) uses a different subset of these
//! helpers — the same per-target-unused scope every `tests/common` module in
//! this workspace carries (see ktir-emulator and scratchy-target-metal's
//! twins); the module-level allow states that scope.
#![allow(dead_code)] // per-target, not per-crate — see the doc above.

use std::collections::HashMap;
use std::path::PathBuf;

use triton_frontend::codegen::{ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;

pub fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn fixture_path(name: &str) -> PathBuf {
    // A SHIPPED spyre kernel is diffed from the file the splice splices, not a test-local
    // copy that could drift (the `triton-numeric` tests' `include_str!` law): its fixture
    // name is `spyre/<kernel>`.
    if let Some(kernel) = name.strip_prefix("spyre/") {
        return crate_dir()
            .join("../../targets/spyre/kernels")
            .join(format!("{kernel}.py"));
    }
    crate_dir()
        .join("../test-fixtures")
        .join(format!("{name}.py"))
}

pub fn fixture_src(name: &str) -> String {
    let p = fixture_path(name);
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", p.display()))
}

/// The RAW ttir golden -- `make_ir` output, which is bridge one's oracle. See
/// `tools/gen_ttir_goldens.py` for why not the post-pass one.
pub fn golden(name: &str) -> Option<String> {
    let p = crate_dir()
        .join("tests/goldens")
        .join(format!("{name}.ttir_raw.mlir"));
    std::fs::read_to_string(p).ok()
}

/// One fixture configuration: the golden name, the source file, and the spec.
pub struct Case {
    /// The golden's basename, which for the multi-configuration fixtures differs from the
    /// source file (`swiglu_mlp_granite` comes from `swiglu_mlp.py`).
    pub name: String,
    pub fixture: String,
    pub spec: KernelSpec,
}

fn build(
    name: &str,
    fixture: &str,
    kernel: &str,
    sig: &[(&str, &str)],
    constexprs: &[(&str, Val)],
) -> Case {
    let mut signature = HashMap::new();
    for (k, v) in sig {
        signature.insert(
            (*k).to_string(),
            ArgSpec::parse(v).unwrap_or_else(|e| panic!("bad signature entry {k}={v}: {e}")),
        );
    }
    let mut ce = HashMap::new();
    for (k, v) in constexprs {
        ce.insert((*k).to_string(), v.clone());
    }
    Case {
        name: name.to_string(),
        fixture: fixture.to_string(),
        spec: KernelSpec {
            kernel: kernel.to_string(),
            signature,
            constexprs: ce,
            file: fixture_path(fixture).to_string_lossy().to_string(),
        },
    }
}

/// A minimal spec for an ad-hoc kernel written inside a test.
pub fn simple_spec(kernel: &str, sig: &[(&str, &str)]) -> KernelSpec {
    let mut signature = HashMap::new();
    let mut constexprs = HashMap::new();
    for (k, v) in sig {
        signature.insert((*k).to_string(), ArgSpec::parse(v).unwrap());
        if *v == "constexpr" {
            constexprs.insert((*k).to_string(), Val::Int(64));
        }
    }
    KernelSpec {
        kernel: kernel.to_string(),
        signature,
        constexprs,
        file: format!("{kernel}_test.py"),
    }
}

/// The signature the three elementwise fixtures share.
const ELEMENTWISE: &[(&str, &str)] = &[
    ("a_ptr", "*fp16"),
    ("b_ptr", "*fp16"),
    ("c_ptr", "*fp16"),
    ("n", "i32"),
    ("BLOCK", "constexpr"),
];

pub fn vector_add() -> Case {
    build(
        "vector_add",
        "vector_add",
        "vector_add_kernel",
        ELEMENTWISE,
        &[("BLOCK", Val::Int(64))],
    )
}

pub fn mul() -> Case {
    build(
        "mul",
        "mul",
        "mul_kernel",
        ELEMENTWISE,
        &[("BLOCK", Val::Int(64))],
    )
}

pub fn bias_add_f32() -> Case {
    build(
        "bias_add_f32",
        "bias_add_f32",
        "bias_add_f32_kernel",
        ELEMENTWISE,
        &[("BLOCK", Val::Int(64))],
    )
}

/// `swiglu_mlp.py`'s `SIGNATURE`.
const SWIGLU: &[(&str, &str)] = &[
    ("desc_x", "*fp16"),
    ("desc_wg", "*fp16"),
    ("desc_wu", "*fp16"),
    ("desc_wd", "*fp16"),
    ("desc_o", "*fp16"),
    ("M", "constexpr"),
    ("D_MODEL", "constexpr"),
    ("D_FF", "constexpr"),
    ("BLOCK_M", "constexpr"),
    ("BLOCK_N", "constexpr"),
    ("BLOCK_K", "constexpr"),
];

/// One SwiGLU configuration, matching `swiglu_mlp.constexprs(...)`.
pub fn swiglu(name: &str, d_model: i128, d_ff: i128, block_k: i128) -> Case {
    build(
        name,
        "swiglu_mlp",
        "swiglu_mlp_fwd",
        SWIGLU,
        &[
            ("M", Val::Int(64)),
            ("D_MODEL", Val::Int(d_model)),
            ("D_FF", Val::Int(d_ff)),
            ("BLOCK_M", Val::Int(64)),
            ("BLOCK_N", Val::Int(64)),
            ("BLOCK_K", Val::Int(block_k)),
        ],
    )
}

/// `embedding.py`'s `SIGNATURE`. The FIRST fixture with a non-float pointer: the token ids
/// are `*i32` because `tt.descriptor_gather` accepts only an int16/int32 index vector.
const EMBEDDING: &[(&str, &str)] = &[
    ("desc_ids", "*i32"),
    ("desc_table", "*fp16"),
    ("desc_o", "*fp16"),
    ("N_TOK", "constexpr"),
    ("V", "constexpr"),
    ("D_MODEL", "constexpr"),
    ("BLOCK_M", "constexpr"),
    ("EMB_SCALE", "constexpr"),
];

/// One embedding configuration, matching `embedding.constexprs(...)`.
pub fn embedding(name: &str, v: i128, d_model: i128) -> Case {
    build(
        name,
        "embedding",
        "embedding_fwd",
        EMBEDDING,
        &[
            ("N_TOK", Val::Int(256)),
            ("V", Val::Int(v)),
            ("D_MODEL", Val::Int(d_model)),
            ("BLOCK_M", Val::Int(64)),
            ("EMB_SCALE", Val::Float(12.0)),
        ],
    )
}

/// `expert_slab.py`'s `SIGNATURE`. The routed ids are `*i32` and read one SCALAR at a time
/// (`tl.load(desc_ids + p)`), the offset of a rank-3 bank's expert axis.
const EXPERT_SLAB: &[(&str, &str)] = &[
    ("desc_ids", "*i32"),
    ("desc_w", "*fp16"),
    ("desc_x", "*fp16"),
    ("desc_o", "*fp16"),
    ("P", "constexpr"),
    ("E", "constexpr"),
    ("N", "constexpr"),
    ("K", "constexpr"),
];

/// The one expert-slab configuration, matching `expert_slab.constexprs()`.
pub fn expert_slab() -> Case {
    build(
        "expert_slab",
        "expert_slab",
        "expert_slab_fwd",
        EXPERT_SLAB,
        &[
            ("P", Val::Int(8)),
            ("E", Val::Int(8)),
            ("N", Val::Int(64)),
            ("K", Val::Int(128)),
        ],
    )
}

/// `argmax_first_max.py`'s signature: the solo greedy decode step's vocab scan. The idx
/// output is `*i32` (the gather ingestion dtype); `PAD_V` is the vocab padded to a whole
/// power-of-two row with -inf lanes and FOUR fp16 index tables staged (the two decreasing
/// ramps pick first-block/first-lane; the two increasing 0-based tables read the numbers
/// back out), plus the [1, BLOCK] ones row (f16) and 64.0 row (f32) the widen and the
/// island's scale read -- HOST obligations the fixture's docstring states.
const ARGMAX: &[(&str, &str)] = &[
    ("logits_ptr", "*fp16"),
    ("bramp_ptr", "*fp16"),
    ("lramp_ptr", "*fp16"),
    ("binc_ptr", "*fp16"),
    ("linc_ptr", "*fp16"),
    ("ones_ptr", "*fp16"),
    ("sixtyfour_ptr", "*fp32"),
    ("idx_ptr", "*i32"),
    ("V", "constexpr"),
    ("PAD_V", "constexpr"),
    ("NB", "constexpr"),
    ("BLOCK", "constexpr"),
];

/// The one argmax configuration, matching `cases.rs`' `argmax_first_max` row: granite-3.1's
/// vocab 49155 (deliberately NOT a multiple of 64, so the pad lanes are real), the
/// power-of-two padded row (the descriptor block's law), and the two-level split's block
/// count NB = PAD_V / BLOCK.
pub fn argmax_first_max() -> Case {
    build(
        "argmax_first_max",
        "argmax_first_max",
        "argmax_first_max_kernel",
        ARGMAX,
        &[
            ("V", Val::Int(49155)),
            ("PAD_V", Val::Int(65536)),
            ("NB", Val::Int(1024)),
            ("BLOCK", Val::Int(64)),
        ],
    )
}

/// `kernels/moe.py`'s signature: the router's input, gain, projection and per-expert scale;
/// the experts' input; the three fp8 banks and their scales; the output; the scratch.
const MOE: &[(&str, &str)] = &[
    ("desc_xr", "*fp16"),
    ("desc_rg", "*fp16"),
    ("desc_rw", "*fp16"),
    ("desc_pes", "*fp16"),
    ("desc_xe", "*fp16"),
    ("desc_wg", "*fp8e4nv"),
    ("desc_sg", "*fp16"),
    ("desc_wu", "*fp8e4nv"),
    ("desc_su", "*fp16"),
    ("desc_wd", "*fp8e4nv"),
    ("desc_sd", "*fp16"),
    ("desc_o", "*fp16"),
    ("desc_ids", "*i32"),
    ("desc_rs", "*fp16"),
    ("desc_act", "*fp16"),
    ("M", "constexpr"),
    ("H", "constexpr"),
    ("I", "constexpr"),
    ("E", "constexpr"),
    ("K", "constexpr"),
    ("EPS", "constexpr"),
    ("INV_H", "constexpr"),
    ("TEMP", "constexpr"),
    ("INV_QMAX", "constexpr"),
    ("SCALE_FLOOR", "constexpr"),
];

/// The fused MoE kernel at the smallest geometry the golden's real-Triton compile admits:
/// one token, `E = 64` (the router dot's `N` at the M=1 dot floor `(1, 64, 64)`), `K = 8`
/// (the `[M, K]` id and weight blocks need 16 bytes a row), `H = 128`, `I = 64`.
pub fn moe() -> Case {
    let h = 128.0f64;
    build(
        "moe",
        "spyre/moe",
        "moe_fwd",
        MOE,
        &[
            ("M", Val::Int(1)),
            ("H", Val::Int(128)),
            ("I", Val::Int(64)),
            ("E", Val::Int(64)),
            ("K", Val::Int(8)),
            ("EPS", Val::Float(1e-6)),
            ("INV_H", Val::Float(1.0 / h)),
            ("TEMP", Val::Float(h.powf(-0.5))),
            ("INV_QMAX", Val::Float(1.0 / 448.0)),
            ("SCALE_FLOOR", Val::Float(1.0 / 448.0 / 448.0)),
        ],
    )
}

/// `kernels/moe_prefill.py`'s signature: `moe.py`'s first twelve, then the dense route.
const MOE_PREFILL: &[(&str, &str)] = &[
    ("desc_xr", "*fp16"),
    ("desc_rg", "*fp16"),
    ("desc_rw", "*fp16"),
    ("desc_pes", "*fp16"),
    ("desc_xe", "*fp16"),
    ("desc_wg", "*fp8e4nv"),
    ("desc_sg", "*fp16"),
    ("desc_wu", "*fp8e4nv"),
    ("desc_su", "*fp16"),
    ("desc_wd", "*fp8e4nv"),
    ("desc_sd", "*fp16"),
    ("desc_o", "*fp16"),
    ("desc_dense", "*fp16"),
    ("T", "constexpr"),
    ("H", "constexpr"),
    ("I", "constexpr"),
    ("E", "constexpr"),
    ("K", "constexpr"),
    ("EPS", "constexpr"),
    ("INV_H", "constexpr"),
    ("TEMP", "constexpr"),
    ("INV_QMAX", "constexpr"),
    ("SCALE_FLOOR", "constexpr"),
];

/// The prompt MoE kernel at the golden's geometry: 16 prompt rows, `E = 64`, `K = 8`,
/// `H = 128`, `I = 64` — `moe()`'s widths with a token tile in place of one token.
pub fn moe_prefill() -> Case {
    let h = 128.0f64;
    build(
        "moe_prefill",
        "spyre/moe_prefill",
        "moe_prefill_fwd",
        MOE_PREFILL,
        &[
            ("T", Val::Int(16)),
            ("H", Val::Int(128)),
            ("I", Val::Int(64)),
            ("E", Val::Int(64)),
            ("K", Val::Int(8)),
            ("EPS", Val::Float(1e-6)),
            ("INV_H", Val::Float(1.0 / h)),
            ("TEMP", Val::Float(h.powf(-0.5))),
            ("INV_QMAX", Val::Float(1.0 / 448.0)),
            ("SCALE_FLOOR", Val::Float(1.0 / 448.0 / 448.0)),
        ],
    )
}

/// `rmsnorm.py`'s `SIGNATURE`.
const RMSNORM: &[(&str, &str)] = &[
    ("desc_x", "*fp16"),
    ("desc_w", "*fp16"),
    ("desc_o", "*fp16"),
    ("M", "constexpr"),
    ("D_MODEL", "constexpr"),
    ("BLOCK_M", "constexpr"),
    ("EPS", "constexpr"),
    ("INV_D", "constexpr"),
];

/// One RMSNorm configuration, matching `rmsnorm.constexprs(...)`.
///
/// `INV_D` is `1.0 / d_model` folded on the HOST (the fixture's delta 2), so it is passed
/// here as the number the kernel actually sees rather than recomputed in the kernel.
pub fn rmsnorm(name: &str, d_model: i128) -> Case {
    build(
        name,
        "rmsnorm",
        "rmsnorm_fwd",
        RMSNORM,
        &[
            ("M", Val::Int(64)),
            ("D_MODEL", Val::Int(d_model)),
            ("BLOCK_M", Val::Int(64)),
            ("EPS", Val::Float(1e-05)),
            ("INV_D", Val::Float(1.0 / d_model as f64)),
        ],
    )
}

/// `rope.py`'s `SIGNATURE`.
const ROPE: &[(&str, &str)] = &[
    ("desc_x", "*fp16"),
    ("desc_cos", "*fp16"),
    ("desc_sin", "*fp16"),
    ("desc_o", "*fp16"),
    ("H", "constexpr"),
    ("N_TOK", "constexpr"),
    ("HEAD_DIM", "constexpr"),
    ("BLOCK_M", "constexpr"),
    ("HALF", "constexpr"),
];

/// One RoPE configuration, matching `rope.constexprs(...)`. `h` is the head count of the
/// plane being rotated -- 32 for Granite's queries, 8 for its keys.
pub fn rope(name: &str, h: i128) -> Case {
    build(
        name,
        "rope",
        "rope_fwd",
        ROPE,
        &[
            ("H", Val::Int(h)),
            ("N_TOK", Val::Int(256)),
            ("HEAD_DIM", Val::Int(128)),
            ("BLOCK_M", Val::Int(64)),
            ("HALF", Val::Int(64)),
        ],
    )
}

/// The constexpr bindings both `decoder_block.py` kernels share, from its `constexprs()`.
///
/// `QK_SCALE` is `attention_multiplier * log2(e)` = `0.0078125 * 1.44269504`, folded on the
/// host so `exp2` needs no extra multiply -- the same trick `attention_flash.py` uses.
fn decoder_constexprs() -> Vec<(&'static str, Val)> {
    vec![
        ("M", Val::Int(64)),
        ("D_MODEL", Val::Int(128)),
        ("D_FF", Val::Int(256)),
        ("BLOCK_N", Val::Int(64)),
        ("HALF", Val::Int(64)),
        ("EPS", Val::Float(1e-05)),
        ("INV_D", Val::Float(1.0 / 128.0)),
        ("QK_SCALE", Val::Float(0.011271055)), // 0.0078125 * 1.44269504, folded; see the comment above
        ("RM", Val::Float(0.22)),
    ]
}

fn decoder_case(name: &str, kernel: &str, ptrs: &[&'static str]) -> Case {
    let mut sig: Vec<(&str, &str)> = ptrs.iter().map(|p| (*p, "*fp16")).collect();
    let ce = decoder_constexprs();
    for (k, _) in &ce {
        sig.push((k, "constexpr"));
    }
    build(name, "decoder_block", kernel, &sig, &ce)
}

/// `decoder_block.py`'s ONE-layer kernel: the whole block, called once.
pub fn decoder_layer() -> Case {
    decoder_case(
        "decoder_layer",
        "decoder_layer_fwd",
        &[
            "desc_x",
            "desc_o",
            "desc_n1",
            "desc_wq",
            "desc_wk",
            "desc_wv",
            "desc_wo",
            "desc_mask",
            "desc_cos",
            "desc_sin",
            "desc_n2",
            "desc_wg",
            "desc_wu",
            "desc_wd",
        ],
    )
}

/// `decoder_block.py`'s TWO-layer kernel: the SAME block function, called twice, with the
/// hidden state between them never touching memory.
pub fn decoder_two_layers() -> Case {
    decoder_case(
        "decoder_two_layers",
        "decoder_two_layers_fwd",
        &[
            "desc_x",
            "desc_o",
            "desc_n1a",
            "desc_wqa",
            "desc_wka",
            "desc_wva",
            "desc_woa",
            "desc_n2a",
            "desc_wga",
            "desc_wua",
            "desc_wda",
            "desc_n1b",
            "desc_wqb",
            "desc_wkb",
            "desc_wvb",
            "desc_wob",
            "desc_n2b",
            "desc_wgb",
            "desc_wub",
            "desc_wdb",
            "desc_mask",
            "desc_cos",
            "desc_sin",
        ],
    )
}

/// `attention_flash.py`'s `SIGNATURE`.
const ATTENTION: &[(&str, &str)] = &[
    ("desc_q", "*fp16"),
    ("desc_k", "*fp16"),
    ("desc_v", "*fp16"),
    ("desc_o", "*fp16"),
    ("desc_mask", "*fp16"),
    ("Z", "constexpr"),
    ("H", "constexpr"),
    ("N_CTX", "constexpr"),
    ("HEAD_DIM", "constexpr"),
    ("BLOCK_M", "constexpr"),
    ("BLOCK_N", "constexpr"),
    ("GQA", "constexpr"),
    ("STAGE", "constexpr"),
    ("sm_scale", "constexpr"),
];

/// One flash-attention configuration. `stage` is 1 for non-causal, 3 for causal.
pub fn attention(name: &str, stage: i128) -> Case {
    build(
        name,
        "attention_flash",
        "attn_fwd",
        ATTENTION,
        &[
            ("Z", Val::Int(1)),
            ("H", Val::Int(4)),
            ("N_CTX", Val::Int(256)),
            ("HEAD_DIM", Val::Int(128)),
            ("BLOCK_M", Val::Int(64)),
            ("BLOCK_N", Val::Int(64)),
            ("GQA", Val::Int(2)),
            ("STAGE", Val::Int(stage)),
            ("sm_scale", Val::Float(1.0)),
        ],
    )
}

/// `attention_flash.py`'s FOUR-pointer signature: `attn_fwd_noncausal`, the `@triton.jit`
/// wrapper the non-causal configuration launches. The five-pointer `attn_fwd` leaves the
/// mask parameter addressed by nothing in that configuration (the inner never loads it at
/// `4 - STAGE == 3`), so the wrapper exists to state the launch that actually happens.
const ATTENTION_NONCAUSAL: &[(&str, &str)] = &[
    ("desc_q", "*fp16"),
    ("desc_k", "*fp16"),
    ("desc_v", "*fp16"),
    ("desc_o", "*fp16"),
    ("sm_scale", "constexpr"),
    ("Z", "constexpr"),
    ("H", "constexpr"),
    ("N_CTX", "constexpr"),
    ("HEAD_DIM", "constexpr"),
    ("BLOCK_M", "constexpr"),
    ("BLOCK_N", "constexpr"),
    ("GQA", "constexpr"),
    ("STAGE", "constexpr"),
];

/// The non-causal flash-attention configuration the case table launches: the four-pointer
/// wrapper at `attn_ce(1)`'s geometry — `triton_ktir_superdsc::cases`'
/// `attention_flash_noncausal` row, whose spec this mirrors so a frontend test and the bake
/// driver state ONE configuration rather than two that could drift.
///
/// ⛔ NOT a `make_ttir.rs` `cases()` entry: the golden set's `attention_flash_noncausal`
/// is the FIVE-POINTER `attn_fwd` compile (the C++ oracle's own configuration), and this
/// spec would diff against it structurally — loudly, by function name and parameter count,
/// but pointlessly. This builder exists for tests that assert on the pipeline's behavior
/// rather than on an oracle.
pub fn attention_noncausal() -> Case {
    build(
        "attention_flash_noncausal",
        "attention_flash",
        "attn_fwd_noncausal",
        ATTENTION_NONCAUSAL,
        &[
            ("sm_scale", Val::Float(1.0)),
            ("Z", Val::Int(1)),
            ("H", Val::Int(4)),
            ("N_CTX", Val::Int(256)),
            ("HEAD_DIM", Val::Int(128)),
            ("BLOCK_M", Val::Int(64)),
            ("BLOCK_N", Val::Int(64)),
            ("GQA", Val::Int(2)),
            ("STAGE", Val::Int(1)),
        ],
    )
}
