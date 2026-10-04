//! Shared fixture plumbing for the golden diff and the status table.
#![allow(dead_code)] // each test binary uses a different subset of these helpers.

use std::collections::HashMap;
use std::path::PathBuf;

use triton_frontend::codegen::{ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;

pub fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn fixture_path(name: &str) -> PathBuf {
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
    build("mul", "mul", "mul_kernel", ELEMENTWISE, &[("BLOCK", Val::Int(64))])
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
            "desc_x", "desc_o", "desc_n1", "desc_wq", "desc_wk", "desc_wv", "desc_wo",
            "desc_mask", "desc_cos", "desc_sin", "desc_n2", "desc_wg", "desc_wu", "desc_wd",
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
            "desc_x", "desc_o", "desc_n1a", "desc_wqa", "desc_wka", "desc_wva", "desc_woa",
            "desc_n2a", "desc_wga", "desc_wua", "desc_wda", "desc_n1b", "desc_wqb",
            "desc_wkb", "desc_wvb", "desc_wob", "desc_n2b", "desc_wgb", "desc_wub",
            "desc_wdb", "desc_mask", "desc_cos", "desc_sin",
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
