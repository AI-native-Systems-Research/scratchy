// SPDX-License-Identifier: Apache-2.0
//! Single source of truth for the FlashInfer per-tuple configuration set.
//!
//! Each [`FlashInferConfig`] describes one `(dtype, head_dim, logits_soft_cap)`
//! specialization that [`build.rs::build_flashinfer_attention`] renders into a
//! per-tuple translation unit. The resulting `libflashinfer_attn.a` exports
//! `fi_plan_<suffix>_new`, `fi_plan_<suffix>_delete`, `fi_run_<suffix>` per entry.
//!
//! Keep this module leaf-level — no cudaforge imports — so that `build.rs` can
//! pull it in via `#[path]` and `lib.rs` can re-export it for downstream crates
//! that emit symbol names (scratchy-target-cuda, scratchy-forward-compiler-macro).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    Bf16,
    // Placeholder for the next tuple-set extension. Unused until the first
    // Fp16 tuple is added to FLASHINFER_CONFIG_SET; the matching extern-C
    // decls and dispatch arms in `scratchy_target_cuda::flashinfer` must be
    // added at the same time.
    #[allow(dead_code)]
    Fp16,
}

impl DType {
    pub fn cpp_ty(self) -> &'static str {
        match self {
            DType::Bf16 => "__nv_bfloat16",
            DType::Fp16 => "half",
        }
    }

    pub fn sym_token(self) -> &'static str {
        match self {
            DType::Bf16 => "bf16",
            DType::Fp16 => "fp16",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashInferConfig {
    pub dtype: DType,
    pub head_dim: u32,
    pub use_logits_soft_cap: bool,
}

impl FlashInferConfig {
    pub fn sym_suffix(&self) -> String {
        let sc = if self.use_logits_soft_cap {
            "softcap"
        } else {
            "nosoftcap"
        };
        format!("{}_h{}_{}", self.dtype.sym_token(), self.head_dim, sc)
    }
}

/// The specialization grid. Start narrow — every added tuple grows
/// `libflashinfer_attn.a` by one CUDA TU, and each Impl's `emit_call` must
/// name an entry present here.
pub const FLASHINFER_CONFIG_SET: &[FlashInferConfig] = &[
    FlashInferConfig {
        dtype: DType::Bf16,
        head_dim: 64,
        use_logits_soft_cap: false,
    },
    FlashInferConfig {
        dtype: DType::Bf16,
        head_dim: 64,
        use_logits_soft_cap: true,
    },
    FlashInferConfig {
        dtype: DType::Bf16,
        head_dim: 128,
        use_logits_soft_cap: false,
    },
    FlashInferConfig {
        dtype: DType::Bf16,
        head_dim: 128,
        use_logits_soft_cap: true,
    },
    FlashInferConfig {
        dtype: DType::Bf16,
        head_dim: 256,
        use_logits_soft_cap: false,
    },
    FlashInferConfig {
        dtype: DType::Bf16,
        head_dim: 256,
        use_logits_soft_cap: true,
    },
];
