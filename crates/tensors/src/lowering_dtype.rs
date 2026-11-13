// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral lowering dtypes.
//!
//! [`ActDtype`] and [`ScaleDtype`] are trivial dependency-free enums a
//! target's lowering and the per-canonical `CanonicalParams` consts
//! speak in. They live here in the cfg-free tensor core so the cfg-free
//! `scratchy-ir` crate can name them from `CanonicalParams`
//! without pulling a target in.

/// The element dtype a model's activations run in — a fact of the model
/// (`CanonicalParams::ACT_DTYPE`) a target lowers each kernel at (a
/// dtype-specialized kernel symbol is picked by it).
///
/// Llama-3.x ships bf16 on disk. Casting to fp16 clips the exponent range
/// and accumulates into nonsense output across deep layer chains (28
/// layers for Llama-3.2-3B).
///
/// `Int4` is a placeholder for the upcoming AWQ / GPTQ dequant path
/// (group-wise int4 weights with bf16 scales / zeros). When that
/// lands the dtype routes through this enum just like bf16 does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActDtype {
    F16,
    Bf16,
    /// Reserved — int4 quantized weight path. Not yet wired through
    /// the lowering kernel-symbol pickers; placed here so callers
    /// can already speak in `ActDtype` terms.
    Int4,
}

impl ActDtype {
    /// The `f16` / `bf16` infix every dtype-specialized kernel symbol
    /// carries. Centralized so symbol naming stays consistent across
    /// the on-path kernels.
    pub fn symbol_infix(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Int4 => "int4",
        }
    }
}

/// Storage dtype the kernel reads `scales` / `biases` device pointers
/// as — the kernel template parameter `T_scale`, cast to the
/// activation dtype in-register. Llama-3.x
/// mlx-community 4bit ships F16 scales; Qwen3-MoE (and other
/// `torch_dtype: bfloat16` exports) ships BF16 scales. MLX templates
/// both natively (`mlx/.../quantized.h:INSTANTIATE_QUANTIZED_FUNCTIONS`
/// instantiates `T_scale ∈ {half, bfloat16_t}`), and a target's kernels
/// mirror that surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScaleDtype {
    F16,
    Bf16,
}

impl ScaleDtype {
    pub fn symbol_infix(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    /// The on-device element dtype the `_s_<scale>_` kernel arm binds the
    /// scale/gain pointer as. A loader that must guarantee the gain buffer
    /// matches the kernel's `T_scale` (e.g. metal's RMSNorm gain, which is
    /// read through a `device const T_scale*`) converts the on-disk tensor
    /// to this dtype.
    pub fn as_dtype(self) -> crate::DType {
        match self {
            Self::F16 => crate::DType::F16,
            Self::Bf16 => crate::DType::BF16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn scale_dtype_maps_to_matching_element_dtype() {
        // The metal RMSNorm gain loader (`take_as_dtype`) relies on this
        // mapping to convert a gain to the dtype the `_s_<scale>_` kernel
        // arm binds. A mismatch here would re-introduce the bf16-gain-read-
        // as-f16 garbage bug on standard HF bf16 checkpoints.
        assert_eq!(ScaleDtype::F16.as_dtype(), DType::F16);
        assert_eq!(ScaleDtype::Bf16.as_dtype(), DType::BF16);
        assert_eq!(ScaleDtype::F16.as_dtype().size_bytes(), 2);
        assert_eq!(ScaleDtype::Bf16.as_dtype().size_bytes(), 2);
    }
}
