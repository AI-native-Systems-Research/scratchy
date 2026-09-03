// SPDX-License-Identifier: Apache-2.0
//! GGML quantized data types — pure data, no CUDA calls.
//!
//! These were lifted out of `scratchy-target-cuda::ggml` so `GpuWeights`
//! (which lives in this crate) can hold a `quantized:
//! HashMap<String, GgmlStorage>` field. The CUDA kernels that
//! consume `GgmlStorage` (mul_mat_vec, dequantize_to_tensor, …) stay
//! in `scratchy-target-cuda`. `scratchy-target-cuda::ggml` re-exports
//! `GgmlDType` and `GgmlStorage` from here for backward compat.

/// GGML quantization data types.
///
/// Matches the GGUF on-disk format tags. Each variant knows its
/// `type_size()` (bytes per block) and `block_size()` (elements per
/// block).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlDType {
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    IQ1M = 29,
}

impl GgmlDType {
    /// Create from the GGUF on-disk u32 tag.
    pub fn from_u32(u: u32) -> Option<Self> {
        match u {
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::IQ2XXS),
            17 => Some(Self::IQ2XS),
            19 => Some(Self::IQ1S),
            20 => Some(Self::IQ4NL),
            21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),
            23 => Some(Self::IQ4XS),
            29 => Some(Self::IQ1M),
            _ => None,
        }
    }

    /// Convert from a vendored GgufDType tag.
    pub fn from_gguf(dt: crate::gguf::GgufDType) -> Option<Self> {
        Self::from_u32(dt.0)
    }

    /// Size in bytes of one quantization block.
    ///
    /// Values match llama.cpp block structs.
    pub const fn type_size(self) -> usize {
        match self {
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36, // 2*sizeof(ggml_half) + QK8_1 = 4 + 32 = 36 (see static_assert in quantized.cu)
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::IQ2XXS => 66,
            Self::IQ2XS => 74,
            Self::IQ1S => 50,
            Self::IQ4NL => 18,
            Self::IQ3S => 110,
            Self::IQ2S => 82,
            Self::IQ4XS => 136,
            Self::IQ1M => 56,
        }
    }

    /// Number of elements per quantization block.
    pub const fn block_size(self) -> usize {
        match self {
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4NL => 32,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::IQ2XXS
            | Self::IQ2XS
            | Self::IQ1S
            | Self::IQ3S
            | Self::IQ2S
            | Self::IQ4XS
            | Self::IQ1M => 256,
        }
    }

    /// Whether this is a "K-quant" type (QK_K=256 block size).
    pub const fn is_k_quant(self) -> bool {
        matches!(
            self,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K
        )
    }

    /// Whether this is an IQ (importance-matrix) quantization type.
    ///
    /// IQ types lack the fused dequant+dot BS=1 kernel — they must always go
    /// through the Q8_1 intermediate quantization path.
    pub const fn is_iq_quant(self) -> bool {
        matches!(
            self,
            Self::IQ4NL
                | Self::IQ4XS
                | Self::IQ1M
                | Self::IQ2XXS
                | Self::IQ2XS
                | Self::IQ1S
                | Self::IQ3S
                | Self::IQ2S
        )
    }
}

impl std::fmt::Display for GgmlDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Q4_0 => write!(f, "Q4_0"),
            Self::Q4_1 => write!(f, "Q4_1"),
            Self::Q5_0 => write!(f, "Q5_0"),
            Self::Q5_1 => write!(f, "Q5_1"),
            Self::Q8_0 => write!(f, "Q8_0"),
            Self::Q8_1 => write!(f, "Q8_1"),
            Self::Q2K => write!(f, "Q2K"),
            Self::Q3K => write!(f, "Q3K"),
            Self::Q4K => write!(f, "Q4K"),
            Self::Q5K => write!(f, "Q5K"),
            Self::Q6K => write!(f, "Q6K"),
            Self::Q8K => write!(f, "Q8K"),
            Self::IQ2XXS => write!(f, "IQ2_XXS"),
            Self::IQ2XS => write!(f, "IQ2_XS"),
            Self::IQ1S => write!(f, "IQ1_S"),
            Self::IQ4NL => write!(f, "IQ4_NL"),
            Self::IQ3S => write!(f, "IQ3_S"),
            Self::IQ2S => write!(f, "IQ2_S"),
            Self::IQ4XS => write!(f, "IQ4_XS"),
            Self::IQ1M => write!(f, "IQ1_M"),
        }
    }
}

/// Raw GGML-quantized weight data on GPU.
///
/// The bytes are in the exact same format as the GGUF file — no dequantization.
/// The CUDA kernels read quantized blocks directly and do fused dequant+matvec.
#[derive(Clone, Copy)]
pub struct GgmlStorage {
    /// Raw GPU pointer to quantized block data.
    pub ptr: *mut u8,
    /// Total size in bytes.
    pub len: usize,
    /// Quantization type.
    pub dtype: GgmlDType,
    /// Number of rows (output features for a weight matrix).
    pub nrows: usize,
    /// Number of columns (input features) — the "logical" element count per row.
    pub ncols: usize,
}

unsafe impl Send for GgmlStorage {}
unsafe impl Sync for GgmlStorage {}

impl GgmlStorage {
    /// Total number of logical elements.
    pub fn numel(&self) -> usize {
        self.nrows * self.ncols
    }

    /// Verify that the stored byte count matches the expected size.
    pub fn verify_size(&self) -> bool {
        let expected_blocks = self.numel() / self.dtype.block_size();
        let expected_bytes = expected_blocks * self.dtype.type_size();
        self.len == expected_bytes
    }
}

impl std::fmt::Debug for GgmlStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GgmlStorage({}, [{}, {}], {} bytes, ptr={:p})",
            self.dtype, self.nrows, self.ncols, self.len, self.ptr
        )
    }
}
