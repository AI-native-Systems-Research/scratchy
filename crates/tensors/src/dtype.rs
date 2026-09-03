// SPDX-License-Identifier: Apache-2.0
//! Data types for GPU tensors — only what LLM inference needs.

/// Data type for GPU tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
    U32 = 3,
    I64 = 4,
    I32 = 5,
    U8 = 6,
    Fp8E4m3 = 7,
}

impl DType {
    /// Size in bytes of a single element.
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::U8 | DType::Fp8E4m3 => 1,
            DType::F16 | DType::BF16 => 2,
            DType::F32 | DType::U32 | DType::I32 => 4,
            DType::I64 => 8,
        }
    }

    /// Returns true if this is an FP8 type.
    pub const fn is_fp8(self) -> bool {
        matches!(self, DType::Fp8E4m3)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::F32 => write!(f, "f32"),
            DType::U32 => write!(f, "u32"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::U8 => write!(f, "u8"),
            DType::Fp8E4m3 => write!(f, "fp8_e4m3"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_size_bytes_f16() {
        assert_eq!(DType::F16.size_bytes(), 2);
    }

    #[test]
    fn test_size_bytes_bf16() {
        assert_eq!(DType::BF16.size_bytes(), 2);
    }

    #[test]
    fn test_size_bytes_f32() {
        assert_eq!(DType::F32.size_bytes(), 4);
    }

    #[test]
    fn test_size_bytes_u32() {
        assert_eq!(DType::U32.size_bytes(), 4);
    }

    #[test]
    fn test_size_bytes_i32() {
        assert_eq!(DType::I32.size_bytes(), 4);
    }

    #[test]
    fn test_size_bytes_u8() {
        assert_eq!(DType::U8.size_bytes(), 1);
    }

    #[test]
    fn test_size_bytes_i64() {
        assert_eq!(DType::I64.size_bytes(), 8);
    }

    #[test]
    fn test_size_bytes_fp8e4m3() {
        assert_eq!(DType::Fp8E4m3.size_bytes(), 1);
    }

    #[test]
    fn test_is_fp8() {
        assert!(DType::Fp8E4m3.is_fp8());
        assert!(!DType::BF16.is_fp8());
        assert!(!DType::F16.is_fp8());
        assert!(!DType::F32.is_fp8());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", DType::F16), "f16");
        assert_eq!(format!("{}", DType::BF16), "bf16");
        assert_eq!(format!("{}", DType::F32), "f32");
        assert_eq!(format!("{}", DType::U32), "u32");
        assert_eq!(format!("{}", DType::I32), "i32");
        assert_eq!(format!("{}", DType::I64), "i64");
        assert_eq!(format!("{}", DType::U8), "u8");
        assert_eq!(format!("{}", DType::Fp8E4m3), "fp8_e4m3");
    }

    #[test]
    fn test_debug() {
        assert_eq!(format!("{:?}", DType::F16), "F16");
        assert_eq!(format!("{:?}", DType::BF16), "BF16");
    }

    #[test]
    fn test_equality() {
        assert_eq!(DType::F16, DType::F16);
        assert_ne!(DType::F16, DType::BF16);
        assert_ne!(DType::F32, DType::U32);
    }

    #[test]
    fn test_copy_clone() {
        let a = DType::BF16;
        let b = a; // Copy
        let c = a; // Clone
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn test_hash_distinct() {
        let mut set = HashSet::new();
        set.insert(DType::F16);
        set.insert(DType::BF16);
        set.insert(DType::F32);
        set.insert(DType::U32);
        set.insert(DType::I32);
        set.insert(DType::I64);
        set.insert(DType::U8);
        set.insert(DType::Fp8E4m3);
        assert_eq!(set.len(), 8);
    }

    #[test]
    fn test_hash_duplicates() {
        let mut set = HashSet::new();
        set.insert(DType::F16);
        set.insert(DType::F16);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_repr_values() {
        // Verify repr(u8) discriminants.
        assert_eq!(DType::F16 as u8, 0);
        assert_eq!(DType::BF16 as u8, 1);
        assert_eq!(DType::F32 as u8, 2);
        assert_eq!(DType::U32 as u8, 3);
        assert_eq!(DType::I64 as u8, 4);
        assert_eq!(DType::I32 as u8, 5);
        assert_eq!(DType::U8 as u8, 6);
        assert_eq!(DType::Fp8E4m3 as u8, 7);
    }

    #[test]
    fn test_size_bytes_is_const() {
        // Ensure size_bytes works in const context.
        const F16_SIZE: usize = DType::F16.size_bytes();
        const F32_SIZE: usize = DType::F32.size_bytes();
        assert_eq!(F16_SIZE, 2);
        assert_eq!(F32_SIZE, 4);
    }
}
