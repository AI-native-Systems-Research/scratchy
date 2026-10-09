// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! The Metal GPU a tape targets: the constants its architecture name carries.
//!
//! `MTLDevice.architecture.name` reads `applegpu_g<GEN><size>` — `applegpu_g17g` is a base M5,
//! `applegpu_g13d` an M1 Ultra. MLX keys every per-device choice on those two facts
//! (`device.cpp`: `get_architecture_gen`, `get_architecture().back()`); so does the lowering, which
//! is generic over an [`AppleGpu<GEN, ULTRA>`](AppleGpu) and reads every setting off its
//! [`MetalTarget`] consts. One tape is baked per target; the device picks its own at load.

use serde::Serialize;

use super::ids::QmvBatchLimit;
use super::kernel_constants::AffineCodesTarget;

/// A Metal target as a value: a device's parsed architecture, or a baked tape's key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct MetalTargetProfile {
    /// The architecture generation: 13 = M1, 14 = M2, 15 = M3, 16 = M4, 17 = M5.
    pub generation: u32,
    /// An Ultra (architecture letter `d`), which MLX tunes apart.
    pub ultra: bool,
}

impl MetalTargetProfile {
    /// The newest generation a tape is baked for: a later one has all its features.
    pub const NEWEST_GENERATION: u32 = 17;

    /// Every target a tape is baked for: each generation from the M1's, Ultra and not.
    pub const BAKED: [Self; 10] = {
        let mut all = [Self {
            generation: 13,
            ultra: false,
        }; 10];
        let mut i = 0;
        while i < all.len() {
            all[i] = Self {
                generation: 13 + i as u32 / 2,
                ultra: i % 2 == 1,
            };
            i += 1;
        }
        all
    };

    /// The target of the GPU whose architecture is `name` (`applegpu_g<GEN><size>`); `None` for a
    /// name of another shape, or a generation before the M1's.
    pub fn of_architecture(name: &str) -> Option<Self> {
        let rest = name.strip_prefix("applegpu_g")?;
        let letter = rest.chars().last()?;
        let generation: u32 = rest[..rest.len() - letter.len_utf8()].parse().ok()?;
        (generation >= 13).then_some(Self {
            generation: generation.min(Self::NEWEST_GENERATION),
            ultra: letter == 'd',
        })
    }

    /// The NAX matrix unit: MLX `is_nax_available`, gen ≥ 17. An M4 runs MPP `matmul2d` emulated on
    /// the simdgroup matmul, with a per-thread layout that is not `BaseNAXFrag`'s.
    pub const fn has_nax(self) -> bool {
        self.generation >= 17
    }

    /// Whether an affine matvec over two or more rows takes `qmv_wide`: MLX `use_qmv_wide`
    /// ("affine qmv_wide only beats qmv on gen-15+").
    pub const fn affine_qmv_wide(self) -> bool {
        self.generation >= 15
    }

    /// Whether bf16 simdgroup MMAs run emulated (the M1s): ~1.7× slower than f16 (4.58 vs 7.74
    /// TF/s on `affine_qmm_t_*_gs_64_b_4_alN_true_batch_0`, M=1024 N=3072 K=3072), so a bf16
    /// model's qmm_t computes in f16 — casting inside the kernel only; the residual stream stays
    /// bf16.
    pub const fn bf16_simdgroup_slow(self) -> bool {
        self.generation == 13
    }

    /// Whether 4-bit MLX-affine codes are stored XOR 0x88 (signed q - 8): the matrix unit's int8 x
    /// int4 lane reads them as stored (the W4A8 GEMM). The weight loader flips them on load and the
    /// lowering tells every other reader to flip them back — one fact, read by both.
    pub const fn stores_affine_b4_offset8(self) -> bool {
        self.has_nax()
    }
}

/// An Apple GPU of architecture generation `GEN`, an Ultra or not: the Metal target as a type.
pub struct AppleGpu<const GEN: u32, const ULTRA: bool>;

/// Every setting a tape is lowered at, a const of its target.
pub trait MetalTarget {
    const PROFILE: MetalTargetProfile;
    const HAS_NAX: bool = Self::PROFILE.has_nax();
    const AFFINE_QMV_WIDE: bool = Self::PROFILE.affine_qmv_wide();
    const BF16_SIMDGROUP_SLOW: bool = Self::PROFILE.bf16_simdgroup_slow();
    const AFFINE_CODES: AffineCodesTarget = AffineCodesTarget::of(Some(Self::PROFILE));
    const QMV_BATCH_LIMITS: [QmvBatchLimit; 3] = super::quantized::qmv_batch_limits(Self::PROFILE);
}

impl<const GEN: u32, const ULTRA: bool> MetalTarget for AppleGpu<GEN, ULTRA> {
    const PROFILE: MetalTargetProfile = MetalTargetProfile {
        generation: GEN,
        ultra: ULTRA,
    };
}

/// What runs at one target, the target a type: the consumer side of [`with_target`].
pub trait OnTarget {
    type Out;
    fn on_target<T: MetalTarget>(self) -> Self::Out;
}

/// The one door from a target value to its type. `None` for a target no tape is baked for.
pub fn with_target<C: OnTarget>(target: MetalTargetProfile, consumer: C) -> Option<C::Out> {
    fn sized<const GEN: u32, C: OnTarget>(ultra: bool, consumer: C) -> C::Out {
        match ultra {
            false => consumer.on_target::<AppleGpu<GEN, false>>(),
            true => consumer.on_target::<AppleGpu<GEN, true>>(),
        }
    }
    Some(match target.generation {
        13 => sized::<13, C>(target.ultra, consumer),
        14 => sized::<14, C>(target.ultra, consumer),
        15 => sized::<15, C>(target.ultra, consumer),
        16 => sized::<16, C>(target.ultra, consumer),
        17 => sized::<17, C>(target.ultra, consumer),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_names_carry_the_target() {
        let at = |generation, ultra| Some(MetalTargetProfile { generation, ultra });
        let of = MetalTargetProfile::of_architecture;
        assert_eq!(of("applegpu_g13s"), at(13, false));
        assert_eq!(of("applegpu_g13d"), at(13, true));
        assert_eq!(of("applegpu_g16g"), at(16, false));
        assert_eq!(of("applegpu_g17g"), at(17, false));
        assert_eq!(of("applegpu_g18s"), at(17, false));
        assert_eq!(of("applegpu_g12g"), None);
        assert_eq!(of("Apple Paravirtual device"), None);
    }

    /// The door hands every baked target its own type.
    #[test]
    fn every_baked_target_has_its_type() {
        struct Profile;
        impl OnTarget for Profile {
            type Out = MetalTargetProfile;
            fn on_target<T: MetalTarget>(self) -> MetalTargetProfile {
                T::PROFILE
            }
        }
        for target in MetalTargetProfile::BAKED {
            assert_eq!(with_target(target, Profile), Some(target));
        }
    }
}
