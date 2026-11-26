// SPDX-License-Identifier: Apache-2.0
//! The model facts a rotary table is built from.
//!
//! They live here in the cfg-free tensor core, which the config loader
//! (`scratchy-forward-compiler-macro`), the shared passes (`scratchy-subtile`) and every target
//! depend on, so each can name the same types.

/// The RoPE base frequency θ: finite and positive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeTheta(f64);

impl RopeTheta {
    /// The base, or `None` unless finite and positive.
    pub const fn try_new(v: f64) -> Option<Self> {
        match v.is_finite() && v > 0.0 {
            true => Some(Self(v)),
            false => None,
        }
    }
    /// The base as a number.
    pub const fn get(self) -> f64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::RopeTheta;

    #[test]
    fn only_a_finite_positive_base_is_a_rope_theta() {
        assert_eq!(RopeTheta::try_new(1e4).map(RopeTheta::get), Some(1e4));
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(RopeTheta::try_new(bad), None, "{bad}");
        }
    }
}
