// SPDX-License-Identifier: Apache-2.0
//! The model facts a rotary table is built from: a head's width, the lanes rotary rotates, and the
//! RoPE base.
//!
//! They live here in the cfg-free tensor core so the shared rotary builder
//! (`scratchy-layers`) and `scratchy-ir`'s model facts, which re-exports [`HeadDim`] and
//! [`RotDim`], name the same types.

/// One head's width. Its storage IS the non-zero its law states, so
/// [`HeadDim::nonzero`](HeadDim::nonzero) is a field read, not a re-proven conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeadDim(std::num::NonZeroU32);
impl HeadDim {
    /// The width, refused at zero.
    pub const fn new(v: u32) -> Self {
        match std::num::NonZeroU32::new(v) {
            Some(nz) => Self(nz),
            None => panic!("HeadDim is a positive quantity"),
        }
    }
    /// The width as a number.
    pub const fn get(self) -> u32 {
        self.0.get()
    }
    /// The width as a non-zero — the storage already is one, so no caller re-proves
    /// the law this type states.
    pub const fn nonzero(self) -> std::num::NonZeroU32 {
        self.0
    }
}

/// The leading lanes of a head that rotary rotates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RotDim(u32);
impl RotDim {
    /// The quantity, refused at zero.
    pub const fn new(v: u32) -> Self {
        assert!(v > 0, "RotDim is a positive quantity");
        Self(v)
    }
    /// The quantity as a number.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The RoPE base frequency θ: finite and positive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeTheta(f64);
impl RopeTheta {
    /// The base, refused unless finite and positive.
    pub const fn new(v: f64) -> Self {
        assert!(
            v.is_finite() && v > 0.0,
            "RopeTheta is a finite positive base"
        );
        Self(v)
    }
    /// The base as a number.
    pub const fn get(self) -> f64 {
        self.0
    }
}
