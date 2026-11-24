// SPDX-License-Identifier: Apache-2.0
//! The model facts a rotary table is built from.
//!
//! They live here in the cfg-free tensor core so the config loader (`scratchy-forward-compiler-macro`),
//! the shared handoff (`scratchy-subtile`) and every target's tape name the same types.

/// The RoPE base frequency θ: finite and positive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeTheta(f64);

/// Finite, so never NaN: `==` is an equivalence.
impl Eq for RopeTheta {}

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

/// `Serialize` for `$ty`, a newtype whose field is private behind a law: as a newtype named by the
/// path of its checked constructor (`const fn try_new`), over its `get()`. A static tape cannot
/// spell the private field, so `scratchy-const-tokens` emits a newtype named by a path as that
/// constructor's call. The path is built from the type's own name, the one the `impl` names, so it
/// cannot name another type.
macro_rules! serialize_via_try_new {
    ($ty:ident) => {
        impl serde::Serialize for $ty {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                const TRY_NEW: &str =
                    concat!("::", module_path!(), "::", stringify!($ty), "::try_new");
                s.serialize_newtype_struct(TRY_NEW, &self.get())
            }
        }
    };
}
serialize_via_try_new!(RopeTheta);

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
