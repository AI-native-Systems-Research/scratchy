// SPDX-License-Identifier: Apache-2.0
//! The `positive!` newtype macro: a count or extent whose constructor refuses zero.

/// A positive count or extent: the constructor refuses zero.
#[macro_export]
macro_rules! positive {
    ($($(#[$m:meta])* $name:ident($t:ty)),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name($t);
        impl $name {
            /// The quantity, refused at zero.
            pub const fn new(v: $t) -> Self {
                assert!(v > 0, concat!(stringify!($name), " is a positive quantity"));
                Self(v)
            }
            /// The quantity as a number.
            pub const fn get(self) -> $t {
                self.0
            }
        }
    )*};
}
