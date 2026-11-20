// SPDX-License-Identifier: Apache-2.0
//! Gated-DeltaNet's layer geometry, as types: the heads, head widths and short convolution a
//! layer's numbers state, and the law that ties the convolution's channels to its streams.
//!
//! Mamba-2 reads as the same geometry. `scratchy-ir` re-exports these from its `model_facts`, so
//! every model's emitted `FACTS` states its layers through the same paths.

use scratchy_tensors::typed::typenum::U2;
use scratchy_tensors::typed::{Add, Dim, DimN, Equals, Mul};

/// A positive count or extent: the constructor refuses zero.
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

positive! {
    /// A Gated-DeltaNet head count (key or value side).
    GdnHeads(u32),
    /// A Gated-DeltaNet head's width (key or value side).
    GdnHeadDim(u32),
    /// The Gated-DeltaNet short convolution's width.
    ConvKernel(u32),
    /// The Gated-DeltaNet convolution's channels.
    ConvDim(usize),
}

/// Gated-DeltaNet (or Mamba-2) layers' geometry. Mamba-2 reads as the same geometry: its groups
/// are the key heads, its heads the value heads, its state size the key head's width and its head
/// dim the value head's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Gdn {
    k_heads: GdnHeads,
    v_heads: GdnHeads,
    k_dim: GdnHeadDim,
    v_dim: GdnHeadDim,
    conv_kernel: ConvKernel,
    conv_dim: ConvDim,
}

impl Gdn {
    /// The layers' geometry of a model's numbers, as types: `KH` key heads of `KD` lanes, `VH`
    /// value heads of `VD`, a short convolution `CK` wide over `CD` channels. Its law is its bound:
    /// the convolution covers exactly its streams, the q and k streams (`2 · KH · KD`) beside the v
    /// stream (`VH · VD`). Mamba-2's conv covers its `x` (`heads · head_dim`) beside its B and C
    /// (`2 · groups · state`): the same law under the reading above. A model's emitted `FACTS`
    /// states its layers through this, so rustc checks the law for every model.
    ///
    /// ```
    /// # use scratchy_state_decoder_gdn::geometry::Gdn;
    /// # use scratchy_tensors::typed::typenum::{U4, U16, U32, U128, U8192};
    /// const G: Gdn = Gdn::of::<U16, U32, U128, U128, U4, U8192>();
    /// assert_eq!(G.conv_dim().get(), 8192);
    /// ```
    ///
    /// A convolution of another width than its streams' (E0277):
    ///
    /// ```compile_fail,E0277
    /// # use scratchy_state_decoder_gdn::geometry::Gdn;
    /// # use scratchy_tensors::typed::typenum::{U4, U16, U32, U128, U4096};
    /// const G: Gdn = Gdn::of::<U16, U32, U128, U128, U4, U4096>();
    /// ```
    pub const fn of<KH: Dim, VH: Dim, KD: Dim, VD: Dim, CK: Dim, CD: Dim>() -> Self
    where
        Add<Mul<U2, Mul<KH, KD>>, Mul<VH, VD>>: Equals<CD>,
    {
        Self::new(
            GdnHeads::new(<KH as DimN>::N),
            GdnHeads::new(<VH as DimN>::N),
            GdnHeadDim::new(<KD as DimN>::N),
            GdnHeadDim::new(<VD as DimN>::N),
            ConvKernel::new(<CK as DimN>::N),
            ConvDim::new(<CD as DimN>::N as usize),
        )
    }

    /// The layers' geometry of values the fill and the tests read. Its law is [`Gdn::of`]'s bound,
    /// which every model's `FACTS` states.
    pub const fn new(
        k_heads: GdnHeads,
        v_heads: GdnHeads,
        k_dim: GdnHeadDim,
        v_dim: GdnHeadDim,
        conv_kernel: ConvKernel,
        conv_dim: ConvDim,
    ) -> Self {
        Self {
            k_heads,
            v_heads,
            k_dim,
            v_dim,
            conv_kernel,
            conv_dim,
        }
    }
    /// Key heads.
    pub const fn k_heads(self) -> GdnHeads {
        self.k_heads
    }
    /// Value heads.
    pub const fn v_heads(self) -> GdnHeads {
        self.v_heads
    }
    /// A key head's width.
    pub const fn k_dim(self) -> GdnHeadDim {
        self.k_dim
    }
    /// A value head's width.
    pub const fn v_dim(self) -> GdnHeadDim {
        self.v_dim
    }
    /// The short convolution's width.
    pub const fn conv_kernel(self) -> ConvKernel {
        self.conv_kernel
    }
    /// The convolution's channels.
    pub const fn conv_dim(self) -> ConvDim {
        self.conv_dim
    }
}
