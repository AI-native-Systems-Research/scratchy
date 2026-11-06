// SPDX-License-Identifier: Apache-2.0
//! A BAKED TAPE'S WEIGHT READS, TYPED.
//!
//! Each model's bake emits one [`Source`] marker per weight family its tapes bind: the family's
//! [`SourceIx`] and the typed handle (`scratchy_tensors::typed::W2<K, N, _>`, …) its `Weights`
//! field holds, reached through [`Source::at`] — the resolver's one path to it. A dense GEMM's
//! record (`Gemm`: its dims and the weight it reads, which nothing else of the command names)
//! takes the weight's index and dims from its slot's one [`GemmSlot::WEIGHT`], read off the
//! handle; each binding of a typed family's tensor states its index as its slot's. So the loader
//! (the field's type), the tape (the GEMM's weight and dims) and the kernel (picked and baked from
//! those dims) read one type, and rustc checks they agree.
//!
//! What rustc refuses, each refusal one edit from the version it accepts. (ii) A GEMM's dims are
//! its slot's handle's — `lm_head`'s are `[2048, 128256]` — and a slot whose handle has no GEMM
//! dims (the norm's rank-1 gain) has none to give:
//!
//! ```
//! # use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
//! # use scratchy_target_metal::tape::typed::{GemmSlot, Source};
//! # use scratchy_tensors::typed::{Dim, W1, W2};
//! # enum Hidden {}
//! # impl Dim for Hidden { type Num = scratchy_tensors::typed::typenum::U2048; }
//! # enum Vocab {}
//! # impl Dim for Vocab { type Num = scratchy_tensors::typed::typenum::Prod<scratchy_tensors::typed::typenum::U1002, scratchy_tensors::typed::typenum::U128>; }
//! # struct Layer;
//! # struct Weights { lm_head: W2<Hidden, Vocab, Layer>, norm: W1<Hidden, Layer> }
//! # enum Norm {}
//! # impl Source for Norm {
//! #     type Model = Weights;
//! #     type Handle = W1<Hidden, Layer>;
//! #     const IX: SourceIx = SourceIx(1);
//! #     fn at(w: &Weights, _: LayerId) -> &Self::Handle { &w.norm }
//! # }
//! # enum LmHead {}
//! # impl Source for LmHead {
//! #     type Model = Weights;
//! #     type Handle = W2<Hidden, Vocab, Layer>;
//! #     const IX: SourceIx = SourceIx(0);
//! #     fn at(w: &Weights, _: LayerId) -> &Self::Handle { &w.lm_head }
//! # }
//! let w = <LmHead as GemmSlot>::WEIGHT;
//! assert_eq!((w.ix, w.n, w.k), (SourceIx(0), 128256, 2048));
//! ```
//!
//! ```compile_fail,E0277
//! # use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
//! # use scratchy_target_metal::tape::typed::{GemmSlot, Source};
//! # use scratchy_tensors::typed::{Dim, W1, W2};
//! # enum Hidden {}
//! # impl Dim for Hidden { type Num = scratchy_tensors::typed::typenum::U2048; }
//! # enum Vocab {}
//! # impl Dim for Vocab { type Num = scratchy_tensors::typed::typenum::Prod<scratchy_tensors::typed::typenum::U1002, scratchy_tensors::typed::typenum::U128>; }
//! # struct Layer;
//! # struct Weights { lm_head: W2<Hidden, Vocab, Layer>, norm: W1<Hidden, Layer> }
//! # enum Norm {}
//! # impl Source for Norm {
//! #     type Model = Weights;
//! #     type Handle = W1<Hidden, Layer>;
//! #     const IX: SourceIx = SourceIx(1);
//! #     fn at(w: &Weights, _: LayerId) -> &Self::Handle { &w.norm }
//! # }
//! let w = <Norm as GemmSlot>::WEIGHT;
//! ```
//!
//! (iii) A slot whose handle is not the type of the field it reaches — `lm_head` declared
//! `[vocab, hidden]` over a `[hidden, vocab]` field:
//!
//! ```
//! # use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
//! # use scratchy_target_metal::tape::typed::{GemmSlot, Source};
//! # use scratchy_tensors::typed::{Dim, W1, W2};
//! # enum Hidden {}
//! # impl Dim for Hidden { type Num = scratchy_tensors::typed::typenum::U2048; }
//! # enum Vocab {}
//! # impl Dim for Vocab { type Num = scratchy_tensors::typed::typenum::Prod<scratchy_tensors::typed::typenum::U1002, scratchy_tensors::typed::typenum::U128>; }
//! # struct Layer;
//! # struct Weights { lm_head: W2<Hidden, Vocab, Layer>, norm: W1<Hidden, Layer> }
//! # enum Norm {}
//! # impl Source for Norm {
//! #     type Model = Weights;
//! #     type Handle = W1<Hidden, Layer>;
//! #     const IX: SourceIx = SourceIx(1);
//! #     fn at(w: &Weights, _: LayerId) -> &Self::Handle { &w.norm }
//! # }
//! # enum LmHead {}
//! impl Source for LmHead {
//!     type Model = Weights;
//!     type Handle = W2<Hidden, Vocab, Layer>;
//!     const IX: SourceIx = SourceIx(0);
//!     fn at(w: &Weights, _: LayerId) -> &Self::Handle { &w.lm_head }
//! }
//! ```
//!
//! ```compile_fail,E0308
//! # use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
//! # use scratchy_target_metal::tape::typed::{GemmSlot, Source};
//! # use scratchy_tensors::typed::{Dim, W1, W2};
//! # enum Hidden {}
//! # impl Dim for Hidden { type Num = scratchy_tensors::typed::typenum::U2048; }
//! # enum Vocab {}
//! # impl Dim for Vocab { type Num = scratchy_tensors::typed::typenum::Prod<scratchy_tensors::typed::typenum::U1002, scratchy_tensors::typed::typenum::U128>; }
//! # struct Layer;
//! # struct Weights { lm_head: W2<Hidden, Vocab, Layer>, norm: W1<Hidden, Layer> }
//! # enum Norm {}
//! # impl Source for Norm {
//! #     type Model = Weights;
//! #     type Handle = W1<Hidden, Layer>;
//! #     const IX: SourceIx = SourceIx(1);
//! #     fn at(w: &Weights, _: LayerId) -> &Self::Handle { &w.norm }
//! # }
//! # enum LmHead {}
//! impl Source for LmHead {
//!     type Model = Weights;
//!     type Handle = W2<Vocab, Hidden, Layer>;
//!     const IX: SourceIx = SourceIx(0);
//!     fn at(w: &Weights, _: LayerId) -> &Self::Handle { &w.lm_head }
//! }
//! ```
//!
//! (iv) An MLX-affine matmul's slot holds a quantized weight, and its codes and dims are the
//! handle's — `lm_head`'s are 4 bits in groups of 64, `[2048, 128256]`:
//!
//! ```
//! # use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
//! # use scratchy_target_metal::tape::typed::{AffineSlot, Source};
//! # use scratchy_tensors::typed::{B4, Dim, G64, Q2};
//! # enum Hidden {}
//! # impl Dim for Hidden { type Num = scratchy_tensors::typed::typenum::U2048; }
//! # enum Vocab {}
//! # impl Dim for Vocab { type Num = scratchy_tensors::typed::typenum::Prod<scratchy_tensors::typed::typenum::U1002, scratchy_tensors::typed::typenum::U128>; }
//! # struct Layer;
//! # type Head = Q2<Hidden, Vocab, B4, G64, Layer>;
//! # struct Weights { lm_head: Head }
//! # enum LmHead {}
//! # impl Source for LmHead {
//! #     type Model = Weights;
//! #     type Handle = Head;
//! #     const IX: SourceIx = SourceIx(0);
//! #     fn at(w: &Weights, _: LayerId) -> &Head { &w.lm_head }
//! # }
//! let w = <LmHead as AffineSlot>::WEIGHT;
//! assert_eq!((w.n, w.k, w.bits.get(), w.group.get()), (128256, 2048, 4, 64));
//! ```
//!
//! and a dense GEMM over it has no dims to read (E0277):
//!
//! ```compile_fail,E0277
//! # use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
//! # use scratchy_target_metal::tape::typed::{GemmSlot, Source};
//! # use scratchy_tensors::typed::{B4, Dim, G64, Q2};
//! # enum Hidden {}
//! # impl Dim for Hidden { type Num = scratchy_tensors::typed::typenum::U2048; }
//! # enum Vocab {}
//! # impl Dim for Vocab { type Num = scratchy_tensors::typed::typenum::Prod<scratchy_tensors::typed::typenum::U1002, scratchy_tensors::typed::typenum::U128>; }
//! # struct Layer;
//! # type Head = Q2<Hidden, Vocab, B4, G64, Layer>;
//! # struct Weights { lm_head: Head }
//! # enum LmHead {}
//! # impl Source for LmHead {
//! #     type Model = Weights;
//! #     type Handle = Head;
//! #     const IX: SourceIx = SourceIx(0);
//! #     fn at(w: &Weights, _: LayerId) -> &Head { &w.lm_head }
//! # }
//! let w = <LmHead as GemmSlot>::WEIGHT;
//! ```
//!
//! (v) A width no affine kernel is instantiated for has no slot (E0277): `B5` is a width MLX
//! writes, not one these kernels take.
//!
//! ```compile_fail,E0277
//! # use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
//! # use scratchy_target_metal::tape::typed::{AffineSlot, Source};
//! # use scratchy_tensors::typed::{B5, Dim, G64, Q2};
//! # enum Hidden {}
//! # impl Dim for Hidden { type Num = scratchy_tensors::typed::typenum::U2048; }
//! # enum Vocab {}
//! # impl Dim for Vocab { type Num = scratchy_tensors::typed::typenum::Prod<scratchy_tensors::typed::typenum::U1002, scratchy_tensors::typed::typenum::U128>; }
//! # struct Layer;
//! # type Head = Q2<Hidden, Vocab, B5, G64, Layer>;
//! # struct Weights { lm_head: Head }
//! # enum LmHead {}
//! # impl Source for LmHead {
//! #     type Model = Weights;
//! #     type Handle = Head;
//! #     const IX: SourceIx = SourceIx(0);
//! #     fn at(w: &Weights, _: LayerId) -> &Head { &w.lm_head }
//! # }
//! let w = <LmHead as AffineSlot>::WEIGHT;
//! ```
//!
//! (vi) The bake's numbers are the slot's, tied in the build. Its codes by a bound: the markers
//! the bake minted must be the handle's (E0271) —
//!
//! ```
//! # use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
//! # use scratchy_target_metal::tape::typed::{Source, affine_codes_agree, affine_dims_agree, nax_group};
//! # use scratchy_tensors::typed::typenum::{Prod, U128, U1002, U2048};
//! # use scratchy_tensors::typed::{B4, Dim, G64, Q2};
//! # enum Hidden {}
//! # impl Dim for Hidden { type Num = scratchy_tensors::typed::typenum::U2048; }
//! # enum Vocab {}
//! # impl Dim for Vocab { type Num = scratchy_tensors::typed::typenum::Prod<scratchy_tensors::typed::typenum::U1002, scratchy_tensors::typed::typenum::U128>; }
//! # struct Layer;
//! # type Head = Q2<Hidden, Vocab, B4, G64, Layer>;
//! # struct Weights { lm_head: Head }
//! # enum LmHead {}
//! # impl Source for LmHead {
//! #     type Model = Weights;
//! #     type Handle = Head;
//! #     const IX: SourceIx = SourceIx(0);
//! #     fn at(w: &Weights, _: LayerId) -> &Head { &w.lm_head }
//! # }
//! const _: () = affine_codes_agree::<LmHead, scratchy_tensors::typed::B4, G64>();
//! const _: () = affine_dims_agree::<LmHead, Prod<U1002, U128>, U2048>();
//! const _: () = nax_group::<LmHead>();
//! ```
//!
//! ```compile_fail,E0271
//! # use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
//! # use scratchy_target_metal::tape::typed::{Source, affine_codes_agree};
//! # use scratchy_tensors::typed::{B4, Dim, G64, Q2};
//! # enum Hidden {}
//! # impl Dim for Hidden { type Num = scratchy_tensors::typed::typenum::U2048; }
//! # enum Vocab {}
//! # impl Dim for Vocab { type Num = scratchy_tensors::typed::typenum::Prod<scratchy_tensors::typed::typenum::U1002, scratchy_tensors::typed::typenum::U128>; }
//! # struct Layer;
//! # type Head = Q2<Hidden, Vocab, B4, G64, Layer>;
//! # struct Weights { lm_head: Head }
//! # enum LmHead {}
//! # impl Source for LmHead {
//! #     type Model = Weights;
//! #     type Handle = Head;
//! #     const IX: SourceIx = SourceIx(0);
//! #     fn at(w: &Weights, _: LayerId) -> &Head { &w.lm_head }
//! # }
//! const _: () = affine_codes_agree::<LmHead, scratchy_tensors::typed::B8, G64>();
//! ```
//!
//! — its n and k by type equality: the numbers the bake baked, as types, must be the handle's
//! dims' (E0271; a dense GEMM's the same, `gemm_dims_agree`) —
//!
//! ```compile_fail,E0271
//! # use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
//! # use scratchy_target_metal::tape::typed::{Source, affine_dims_agree};
//! # use scratchy_tensors::typed::typenum::{Prod, U128, U1000, U2048};
//! # use scratchy_tensors::typed::{B4, Dim, G64, Q2};
//! # enum Hidden {}
//! # impl Dim for Hidden { type Num = scratchy_tensors::typed::typenum::U2048; }
//! # enum Vocab {}
//! # impl Dim for Vocab { type Num = scratchy_tensors::typed::typenum::Prod<scratchy_tensors::typed::typenum::U1002, scratchy_tensors::typed::typenum::U128>; }
//! # struct Layer;
//! # type Head = Q2<Hidden, Vocab, B4, G64, Layer>;
//! # struct Weights { lm_head: Head }
//! # enum LmHead {}
//! # impl Source for LmHead {
//! #     type Model = Weights;
//! #     type Handle = Head;
//! #     const IX: SourceIx = SourceIx(0);
//! #     fn at(w: &Weights, _: LayerId) -> &Head { &w.lm_head }
//! # }
//! const _: () = affine_dims_agree::<LmHead, Prod<U1000, U128>, U2048>();
//! ```
//!
//! — and a matrix-unit kernel's group by its bound (E0277): those take no group of 32.
//!
//! ```compile_fail,E0277
//! # use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
//! # use scratchy_target_metal::tape::typed::{Source, nax_group};
//! # use scratchy_tensors::typed::{B4, Dim, G32, Q2};
//! # enum Hidden {}
//! # impl Dim for Hidden { type Num = scratchy_tensors::typed::typenum::U2048; }
//! # enum Vocab {}
//! # impl Dim for Vocab { type Num = scratchy_tensors::typed::typenum::Prod<scratchy_tensors::typed::typenum::U1002, scratchy_tensors::typed::typenum::U128>; }
//! # struct Layer;
//! # type Head = Q2<Hidden, Vocab, B4, G32, Layer>;
//! # struct Weights { lm_head: Head }
//! # enum LmHead {}
//! # impl Source for LmHead {
//! #     type Model = Weights;
//! #     type Handle = Head;
//! #     const IX: SourceIx = SourceIx(0);
//! #     fn at(w: &Weights, _: LayerId) -> &Head { &w.lm_head }
//! # }
//! const _: () = nax_group::<LmHead>();
//! ```

use scratchy_tensors::typed::{
    B2, B3, B4, B8, Bits, Dim, DimN, G32, G64, G128, Group, QShape2, Shape2,
};

use super::ids::{AffineBits, AffineGroupSize, LayerId, SourceIx};

/// A code width the MLX-affine kernels are instantiated for, as a type: its [`AffineBits`].
pub trait AffineBitsOf: Bits {
    const BITS: AffineBits;
}

/// A group the MLX-affine kernels are instantiated for, as a type: its [`AffineGroupSize`].
pub trait AffineGroupOf: Group {
    const GROUP: AffineGroupSize;
}

/// The width of the kernels instantiated for int4 codes alone (`qmm_t_splitk`, `qmm_n`, `qvm`,
/// the W4A8 pair).
pub trait Int4: AffineBitsOf {}

/// A group the matrix-unit kernels (`quantized_qmm_nax`) are instantiated for.
pub trait NaxGroup: AffineGroupOf {}

/// ⭐ A KERNEL SET, STATED ONCE: the marker types a kernel family is instantiated for, each
/// implementing its trait, and the table of their numbers the refined value type
/// ([`AffineBits`], …) is minted against. The impls and the table are one list.
macro_rules! kernel_set {
    ($(#[$m:meta])* $table:ident: $carrier:ident, $set:ident { $($marker:ident),+ }
        $(=> $assoc:ident: $value:ident)?) => {
        $(#[$m])*
        pub const $table: &[u32] = &[$(<$marker as DimN>::N),+];
        kernel_set!(@impls $set [$($assoc $value)?] $carrier $($marker)+);
    };
    (@impls $set:ident [] $carrier:ident $($marker:ident)+) => {
        $(impl $set for $marker {})+
    };
    (@impls $set:ident [$assoc:ident $value:ident] $carrier:ident $($marker:ident)+) => {
        $(impl $set for $marker {
            const $assoc: $value = $value::of(<$marker as DimN>::N);
        })+
    };
}

kernel_set! {
    /// The widths the affine kernels are instantiated for (`_b_{2,3,4,8}` in `quantized*.metal`).
    AFFINE_BITS: Bits, AffineBitsOf { B2, B3, B4, B8 } => BITS: AffineBits
}
kernel_set! {
    /// The groups the affine kernels are instantiated for (`_gs_{32,64,128}`).
    AFFINE_GROUP_SIZES: Group, AffineGroupOf { G32, G64, G128 } => GROUP: AffineGroupSize
}
kernel_set! {
    /// The width the int4-only kernels are instantiated for.
    INT4_BITS: Bits, Int4 { B4 }
}
kernel_set! {
    /// The groups the matrix-unit kernels are instantiated for.
    NAX_GROUP_SIZES: Group, NaxGroup { G64, G128 }
}

/// A weight family of model `Model`: its index in the model's source manifest, and the typed
/// handle its field holds.
pub trait Source {
    type Model;
    type Handle;
    const IX: SourceIx;
    fn at(model: &Self::Model, layer: LayerId) -> &Self::Handle;
}

/// A GEMM's weight, as its slot states it: the slot's index and the handle's dims, `n` (output
/// columns) and `k` (the inner dim).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmSlotWeight {
    pub ix: SourceIx,
    pub n: u32,
    pub k: u32,
}

/// A slot a GEMM reads: only a slot whose handle is a rank-2 weight has a [`GemmSlotWeight`].
pub trait GemmSlot {
    const WEIGHT: GemmSlotWeight;
}

impl<S: Source<Handle: Shape2>> GemmSlot for S {
    const WEIGHT: GemmSlotWeight = GemmSlotWeight {
        ix: S::IX,
        n: <<S::Handle as Shape2>::N as DimN>::N,
        k: <<S::Handle as Shape2>::K as DimN>::N,
    };
}

/// The bake's `(N, K)` are slot `S`'s: `N` and `K` the numbers the bake baked its GEMM at, as types,
/// which must be the numbers of the slot's handle's dims (E0271 where they are not).
pub const fn gemm_dims_agree<S, N, K>()
where
    S: Source<Handle: Shape2>,
    <S::Handle as Shape2>::N: Dim<Num = N>,
    <S::Handle as Shape2>::K: Dim<Num = K>,
{
}

/// An MLX-affine matmul's weight, as its slot states it: the slot's index, the handle's dims and
/// its codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineSlotWeight {
    pub ix: SourceIx,
    pub n: u32,
    pub k: u32,
    pub bits: AffineBits,
    pub group: AffineGroupSize,
}

/// A slot an MLX-affine matmul reads: only a slot whose handle is a quantized rank-2 weight of
/// codes the affine kernels are instantiated for has an [`AffineSlotWeight`].
pub trait AffineSlot {
    type Bits: AffineBitsOf;
    type Group: AffineGroupOf;
    const WEIGHT: AffineSlotWeight;
}

impl<S> AffineSlot for S
where
    S: Source<Handle: QShape2>,
    <S::Handle as QShape2>::Bits: AffineBitsOf,
    <S::Handle as QShape2>::Group: AffineGroupOf,
{
    type Bits = <S::Handle as QShape2>::Bits;
    type Group = <S::Handle as QShape2>::Group;
    const WEIGHT: AffineSlotWeight = AffineSlotWeight {
        ix: S::IX,
        n: <<S::Handle as QShape2>::N as DimN>::N,
        k: <<S::Handle as QShape2>::K as DimN>::N,
        bits: <Self::Bits as AffineBitsOf>::BITS,
        group: <Self::Group as AffineGroupOf>::GROUP,
    };
}

/// The bake's codes for slot `S` are `S`'s: `B` and `G` the markers the bake minted, which must
/// be the slot's handle's (E0271 where they are not).
pub const fn affine_codes_agree<S: AffineSlot<Bits = B, Group = G>, B, G>() {}

/// The bake's `(N, K)` for slot `S`'s affine matmuls are `S`'s: `N` and `K` the numbers it baked
/// the kernels at, as types, which must be the numbers of the slot's handle's dims (E0271 where
/// they are not).
pub const fn affine_dims_agree<S, N, K>()
where
    S: AffineSlot + Source<Handle: QShape2>,
    <S::Handle as QShape2>::N: Dim<Num = N>,
    <S::Handle as QShape2>::K: Dim<Num = K>,
{
}

/// Slot `S` is read by an int4-only kernel: its codes must be 4 bits (E0277 where not).
pub const fn int4_codes<S: AffineSlot<Bits: Int4>>() {}

/// Slot `S` is read by a matrix-unit kernel: its group must be one they take (E0277 where not).
pub const fn nax_group<S: AffineSlot<Group: NaxGroup>>() {}
pub use scratchy_tensors::typed::{Act, Extent, ExtentAt, Hosts};

/// An arena colour: its slot, and the bytes its buffer holds, the largest of the extents it hosts.
pub trait Colour {
    const SLOT: u32;
    const BYTES: u64;
}

/// Colour `C`'s slot, as a binding of extent `E` names it: only a colour hosting `E` has one for
/// it. A binding's slot is this call's, so it binds a colour the colouring put its extent in.
pub const fn slot<C: Colour + Hosts<E>, E>() -> u32 {
    C::SLOT
}

/// That colour `C` hosts extent `E`: a binding naming `C`'s slot as `E` through another colour
/// type than the one its [`slot`] is read from.
pub const fn hosts<C: Hosts<E>, E>() {}

/// The bytes of a colour hosting extents of `elems` elements: the largest, at metal's two bytes
/// an element (its kernels are f16), and one byte for a colour that hosts none.
pub const fn colour_bytes(elems: &[u64]) -> u64 {
    let mut bytes = 1;
    let mut i = 0;
    while i < elems.len() {
        let b = elems[i].saturating_mul(2);
        if b > bytes {
            bytes = b;
        }
        i += 1;
    }
    bytes
}

/// A bucket's arena: each slot's bytes, its colour's ([`Colour::BYTES`]), at the bucket's rows.
pub trait Arena {
    const BYTES: &'static [u64];
}

/// A model's arena extents, each a marker ([`ExtentAt`]): `<M, S> E3 = Act<M, D>;` declares `E3`,
/// the extent `Act<M, D>` at a bucket's rows `M` (its `sk_bucket` `S`).
#[macro_export]
macro_rules! arena_extents {
    (<$m:ident, $s:ident> $($extent:ident = $at:ty;)*) => {$(
        pub enum $extent {}
        impl<$m, $s> $crate::tape::typed::ExtentAt<$m, $s> for $extent
        where
            $at: $crate::tape::typed::Extent,
        {
            type At = $at;
        }
    )*};
}

/// ⭐ A MODEL'S ARENA COLOURS, EACH FROM ONE LIST, GENERIC OVER A BUCKET'S ROWS:
/// `<M, S> K3 = 3 [E0, …] [E5, …]` declares colour `K3<M, S>` — slot 3 of any bucket whose rows
/// are `M` (its `sk_bucket` `S`) that the colouring gave these extents — hosting each extent of its
/// two lists (`Hosts`, over [`arena_extents!`] markers). Its [`Colour::BYTES`] is the largest of
/// the first list at the bucket's rows. The second holds what the colour hosts that its bytes do
/// not count: what another of the bucket's sizes covers (its logits rows). Both lists' extents
/// must be extents at the bucket's rows. The colouring's assignment is the lists, so the bytes are
/// the extents', not a number beside them, and a binding's slot ([`slot`]) is a colour that hosts
/// its extent.
///
/// ```
/// # use scratchy_target_metal::tape::typed::{Act, Arena, slot};
/// # use scratchy_tensors::typed::typenum::{U0, U1, U8, U2048, U4096, U8192};
/// scratchy_target_metal::arena_extents! { <M, S>
///     E0 = Act<M, U2048>; E1 = Act<M, U8192>; E2 = Act<U1, U4096>;
/// }
/// scratchy_target_metal::arena_colours! { <M, S> K0 = 0 [E0, E1] [E2] K1 = 1 [] [] }
/// scratchy_target_metal::arena_schemas! { <M, S> S0 [K0, K1] }
/// assert_eq!(<S0<U8, U0> as Arena>::BYTES, &[8 * 8192 * 2, 1]);
/// assert_eq!(slot::<K0<U1, U0>, E2>(), 0);
/// ```
///
/// A colour hosts only the extents of its lists:
///
/// ```compile_fail,E0277
/// # use scratchy_target_metal::tape::typed::{Act, slot};
/// # use scratchy_tensors::typed::typenum::{U0, U8, U2048, U8192};
/// scratchy_target_metal::arena_extents! { <M, S> E0 = Act<M, U2048>; E1 = Act<M, U8192>; }
/// scratchy_target_metal::arena_colours! { <M, S> K0 = 0 [E0] [] }
/// let _ = slot::<K0<U8, U0>, E1>();
/// ```
#[macro_export]
macro_rules! arena_colours {
    (<$m:ident, $s:ident>
     $($colour:ident = $slot:literal [$($sized:ident),* $(,)?] [$($hosted:ident),* $(,)?])*) => {$(
        pub struct $colour<$m, $s>(::core::marker::PhantomData<fn() -> ($m, $s)>);
        $(impl<$m, $s> $crate::tape::typed::Hosts<$sized> for $colour<$m, $s> {})*
        $(impl<$m, $s> $crate::tape::typed::Hosts<$hosted> for $colour<$m, $s> {})*
        impl<$m, $s> $crate::tape::typed::Colour for $colour<$m, $s>
        where
            $($sized: $crate::tape::typed::ExtentAt<$m, $s>,)*
            $($hosted: $crate::tape::typed::ExtentAt<$m, $s>,)*
        {
            const SLOT: u32 = $slot;
            const BYTES: u64 = $crate::tape::typed::colour_bytes(&[$(
                <<$sized as $crate::tape::typed::ExtentAt<$m, $s>>::At
                    as $crate::tape::typed::Extent>::ELEMS
            ),*]);
        }
    )*};
}

/// A model's bucket arenas, each a colouring: `<M, S> S0 [K0, K1, …]` declares arena `S0<M, S>`,
/// slot `k` its `k`th colour, its [`Arena::BYTES`] theirs at the bucket's rows `M`. A bucket is
/// its arena at its rows: `S0<U8, U128>`.
#[macro_export]
macro_rules! arena_schemas {
    (<$m:ident, $s:ident> $($arena:ident [$($colour:ident),* $(,)?])*) => {$(
        pub struct $arena<$m, $s>(::core::marker::PhantomData<fn() -> ($m, $s)>);
        impl<$m, $s> $crate::tape::typed::Arena for $arena<$m, $s>
        where
            $($colour<$m, $s>: $crate::tape::typed::Colour,)*
        {
            const BYTES: &'static [u64] =
                &[$(<$colour<$m, $s> as $crate::tape::typed::Colour>::BYTES),*];
        }
    )*};
}
