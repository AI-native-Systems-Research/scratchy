// SPDX-License-Identifier: Apache-2.0
//! ⭐ A REQUEST'S BUCKET, AS A TYPE, CHOSEN AT ONE DOOR.
//!
//! Each baked bucket row of a model is a type ([`BakedBucket`]): its rows `M`, its fold band, and
//! its arena at those rows (`__arena::S<n><M, S>`), the arena's rows bound to the bucket's. The
//! `#[forward]` bake declares them, and the model's bucket enum of them ([`baked_buckets!`]), one
//! variant per row of its `METAL_BUCKETS`.
//!
//! A variant holds a [`Fits`] proof: that the request's tokens fit its bucket's rows, and that the
//! bucket is one the caller's rows keep. Its fields are private and [`bucket`] alone mints one —
//! the one runtime check between a request's token count and the bucket it runs, as a weight's
//! mint is between a file's numbers and its handle's type. Everything that plays a bucket takes
//! the model's enum ([`BakedBuckets::Bucket`]), and the token count it plays is the proof's: a
//! request longer than its bucket's rows, or a bucket of another model, does not reach a tape.
//!
//! A model's buckets, each at its own rows' arena; the door hands out a bucket of the rows it is
//! handed, and none of none:
//!
//! ```
//! # use scratchy_target_metal::interpreter::metal::{ForwardError, bucket};
//! # scratchy_target_metal::arena_schemas! { <M, S> S0 [] }
//! # use scratchy_tensors::typed::typenum as t;
//! scratchy_target_metal::baked_buckets! {
//!     pub enum Bucket { B0 = (t::U1, Narrow, S0<t::U1, t::U0>), B1 = (t::U8, Wide, S0<t::U8, t::U0>) }
//! }
//! let none = bucket::<Bucket>(&[], 1).map(|_| ());
//! assert!(matches!(none, Err(ForwardError::NoBucketFits { num_tokens: 1, max_bucket: 0 })));
//! ```
//!
//! A proof is the door's alone: no request reaches a bucket's tape without it.
//!
//! ```compile_fail,E0451
//! # use scratchy_target_metal::interpreter::metal::Fits;
//! # scratchy_target_metal::arena_schemas! { <M, S> S0 [] }
//! # use scratchy_tensors::typed::typenum as t;
//! scratchy_target_metal::baked_buckets! { pub enum Bucket { B0 = (t::U1, Narrow, S0<t::U1, t::U0>) } }
//! let too_long = Fits::<B0> { picked: todo!(), bucket: ::core::marker::PhantomData };
//! ```
//!
//! A bucket's arena is the arena at its own rows: another bucket's does not compile.
//!
//! ```compile_fail,E0271
//! # scratchy_target_metal::arena_schemas! { <M, S> S0 [] }
//! # use scratchy_tensors::typed::typenum as t;
//! scratchy_target_metal::baked_buckets! { pub enum Bucket { B0 = (t::U1, Narrow, S0<t::U8, t::U0>) } }
//! ```

use std::marker::PhantomData;
use std::num::NonZeroU32;

use scratchy_tensors::typed::{Dim, DimN};

use super::forward::ForwardError;
use super::pool::{FoldBand, MetalBucketSpec};
use crate::tape::typed::Arena;

/// One baked bucket row of a model, as a type: its rows `M`, its fold band, and its arena, which
/// is the arena at its rows.
pub trait BakedBucket: 'static {
    /// The bucket's rows, `M`.
    type Rows: Dim;
    /// The bucket's arena: its colouring at its rows.
    type Arena: Arena<Rows = Self::Rows>;
    /// The fold band its row was coloured under.
    const BAND: FoldBand;
}

/// What a [`Fits`] proves, whatever its bucket's type: the row a request runs, where its caller's
/// rows keep it, and the request's tokens, which fit the row's rows.
#[derive(Clone, Copy)]
pub struct Picked {
    index: usize,
    row: &'static MetalBucketSpec,
    tokens: NonZeroU32,
}

impl Picked {
    /// The bucket's row.
    pub fn row(&self) -> &'static MetalBucketSpec {
        self.row
    }

    /// The request's tokens: at least one, and at most the bucket's rows.
    pub fn tokens(&self) -> NonZeroU32 {
        self.tokens
    }

    /// Where the rows the door was handed keep the bucket: the pool's bucket, its tape and baking.
    pub(crate) fn index(&self) -> usize {
        self.index
    }
}

/// ⭐ THE PROOF A REQUEST FITS BUCKET `B`: its tokens at most `B`'s rows, `B` a row its caller
/// keeps. Minted by [`bucket`] alone (its fields are private).
pub struct Fits<B> {
    picked: Picked,
    bucket: PhantomData<fn() -> B>,
}

impl<B> Clone for Fits<B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B> Copy for Fits<B> {}

impl<B: BakedBucket> Fits<B> {
    /// What the proof proves.
    pub fn picked(&self) -> &Picked {
        &self.picked
    }

    /// The bytes of each slot of the bucket's arena, at its rows.
    pub fn arena_bytes(&self) -> &'static [u64] {
        <B::Arena as Arena>::BYTES
    }
}

/// The rows and the request a door call checks: what [`Buckets::fitting`] asks of each bucket.
pub struct Door<'a> {
    kept: &'a [&'static MetalBucketSpec],
    tokens: NonZeroU32,
}

impl Door<'_> {
    /// The proof the request fits `B`, when it does: its tokens at most `B`'s rows, and `B` a row
    /// kept here — its rows and band, and its arena's bytes at its rows.
    pub fn fits<B: BakedBucket>(&self) -> Option<Fits<B>> {
        let rows = <B::Rows as DimN>::N;
        if rows < self.tokens.get() {
            return None;
        }
        let bytes = <B::Arena as Arena>::BYTES;
        let index = (self.kept.iter())
            .position(|r| r.bucket_m == rows && r.band == B::BAND && r.arena_bytes == bytes)?;
        Some(Fits {
            picked: Picked {
                index,
                row: self.kept[index],
                tokens: self.tokens,
            },
            bucket: PhantomData,
        })
    }
}

/// A model's buckets, one variant a baked row, each holding its [`Fits`] ([`baked_buckets!`]).
pub trait Buckets: Copy + Sized + 'static {
    /// Each bucket the door finds the request fits, in the model's row order.
    fn fitting(door: &Door<'_>) -> impl Iterator<Item = Self>;

    /// What this bucket's proof proves.
    fn picked(&self) -> &Picked;

    /// The bucket's row.
    fn row(&self) -> &'static MetalBucketSpec {
        self.picked().row()
    }

    /// The request's tokens, which the bucket's rows hold.
    fn tokens(&self) -> NonZeroU32 {
        self.picked().tokens()
    }
}

/// A model whose forwards play baked buckets: its bucket enum.
pub trait BakedBuckets {
    type Bucket: Buckets;
}

/// ⭐ THE DOOR: the bucket a request of `num_tokens` runs, of the rows `kept` — the smallest whose
/// rows hold it, the first in `K`'s row order among equals — as `K`'s variant holding the proof.
pub fn bucket<K: Buckets>(
    kept: &[&'static MetalBucketSpec],
    num_tokens: u32,
) -> Result<K, ForwardError> {
    let tokens = NonZeroU32::new(num_tokens).ok_or(ForwardError::ZeroTokens)?;
    let door = Door { kept, tokens };
    (K::fitting(&door).min_by_key(|b| b.row().bucket_m)).ok_or_else(|| ForwardError::NoBucketFits {
        num_tokens,
        max_bucket: kept.iter().map(|r| r.bucket_m).max().unwrap_or(0),
    })
}

/// [`bucket`]'s row: what a caller holding no bucket enum of its own reads (a model's
/// [`super::MetalRungs::bucket_row`]).
pub fn bucket_row<K: Buckets>(
    kept: &[&'static MetalBucketSpec],
    num_tokens: u32,
) -> Result<&'static MetalBucketSpec, ForwardError> {
    bucket::<K>(kept, num_tokens).map(|b| b.row())
}

/// ⭐ A MODEL'S BUCKETS, EACH A TYPE, AND ITS BUCKET ENUM OF THEM: `pub enum Bucket { B0 = (U1,
/// Narrow, __arena::S0<U1, U0>), … }` declares bucket type `B0` — its rows, band and arena — and
/// `Bucket::B0(Fits<B0>)`, in the model's row order. The arena must be the arena at the bucket's
/// rows (E0271 where not).
#[macro_export]
macro_rules! baked_buckets {
    ($vis:vis enum $name:ident { $($b:ident = ($rows:ty, $band:ident, $arena:ty)),* $(,)? }) => {
        $(
            $vis enum $b {}
            impl $crate::interpreter::metal::BakedBucket for $b {
                type Rows = $rows;
                type Arena = $arena;
                const BAND: $crate::interpreter::metal::FoldBand =
                    $crate::interpreter::metal::FoldBand::$band;
            }
        )*
        /// The model's buckets, each holding the proof a request fits it.
        #[derive(Clone, Copy)]
        $vis enum $name {
            $($b($crate::interpreter::metal::Fits<$b>)),*
        }
        impl $crate::interpreter::metal::Buckets for $name {
            fn fitting(
                door: &$crate::interpreter::metal::Door<'_>,
            ) -> impl ::core::iter::Iterator<Item = Self> {
                [$(door.fits::<$b>().map(Self::$b)),*].into_iter().flatten()
            }

            fn picked(&self) -> &$crate::interpreter::metal::Picked {
                match self {
                    $(Self::$b(fits) => fits.picked()),*
                }
            }
        }
    };
}
