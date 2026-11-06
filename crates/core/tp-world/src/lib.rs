// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE TENSOR-PARALLEL WORLD, AS A TYPE.
//!
//! A build's `tp/N` names ONE world, [`TpWorld<N>`], and every structure and function of the ladder
//! that a world shapes is generic over it (`T: World`). Nothing stores a world size or reads one
//! back: every count that depends on it is an associated const of `T`, a rank is minted by its
//! world ([`ranks`], [`Rank::of`], [`Rank::baked`]) and carries it as its type, so a value of one
//! world's types cannot reach another world's code and no world is ever compared at run time.
//!
//! There is nothing special about any `N`: a world of one is the same law over one rank.
#![no_std]

use core::marker::PhantomData;

/// A tensor-parallel world. Its one implementor is [`TpWorld<N>`]; a target states its own limits
/// on a world as a trait over this one (spyre's `ktir_superdsc::collective::Tp`).
pub trait World:
    Copy
    + Clone
    + Default
    + core::fmt::Debug
    + PartialEq
    + Eq
    + core::hash::Hash
    + Send
    + Sync
    + 'static
{
    /// The rank count.
    const RANKS: u32;
    /// The law every world keeps, proven at build: at least one rank.
    const NONEMPTY: () = assert!(
        Self::RANKS >= 1,
        "a tensor-parallel world has at least one rank"
    );
    /// Whether this world exchanges anything at all.
    const SHARDED: bool = Self::RANKS > 1;
}

/// ⭐ THE WORLD OF `N` RANKS — the newtype a build's `tp/N` names (`tp/1` is the default, a world
/// of one). `N = 0` does not compile:
///
/// ```compile_fail
/// use scratchy_tp_world::{World, TpWorld};
/// let () = <TpWorld<0> as World>::NONEMPTY;
/// ```
///
/// ```
/// use scratchy_tp_world::{World, TpWorld};
/// let () = <TpWorld<8> as World>::NONEMPTY;
/// assert!(<TpWorld<8> as World>::SHARDED && !<TpWorld<1> as World>::SHARDED);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TpWorld<const N: u32>;

impl<const N: u32> World for TpWorld<N> {
    const RANKS: u32 = N;
}

/// Every rank of world `T`, in rank order.
pub fn ranks<T: World>() -> impl Iterator<Item = Rank<T>> {
    let () = T::NONEMPTY;
    (0..T::RANKS).map(|r| Rank(r, PhantomData))
}

/// ⭐ RANK `r` OF WORLD `T` — its world is its type. Minted only by [`ranks`], [`Rank::of`] and
/// [`Rank::baked`], so `r < T::RANKS` holds by construction and a rank of one world is not a rank of
/// another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rank<T: World>(u32, PhantomData<T>);

impl<T: World> Rank<T> {
    /// Rank `R`, proven below the world at build.
    ///
    /// ```compile_fail
    /// use scratchy_tp_world::{Rank, TpWorld};
    /// let _ = Rank::<TpWorld<4>>::of::<4>();
    /// ```
    pub const fn of<const R: u32>() -> Self {
        const { assert!(R < T::RANKS, "a rank is below its world") };
        Self(R, PhantomData)
    }

    /// Re-mint from the rank the `#[forward]` macro bakes into a `static` — a `const fn`, so a rank
    /// past the world fails the BUILD of the crate that bakes it.
    ///
    /// ```compile_fail
    /// use scratchy_tp_world::{Rank, TpWorld};
    /// const R: Rank<TpWorld<4>> = Rank::baked(4);
    /// ```
    pub const fn baked(r: u32) -> Self {
        assert!(r < T::RANKS, "a baked rank is below its world");
        Self(r, PhantomData)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    /// Every OTHER rank of the world — the peers this rank sends to and hears from, in rank order.
    /// Empty in a world of one.
    pub fn peers(self) -> impl Iterator<Item = Rank<T>> {
        ranks::<T>().filter(move |p| *p != self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A world's ranks are its own, in order, and each rank's peers are every other rank — at every
    /// width, the same law.
    #[test]
    fn a_ranks_peers_are_every_other_rank() {
        fn check<T: World>() {
            let mut n = 0;
            for (i, me) in ranks::<T>().enumerate() {
                assert_eq!(me.get(), i as u32);
                assert_eq!(me.peers().count() as u32, T::RANKS - 1);
                assert!(me.peers().all(|p| p != me && p.get() < T::RANKS));
                n += 1;
            }
            assert_eq!(n, T::RANKS);
        }
        check::<TpWorld<1>>();
        check::<TpWorld<2>>();
        check::<TpWorld<3>>();
        check::<TpWorld<4>>();
        check::<TpWorld<8>>();
        check::<TpWorld<32>>();
    }
}
