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

extern crate alloc;

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

/// The rank count of the world this build names — `tp-N`, or one when it names none.
const BUILD_RANKS: u32 = {
    let named = [
        cfg!(feature = "tp-2"),
        cfg!(feature = "tp-4"),
        cfg!(feature = "tp-8"),
    ];
    let worlds = [2, 4, 8];
    let mut ranks = 1;
    let mut i = 0;
    while i < named.len() {
        if named[i] {
            assert!(
                ranks == 1,
                "a build names one tensor-parallel world: one `tp/<N>`"
            );
            ranks = worlds[i];
        }
        i += 1;
    }
    ranks
};

/// ⭐ THE WORLD THIS BUILD BAKES AND SERVES — the type every top-level instantiation names. The one
/// place a world size is spelled as a number.
pub type Build = TpWorld<BUILD_RANKS>;

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

/// ⭐ ONE `X` PER RANK OF WORLD `T` — minted only by [`PerRank::from_fn`] (or its fallible twin)
/// over [`ranks`], so it holds exactly `T::RANKS` values by construction and indexing it by a
/// [`Rank<T>`] is always in bounds. A list of cards, sessions or weight sets per rank is one of
/// these: no length to check, no index to refuse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PerRank<T: World, X> {
    items: alloc::boxed::Box<[X]>,
    world: PhantomData<T>,
}

impl<T: World, X> PerRank<T, X> {
    /// `f(r)` for every rank, in rank order.
    pub fn from_fn(f: impl FnMut(Rank<T>) -> X) -> Self {
        Self {
            items: ranks::<T>().map(f).collect(),
            world: PhantomData,
        }
    }

    /// `f(r)` for every rank, in rank order; the first error stops it.
    ///
    /// # Errors
    /// The first rank's error.
    pub fn try_from_fn<E>(f: impl FnMut(Rank<T>) -> Result<X, E>) -> Result<Self, E> {
        Ok(Self {
            items: ranks::<T>().map(f).collect::<Result<_, E>>()?,
            world: PhantomData,
        })
    }

    /// Rank `r`'s value.
    pub fn get(&self, r: Rank<T>) -> &X {
        // `r < T::RANKS == self.items.len()` by both types' construction.
        &self.items[r.get() as usize]
    }

    /// Rank `r`'s value, mutably.
    pub fn get_mut(&mut self, r: Rank<T>) -> &mut X {
        &mut self.items[r.get() as usize]
    }

    /// Every rank with its value, in rank order.
    pub fn iter(&self) -> impl Iterator<Item = (Rank<T>, &X)> {
        ranks::<T>().zip(self.items.iter())
    }

    /// Every rank with its value, mutably, in rank order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Rank<T>, &mut X)> {
        ranks::<T>().zip(self.items.iter_mut())
    }

    /// Every value, in rank order.
    pub fn values(&self) -> impl Iterator<Item = &X> {
        self.items.iter()
    }

    /// Every value, mutably, in rank order.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut X> {
        self.items.iter_mut()
    }

    /// Every value, by value, in rank order.
    pub fn into_values(self) -> impl Iterator<Item = X> {
        alloc::vec::Vec::from(self.items).into_iter()
    }

    /// Rank 0's value, mutably.
    pub fn first_mut(&mut self) -> &mut X {
        &mut self.items[0]
    }

    /// Rank 0's value — every world has one.
    pub fn first(&self) -> &X {
        &self.items[0]
    }

    /// Rank 0's value, by value — every world has one (a world of one's only value).
    pub fn into_first(self) -> X {
        // `T::RANKS >= 1` (`World::NONEMPTY`), so there is a rank 0.
        let () = T::NONEMPTY;
        match alloc::vec::Vec::from(self.items).into_iter().next() {
            Some(x) => x,
            None => unreachable!("a world has at least one rank"),
        }
    }

    /// `f` of every rank's value, as the same world's.
    pub fn map<Y>(&self, mut f: impl FnMut(Rank<T>, &X) -> Y) -> PerRank<T, Y> {
        PerRank::from_fn(|r| f(r, self.get(r)))
    }

    /// `f` of every rank's value, by value, as the same world's; the first error stops it.
    ///
    /// # Errors
    /// The first rank's error.
    pub fn try_map_into<Y, E>(
        self,
        mut f: impl FnMut(Rank<T>, X) -> Result<Y, E>,
    ) -> Result<PerRank<T, Y>, E> {
        Ok(PerRank {
            items: ranks::<T>()
                .zip(alloc::vec::Vec::from(self.items))
                .map(|(r, x)| f(r, x))
                .collect::<Result<_, E>>()?,
            world: PhantomData,
        })
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

    /// A per-rank list holds one value per rank, indexed by the rank itself.
    #[test]
    fn a_per_rank_list_is_one_value_per_rank() {
        fn check<T: World>() {
            let v = PerRank::<T, u32>::from_fn(|r| r.get() * 10);
            assert_eq!(v.values().count() as u32, T::RANKS);
            for r in ranks::<T>() {
                assert_eq!(*v.get(r), r.get() * 10);
            }
            assert_eq!(*v.first(), 0);
            let e: Result<PerRank<T, u32>, u32> = PerRank::try_from_fn(|r| {
                if r.get() + 1 == T::RANKS {
                    Err(r.get())
                } else {
                    Ok(0)
                }
            });
            assert_eq!(e, Err(T::RANKS - 1));
        }
        check::<TpWorld<1>>();
        check::<TpWorld<2>>();
        check::<TpWorld<3>>();
        check::<TpWorld<8>>();
        check::<TpWorld<32>>();
    }
}
