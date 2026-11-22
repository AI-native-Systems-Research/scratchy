// SPDX-License-Identifier: Apache-2.0
//! The lowering's quantities, each its own type: positions along an axis ([`Extent`]), the
//! elements between consecutive positions ([`Stride`]), element counts and offsets in a buffer
//! ([`Elems`]), the elements one stick holds ([`Lanes`]), bytes ([`Bytes`]) and cores ([`Cores`]).
//! A quantity crosses an API only as its type; a function body reads `.get()` where it does
//! arithmetic the types do not state.

/// A number of positions along an axis — an axis's extent, a window's corner, a split's share.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Extent(u64);

impl Extent {
    pub const ZERO: Extent = Extent(0);
    pub const ONE: Extent = Extent(1);

    pub const fn new(n: u64) -> Extent {
        Extent(n)
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    /// The extent rounded up to whole sticks of `lanes`.
    pub const fn whole_sticks(self, lanes: Lanes) -> Extent {
        Extent(self.0.next_multiple_of(lanes.0))
    }

    /// The whole sticks of `lanes` this extent spans.
    pub const fn sticks(self, lanes: Lanes) -> Extent {
        Extent(self.0 / lanes.0)
    }

    /// Whether the extent is whole sticks of `lanes`.
    pub const fn is_whole_sticks(self, lanes: Lanes) -> bool {
        self.0.is_multiple_of(lanes.0)
    }

    /// The product of `extents`: a shape's element count, as positions of its flattening.
    pub fn product(extents: &[Extent]) -> Extent {
        Extent(extents.iter().map(|e| e.0).product())
    }

    /// Element count of a dense run of these positions.
    pub const fn elems(self) -> Elems {
        Elems(self.0)
    }
}

/// Elements between consecutive positions of an axis; `0` repeats one element (a broadcast).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Stride(u64);

impl Stride {
    pub const BROADCAST: Stride = Stride(0);
    pub const UNIT: Stride = Stride(1);

    pub const fn new(n: u64) -> Stride {
        Stride(n)
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    /// The element distance `n` positions span.
    pub const fn span(self, n: Extent) -> Elems {
        Elems(self.0 * n.0)
    }

    /// The stride of an axis outside one of `n` positions at this stride.
    pub const fn times(self, n: Extent) -> Stride {
        Stride(self.0 * n.0)
    }

    /// The stride `lanes` elements make — a stick's.
    pub const fn of_lanes(lanes: Lanes) -> Stride {
        Stride(lanes.0)
    }

    /// Whether the stride is whole sticks of `lanes`.
    pub const fn is_whole_sticks(self, lanes: Lanes) -> bool {
        self.0.is_multiple_of(lanes.0)
    }
}

/// A number of elements, or an element offset from a buffer's origin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Elems(u64);

impl Elems {
    pub const ZERO: Elems = Elems(0);

    pub const fn new(n: u64) -> Elems {
        Elems(n)
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    /// Bytes these elements take at `word` bytes each.
    pub const fn bytes(self, word: Bytes) -> Bytes {
        Bytes(self.0 * word.0)
    }
}

impl std::ops::Add for Elems {
    type Output = Elems;
    fn add(self, o: Elems) -> Elems {
        Elems(self.0 + o.0)
    }
}

impl std::ops::AddAssign for Elems {
    fn add_assign(&mut self, o: Elems) {
        self.0 += o.0;
    }
}

impl std::iter::Sum for Elems {
    fn sum<I: Iterator<Item = Elems>>(it: I) -> Elems {
        Elems(it.map(|e| e.0).sum())
    }
}

/// Elements one stick of a format holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lanes(u64);

impl Lanes {
    pub const fn new(n: u64) -> Lanes {
        Lanes(n)
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    /// The lanes as positions along the axis they hold.
    pub const fn extent(self) -> Extent {
        Extent(self.0)
    }
}

/// A number of bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bytes(u64);

impl Bytes {
    pub const ZERO: Bytes = Bytes(0);

    pub const fn new(n: u64) -> Bytes {
        Bytes(n)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl std::ops::Add for Bytes {
    type Output = Bytes;
    fn add(self, o: Bytes) -> Bytes {
        Bytes(self.0 + o.0)
    }
}

impl std::iter::Sum for Bytes {
    fn sum<I: Iterator<Item = Bytes>>(it: I) -> Bytes {
        Bytes(it.map(|b| b.0).sum())
    }
}

/// A number of cores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cores(u64);

impl Cores {
    pub const ONE: Cores = Cores(1);

    pub const fn new(n: u64) -> Cores {
        Cores(n)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}
