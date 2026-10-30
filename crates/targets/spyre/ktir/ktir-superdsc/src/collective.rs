// SPDX-License-Identifier: Apache-2.0
//! THE COLLECTIVE PLAN — tensor parallelism's control-block pipeline, as baked data.
//!
//! The shared analysis (`scratchy_subtile::tensor_parallel::shard`) decides WHERE the forward's
//! collectives are. This module is how a spyre bundle EXECUTES them, entirely in the compute
//! stream: each collective's consume is a launch group of its own, preceded by a p2p send of this
//! rank's operand into the peer's receive window and a wait on the write-done counter the peer's
//! twin send increments. No host step sits between a producer, its collective and its consumer.
//!
//! Everything here is a BAKE-TIME constant. The one fact bound at load is which physical card is
//! the peer, which this plan never names: the window lives at the same place on every card (the
//! allocator's region layout is card-symmetric), so a rank addresses its peer's window by its OWN
//! window's placement and no address is ever exchanged.
//!
//! The protocol is #1007 §3's, and every law there that a value can break is a type here:
//! - the world is a member of [`SPYRE_TP_WORLDS`] or [`TpWorld::of`] does not compile;
//! - a rank is minted by its world ([`TpWorld::ranks`]), so it is in range and carries the world
//!   it belongs to ([`TpIdentity`] is one fact, never a rank and a world passed apart);
//! - a p2p payload is a whole number of flits no wider than the RDMA message cap ([`WireBytes`]),
//!   and a wider operand is cut at bake into runs of whole stick columns — the only contiguous
//!   slices of a stick-major tensor ([`fragments`]);
//! - the receive window is ONE `(segment, offset, size)` triple ([`RecvWindow`]), flit-aligned, in
//!   the weight segment;
//! - write-done signals use ONE sid per rank ([`Sid::of`]) — never per collective (a forward holds
//!   more collectives than the card has sids);
//! - a bundle's control blocks are derived from its own launch groups by ONE function (the bundle
//!   crate's `collective::cb_pipeline`, which needs the launch group and so lives above this leaf):
//!   a consume group is `send, wait, launch`, so a wait can never queue ahead of its own send (the
//!   symmetric deadlock: both ranks waiting, neither sending) and every send has exactly one wait.
//!
//! ⭐ THE VALUES LIVE IN THIS LEAF because the emitter marks a consume on its [`EmittedOp`]
//! (`crate::emit::EmittedOp::tp_consume`) before any launch group exists, and the bundle crate
//! sits above this one; the bundle re-exports every name here.
//!
//! [`EmittedOp`]: crate::emit::EmittedOp

use crate::superdsc_opspec::{DataFormat, Fp16};

/// The world sizes a spyre bundle can be baked for — the target's const row. A world of one is a
/// forward with no collectives; two is one peer per card.
pub const SPYRE_TP_WORLDS: &[u32] = &[1, 2];

/// The widest single p2p message, in bytes — the cross-card transfer cap (`CopyToRemote`'s
/// validation; flex-rs `multi_device::MAX_HDMA_MESSAGE_BYTES`, which the target crate pins equal).
pub const RDMA_MESSAGE_CAP_BYTES: u64 = 4 * 1024 * 1024;

/// One flit, the unit every CB address and length field counts in (flex-rs
/// `control_block_wire::Flits::BYTES`, which the target crate pins equal).
pub const FLIT_BYTES: u64 = 128;

/// How many write-done sids a card has.
pub const MAX_SIDS: u32 = 64;

/// The elements of one fp16 stick.
pub const STICK_ELEMS: u32 = <Fp16 as DataFormat>::ELEMS_PER_STICK;

/// The bytes one fp16 STICK holds — one flit. A `[rows, cols]` tensor is laid out stick-major
/// (`(j / stick) · (rows · stick) + i · stick + j % stick`), so ONE STICK COLUMN — every row's
/// elements `[k · stick, (k + 1) · stick)` — is `rows · STICK_BYTES` contiguous bytes, and a run of
/// whole stick columns is the only slice of a tensor that is one contiguous transfer.
pub const STICK_BYTES: u64 = FLIT_BYTES;

const _: () = assert!(
    STICK_ELEMS as u64 * <Fp16 as DataFormat>::WORD_LENGTH as u64 == STICK_BYTES,
    "one fp16 stick is one flit"
);

const fn is_spyre_world(w: u32) -> bool {
    let mut i = 0;
    while i < SPYRE_TP_WORLDS.len() {
        if SPYRE_TP_WORLDS[i] == w {
            return true;
        }
        i += 1;
    }
    false
}

/// A tensor-parallel world a spyre bundle was baked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TpWorld(u32);

impl TpWorld {
    /// The world of `W` ranks. A `W` outside [`SPYRE_TP_WORLDS`] does not compile:
    ///
    /// ```compile_fail
    /// let _ = ktir_superdsc::collective::TpWorld::of::<4>();
    /// ```
    ///
    /// ```
    /// assert_eq!(ktir_superdsc::collective::TpWorld::of::<2>().get(), 2);
    /// ```
    pub const fn of<const W: u32>() -> Self {
        const {
            assert!(
                is_spyre_world(W),
                "spyre bakes only the worlds in SPYRE_TP_WORLDS"
            );
            assert!(
                W <= MAX_SIDS,
                "one sid per rank: a world wider than the card's sids"
            );
        }
        Self(W)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    /// Every rank of this world, each carrying the world it belongs to.
    pub fn ranks(self) -> impl Iterator<Item = TpIdentity> {
        (0..self.0).map(move |rank| TpIdentity { rank, world: self })
    }

    /// Whether this world exchanges anything at all.
    pub const fn is_sharded(self) -> bool {
        self.0 > 1
    }
}

/// THE RANK, WITH ITS WORLD — one fact. Minted only by [`TpWorld::ranks`], so `rank < world`
/// holds by construction, and a session constructed for one world cannot be handed a bundle baked
/// for another without [`TpIdentity::world`] saying so.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TpIdentity {
    rank: u32,
    world: TpWorld,
}

impl TpIdentity {
    pub const fn rank(self) -> u32 {
        self.rank
    }

    pub const fn world(self) -> TpWorld {
        self.world
    }

    /// The other rank of a two-rank world — the one peer this card sends to and hears from.
    /// `None` in a world of one, where there is no peer.
    pub const fn peer(self) -> Option<TpIdentity> {
        match self.world.0 {
            2 => Some(TpIdentity {
                rank: 1 - self.rank,
                world: self.world,
            }),
            _ => None,
        }
    }
}

/// A p2p payload: a whole, non-zero number of flits, no wider than [`RDMA_MESSAGE_CAP_BYTES`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WireBytes(u64);

impl WireBytes {
    /// `Some` iff `bytes` is a non-zero flit multiple within the message cap.
    pub const fn new(bytes: u64) -> Option<Self> {
        if bytes == 0 || !bytes.is_multiple_of(FLIT_BYTES) || bytes > RDMA_MESSAGE_CAP_BYTES {
            return None;
        }
        Some(Self(bytes))
    }

    /// The widest payload one message can carry.
    pub const CAP: Self = Self(RDMA_MESSAGE_CAP_BYTES);

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// THE RECEIVE WINDOW — one per card, where the peer's sends land. One typed triple, the same in
/// every bundle of a forward (prefill, decode, every rung), so a send baked into one bundle and the
/// consume baked into another always name the same bytes.
///
/// It lives in the weight segment ([`RecvWindow::SEGMENT`]), the one process-shared,
/// placement-stable segment, above the packed weights. Its size is the widest fragment any
/// collective of the forward sends — never one bundle's own need.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RecvWindow {
    offset: u64,
    size: WireBytes,
}

impl RecvWindow {
    /// The weight segment (seg1).
    pub const SEGMENT: u32 = 1;

    /// `Some` iff `offset` is flit-aligned.
    pub const fn new(offset: u64, size: WireBytes) -> Option<Self> {
        if !offset.is_multiple_of(FLIT_BYTES) {
            return None;
        }
        Some(Self { offset, size })
    }

    pub const fn offset(self) -> u64 {
        self.offset
    }

    pub const fn size(self) -> WireBytes {
        self.size
    }

    /// One past the window's last byte — the weight segment must reserve up to here.
    pub const fn end(self) -> u64 {
        self.offset + self.size.0
    }
}

/// A write-done sid. ONE PER RANK: the receiver's counter for "my peer's send landed" is the
/// sending rank's, whatever the collective — so the count of collectives in a forward never meets
/// the card's sid budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sid(u8);

impl Sid {
    /// The sid a rank's sends signal on.
    pub const fn of(sender: TpIdentity) -> Self {
        // `TpWorld::of` bounds every world by `MAX_SIDS`, so a rank always fits.
        Self(sender.rank as u8)
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

/// Which collective a consume group runs. Mirrors the shared analysis's kinds; the emitter maps
/// them exhaustively.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CollectiveKind {
    /// The consume adds the peer's partial to this rank's.
    AllReduce,
    /// The consume places the peer's block beside this rank's.
    AllGather,
}

/// One fragment of a collective's operand: a run of whole stick columns, which is one contiguous
/// transfer. Minted only by [`fragments`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fragment {
    first_stick: u32,
    sticks: u32,
    rows: u32,
    bytes: WireBytes,
}

impl Fragment {
    /// Re-mint a fragment from the parts the `#[forward]` macro bakes as tokens. A `const fn` and
    /// the bundle is a `static`, so parts that are not a whole-flit payload within one message —
    /// parts no cut could have produced — fail the BUILD of the crate that bakes them:
    ///
    /// ```compile_fail
    /// use ktir_superdsc::collective::Fragment;
    /// // One stick column of 40 000 rows is 5 MB — past one message.
    /// const F: Fragment = Fragment::baked(0, 1, 40_000);
    /// ```
    pub const fn baked(first_stick: u32, sticks: u32, rows: u32) -> Self {
        match WireBytes::new(sticks as u64 * rows as u64 * STICK_BYTES) {
            Some(bytes) => Self {
                first_stick,
                sticks,
                rows,
                bytes,
            },
            None => panic!("a baked fragment that is not one whole-flit message"),
        }
    }

    /// The first stick column it carries.
    pub const fn first_stick(self) -> u32 {
        self.first_stick
    }

    /// How many stick columns it carries.
    pub const fn sticks(self) -> u32 {
        self.sticks
    }

    /// The fragment's first column (an element index, stick-aligned).
    pub const fn first_col(self) -> u32 {
        self.first_stick * STICK_ELEMS
    }

    /// How many columns it carries (whole sticks).
    pub const fn cols(self) -> u32 {
        self.sticks * STICK_ELEMS
    }

    /// The rows every stick column holds.
    pub const fn rows(self) -> u32 {
        self.rows
    }

    /// Where it starts inside the operand's placement.
    pub const fn byte_offset(self) -> u64 {
        self.first_stick as u64 * self.rows as u64 * STICK_BYTES
    }

    /// What one send carries.
    pub const fn bytes(self) -> WireBytes {
        self.bytes
    }
}

/// Why an operand cannot be cut into fragments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FragmentRefusal {
    /// The operand's columns are not whole sticks — not a layout a stick-column run can cover. The
    /// shared analysis splits only at whole sticks, so this refuses a shape it never mints.
    RaggedCols { cols: u32 },
    /// One stick column alone is wider than one message.
    StickColumnPastCap { rows: u32 },
}

/// A `[rows, cols]` fp16 operand cut into fragments of whole stick columns, each at most `width`
/// bytes, tiling it exactly in column order.
pub fn fragments(rows: u32, cols: u32, width: WireBytes) -> Result<Vec<Fragment>, FragmentRefusal> {
    if cols == 0 || !cols.is_multiple_of(STICK_ELEMS) {
        return Err(FragmentRefusal::RaggedCols { cols });
    }
    let column_bytes = u64::from(rows) * STICK_BYTES;
    let per = width.get().checked_div(column_bytes).unwrap_or(0);
    if per == 0 {
        return Err(FragmentRefusal::StickColumnPastCap { rows });
    }
    let total = cols / STICK_ELEMS;
    let mut out = Vec::new();
    let mut first_stick = 0;
    while first_stick < total {
        let sticks = per.min(u64::from(total - first_stick)) as u32;
        // Whole stick columns, each a flit multiple, within `width` — always a payload.
        let Some(bytes) = WireBytes::new(u64::from(sticks) * column_bytes) else {
            unreachable!("whole stick columns within the width are a payload")
        };
        out.push(Fragment {
            first_stick,
            sticks,
            rows,
            bytes,
        });
        first_stick += sticks;
    }
    Ok(out)
}

/// ⭐ WHAT MAKES A LAUNCH GROUP A COLLECTIVE'S CONSUME — carried on the group, so a bundle's CB
/// pipeline is a function of its own groups (the bundle crate's `collective::cb_pipeline`) and a
/// rolled body replays its collectives per layer exactly as it replays its launches.
///
/// ⭐ AND IT NAMES THE SEND. The send in front of a consume is not a program but an F-send control
/// block (device p2p RDMA, flex-rs `ControlBlockWire::rdma_unicast_send`): this rank's `fragment`
/// of `operand`, from its own placement, into the peer's receive window. Everything it moves is
/// named here, so a consume and its send cannot come apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TpConsume {
    pub kind: CollectiveKind,
    /// The tensor this rank SENDS — the collective's operand, read from its own placement.
    pub operand: crate::place::PlaceId,
    /// Which run of the operand's stick columns this rank sends, and this consume covers.
    pub fragment: Fragment,
    /// `Some(rank)` when only that rank launches this consume — an all-gather's, whose blocks land
    /// at columns that depend on which rank is which, so the bake carries one consume per rank.
    /// `None` when every rank launches it (an all-reduce's sum is the same on every rank).
    pub plays_at: Option<PlaysAt>,
}

/// The one rank that launches a rank-selected consume. Minted from a [`TpIdentity`], or re-minted
/// from baked tokens through [`PlaysAt::baked`], which refuses at BUILD a rank its world does not
/// have — a consume for rank 2 of a two-rank world would never launch, silently.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaysAt(TpIdentity);

impl PlaysAt {
    pub const fn of(rank: TpIdentity) -> Self {
        Self(rank)
    }

    /// Re-mint from the parts the `#[forward]` macro bakes.
    ///
    /// ```compile_fail
    /// use ktir_superdsc::collective::{PlaysAt, TpWorld};
    /// const P: PlaysAt = PlaysAt::baked(TpWorld::of::<2>(), 2);
    /// ```
    pub const fn baked(world: TpWorld, rank: u32) -> Self {
        assert!(
            rank < world.0,
            "a rank-selected consume names a rank its world does not have"
        );
        Self(TpIdentity { rank, world })
    }

    pub const fn rank(self) -> TpIdentity {
        self.0
    }

    /// Whether the session running as `session` launches this consume.
    pub fn launches_on(self, session: TpIdentity) -> bool {
        self.0 == session
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rank_carries_its_world_and_its_peer() {
        let w = TpWorld::of::<2>();
        let ids: Vec<TpIdentity> = w.ranks().collect();
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0].peer(), Some(ids[1]));
        assert_eq!(ids[1].peer(), Some(ids[0]));
        assert_eq!(TpWorld::of::<1>().ranks().next().unwrap().peer(), None);
        assert_ne!(Sid::of(ids[0]), Sid::of(ids[1]));
    }

    #[test]
    fn a_payload_is_whole_flits_within_the_cap() {
        assert!(WireBytes::new(0).is_none());
        assert!(WireBytes::new(FLIT_BYTES - 1).is_none());
        assert!(WireBytes::new(RDMA_MESSAGE_CAP_BYTES + FLIT_BYTES).is_none());
        assert_eq!(WireBytes::new(FLIT_BYTES).unwrap().get(), FLIT_BYTES);
        assert_eq!(WireBytes::CAP.get(), RDMA_MESSAGE_CAP_BYTES);
    }

    /// An operand is cut at whole stick columns, each fragment one contiguous run within the cap,
    /// tiling the columns exactly; a ragged width and a too-tall column are refusals.
    #[test]
    fn fragments_are_whole_stick_columns_within_the_cap() {
        // A 512-row prefill chunk of a 4096-wide hidden: 4 MiB, one fragment at the cap.
        let one = fragments(512, 4096, WireBytes::CAP).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].bytes(), WireBytes::CAP);
        // A 1024-row chunk: 8 MiB, two fragments of 32 stick columns each.
        let f = fragments(1024, 4096, WireBytes::CAP).unwrap();
        assert_eq!(f.len(), 2);
        assert_eq!((f[0].first_col(), f[0].cols()), (0, 2048));
        assert_eq!((f[1].first_col(), f[1].cols()), (2048, 2048));
        assert_eq!(f[1].byte_offset(), 32 * 1024 * STICK_BYTES);
        assert!(f.iter().all(|x| x.bytes() == WireBytes::CAP));
        // The baked re-mint reproduces every cut exactly — what the macro's tokens rebuild.
        for x in f.iter().chain(&one) {
            assert_eq!(Fragment::baked(x.first_stick(), x.sticks(), x.rows()), *x);
        }
        // Decode: one row, one fragment.
        let d = fragments(1, 2048, WireBytes::CAP).unwrap();
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].bytes().get(), 2048 * 2);
        assert_eq!(
            fragments(1, 100, WireBytes::CAP),
            Err(FragmentRefusal::RaggedCols { cols: 100 })
        );
        assert_eq!(
            fragments(40_000, 64, WireBytes::CAP),
            Err(FragmentRefusal::StickColumnPastCap { rows: 40_000 })
        );
    }
}
