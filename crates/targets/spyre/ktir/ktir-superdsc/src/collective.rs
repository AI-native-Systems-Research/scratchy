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
//! - a send moves a [`Payload`] (a whole number of flits) as a run of messages, each within the
//!   RDMA message cap ([`WireBytes`], [`Payload::messages`]); an operand is cut at bake into runs
//!   of whole stick columns — the only contiguous slices of a stick-major tensor ([`fragments`]);
//! - the receive window is ONE `(segment, offset, size)` triple ([`RecvWindow`]), flit-aligned, in
//!   the weight segment, tiled by fixed landing slots ([`LandingSlot`]);
//! - a message lands where its sid's `XSEG` entry points and nowhere else (an F-send's own word
//!   decides nothing — measured, #1009 E8a), so every landing slot has its OWN sid per sending rank
//!   ([`Sid::of`]), each entry programmed once at link: a bounded, static budget per rank, never
//!   one per collective (a forward holds more collectives than the card has sids);
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
/// forward with no collectives; a wider world is `W − 1` peers per card.
pub const SPYRE_TP_WORLDS: &[u32] = &[1, 2, 4];

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
    /// let _ = ktir_superdsc::collective::TpWorld::of::<3>();
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
                W * LandingSlot::SLOTS <= MAX_SIDS,
                "every rank's landing slots take a sid of their own: a world wider than the \
                 card's sids"
            );
        }
        Self(W)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    /// The world of `w` ranks when `w` is only known at run time (a launcher's `--tensor-parallel-size`):
    /// `Some` iff [`TpWorld::of`] would compile for it.
    pub const fn named(w: u32) -> Option<Self> {
        if is_spyre_world(w) && w * LandingSlot::SLOTS <= MAX_SIDS {
            Some(Self(w))
        } else {
            None
        }
    }

    /// Rank `r` of this world, `None` past its last rank.
    pub const fn rank(self, r: u32) -> Option<TpIdentity> {
        if r < self.0 {
            Some(TpIdentity {
                rank: r,
                world: self,
            })
        } else {
            None
        }
    }

    /// Every rank of this world, each carrying the world it belongs to.
    pub fn ranks(self) -> impl Iterator<Item = TpIdentity> {
        (0..self.0).map(move |rank| TpIdentity { rank, world: self })
    }

    /// Whether this world exchanges anything at all.
    pub const fn is_sharded(self) -> bool {
        self.0 > 1
    }

    /// How many peers each rank hears from — the receive window's regions per cut.
    pub const fn peers(self) -> u32 {
        self.0 - 1
    }
}

/// The widest world [`SPYRE_TP_WORLDS`] names.
const WIDEST_WORLD: u32 = {
    let mut w = 0;
    let mut i = 0;
    while i < SPYRE_TP_WORLDS.len() {
        if SPYRE_TP_WORLDS[i] > w {
            w = SPYRE_TP_WORLDS[i];
        }
        i += 1;
    }
    w
};

/// ⭐ ONE SENDING PEER'S PLACE IN A RECEIVE WINDOW. Every cut of a card's window holds one region
/// per peer, in rank order with the card's own rank left out: on rank `r`, rank `s`'s region is
/// `s` below `r` and `s − 1` above it. Minted only by [`TpIdentity::region_of`], so it is below the
/// world's peer count by construction. In a world of two the one peer's region is 0, so the window
/// is exactly one region per cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PeerRegion(u32);

impl PeerRegion {
    /// The most regions any baked world's window has per cut.
    pub const MAX: u32 = WIDEST_WORLD - 1;

    pub const fn index(self) -> u32 {
        self.0
    }

    /// Every region of `world`'s windows, in rank order of the peers they hold.
    pub fn of_world(world: TpWorld) -> impl Iterator<Item = PeerRegion> {
        (0..world.peers()).map(PeerRegion)
    }

    /// Re-mint from the index the `#[forward]` macro bakes; refuses at BUILD a region no baked world
    /// has.
    ///
    /// ```compile_fail
    /// use ktir_superdsc::collective::PeerRegion;
    /// const R: PeerRegion = PeerRegion::baked(PeerRegion::MAX);
    /// ```
    pub const fn baked(region: u32) -> Self {
        assert!(
            region < Self::MAX,
            "a receive-window region no baked world has"
        );
        Self(region)
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

    /// Every other rank of this world, in rank order — the peers this card sends to and hears from.
    pub fn peers(self) -> impl Iterator<Item = TpIdentity> {
        self.world.ranks().filter(move |r| r.rank != self.rank)
    }

    /// The region of this card's receive window that `sender`'s messages land in. `None` for this
    /// rank itself (a rank never sends to itself) or a rank of another world.
    pub const fn region_of(self, sender: TpIdentity) -> Option<PeerRegion> {
        if sender.world.0 != self.world.0 || sender.rank == self.rank {
            return None;
        }
        Some(PeerRegion(if sender.rank > self.rank {
            sender.rank - 1
        } else {
            sender.rank
        }))
    }
}

/// ONE p2p MESSAGE: a whole, non-zero number of flits, no wider than [`RDMA_MESSAGE_CAP_BYTES`]
/// (the send section's 15-bit flit length field). A wider [`Payload`] is cut into these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WireBytes(u64);

/// WHAT ONE SEND MOVES: a whole, non-zero number of flits, of any width. On the wire it is a run of
/// [`Message`]s — the vendor's windowed P2P loop: one F-send per [`RDMA_MESSAGE_CAP_BYTES`] chunk,
/// `first` on the first, `last` on the last ([`Payload::messages`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Payload(u64);

impl Payload {
    /// `Some` iff `bytes` is a non-zero flit multiple.
    pub const fn new(bytes: u64) -> Option<Self> {
        if bytes == 0 || !bytes.is_multiple_of(FLIT_BYTES) {
            return None;
        }
        Some(Self(bytes))
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    /// How many messages carry it.
    pub const fn message_count(self) -> u32 {
        self.0.div_ceil(RDMA_MESSAGE_CAP_BYTES) as u32
    }

    /// Its messages, in order: whole caps, then the remainder. They tile the payload exactly.
    pub fn messages(self) -> impl Iterator<Item = Message> {
        let n = self.message_count();
        (0..n).map(move |i| {
            let offset = u64::from(i) * RDMA_MESSAGE_CAP_BYTES;
            let Some(bytes) = WireBytes::new((self.0 - offset).min(RDMA_MESSAGE_CAP_BYTES)) else {
                unreachable!("a flit-multiple remainder within the cap is one message")
            };
            Message {
                offset,
                bytes,
                first: i == 0,
                last: i + 1 == n,
            }
        })
    }
}

impl From<WireBytes> for Payload {
    fn from(w: WireBytes) -> Self {
        Self(w.0)
    }
}

/// One message of a [`Payload`]: where it starts inside the payload, how much it carries, and its
/// place in the run (`first` / `last`, the send section's own protocol bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Message {
    pub offset: u64,
    pub bytes: WireBytes,
    pub first: bool,
    pub last: bool,
}

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

/// The rows a receive window holds: the most rows any bundle of a forward hands a collective. The
/// pool's prompt window (`PagedKvPool::PREFILL_CHUNK_SLOTS`) bounds every prefill chunk, and every
/// decode batch is narrower (asserted below).
pub const WINDOW_ROWS: u32 = crate::sdsc_abstract::PagedKvPool::PREFILL_CHUNK_SLOTS as u32;

const _: () = assert!(
    crate::sdsc_abstract::PagedKvPool::WIDEST_BATCH_RUNG <= WINDOW_ROWS,
    "a decode batch wider than the prompt window would hand a collective more rows than the \
     receive window holds"
);

/// How many disjoint cuts the receive window is — one placement per cut
/// ([`crate::reserved_tids::TP_RECV_CUT_TIDS`]).
pub const WINDOW_CUTS: usize = crate::reserved_tids::TP_RECV_CUT_TIDS.len();

/// ⭐ WHICH CUT OF THE RECEIVE WINDOW A COLLECTIVE USES — its send lands there, its consume reads
/// there. The forward's `k`-th collective takes cut `k mod WINDOW_CUTS`: consecutive collectives
/// alternate. A baked fact of the consume ([`TpConsume::cut`]), never re-derived at load.
///
/// Why two are enough, by the pipeline's own order: a send carries the CTRL barrier, so this
/// rank's send for collective `k + 1` cannot start until its consume of `k` has completed; the
/// peer cannot pass its wait on `k + 1` until that send lands; and only past that wait can the
/// peer issue its send for `k + 2` — the next one into `k`'s cut. So no send lands in a cut its
/// receiver is still reading, with no reverse signal (#1009 review, item 4: disjoint cuts of the
/// one window). Each cut's messages land through sids of their own ([`LandingSlot`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WindowCut(u32);

impl WindowCut {
    /// The cut of the forward's `k`-th collective.
    pub const fn of_collective(k: usize) -> Self {
        Self((k % WINDOW_CUTS) as u32)
    }

    /// Re-mint from the index the `#[forward]` macro bakes.
    ///
    /// ```compile_fail
    /// use ktir_superdsc::collective::WindowCut;
    /// const C: WindowCut = WindowCut::baked(2);
    /// ```
    pub const fn baked(cut: u32) -> Self {
        assert!(
            (cut as usize) < WINDOW_CUTS,
            "a consume names a window cut the window does not have"
        );
        Self(cut)
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// The tid `region`'s placement in this cut carries — what a consume program binds to read that
    /// peer's fragment.
    pub const fn tid(self, region: PeerRegion) -> u32 {
        crate::reserved_tids::tp_recv_tid(self.0, region.0)
    }
}

/// Two consecutive collectives, in the order the card runs them, that use ONE window cut — the
/// second's send could land while the first's consume is still reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CutReused {
    /// The later collective's position in run order.
    pub collective: usize,
    pub cut: WindowCut,
}

/// ⭐ THE LAW THE TWO CUTS STAND ON: consecutive collectives, in the order the card RUNS them, use
/// different cuts. [`WindowCut`]'s proof needs exactly this — the peer may send `k + 2` into `k`'s
/// cut only because `k + 1`, in the other cut, sits between them. `cuts` is one entry per
/// collective, in run order (a rolled body counts once per layer it is replayed for).
///
/// # Errors
/// [`CutReused`]: the first collective whose cut is its predecessor's.
pub fn cuts_alternate(cuts: impl IntoIterator<Item = WindowCut>) -> Result<(), CutReused> {
    let mut last = None;
    for (collective, cut) in cuts.into_iter().enumerate() {
        if last == Some(cut) {
            return Err(CutReused { collective, cut });
        }
        last = Some(cut);
    }
    Ok(())
}

/// The most messages one window cut holds — the width of a rank's landing-slot budget. A forward
/// whose widest collective operand needs more refuses at bake ([`RecvWindow::cut_bytes_for`]).
pub const MAX_CUT_MESSAGES: u32 = 8;

/// ⭐ ONE MESSAGE'S FIXED PLACE IN THE RECEIVE WINDOW: message `message` of a send into cut `cut`,
/// at the cut's offset plus `message` whole messages ([`RecvWindow::slot_offset`]).
///
/// WHY A SLOT IS A SID. An F-send lands where the receiver's `XSEG[sid]` entry points, and the
/// engine adds nothing — the send's own xseg word decides no landing (measured on card, #1009: E8a;
/// "one sid plus one static table entry is one landing address"). So two messages that must land
/// apart must signal on two sids. Each slot's entry is programmed ONCE, at link, to the slot's
/// address; every send and every wait names its slot's sid; nothing re-points a table entry at run
/// time (the T-send's re-pointing fires an unpaired write-done count, #1009).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LandingSlot {
    cut: WindowCut,
    message: u32,
}

impl LandingSlot {
    /// Every slot a rank's sends can land in: each cut's run of messages.
    pub const SLOTS: u32 = WINDOW_CUTS as u32 * MAX_CUT_MESSAGES;

    /// Message `message` of a send into `cut`; `None` past [`MAX_CUT_MESSAGES`].
    pub const fn of(cut: WindowCut, message: u32) -> Option<Self> {
        if message < MAX_CUT_MESSAGES {
            Some(Self { cut, message })
        } else {
            None
        }
    }

    pub const fn cut(self) -> WindowCut {
        self.cut
    }

    pub const fn message(self) -> u32 {
        self.message
    }

    /// The slot's place in a rank's budget, `< SLOTS`.
    pub const fn index(self) -> u32 {
        self.cut.0 * MAX_CUT_MESSAGES + self.message
    }
}

/// WHAT ONE COLLECTIVE'S SEND MOVES: a [`Payload`] of at most [`MAX_CUT_MESSAGES`] messages — one
/// cut's worth of slots, so every message it is cut into has a slot to land in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CutPayload(Payload);

impl CutPayload {
    /// `Some` iff `bytes` is at most [`MAX_CUT_MESSAGES`] messages.
    pub const fn new(bytes: Payload) -> Option<Self> {
        if bytes.message_count() <= MAX_CUT_MESSAGES {
            Some(Self(bytes))
        } else {
            None
        }
    }

    pub const fn get(self) -> Payload {
        self.0
    }

    /// Its messages ([`Payload::messages`]), each with the slot of `cut` it lands in: message `m`
    /// in slot `(cut, m)`.
    pub fn landings(self, cut: WindowCut) -> impl Iterator<Item = (LandingSlot, Message)> {
        // `new` bounds the count by `MAX_CUT_MESSAGES`, so every index is a slot.
        (0..)
            .zip(self.0.messages())
            .map(move |(message, m)| (LandingSlot { cut, message }, m))
    }
}

/// THE RECEIVE WINDOW — one per card, where its peers' sends land. One typed triple, the same in
/// every bundle of a forward (prefill, decode, every rung), so a send baked into one bundle and the
/// consume baked into another always name the same bytes.
///
/// It lives in the weight segment ([`RecvWindow::SEGMENT`]), the one process-shared,
/// placement-stable segment, above the packed weights, and it is [`WINDOW_CUTS`] equal, adjacent
/// cuts. A cut holds one REGION per peer ([`PeerRegion`], in rank order), each region
/// [`RecvWindow::cut_bytes_for`] the forward's widest collective operand — never one bundle's own
/// need. Each region is a run of fixed landing slots ([`RecvWindow::slots`]), one `XSEG` entry per
/// slot, each pointing at its slot. In a world of two a cut is one region, and the window is
/// exactly the one it has always been.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RecvWindow {
    offset: u64,
    region: Payload,
    world: TpWorld,
}

impl RecvWindow {
    /// The weight segment (seg1).
    pub const SEGMENT: u32 = 1;

    /// `world`'s window at `offset`: [`WINDOW_CUTS`] cuts of one `region`-byte region per peer.
    /// `Some` iff the world exchanges at all, `offset` is flit-aligned, and a region is at most
    /// [`MAX_CUT_MESSAGES`] messages — a slot each.
    pub const fn new(offset: u64, region: Payload, world: TpWorld) -> Option<Self> {
        if !world.is_sharded()
            || !offset.is_multiple_of(FLIT_BYTES)
            || region.message_count() > MAX_CUT_MESSAGES
        {
            return None;
        }
        Some(Self {
            offset,
            region,
            world,
        })
    }

    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// One region's bytes — the most one peer's send into a cut can carry.
    pub const fn region(self) -> Payload {
        self.region
    }

    /// The world whose peers land here.
    pub const fn world(self) -> TpWorld {
        self.world
    }

    /// One cut's bytes: a region per peer.
    const fn cut_bytes(self) -> u64 {
        self.world.peers() as u64 * self.region.0
    }

    /// Where `cut` starts in the segment — its first region.
    pub const fn cut_offset(self, cut: WindowCut) -> u64 {
        self.offset + cut.0 as u64 * self.cut_bytes()
    }

    /// Where `region` of `cut` starts in the segment.
    pub const fn region_offset(self, cut: WindowCut, region: PeerRegion) -> u64 {
        self.cut_offset(cut) + region.0 as u64 * self.region.0
    }

    /// Where `slot` of `region` starts in the segment: its region, plus its whole messages before
    /// it.
    pub const fn slot_offset(self, region: PeerRegion, slot: LandingSlot) -> u64 {
        self.region_offset(slot.cut, region) + slot.message as u64 * RDMA_MESSAGE_CAP_BYTES
    }

    /// ⭐ EVERY SLOT `region` HAS, with its offset in the segment — each cut's run of messages. What
    /// the receiver programs for the peer landing there, one `XSEG` entry per slot, at link.
    pub fn slots(self, region: PeerRegion) -> impl Iterator<Item = (LandingSlot, u64)> {
        (0..WINDOW_CUTS).flat_map(move |k| {
            let cut = WindowCut::of_collective(k);
            (0..self.region.message_count()).filter_map(move |m| {
                LandingSlot::of(cut, m).map(|slot| (slot, self.slot_offset(region, slot)))
            })
        })
    }

    /// The whole window's bytes — every cut.
    pub const fn size(self) -> u64 {
        WINDOW_CUTS as u64 * self.cut_bytes()
    }

    /// One past the window's last byte — the weight segment must reserve up to here.
    pub const fn end(self) -> u64 {
        self.offset + self.size()
    }

    /// ⭐ ONE CUT A FORWARD'S COLLECTIVES NEED: [`WINDOW_ROWS`] rows of its widest collective
    /// operand (`widest_cols`, stick-rounded fp16). It reads only the operand's COLUMNS (the
    /// forward's) and the row cap (the target's) — never a bundle's own row count — so every
    /// bundle of one forward places the same window, as the link's `XSEG` entries and the shared
    /// weight segment require. Any fragment of any collective of the forward fits a cut whole, so
    /// a collective is ONE fragment and its send is a run of messages ([`Payload::messages`]),
    /// each landing in a slot of its own. `None` for no columns, or for a cut wider than
    /// [`MAX_CUT_MESSAGES`] messages — more slots than a rank's sid budget holds.
    pub const fn cut_bytes_for(widest_cols: u32) -> Option<Payload> {
        let cols = widest_cols.next_multiple_of(STICK_ELEMS) as u64;
        match Payload::new(cols * <Fp16 as DataFormat>::WORD_LENGTH as u64 * WINDOW_ROWS as u64) {
            Some(cut) if cut.message_count() <= MAX_CUT_MESSAGES => Some(cut),
            _ => None,
        }
    }
}

/// A write-done sid — and the `XSEG` entry its message lands through. ONE PER (SENDING RANK,
/// LANDING SLOT): the receiver's counter for "this slot of my peer's landed" and the entry pointing
/// at that slot, whatever the collective — so the count of collectives in a forward never meets the
/// card's sid budget, and the budget a world takes is a constant (`TpWorld::of` proves it fits).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sid(u8);

impl Sid {
    /// The sid `sender`'s message into `slot` signals on.
    pub const fn of(sender: TpIdentity, slot: LandingSlot) -> Self {
        // `TpWorld::of` bounds `world · SLOTS` by `MAX_SIDS`, and `slot.index() < SLOTS`.
        Self((sender.rank * LandingSlot::SLOTS + slot.index()) as u8)
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
    bytes: Payload,
}

impl Fragment {
    /// Re-mint a fragment from the parts the `#[forward]` macro bakes as tokens. A `const fn` and
    /// the bundle is a `static`, so parts that carry no payload — parts no cut could have produced —
    /// fail the BUILD of the crate that bakes them:
    ///
    /// ```compile_fail
    /// use ktir_superdsc::collective::Fragment;
    /// // A fragment of no rows moves nothing.
    /// const F: Fragment = Fragment::baked(0, 1, 0);
    /// ```
    ///
    /// A fragment wider than one message is a payload like any other — its send is several:
    ///
    /// ```
    /// use ktir_superdsc::collective::Fragment;
    /// // One stick column of 40 000 rows is 5 MB: two messages.
    /// const F: Fragment = Fragment::baked(0, 1, 40_000);
    /// assert_eq!(F.bytes().message_count(), 2);
    /// ```
    pub const fn baked(first_stick: u32, sticks: u32, rows: u32) -> Self {
        match Payload::new(sticks as u64 * rows as u64 * STICK_BYTES) {
            Some(bytes) => Self {
                first_stick,
                sticks,
                rows,
                bytes,
            },
            None => panic!("a baked fragment that carries no whole-flit payload"),
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
    pub const fn bytes(self) -> Payload {
        self.bytes
    }
}

/// Why an operand cannot be cut into fragments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FragmentRefusal {
    /// The operand's columns are not whole sticks — not a layout a stick-column run can cover. The
    /// shared analysis splits only at whole sticks, so this refuses a shape it never mints.
    RaggedCols { cols: u32 },
    /// One stick column alone is wider than the cut's width.
    StickColumnPastWidth { rows: u32 },
}

/// A `[rows, cols]` fp16 operand cut into fragments of whole stick columns, each at most `width`
/// bytes, tiling it exactly in column order.
pub fn fragments(rows: u32, cols: u32, width: Payload) -> Result<Vec<Fragment>, FragmentRefusal> {
    if cols == 0 || !cols.is_multiple_of(STICK_ELEMS) {
        return Err(FragmentRefusal::RaggedCols { cols });
    }
    let column_bytes = u64::from(rows) * STICK_BYTES;
    let per = width.get().checked_div(column_bytes).unwrap_or(0);
    if per == 0 {
        return Err(FragmentRefusal::StickColumnPastWidth { rows });
    }
    let total = cols / STICK_ELEMS;
    let mut out = Vec::new();
    let mut first_stick = 0;
    while first_stick < total {
        let sticks = per.min(u64::from(total - first_stick)) as u32;
        // Whole stick columns, each a flit multiple, within `width` — always a payload.
        let Some(bytes) = Payload::new(u64::from(sticks) * column_bytes) else {
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
/// block (device p2p RDMA, flex-rs `scheduler::build_rdma_send_cb`): this rank's `fragment`
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
    /// The window cut the send lands in and the consume reads.
    pub cut: WindowCut,
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

    /// A run-time world is exactly a compile-time one, and a run-time rank is in range.
    #[test]
    fn a_launchers_world_and_rank_are_the_baked_ones_or_nothing() {
        assert_eq!(TpWorld::named(1), Some(TpWorld::of::<1>()));
        assert_eq!(TpWorld::named(2), Some(TpWorld::of::<2>()));
        assert_eq!(TpWorld::named(0), None);
        assert_eq!(TpWorld::named(3), None);
        assert_eq!(TpWorld::named(4), Some(TpWorld::of::<4>()));
        let w = TpWorld::of::<2>();
        assert_eq!(w.rank(1), w.ranks().nth(1));
        assert_eq!(w.rank(2), None);
    }

    #[test]
    fn a_rank_carries_its_world_and_its_peers() {
        let w = TpWorld::of::<2>();
        let ids: Vec<TpIdentity> = w.ranks().collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.iter().all(|id| id.world() == w));
        assert_eq!(ids[0].peers().collect::<Vec<_>>(), [ids[1]]);
        assert_eq!(ids[1].peers().collect::<Vec<_>>(), [ids[0]]);
        assert_eq!(
            TpWorld::of::<1>().ranks().next().unwrap().peers().count(),
            0
        );
    }

    /// Consecutive collectives in run order must alternate cuts: an even run per replayed layer
    /// passes, and a layer with ONE collective replayed twice reuses its cut, refused at the second.
    #[test]
    fn consecutive_collectives_alternate_cuts_or_refuse() {
        let c = WindowCut::of_collective;
        assert_eq!(cuts_alternate([c(0), c(1), c(0), c(1)]), Ok(()));
        assert_eq!(cuts_alternate([]), Ok(()));
        assert_eq!(
            cuts_alternate([c(0), c(0)]),
            Err(CutReused {
                collective: 1,
                cut: c(0)
            })
        );
        assert_eq!(
            cuts_alternate([c(0), c(1), c(1)]),
            Err(CutReused {
                collective: 2,
                cut: c(1)
            })
        );
    }

    /// Every (rank, slot) of a two-rank world signals on a sid of its own, all within the card's.
    #[test]
    fn every_rank_and_slot_has_its_own_sid() {
        let slots: Vec<LandingSlot> = (0..WINDOW_CUTS)
            .flat_map(|k| {
                (0..MAX_CUT_MESSAGES)
                    .map(move |m| LandingSlot::of(WindowCut::of_collective(k), m).unwrap())
            })
            .collect();
        assert_eq!(slots.len() as u32, LandingSlot::SLOTS);
        let sids: std::collections::BTreeSet<u8> = TpWorld::of::<2>()
            .ranks()
            .flat_map(|r| slots.iter().map(move |&s| Sid::of(r, s).get()))
            .collect();
        assert_eq!(sids.len() as u32, 2 * LandingSlot::SLOTS);
        assert!(sids.iter().all(|&s| u32::from(s) < MAX_SIDS));
        assert_eq!(
            LandingSlot::of(WindowCut::of_collective(0), MAX_CUT_MESSAGES),
            None
        );
    }

    /// A window's slots tile each cut message by message, and a cut wider than the slot budget is
    /// no window at all.
    #[test]
    fn a_windows_slots_tile_its_cuts() {
        let two = TpWorld::of::<2>();
        let region = Payload::new(2 * RDMA_MESSAGE_CAP_BYTES + FLIT_BYTES).unwrap();
        let w = RecvWindow::new(1 << 20, region, two).unwrap();
        let only = PeerRegion::of_world(two).collect::<Vec<_>>();
        assert_eq!(only.len(), 1);
        let slots: Vec<(u32, u32, u64)> = w
            .slots(only[0])
            .map(|(s, at)| (s.cut().index() as u32, s.message(), at))
            .collect();
        let at = |c: u64, m: u64| (1 << 20) + c * region.get() + m * RDMA_MESSAGE_CAP_BYTES;
        assert_eq!(
            slots,
            [
                (0, 0, at(0, 0)),
                (0, 1, at(0, 1)),
                (0, 2, at(0, 2)),
                (1, 0, at(1, 0)),
                (1, 1, at(1, 1)),
                (1, 2, at(1, 2)),
            ]
        );
        assert_eq!(w.size(), WINDOW_CUTS as u64 * region.get());
        let widest = Payload::new(u64::from(MAX_CUT_MESSAGES) * RDMA_MESSAGE_CAP_BYTES).unwrap();
        assert!(RecvWindow::new(0, widest, two).is_some());
        let past = Payload::new(widest.get() + FLIT_BYTES).unwrap();
        assert!(RecvWindow::new(0, past, two).is_none());
        assert!(RecvWindow::new(0, widest, TpWorld::of::<1>()).is_none());
        assert!(RecvWindow::cut_bytes_for(1 << 20).is_none());
    }

    /// Each rank hears its peers in rank order, its own rank left out: rank `s` lands in region
    /// `s` below the receiver and `s − 1` above it, and a region's tid in cut 0 is the cut's own.
    #[test]
    fn a_peer_lands_in_its_rank_ordered_region() {
        let two: Vec<TpIdentity> = TpWorld::of::<2>().ranks().collect();
        assert_eq!(two[0].peers().collect::<Vec<_>>(), [two[1]]);
        assert_eq!(two[1].peers().collect::<Vec<_>>(), [two[0]]);
        assert_eq!(two[0].region_of(two[1]).map(PeerRegion::index), Some(0));
        assert_eq!(two[1].region_of(two[0]).map(PeerRegion::index), Some(0));
        assert_eq!(two[0].region_of(two[0]), None);
        let one = TpWorld::of::<1>().ranks().next().unwrap();
        assert_eq!(one.peers().count(), 0);
        assert_eq!(two[0].region_of(one), None);
        for (k, &tid) in crate::reserved_tids::TP_RECV_CUT_TIDS.iter().enumerate() {
            assert_eq!(WindowCut::of_collective(k).tid(PeerRegion(0)), tid);
        }
    }

    #[test]
    fn a_payload_is_whole_flits_within_the_cap() {
        assert!(WireBytes::new(0).is_none());
        assert!(WireBytes::new(FLIT_BYTES - 1).is_none());
        assert!(WireBytes::new(RDMA_MESSAGE_CAP_BYTES + FLIT_BYTES).is_none());
        assert_eq!(WireBytes::new(FLIT_BYTES).unwrap().get(), FLIT_BYTES);
        assert_eq!(WireBytes::CAP.get(), RDMA_MESSAGE_CAP_BYTES);
    }

    /// A payload is cut into the vendor loop's messages: whole caps then the remainder, tiling it
    /// exactly, `first` on the first alone and `last` on the last alone — one message is both.
    #[test]
    fn a_payload_goes_out_as_messages_within_the_cap() {
        assert!(Payload::new(0).is_none());
        assert!(Payload::new(FLIT_BYTES + 1).is_none());
        let cap = RDMA_MESSAGE_CAP_BYTES;
        // ~12.6 MiB — a prefill gather's fragment: three whole caps and a remainder.
        let p = Payload::new(3 * cap + 600 * FLIT_BYTES).unwrap();
        let m: Vec<Message> = p.messages().collect();
        assert_eq!(m.len(), 4);
        assert_eq!(p.message_count(), 4);
        assert_eq!(
            m.iter().map(|x| x.offset).collect::<Vec<_>>(),
            [0, cap, 2 * cap, 3 * cap]
        );
        assert!(m[..3].iter().all(|x| x.bytes == WireBytes::CAP));
        assert_eq!(m[3].bytes.get(), 600 * FLIT_BYTES);
        assert_eq!(
            m.iter().map(|x| (x.first, x.last)).collect::<Vec<_>>(),
            [(true, false), (false, false), (false, false), (false, true)]
        );
        assert_eq!(m.iter().map(|x| x.bytes.get()).sum::<u64>(), p.get());
        // Exactly one cap is one message, both first and last.
        let one: Vec<Message> = Payload::from(WireBytes::CAP).messages().collect();
        assert_eq!(one.len(), 1);
        assert!(one[0].first && one[0].last);
    }

    /// An operand is cut at whole stick columns, each fragment one contiguous run within the width,
    /// tiling the columns exactly; a ragged width and a too-tall column are refusals.
    #[test]
    fn fragments_are_whole_stick_columns_within_the_width() {
        let cap = Payload::from(WireBytes::CAP);
        // A 512-row prefill chunk of a 4096-wide hidden: 4 MiB, one fragment at the width.
        let one = fragments(512, 4096, cap).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].bytes(), cap);
        // A 1024-row chunk: 8 MiB, two fragments of 32 stick columns each.
        let f = fragments(1024, 4096, cap).unwrap();
        assert_eq!(f.len(), 2);
        assert_eq!((f[0].first_col(), f[0].cols()), (0, 2048));
        assert_eq!((f[1].first_col(), f[1].cols()), (2048, 2048));
        assert_eq!(f[1].byte_offset(), 32 * 1024 * STICK_BYTES);
        assert!(f.iter().all(|x| x.bytes() == cap));
        // The baked re-mint reproduces every cut exactly — what the macro's tokens rebuild.
        for x in f.iter().chain(&one) {
            assert_eq!(Fragment::baked(x.first_stick(), x.sticks(), x.rows()), *x);
        }
        // Decode: one row, one fragment.
        let d = fragments(1, 2048, cap).unwrap();
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].bytes().get(), 2048 * 2);
        assert_eq!(
            fragments(1, 100, cap),
            Err(FragmentRefusal::RaggedCols { cols: 100 })
        );
        assert_eq!(
            fragments(40_000, 64, cap),
            Err(FragmentRefusal::StickColumnPastWidth { rows: 40_000 })
        );
    }
}
