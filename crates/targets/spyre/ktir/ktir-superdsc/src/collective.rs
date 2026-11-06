// SPDX-License-Identifier: Apache-2.0
//! THE COLLECTIVE PLAN — tensor parallelism's control-block pipeline, as baked data.
//!
//! The shared analysis (`scratchy_subtile::tensor_parallel::shard`) decides WHERE the forward's
//! collectives are. This module is how a spyre bundle EXECUTES them, entirely in the compute
//! stream: each collective's consume is a launch group of its own, preceded by a p2p send of this
//! rank's operand into EVERY peer's receive window and a wait on the write-done counters EVERY
//! peer's send increments. No host step sits between a producer, its collective and its consumer.
//!
//! ⭐ ONE PROTOCOL FOR EVERY WORLD. Nothing here names a world size: a world of `W` ranks is the
//! same law over `W` senders. Every receive window holds one block per SENDING rank, at the
//! sender's rank — the same block on every card — so rank `s`'s fragment lands at block `s` of
//! every peer's window, and a consume reads block `s` for rank `s`'s share.
//!
//! Everything here is a BAKE-TIME constant. The one fact bound at load is which physical card each
//! rank is, which this plan never names: the window lives at the same place on every card (the
//! allocator's region layout is card-symmetric), so a rank addresses a peer's window by its OWN
//! window's placement and no address is ever exchanged.
//!
//! The protocol is #1007 §3's, and every law there that a value can break is a type here:
//! - a world fits the card's write-done sids or [`TpWorld::of`] does not compile;
//! - a rank is minted by its world ([`TpWorld::ranks`]), so it is in range and carries the world
//!   it belongs to ([`TpIdentity`] is one fact, never a rank and a world passed apart);
//! - a send moves a [`Payload`] (a whole number of flits) as a run of messages, each within the
//!   RDMA message cap ([`WireBytes`], [`Payload::messages`]); an operand is cut at bake into runs
//!   of whole stick columns — the only contiguous slices of a stick-major tensor ([`fragments`]);
//! - the receive window is ONE `(segment, offset, size)` triple ([`RecvWindow`]), flit-aligned, in
//!   the weight segment: [`WINDOW_CUTS`] cuts, each one block per rank, each block tiled by fixed
//!   landing slots ([`LandingSlot`]);
//! - a message lands where its sid's `XSEG` entry points and nowhere else (an F-send's own word
//!   decides nothing — measured, #1009 E8a), so every landing slot has its OWN sid per sending rank
//!   ([`Sid::of`]), each entry programmed once at link: a bounded, static budget per rank, never
//!   one per collective (a forward holds more collectives than the card has sids). A world of `W`
//!   splits the card's sids `W` ways ([`TpWorld::messages_per_cut`]);
//! - a bundle's control blocks are derived from its own launch groups by ONE function (the bundle
//!   crate's `collective::cb_pipeline`, which needs the launch group and so lives above this leaf):
//!   a consume group is `sends, waits, launch`, so a wait can never queue ahead of its own sends
//!   (the symmetric deadlock: every rank waiting, none sending) and every message has one wait.
//!
//! ⭐ THE VALUES LIVE IN THIS LEAF because the emitter marks a consume on its [`EmittedOp`]
//! (`crate::emit::EmittedOp::tp_consume`) before any launch group exists, and the bundle crate
//! sits above this one; the bundle re-exports every name here.
//!
//! [`EmittedOp`]: crate::emit::EmittedOp

use crate::superdsc_opspec::{DataFormat, Fp16};

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

/// The widest world a card's write-done sids can carry: every rank's sends need at least one
/// landing slot per window cut, each its own sid.
pub const MAX_WORLD: u32 = MAX_SIDS / WINDOW_CUTS as u32;

const _: () = assert!(
    MAX_WORLD == crate::reserved_tids::TP_MAX_WORLD,
    "the window's reserved block tids cover exactly the widest world the sids carry"
);

/// A tensor-parallel world a spyre bundle was baked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TpWorld(u32);

impl TpWorld {
    /// The world of `W` ranks. `W = 0`, and a `W` whose ranks the card's sids cannot each give a
    /// landing slot per cut, do not compile:
    ///
    /// ```compile_fail
    /// let _ = ktir_superdsc::collective::TpWorld::of::<0>();
    /// ```
    ///
    /// ```compile_fail
    /// use ktir_superdsc::collective::{MAX_WORLD, TpWorld};
    /// let _ = TpWorld::of::<{ MAX_WORLD + 1 }>();
    /// ```
    ///
    /// ```
    /// use ktir_superdsc::collective::TpWorld;
    /// assert_eq!(TpWorld::of::<8>().get(), 8);
    /// ```
    pub const fn of<const W: u32>() -> Self {
        const {
            assert!(W >= 1, "a tensor-parallel world has at least one rank");
            assert!(
                W <= MAX_WORLD,
                "every rank's landing slots take a sid of their own: a world wider than the \
                 card's sids"
            );
        }
        Self(W)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    /// The world of `ranks` ranks the shared analysis sharded a forward for (its typed
    /// `WorldSize`, read off the collective ops at bake) — `None` past [`MAX_WORLD`], which the
    /// bake refuses by name. The bake's one bridge from the analysis's world to the plan's.
    pub const fn of_analysis_ranks(ranks: u32) -> Option<Self> {
        if ranks >= 1 && ranks <= MAX_WORLD {
            Some(Self(ranks))
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

    /// ⭐ HOW MANY MESSAGES ONE RANK'S SEND INTO ONE CUT MAY BE — the card's sids split evenly
    /// over every (sending rank, cut): each message lands through a sid of its own
    /// ([`Sid::of`]). A wider world leaves each send fewer, never fewer than one.
    pub const fn messages_per_cut(self) -> u32 {
        MAX_SIDS / (self.0 * WINDOW_CUTS as u32)
    }

    /// How many landing slots one rank's sends take: every cut's messages.
    pub const fn slots_per_rank(self) -> u32 {
        WINDOW_CUTS as u32 * self.messages_per_cut()
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

    /// Every OTHER rank of the world — the peers this rank sends to and hears from, in rank order.
    /// Empty in a world of one.
    pub fn peers(self) -> impl Iterator<Item = TpIdentity> {
        self.world.ranks().filter(move |p| p.rank != self.rank)
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

/// How many disjoint cuts the receive window is ([`crate::reserved_tids::TP_WINDOW_CUTS`]).
pub const WINDOW_CUTS: usize = crate::reserved_tids::TP_WINDOW_CUTS as usize;

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

    /// The tid sender `from`'s block of this cut carries — what a consume program binds to read
    /// `from`'s fragment, and what the layout places at [`RecvWindow::block_offset`].
    pub const fn block_tid(self, from: TpIdentity) -> u32 {
        crate::reserved_tids::tp_recv_block_tid(self.0, from.rank)
    }
}

/// ⭐ ONE MESSAGE'S FIXED PLACE IN A SENDER'S BLOCK OF THE RECEIVE WINDOW: message `message` of a
/// send into cut `cut`, at the sender's block of that cut plus `message` whole messages
/// ([`RecvWindow::slot_offset`]). The slot is the SENDER'S budget: which rank sent it is the sid
/// ([`Sid::of`]).
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
    /// The world's [`TpWorld::messages_per_cut`] — the slot's stride between cuts.
    per_cut: u32,
}

impl LandingSlot {
    /// Message `message` of a send into `cut` in `world`; `None` past the world's
    /// [`TpWorld::messages_per_cut`].
    pub const fn of(world: TpWorld, cut: WindowCut, message: u32) -> Option<Self> {
        if message < world.messages_per_cut() {
            Some(Self {
                cut,
                message,
                per_cut: world.messages_per_cut(),
            })
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

    /// The slot's place in a rank's budget, `< world.slots_per_rank()`.
    pub const fn index(self) -> u32 {
        self.cut.0 * self.per_cut + self.message
    }
}

/// WHAT ONE COLLECTIVE'S SEND MOVES: a [`Payload`] of at most the world's
/// [`TpWorld::messages_per_cut`] messages — one block's worth of slots, so every message it is
/// cut into has a slot to land in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CutPayload {
    bytes: Payload,
    world: TpWorld,
}

impl CutPayload {
    /// `Some` iff `bytes` is at most `world`'s messages per cut.
    pub const fn new(world: TpWorld, bytes: Payload) -> Option<Self> {
        if bytes.message_count() <= world.messages_per_cut() {
            Some(Self { bytes, world })
        } else {
            None
        }
    }

    pub const fn get(self) -> Payload {
        self.bytes
    }

    /// Its messages ([`Payload::messages`]), each with the slot of `cut` it lands in: message `m`
    /// in slot `(cut, m)`.
    pub fn landings(self, cut: WindowCut) -> impl Iterator<Item = (LandingSlot, Message)> {
        // `new` bounds the count by the world's messages per cut, so every index is a slot.
        let per_cut = self.world.messages_per_cut();
        (0..).zip(self.bytes.messages()).map(move |(message, m)| {
            (
                LandingSlot {
                    cut,
                    message,
                    per_cut,
                },
                m,
            )
        })
    }
}

/// THE RECEIVE WINDOW — one per card, where every peer's sends land. One typed triple, the same in
/// every bundle of a forward (prefill, decode, every rung), so a send baked into one bundle and the
/// consume baked into another always name the same bytes.
///
/// It lives in the weight segment ([`RecvWindow::SEGMENT`]), the one process-shared,
/// placement-stable segment, above the packed weights: [`WINDOW_CUTS`] equal, adjacent cuts, each
/// ONE BLOCK PER RANK of the world, each block [`RecvWindow::block_bytes_for`] the forward's widest
/// collective operand — never one bundle's own need. Rank `s`'s fragment lands in block `s` on
/// every card (a card's own block goes unused: its share is read from its own operand), through
/// the block's run of fixed landing slots ([`RecvWindow::slots_from`]), one `XSEG` entry per slot
/// per peer, each pointing at its slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RecvWindow {
    offset: u64,
    block: Payload,
    world: TpWorld,
}

impl RecvWindow {
    /// The weight segment (seg1).
    pub const SEGMENT: u32 = 1;

    /// The window of `world` at `offset`, of [`WINDOW_CUTS`] cuts of one `block`-byte block per
    /// rank. `Some` iff `offset` is flit-aligned and a block is at most the world's messages per
    /// cut — a slot each.
    pub const fn new(world: TpWorld, offset: u64, block: Payload) -> Option<Self> {
        if !offset.is_multiple_of(FLIT_BYTES) || block.message_count() > world.messages_per_cut() {
            return None;
        }
        Some(Self {
            offset,
            block,
            world,
        })
    }

    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// The world whose ranks' blocks the window holds.
    pub const fn world(self) -> TpWorld {
        self.world
    }

    /// One sender's block: what one fragment may be.
    pub const fn block(self) -> Payload {
        self.block
    }

    /// One cut's bytes — a block per rank.
    pub const fn cut_bytes(self) -> u64 {
        self.world.0 as u64 * self.block.0
    }

    /// Where `cut` starts in the segment.
    pub const fn cut_offset(self, cut: WindowCut) -> u64 {
        self.offset + cut.0 as u64 * self.cut_bytes()
    }

    /// Where `from`'s block of `cut` starts in the segment — where rank `from`'s send into `cut`
    /// lands, on every card.
    pub const fn block_offset(self, cut: WindowCut, from: TpIdentity) -> u64 {
        self.cut_offset(cut) + from.rank as u64 * self.block.0
    }

    /// Where `slot` of `from`'s block starts in the segment: the block, plus the whole messages
    /// before it.
    pub const fn slot_offset(self, from: TpIdentity, slot: LandingSlot) -> u64 {
        self.block_offset(slot.cut, from) + slot.message as u64 * RDMA_MESSAGE_CAP_BYTES
    }

    /// ⭐ EVERY SLOT `from`'S SENDS CAN LAND IN, with its offset in the segment — each cut's run
    /// of messages in `from`'s block. What a receiver programs for that sender, one `XSEG` entry
    /// per slot, at link.
    pub fn slots_from(self, from: TpIdentity) -> impl Iterator<Item = (LandingSlot, u64)> {
        (0..WINDOW_CUTS).flat_map(move |k| {
            let cut = WindowCut::of_collective(k);
            (0..self.block.message_count()).filter_map(move |m| {
                LandingSlot::of(self.world, cut, m)
                    .map(|slot| (slot, self.slot_offset(from, slot)))
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

    /// ⭐ ONE BLOCK A FORWARD'S COLLECTIVES NEED: [`WINDOW_ROWS`] rows of its widest collective
    /// operand (`widest_cols`, stick-rounded fp16). It reads only the operand's COLUMNS (the
    /// forward's) and the row cap (the target's) — never a bundle's own row count — so every
    /// bundle of one forward places the same window, as the link's `XSEG` entries and the shared
    /// weight segment require. Any fragment of any collective of the forward fits a block whole, so
    /// a collective is ONE fragment and its send is a run of messages ([`Payload::messages`]),
    /// each landing in a slot of its own. `None` for no columns, or for a block wider than
    /// `world`'s messages per cut — more slots than a rank's sid budget holds.
    pub const fn block_bytes_for(world: TpWorld, widest_cols: u32) -> Option<Payload> {
        let cols = widest_cols.next_multiple_of(STICK_ELEMS) as u64;
        match Payload::new(cols * <Fp16 as DataFormat>::WORD_LENGTH as u64 * WINDOW_ROWS as u64) {
            Some(block) if block.message_count() <= world.messages_per_cut() => Some(block),
            _ => None,
        }
    }
}

/// A write-done sid — and the `XSEG` entry its message lands through. ONE PER (SENDING RANK,
/// LANDING SLOT): the receiver's counter for "this slot of that sender's landed" and the entry
/// pointing at that slot of the sender's block, whatever the collective — so the count of
/// collectives in a forward never meets the card's sid budget, and a world takes at most every sid
/// (`TpWorld::messages_per_cut` divides them).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sid(u8);

impl Sid {
    /// The sid `sender`'s message into `slot` signals on.
    pub const fn of(sender: TpIdentity, slot: LandingSlot) -> Self {
        // `rank < world`, `slot.index() < world.slots_per_rank()`, and
        // `world · slots_per_rank ≤ MAX_SIDS ≤ 256`.
        Self((sender.rank * sender.world.slots_per_rank() + slot.index()) as u8)
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

/// Which collective a consume group runs. Mirrors the shared analysis's kinds; the emitter maps
/// them exhaustively.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CollectiveKind {
    /// The consume sums every rank's partial, in rank order.
    AllReduce,
    /// The consume places every rank's block at that rank's columns.
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
    /// The one rank that launches this consume. A consume reads its own share from its own operand
    /// and every peer's from the window, so which operand is whose depends on the rank: the bake
    /// carries one consume per rank, and each session plays its own.
    pub plays_at: PlaysAt,
    /// The window cut the send lands in and the consume reads.
    pub cut: WindowCut,
}

/// The one rank that launches a rank-selected consume. Minted from a [`TpIdentity`], or re-minted
/// from baked tokens through [`PlaysAt::baked`], which refuses at BUILD a rank its world does not
/// have — a consume for a rank past the world would never launch, silently.
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
    /// const P: PlaysAt = PlaysAt::baked(TpWorld::of::<4>(), 4);
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

    /// A rank is in range, carries its world, and its peers are every other rank of it.
    #[test]
    fn a_rank_carries_its_world_and_its_peers() {
        fn check<const W: u32>() {
            let w = TpWorld::of::<W>();
            let ids: Vec<TpIdentity> = w.ranks().collect();
            assert_eq!(ids.len() as u32, W);
            assert_eq!(w.rank(W - 1), ids.last().copied());
            assert_eq!(w.rank(W), None);
            for id in &ids {
                let peers: Vec<u32> = id.peers().map(TpIdentity::rank).collect();
                let want: Vec<u32> = (0..W).filter(|&r| r != id.rank()).collect();
                assert_eq!(peers, want);
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
        check::<8>();
        check::<MAX_WORLD>();
    }

    /// Every (rank, slot) of a world signals on a sid of its own, all within the card's — at
    /// every world width.
    #[test]
    fn every_rank_and_slot_has_its_own_sid() {
        fn check<const W: u32>() {
            let w = TpWorld::of::<W>();
            assert!(w.messages_per_cut() >= 1);
            let slots: Vec<LandingSlot> = (0..WINDOW_CUTS)
                .flat_map(|k| {
                    (0..w.messages_per_cut())
                        .map(move |m| LandingSlot::of(w, WindowCut::of_collective(k), m).unwrap())
                })
                .collect();
            assert_eq!(slots.len() as u32, w.slots_per_rank());
            let sids: std::collections::BTreeSet<u8> = w
                .ranks()
                .flat_map(|r| slots.iter().map(move |&s| Sid::of(r, s).get()))
                .collect();
            assert_eq!(sids.len() as u32, W * w.slots_per_rank());
            assert!(sids.iter().all(|&s| u32::from(s) < MAX_SIDS));
            assert_eq!(
                LandingSlot::of(w, WindowCut::of_collective(0), w.messages_per_cut()),
                None
            );
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
        check::<8>();
        check::<MAX_WORLD>();
    }

    /// A window's blocks sit at their sender's rank in every cut, its slots tile each block
    /// message by message, and a block wider than the world's slot budget is no window at all.
    #[test]
    fn a_windows_blocks_and_slots_tile_its_cuts() {
        let w = TpWorld::of::<4>();
        let block = Payload::new(2 * RDMA_MESSAGE_CAP_BYTES + FLIT_BYTES).unwrap();
        let win = RecvWindow::new(w, 1 << 20, block).unwrap();
        assert_eq!(win.cut_bytes(), 4 * block.get());
        assert_eq!(win.size(), WINDOW_CUTS as u64 * 4 * block.get());
        let r2 = w.rank(2).unwrap();
        let slots: Vec<(u32, u32, u64)> = win
            .slots_from(r2)
            .map(|(s, at)| (s.cut().index() as u32, s.message(), at))
            .collect();
        let at = |c: u64, m: u64| {
            (1 << 20) + c * win.cut_bytes() + 2 * block.get() + m * RDMA_MESSAGE_CAP_BYTES
        };
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
        let widest =
            Payload::new(u64::from(w.messages_per_cut()) * RDMA_MESSAGE_CAP_BYTES).unwrap();
        assert!(RecvWindow::new(w, 0, widest).is_some());
        let past = Payload::new(widest.get() + FLIT_BYTES).unwrap();
        assert!(RecvWindow::new(w, 0, past).is_none());
        assert!(RecvWindow::block_bytes_for(w, 1 << 20).is_none());
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
