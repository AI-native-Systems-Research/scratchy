// SPDX-License-Identifier: Apache-2.0
//! THE COLLECTIVE PLAN — tensor parallelism's control-block pipeline, as baked data: the world,
//! the rank, the p2p payload and its stick-column fragments, the receive window, the sids, the
//! [`TpConsume`](crate::collective::TpConsume) a launch group carries, and the bundle's CB pipeline derived from its groups.

pub use scratchy_tp_world::{Build, PerRank, Rank, TpWorld, World, ranks};

/// The widest single p2p message, in bytes — the vendor's per-message RDMA window
/// (`pf_runtime_scheduler.cpp:50`'s `P2P_RDMA_WINDOW_SIZE`, flex-rs
/// `scheduler::P2P_RDMA_WINDOW_SIZE`, which the target crate pins equal): one F-send moves at most
/// this many bytes, and a longer send is a run of messages. The value is READ
/// from the baked generation's table (the vendor's per-message RDMA window,
/// under its citation there).
pub const RDMA_MESSAGE_CAP_BYTES: u64 =
    <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::RDMA_MESSAGE_CAP_BYTES;

/// One flit, the unit every CB address and length field counts in (flex-rs
/// `control_block_wire::Flits::BYTES`, which the target crate pins equal) —
/// READ from the baked generation's table (senlib's `FLIT_SHIFT` = 7 and the
/// vendor's `STICK_BYTESIZE` = 128, under its citation).
pub const FLIT_BYTES: u64 =
    <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::FLIT_BYTES;

/// How many write-done sids a card has — READ from the baked generation's
/// table (`<Baked as DeviceGen>::MAX_SIDS`, senlib's
/// `BASE_ADDR_TABLE_ENTRIES` = 64, under its citation).
pub const MAX_SIDS: u32 =
    <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::MAX_SIDS;

/// The elements of one fp16 stick.
pub const STICK_ELEMS: u32 = crate::rung::FP16_LANES;

/// The bytes one fp16 STICK holds — one flit. A `[rows, cols]` tensor is laid out stick-major
/// (`(j / stick) · (rows · stick) + i · stick + j % stick`), so ONE STICK COLUMN — every row's
/// elements `[k · stick, (k + 1) · stick)` — is `rows · STICK_BYTES` contiguous bytes, and a run of
/// whole stick columns is the only slice of a tensor that is one contiguous transfer.
pub const STICK_BYTES: u64 = FLIT_BYTES;

const _: () = assert!(
    STICK_ELEMS as u64 * 2 == STICK_BYTES,
    "one fp16 stick is one flit"
);

/// The widest world a card's write-done sids can carry: every rank's sends need at least one
/// landing slot per window cut, each its own sid.
pub const MAX_WORLD: u32 = MAX_SIDS / WINDOW_CUTS as u32;

const _: () = assert!(
    MAX_WORLD == crate::reserved_tids::TP_MAX_WORLD,
    "the window's reserved block tids cover exactly the widest world the sids carry"
);

/// ⭐ A WORLD THIS CARD CAN CARRY — the world type ([`World`], named once by the build's `tp/N`)
/// together with the card's own law on it: every rank's landing slots take sids of their own, so a
/// world no wider than [`MAX_WORLD`]. Every count the protocol derives from the world is an
/// associated const here, so nothing stores a world size or reads one back.
///
/// Implemented for every [`World`]; a world past the card's sids fails the build at the first use
/// of [`Tp::CHECKED`]:
///
/// ```compile_fail
/// use scratchy_spyre_bundle::collective::{Tp, TpWorld};
/// let () = <TpWorld<0> as Tp>::CHECKED;
/// ```
///
/// ```compile_fail
/// use scratchy_spyre_bundle::collective::{MAX_WORLD, Tp, TpWorld};
/// let () = <TpWorld<{ MAX_WORLD + 1 }> as Tp>::CHECKED;
/// ```
///
/// ```
/// use scratchy_spyre_bundle::collective::{Tp, TpWorld};
/// assert_eq!(<TpWorld<8> as Tp>::MESSAGES_PER_CUT, 4);
/// ```
pub trait Tp: World {
    /// The world's laws on this card, proven at build: at least one rank, and no wider than the
    /// card's sids.
    const CHECKED: () = {
        let () = Self::NONEMPTY;
        assert!(
            Self::RANKS <= MAX_WORLD,
            "every rank's landing slots take a sid of their own: a world wider than the card's sids"
        );
    };
    /// ⭐ HOW MANY MESSAGES ONE RANK'S SEND INTO ONE CUT MAY BE — the card's sids split evenly over
    /// every (sending rank, cut): each message lands through a sid of its own ([`Sid::of`]).
    const MESSAGES_PER_CUT: u32 = {
        let () = Self::CHECKED;
        MAX_SIDS / (Self::RANKS * WINDOW_CUTS as u32)
    };
    /// How many landing slots one rank's sends take: every cut's messages.
    const SLOTS_PER_RANK: u32 = WINDOW_CUTS as u32 * Self::MESSAGES_PER_CUT;
}

impl<T: World> Tp for T {}

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
pub const WINDOW_ROWS: u32 = crate::rung::PagedKvPool::PREFILL_CHUNK_SLOTS as u32;

const _: () = assert!(
    crate::rung::PagedKvPool::WIDEST_BATCH_RUNG <= WINDOW_ROWS,
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
    /// use scratchy_spyre_bundle::collective::WindowCut;
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
    pub const fn block_tid<T: Tp>(self, from: Rank<T>) -> u32 {
        crate::reserved_tids::tp_recv_block_tid(self.0, from.get())
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
pub struct LandingSlot<T: Tp> {
    cut: WindowCut,
    message: u32,
    world: core::marker::PhantomData<T>,
}

impl<T: Tp> LandingSlot<T> {
    /// Message `message` of a send into `cut`; `None` past [`Tp::MESSAGES_PER_CUT`].
    pub const fn of(cut: WindowCut, message: u32) -> Option<Self> {
        if message < T::MESSAGES_PER_CUT {
            Some(Self {
                cut,
                message,
                world: core::marker::PhantomData,
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

    /// The slot's place in a rank's budget, `< T::SLOTS_PER_RANK`.
    pub const fn index(self) -> u32 {
        self.cut.0 * T::MESSAGES_PER_CUT + self.message
    }
}

/// WHAT ONE COLLECTIVE'S SEND MOVES: a [`Payload`] of at most [`Tp::MESSAGES_PER_CUT`] messages
/// — one block's worth of slots, so every message it is cut into has a slot to land in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CutPayload<T: Tp>(Payload, core::marker::PhantomData<T>);

impl<T: Tp> CutPayload<T> {
    /// `Some` iff `bytes` is at most the world's messages per cut.
    pub const fn new(bytes: Payload) -> Option<Self> {
        if bytes.message_count() <= T::MESSAGES_PER_CUT {
            Some(Self(bytes, core::marker::PhantomData))
        } else {
            None
        }
    }

    pub const fn get(self) -> Payload {
        self.0
    }

    /// Its messages ([`Payload::messages`]), each with the slot of `cut` it lands in: message `m`
    /// in slot `(cut, m)`.
    pub fn landings(self, cut: WindowCut) -> impl Iterator<Item = (LandingSlot<T>, Message)> {
        // `new` bounds the count by the world's messages per cut, so every index is a slot.
        (0..).zip(self.0.messages()).map(move |(message, m)| {
            (
                LandingSlot {
                    cut,
                    message,
                    world: core::marker::PhantomData,
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
/// It lives in the weight segment ([`WINDOW_SEGMENT`]), the one process-shared,
/// placement-stable segment, above the packed weights: [`WINDOW_CUTS`] equal, adjacent cuts, each
/// ONE BLOCK PER RANK of the world, each block [`RecvWindow::block_bytes_for`] the forward's widest
/// collective operand — never one bundle's own need. Rank `s`'s fragment lands in block `s` on
/// every card (a card's own block goes unused: its share is read from its own operand), through
/// the block's run of fixed landing slots ([`RecvWindow::slots_from`]), one `XSEG` entry per slot
/// per peer, each pointing at its slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RecvWindow<T: Tp> {
    offset: u64,
    block: Payload,
    world: core::marker::PhantomData<T>,
}

/// The segment every world's receive window lives in: the weight segment (seg1).
pub const WINDOW_SEGMENT: u32 = 1;

impl<T: Tp> RecvWindow<T> {
    /// The window at `offset`, of [`WINDOW_CUTS`] cuts of one `block`-byte block per rank. `Some`
    /// iff `offset` is flit-aligned and a block is at most the world's messages per cut — a slot
    /// each.
    pub const fn new(offset: u64, block: Payload) -> Option<Self> {
        if !offset.is_multiple_of(FLIT_BYTES) || block.message_count() > T::MESSAGES_PER_CUT {
            return None;
        }
        Some(Self {
            offset,
            block,
            world: core::marker::PhantomData,
        })
    }

    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// One sender's block: what one fragment may be.
    pub const fn block(self) -> Payload {
        self.block
    }

    /// One cut's bytes — a block per rank.
    pub const fn cut_bytes(self) -> u64 {
        T::RANKS as u64 * self.block.0
    }

    /// Where `cut` starts in the segment.
    pub const fn cut_offset(self, cut: WindowCut) -> u64 {
        self.offset + cut.0 as u64 * self.cut_bytes()
    }

    /// Where `from`'s block of `cut` starts in the segment — where rank `from`'s send into `cut`
    /// lands, on every card.
    pub const fn block_offset(self, cut: WindowCut, from: Rank<T>) -> u64 {
        self.cut_offset(cut) + from.get() as u64 * self.block.0
    }

    /// Where `slot` of `from`'s block starts in the segment: the block, plus the whole messages
    /// before it.
    pub const fn slot_offset(self, from: Rank<T>, slot: LandingSlot<T>) -> u64 {
        self.block_offset(slot.cut, from) + slot.message as u64 * RDMA_MESSAGE_CAP_BYTES
    }

    /// ⭐ EVERY SLOT `from`'S SENDS CAN LAND IN, with its offset in the segment — each cut's run
    /// of messages in `from`'s block. What a receiver programs for that sender, one `XSEG` entry
    /// per slot, at link.
    pub fn slots_from(self, from: Rank<T>) -> impl Iterator<Item = (LandingSlot<T>, u64)> {
        (0..WINDOW_CUTS).flat_map(move |k| {
            let cut = WindowCut::of_collective(k);
            (0..self.block.message_count()).filter_map(move |m| {
                LandingSlot::of(cut, m).map(|slot| (slot, self.slot_offset(from, slot)))
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
    /// each landing in a slot of its own. `None` for no columns, or for a block wider than the
    /// world's messages per cut — more slots than a rank's sid budget holds.
    pub const fn block_bytes_for(widest_cols: u32) -> Option<Payload> {
        let cols = widest_cols.next_multiple_of(STICK_ELEMS) as u64;
        match Payload::new(cols * 2 * WINDOW_ROWS as u64) {
            Some(block) if block.message_count() <= T::MESSAGES_PER_CUT => Some(block),
            _ => None,
        }
    }
}

/// A write-done sid — and the `XSEG` entry its message lands through. ONE PER (SENDING RANK,
/// LANDING SLOT): the receiver's counter for "this slot of that sender's landed" and the entry
/// pointing at that slot of the sender's block, whatever the collective — so the count of
/// collectives in a forward never meets the card's sid budget, and a world takes at most every sid
/// ([`Tp::MESSAGES_PER_CUT`] divides them).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sid(u8);

impl Sid {
    /// The sid `sender`'s message into `slot` signals on.
    pub const fn of<T: Tp>(sender: Rank<T>, slot: LandingSlot<T>) -> Self {
        // `rank < RANKS`, `slot.index() < SLOTS_PER_RANK`, and `RANKS · SLOTS_PER_RANK ≤ MAX_SIDS`.
        Self((sender.get() * T::SLOTS_PER_RANK + slot.index()) as u8)
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
    /// use scratchy_spyre_bundle::collective::Fragment;
    /// // A fragment of no rows moves nothing.
    /// const F: Fragment = Fragment::baked(0, 1, 0);
    /// ```
    ///
    /// A fragment wider than one message is a payload like any other — its send is several:
    ///
    /// ```
    /// use scratchy_spyre_bundle::collective::Fragment;
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
pub struct TpConsume<T: Tp> {
    pub kind: CollectiveKind,
    /// The tensor this rank SENDS — the collective's operand, read from its own placement.
    pub operand: crate::place::PlaceId,
    /// Which run of the operand's stick columns this rank sends, and this consume covers.
    pub fragment: Fragment,
    /// The one rank that launches this consume. A consume reads its own share from its own operand
    /// and every peer's from the window, so which operand is whose depends on the rank: the bake
    /// carries one consume per rank, and each session plays its own.
    pub plays_at: Rank<T>,
    /// The window cut the sends land in and the consume reads.
    pub cut: WindowCut,
}

impl<T: Tp> TpConsume<T> {
    /// Whether the session running as `session` launches this consume.
    pub fn launches_on(self, session: Rank<T>) -> bool {
        self.plays_at == session
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A world's ranks are its own, in order, and each rank's peers are every other rank.
    #[test]
    fn a_ranks_peers_are_every_other_rank() {
        fn check<T: Tp>() {
            let ids: Vec<Rank<T>> = ranks::<T>().collect();
            assert_eq!(ids.len() as u32, T::RANKS);
            assert_eq!(ids.last().map(|r| r.get()), Some(T::RANKS - 1));
            for id in &ids {
                let peers: Vec<u32> = id.peers().map(Rank::get).collect();
                let want: Vec<u32> = (0..T::RANKS).filter(|&r| r != id.get()).collect();
                assert_eq!(peers, want);
            }
        }
        check::<TpWorld<1>>();
        check::<TpWorld<2>>();
        check::<TpWorld<3>>();
        check::<TpWorld<4>>();
        check::<TpWorld<8>>();
        check::<TpWorld<MAX_WORLD>>();
    }

    /// Every (rank, slot) of a world signals on a sid of its own, all within the card's — at every
    /// world width.
    #[test]
    fn every_rank_and_slot_has_its_own_sid() {
        fn check<T: Tp>() {
            assert!(T::MESSAGES_PER_CUT >= 1);
            let slots: Vec<LandingSlot<T>> = (0..WINDOW_CUTS)
                .flat_map(|k| {
                    (0..T::MESSAGES_PER_CUT)
                        .map(move |m| LandingSlot::of(WindowCut::of_collective(k), m).unwrap())
                })
                .collect();
            assert_eq!(slots.len() as u32, T::SLOTS_PER_RANK);
            let sids: std::collections::BTreeSet<u8> = ranks::<T>()
                .flat_map(|r| slots.iter().map(move |&s| Sid::of(r, s).get()))
                .collect();
            assert_eq!(sids.len() as u32, T::RANKS * T::SLOTS_PER_RANK);
            assert!(sids.iter().all(|&s| u32::from(s) < MAX_SIDS));
            assert_eq!(
                LandingSlot::<T>::of(WindowCut::of_collective(0), T::MESSAGES_PER_CUT),
                None
            );
        }
        check::<TpWorld<1>>();
        check::<TpWorld<2>>();
        check::<TpWorld<3>>();
        check::<TpWorld<4>>();
        check::<TpWorld<8>>();
        check::<TpWorld<MAX_WORLD>>();
    }

    /// A window's blocks sit at their sender's rank in every cut, its slots tile each block message
    /// by message, and a block wider than the world's slot budget is no window at all.
    #[test]
    fn a_windows_blocks_and_slots_tile_its_cuts() {
        let block = Payload::new(2 * RDMA_MESSAGE_CAP_BYTES + FLIT_BYTES).unwrap();
        let win = RecvWindow::<TpWorld<4>>::new(1 << 20, block).unwrap();
        assert_eq!(win.cut_bytes(), 4 * block.get());
        assert_eq!(win.size(), WINDOW_CUTS as u64 * 4 * block.get());
        let slots: Vec<(u32, u32, u64)> = win
            .slots_from(Rank::<TpWorld<4>>::of::<2>())
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
            Payload::new(u64::from(<TpWorld<4> as Tp>::MESSAGES_PER_CUT) * RDMA_MESSAGE_CAP_BYTES)
                .unwrap();
        assert!(RecvWindow::<TpWorld<4>>::new(0, widest).is_some());
        let past = Payload::new(widest.get() + FLIT_BYTES).unwrap();
        assert!(RecvWindow::<TpWorld<4>>::new(0, past).is_none());
        assert!(RecvWindow::<TpWorld<4>>::block_bytes_for(1 << 20).is_none());
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

/// One control block of a bundle's pipeline, in the order the compute FIFO runs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CbStep<T: Tp> {
    /// The F-send in front of consume group `group` to peer `to`: this rank's fragment of the
    /// consume's operand, from its own placement into this rank's block of `to`'s receive window,
    /// on this card's DMA engine — device p2p RDMA (flex-rs `scheduler::build_rdma_send_cb`), one
    /// F-send per message. Each message's landing increments `to`'s write-done counter on this
    /// rank's sid for its landing slot.
    Send {
        group: usize,
        consume: TpConsume<T>,
        to: Rank<T>,
    },
    /// Wait for peer `from`'s fragment to land in `from`'s block of this rank's window (consumes
    /// one write-done per message on `from`'s sid for its slot; flex-rs
    /// `ControlBlockWire::p2p_rdma_wait`).
    Wait {
        group: usize,
        consume: TpConsume<T>,
        from: Rank<T>,
    },
    /// Launch group `group`'s programs.
    Launch { group: usize },
}

/// Whether the session running as `session` plays a group carrying `tp` at all.
///
/// ⭐ A CONSUME PLAYS ON ITS RANK ALONE. A consume reads its own share from its own operand and
/// every peer's from the window, so the bake carries one consume per rank
/// ([`TpConsume::plays_at`]); every other group plays everywhere. The one law the CB pipeline and a
/// loader's launch list both read, so the steps a bundle names and the launches a session binds
/// cannot disagree.
pub fn plays_on<T: Tp>(tp: Option<TpConsume<T>>, session: Rank<T>) -> bool {
    tp.is_none_or(|c| c.launches_on(session))
}

/// One group's control blocks, in order: a consume group is `send to every peer, wait for every
/// peer, launch`, every other group one launch. A collective's sends are never a group of their
/// own: the consume names everything they move, so every consume has its sends in front of it and
/// a wait never queues ahead of its own rank's sends (the symmetric deadlock: every rank waiting,
/// none sending).
pub fn group_steps<T: Tp>(
    group: usize,
    tp: Option<TpConsume<T>>,
) -> impl Iterator<Item = CbStep<T>> {
    let steps: Vec<CbStep<T>> = match tp {
        Some(consume) => {
            let me = consume.plays_at;
            me.peers()
                .map(|to| CbStep::Send { group, consume, to })
                .chain(me.peers().map(|from| CbStep::Wait {
                    group,
                    consume,
                    from,
                }))
                .chain(std::iter::once(CbStep::Launch { group }))
                .collect()
        }
        None => vec![CbStep::Launch { group }],
    };
    steps.into_iter()
}

/// THE CB PIPELINE OF A BUNDLE — the groups `session` plays ([`plays_on`]), each as its
/// [`group_steps`], in launch order. Each rank sends once to and waits once on every peer per
/// collective, and the write-done counter returns to where it started after every replay of the
/// bundle.
pub fn cb_pipeline<'g, T: Tp>(
    groups: &'g [crate::LaunchGroup<'_, T>],
    session: Rank<T>,
) -> impl Iterator<Item = CbStep<T>> + 'g {
    groups
        .iter()
        .enumerate()
        .filter(move |(_, g)| plays_on(g.kv.tp, session))
        .flat_map(|(group, g)| group_steps(group, g.kv.tp))
}

/// WHERE A CONSUME'S SENDS READ AND LAND — segment-relative, as the bundle baked it. The source is
/// the consume's fragment inside the operand's placement; the landing is the SENDING rank's block of
/// the window cut, on every peer (the layout is card-symmetric, so every peer's window is at the
/// same place, and the sender's block is the same block on every one). Only the segment BASES are
/// bound at load: the launch the sends are aimed from supplies them, exactly as it supplies them to
/// the consume's own program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpSend<T: Tp> {
    /// The program segment the operand lives in.
    pub source_segment: u8,
    /// The fragment's first byte inside that segment.
    pub source_offset: u64,
    /// The sending rank's block of the window cut, inside [`WINDOW_SEGMENT`].
    pub window_offset: u64,
    pub bytes: CutPayload<T>,
    /// The cut it lands in — message `m` lands in slot `(cut, m)`, signalling that slot's sid.
    pub cut: WindowCut,
}

/// Why a consume's send names no baked bytes. The bake refuses every one of these (it resolves
/// every consume of a bundle through [`TpSend::resolve`] before baking it), so a loaded bundle meets
/// none of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SendUnplaced {
    /// The operand has no placement in the bundle.
    Operand(crate::PlaceId),
    /// The bundle carries consumes but no receive window.
    Window,
    /// A window cut is not bank 0 of the weight segment — where every consume reads it.
    WindowMisplaced { segment: u32, bank: u32 },
    /// The window's cuts are not equal, adjacent and in order — not one window.
    WindowCutsApart { cut: usize, offset: u64, size: u64 },
    /// The fragment is wider than one block of the window.
    WindowTooSmall { window: u64, bytes: u64 },
    /// The operand lives in the weight segment, which a consume launch pins to layer 0's bank-0
    /// copy — a layer-shifted operand would be read at the wrong layer.
    OperandInWeightSegment(crate::PlaceId),
    /// The fragment runs past the operand's placement.
    FragmentPastOperand {
        operand: crate::PlaceId,
        end: u64,
        size: u64,
    },
    /// An address that is not a whole number of flits.
    Misaligned { offset: u64 },
}

/// ⭐ THE WINDOW `layout` BAKES FOR WORLD `T` — its blocks' placements read back as the ONE window
/// they tile (every cut, every sender's block of it, adjacent and equal, in order), or `None` for a
/// bundle with no collective. The bake's send gate and the loader's link read it here, so they
/// cannot disagree about where a peer's sends land.
///
/// # Errors
/// A block missing or misplaced, blocks that are not one window, or an address that is not whole
/// flits — by name.
pub fn baked_window<T: Tp>(
    layout: &crate::BundleLayout<'_>,
) -> Result<Option<RecvWindow<T>>, SendUnplaced> {
    let tid = |cut: usize, from: Rank<T>| WindowCut::of_collective(cut).block_tid(from);
    if !T::SHARDED {
        return Ok(None);
    }
    let Some(first) = layout.place(crate::PlaceId::Act(tid(0, Rank::of::<0>()))) else {
        return Ok(None);
    };
    for (i, (cut, from)) in (0..WINDOW_CUTS)
        .flat_map(|c| ranks::<T>().map(move |r| (c, r)))
        .enumerate()
    {
        let block = layout
            .place(crate::PlaceId::Act(tid(cut, from)))
            .ok_or(SendUnplaced::Window)?;
        if block.segment != WINDOW_SEGMENT || block.bank != 0 {
            return Err(SendUnplaced::WindowMisplaced {
                segment: block.segment,
                bank: block.bank,
            });
        }
        if block.size != first.size || block.offset != first.offset + i as u64 * first.size {
            return Err(SendUnplaced::WindowCutsApart {
                cut,
                offset: block.offset,
                size: block.size,
            });
        }
    }
    Payload::new(first.size)
        .and_then(|block| RecvWindow::new(first.offset, block))
        .map(Some)
        .ok_or(SendUnplaced::Misaligned {
            offset: first.offset,
        })
}

impl<T: Tp> TpSend<T> {
    /// Read `consume`'s send — landing in its window cut — off `layout`'s placements.
    ///
    /// # Errors
    /// [`SendUnplaced`], by name.
    pub fn resolve(
        layout: &crate::BundleLayout<'_>,
        consume: TpConsume<T>,
    ) -> Result<Self, SendUnplaced> {
        let me = consume.plays_at;
        let window = baked_window::<T>(layout)?.ok_or(SendUnplaced::Window)?;
        let bytes = consume.fragment.bytes();
        if bytes.get() > window.block().get() {
            return Err(SendUnplaced::WindowTooSmall {
                window: window.block().get(),
                bytes: bytes.get(),
            });
        }
        let landing = window.block_offset(consume.cut, me);
        let operand = layout
            .place(consume.operand)
            .ok_or(SendUnplaced::Operand(consume.operand))?;
        if operand.segment == WINDOW_SEGMENT {
            return Err(SendUnplaced::OperandInWeightSegment(consume.operand));
        }
        let end = consume.fragment.byte_offset() + bytes.get();
        if end > operand.size {
            return Err(SendUnplaced::FragmentPastOperand {
                operand: consume.operand,
                end,
                size: operand.size,
            });
        }
        let source_offset = operand.offset + consume.fragment.byte_offset();
        for offset in [source_offset, landing] {
            if !offset.is_multiple_of(FLIT_BYTES) {
                return Err(SendUnplaced::Misaligned { offset });
            }
        }
        Ok(Self {
            // `Placement::segment` is below `NUM_SEGMENTS`.
            source_segment: operand.segment as u8,
            source_offset,
            window_offset: landing,
            // A fragment within one block is within the block's slots (`RecvWindow::new`).
            bytes: CutPayload::new(bytes).ok_or(SendUnplaced::WindowTooSmall {
                window: window.block().get(),
                bytes: bytes.get(),
            })?,
            cut: consume.cut,
        })
    }
}

/// ONE F-SEND of a [`TpSend`]: one message of its payload, at the matching offset on both ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpMessage<T: Tp> {
    pub source_offset: u64,
    pub window_offset: u64,
    pub bytes: WireBytes,
    /// The send section's own protocol bits: this message's place in the run.
    pub first: bool,
    pub last: bool,
    /// Where it lands — and so the sid it signals on and every peer waits on.
    pub slot: LandingSlot<T>,
}

impl<T: Tp> TpSend<T> {
    /// ⭐ THE CUT: the send as the vendor's windowed P2P loop issues it — one F-send per message
    /// ([`Payload::messages`]), each advancing the source and the landing together, and each in a
    /// landing slot of its own ([`CutPayload::landings`]). Every one increments its slot's
    /// write-done counter on the peer it lands on once, so a send of N messages is waited for N
    /// times on that peer, once per slot (a wait consumes exactly one count).
    pub fn messages(self) -> impl Iterator<Item = TpMessage<T>> {
        self.bytes
            .landings(self.cut)
            .map(move |(slot, m)| TpMessage {
                source_offset: self.source_offset + m.offset,
                window_offset: self.window_offset + m.offset,
                bytes: m.bytes,
                first: m.first,
                last: m.last,
                slot,
            })
    }
}

/// One consume group's send, read off its bundle's placements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupSend<T: Tp> {
    pub group: usize,
    pub send: TpSend<T>,
}

/// A consume group whose send names no baked bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnplacedGroup {
    pub group: usize,
    pub why: SendUnplaced,
}

/// Every consume group of `groups`, with its send read off `layout` — the bake's gate (a bundle
/// whose send names no baked bytes does not bake) and the loader's read, one function.
///
/// # Errors
/// The first consume whose send is unplaced, with its group.
pub fn sends<T: Tp>(
    layout: &crate::BundleLayout<'_>,
    groups: &[crate::LaunchGroup<'_, T>],
) -> Result<Vec<GroupSend<T>>, UnplacedGroup> {
    groups
        .iter()
        .enumerate()
        .filter_map(|(group, g)| g.kv.tp.map(|c| (group, c)))
        .map(|(group, c)| {
            TpSend::resolve(layout, c)
                .map(|send| GroupSend { group, send })
                .map_err(|why| UnplacedGroup { group, why })
        })
        .collect()
}

/// The write-done ledger over one rank's pipeline: +1 per send landing on a peer, −1 per wait.
pub fn wdone_balance<T: Tp>(steps: impl Iterator<Item = CbStep<T>>) -> i64 {
    steps
        .map(|s| match s {
            CbStep::Send { .. } => 1,
            CbStep::Wait { .. } => -1,
            CbStep::Launch { .. } => 0,
        })
        .sum()
}

#[cfg(test)]
mod pipeline_tests {
    use super::*;

    fn frag() -> Fragment {
        fragments(1, 2048, Payload::from(WireBytes::CAP)).unwrap()[0]
    }

    fn consume_group<T: Tp>(consume: TpConsume<T>) -> crate::LaunchGroup<'static, T> {
        crate::LaunchGroup {
            kv: crate::KvShifts {
                tp: Some(consume),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn consume_at<T: Tp>(kind: CollectiveKind, me: Rank<T>, cut: usize) -> TpConsume<T> {
        TpConsume {
            kind,
            operand: crate::PlaceId::Act(7),
            fragment: frag(),
            plays_at: me,
            cut: WindowCut::of_collective(cut),
        }
    }

    /// The pipeline law, at every world width: a rank plays its own consume group — a send to
    /// every peer, then a wait on every peer, then the launch — and skips every other rank's;
    /// ordinary groups launch; order is group order; the write-done ledger closes.
    #[test]
    fn a_collective_is_sends_then_waits_then_consume_in_group_order() {
        fn check<T: Tp>() {
            let ranks: Vec<Rank<T>> = ranks::<T>().collect();
            let mut groups = vec![crate::LaunchGroup::default()];
            groups.extend(
                ranks
                    .iter()
                    .map(|&r| consume_group(consume_at(CollectiveKind::AllReduce, r, 0))),
            );
            groups.push(crate::LaunchGroup::default());
            let last = groups.len() - 1;
            for (i, &me) in ranks.iter().enumerate() {
                let mine = 1 + i;
                let consume = consume_at(CollectiveKind::AllReduce, me, 0);
                let mut want = vec![CbStep::Launch { group: 0 }];
                want.extend(me.peers().map(|to| CbStep::Send {
                    group: mine,
                    consume,
                    to,
                }));
                want.extend(me.peers().map(|from| CbStep::Wait {
                    group: mine,
                    consume,
                    from,
                }));
                want.push(CbStep::Launch { group: mine });
                want.push(CbStep::Launch { group: last });
                let steps: Vec<CbStep<T>> = cb_pipeline(&groups, me).collect();
                assert_eq!(steps, want, "rank {} of {}", me.get(), T::RANKS);
                assert_eq!(wdone_balance(cb_pipeline(&groups, me)), 0);
            }
        }
        check::<TpWorld<1>>();
        check::<TpWorld<2>>();
        check::<TpWorld<3>>();
        check::<TpWorld<4>>();
        check::<TpWorld<8>>();
        check::<TpWorld<MAX_WORLD>>();
    }

    /// A consume names the window absolutely — shift 0, bank 0 — whatever layer or bank the launch
    /// is for.
    #[test]
    fn a_consume_reads_the_window_unshifted_and_unbanked() {
        let plain = crate::KvShifts::<TpWorld<4>>::default();
        let me = Rank::<TpWorld<4>>::of::<1>();
        let consuming = consume_group(consume_at(CollectiveKind::AllReduce, me, 0)).kv;
        assert_eq!(plain.weight_slot(4096, 2), (4096, 2));
        assert_eq!(consuming.weight_slot(4096, 2), (0, 0));
    }

    fn placed(id: crate::PlaceId, segment: u32, offset: u64, size: u64) -> crate::Placement {
        crate::Placement {
            id,
            segment,
            bank: 0,
            offset,
            size,
            is_logits: false,
        }
    }

    /// The window as the bake places it for world `T`: `WINDOW_CUTS` adjacent cuts at 1 MiB, each
    /// one `block`-byte placement per sending rank.
    fn window_of<T: Tp>(block: u64) -> Vec<crate::Placement> {
        (0..WINDOW_CUTS)
            .flat_map(|c| ranks::<T>().map(move |r| (c, r)))
            .enumerate()
            .map(|(i, (c, r))| {
                placed(
                    crate::PlaceId::Act(WindowCut::of_collective(c).block_tid(r)),
                    WINDOW_SEGMENT,
                    (1 << 20) + i as u64 * block,
                    block,
                )
            })
            .collect()
    }

    fn with(mut window: Vec<crate::Placement>, more: crate::Placement) -> Vec<crate::Placement> {
        window.push(more);
        window
    }

    fn layout(places: Vec<crate::Placement>) -> crate::BundleLayout<'static> {
        crate::BundleLayout {
            places: places.into(),
            ..Default::default()
        }
    }

    /// The send reads the consume's fragment at the operand's own placement — its segment, its
    /// offset plus the fragment's — and lands at the SENDER'S block of its cut, at every rank of
    /// every world.
    #[test]
    fn a_send_lands_in_its_senders_block_of_its_cut() {
        fn check<T: Tp>() {
            let fragment = fragments(4, 256, Payload::new(1024).unwrap()).unwrap()[1];
            let block = WireBytes::CAP.get();
            let operand = placed(crate::PlaceId::Act(7), 0, 8192, 4 * 256 * 2);
            let places = layout(with(window_of::<T>(block), operand));
            for me in ranks::<T>() {
                for k in 0..WINDOW_CUTS {
                    let cut = WindowCut::of_collective(k);
                    let consume = TpConsume {
                        fragment,
                        ..consume_at(CollectiveKind::AllReduce, me, k)
                    };
                    assert_eq!(
                        TpSend::resolve(&places, consume).unwrap(),
                        TpSend {
                            source_segment: 0,
                            source_offset: 8192 + fragment.byte_offset(),
                            window_offset: (1 << 20)
                                + k as u64 * u64::from(T::RANKS) * block
                                + u64::from(me.get()) * block,
                            bytes: CutPayload::new(fragment.bytes()).unwrap(),
                            cut,
                        }
                    );
                }
            }
        }
        check::<TpWorld<2>>();
        check::<TpWorld<3>>();
        check::<TpWorld<4>>();
        check::<TpWorld<8>>();
        check::<TpWorld<MAX_WORLD>>();
    }

    /// The window reads back off the bake as the one window its cuts tile — a block per rank of
    /// its world — and a bundle with no cut placed has none.
    #[test]
    fn the_baked_window_is_its_cuts_read_back() {
        type W4 = TpWorld<4>;
        let window = baked_window::<W4>(&layout(window_of::<W4>(WireBytes::CAP.get())))
            .unwrap()
            .unwrap();
        assert_eq!(window.offset(), 1 << 20);
        assert_eq!(window.block(), Payload::from(WireBytes::CAP));
        assert_eq!(window.size(), WINDOW_CUTS as u64 * 4 * WireBytes::CAP.get());
        assert_eq!(baked_window::<W4>(&layout(vec![])), Ok(None));
        assert_eq!(
            baked_window::<TpWorld<1>>(&layout(window_of::<W4>(WireBytes::CAP.get()))),
            Ok(None),
            "a world of one has no window"
        );
    }

    /// Consecutive collectives alternate cuts, and each one's send lands in its own; the forward's
    /// third collective is back in the first's cut.
    #[test]
    fn consecutive_collectives_send_into_alternate_cuts() {
        type W4 = TpWorld<4>;
        let me = Rank::<W4>::of::<0>();
        let block = WireBytes::CAP.get();
        let operand = placed(crate::PlaceId::Act(7), 0, 0, frag().bytes().get());
        let places = layout(with(window_of::<W4>(block), operand));
        let groups = [
            consume_group(consume_at(CollectiveKind::AllReduce, me, 0)),
            crate::LaunchGroup::<W4>::default(),
            consume_group(consume_at(CollectiveKind::AllReduce, me, 1)),
            consume_group(consume_at(CollectiveKind::AllReduce, me, 2)),
        ];
        let landed: Vec<(usize, u64)> = sends(&places, &groups)
            .unwrap()
            .into_iter()
            .map(|s| (s.group, s.send.window_offset))
            .collect();
        let cut = 4 * block;
        assert_eq!(landed, [(0, 1 << 20), (2, (1 << 20) + cut), (3, 1 << 20)]);
        assert_eq!(WindowCut::baked(1), WindowCut::of_collective(1));
        assert_eq!(
            WindowCut::of_collective(1).block_tid(me),
            crate::reserved_tids::tp_recv_block_tid(1, 0)
        );
    }

    /// A send wider than one message is the vendor loop's run: each message reads and lands at
    /// the same offset into the payload, in a slot of its own, the run tiles it, and only its ends
    /// carry first / last; a payload past the world's slots per cut is no send.
    #[test]
    fn a_wide_send_is_a_run_of_messages_at_matching_offsets() {
        type W2 = TpWorld<2>;
        let cap = WireBytes::CAP.get();
        let cut = WindowCut::of_collective(1);
        let send = TpSend::<W2> {
            source_segment: 0,
            source_offset: 8192,
            window_offset: 1 << 30,
            bytes: CutPayload::new(Payload::new(2 * cap + 256).unwrap()).unwrap(),
            cut,
        };
        let m: Vec<TpMessage<W2>> = send.messages().collect();
        assert_eq!(m.len(), 3);
        for (k, x) in m.iter().enumerate() {
            let at = k as u64 * cap;
            assert_eq!(
                (x.source_offset, x.window_offset),
                (8192 + at, (1 << 30) + at)
            );
            assert_eq!((x.first, x.last), (k == 0, k == 2));
            assert_eq!(x.slot, LandingSlot::of(cut, k as u32).unwrap());
        }
        assert_eq!(m[2].bytes.get(), 256);
        assert_eq!(
            m.iter().map(|x| x.bytes.get()).sum::<u64>(),
            send.bytes.get().get()
        );
        fn past<T: Tp>() {
            let cap = WireBytes::CAP.get();
            let past = Payload::new(u64::from(T::MESSAGES_PER_CUT) * cap + FLIT_BYTES).unwrap();
            assert_eq!(
                CutPayload::<T>::new(past),
                None,
                "more messages than a block has slots"
            );
        }
        past::<TpWorld<2>>();
        past::<TpWorld<4>>();
        past::<TpWorld<8>>();
        past::<TpWorld<MAX_WORLD>>();
    }

    /// Every way a send can name no baked bytes refuses by name.
    #[test]
    fn an_unplaced_send_refuses_by_name() {
        type W2 = TpWorld<2>;
        let consume = consume_at(CollectiveKind::AllReduce, Rank::<W2>::of::<0>(), 0);
        let whole = frag().bytes().get();
        let window = || window_of::<W2>(WireBytes::CAP.get());
        let operand = || placed(crate::PlaceId::Act(7), 0, 0, whole);
        let refusal = |places| TpSend::resolve(&layout(places), consume).unwrap_err();
        assert_eq!(refusal(vec![operand()]), SendUnplaced::Window);
        assert_eq!(
            refusal(window()),
            SendUnplaced::Operand(crate::PlaceId::Act(7))
        );
        let mut one_cut = window();
        one_cut.pop();
        assert_eq!(refusal(with(one_cut, operand())), SendUnplaced::Window);
        let mut banked = window();
        banked[1].bank = 1;
        assert_eq!(
            refusal(with(banked, operand())),
            SendUnplaced::WindowMisplaced {
                segment: WINDOW_SEGMENT,
                bank: 1
            }
        );
        // The first block of cut 1 — sender 0's — moved off the window.
        let cut1 = <W2 as World>::RANKS as usize;
        let mut apart = window();
        apart[cut1].offset += FLIT_BYTES;
        assert_eq!(
            refusal(with(apart.clone(), operand())),
            SendUnplaced::WindowCutsApart {
                cut: 1,
                offset: apart[cut1].offset,
                size: apart[cut1].size
            }
        );
        assert_eq!(
            refusal(with(window_of::<W2>(whole - FLIT_BYTES), operand())),
            SendUnplaced::WindowTooSmall {
                window: whole - FLIT_BYTES,
                bytes: whole
            }
        );
        assert_eq!(
            refusal(with(
                window(),
                placed(crate::PlaceId::Act(7), WINDOW_SEGMENT, 0, whole)
            )),
            SendUnplaced::OperandInWeightSegment(crate::PlaceId::Act(7))
        );
        assert_eq!(
            refusal(with(
                window(),
                placed(crate::PlaceId::Act(7), 0, 0, whole - FLIT_BYTES)
            )),
            SendUnplaced::FragmentPastOperand {
                operand: crate::PlaceId::Act(7),
                end: whole,
                size: whole - FLIT_BYTES
            }
        );
        assert_eq!(
            refusal(with(window(), placed(crate::PlaceId::Act(7), 0, 64, whole))),
            SendUnplaced::Misaligned { offset: 64 }
        );
    }
}
