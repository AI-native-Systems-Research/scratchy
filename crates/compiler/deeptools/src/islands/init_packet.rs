//! ISLAND 4 — the tape of PACKETS, which serialise to `init_binary.bin`.
//!
//! `AtThisArch<Packet>`. Bridge 3 laid island 3's ops out into packets, and this is where the tape stops being a
//! sequence of instructions: a packet is ADDRESSED, so "the op after this one" is no longer expressible. Anything
//! depending on op order — every sync — must therefore be resolved before bridge 3, not after.
//!
//! # The geometry, ported from `Dip::runDip` (`deeptools/dip/dip.cpp:345-388`)
//!
//! A file is a flat sequence of **128-byte flits**, each **8 × 16-byte slices** (`SixteenByteStream = [u32; 4]`),
//! serialised as little-endian `u32`. So the byte at `(flit f, slice s, word w)` is `f*128 + s*16 + w*4`. Flit 0
//! slice 0 is the [`QgiHeader`]; later flits carry the per-unit HEADER / SPR / LRF / IBUFF slices, one unit per
//! slice COLUMN.
//!
//! ⛔ AND `init_binary.bin` IS A CONCATENATION, NOT ONE PACKET. `dip.cpp:126` writes one QGI header per SuperDSC,
//! and each QGI's `myflits` counts the flits AFTER it — so the next packet starts at `f + myflits + 1`. Over the
//! nine captured fixtures that walk closes exactly on the final flit every time, and `de8ab8a9` is the clearest:
//! 18 packets for the group's 18 ops. **ONE PACKET PER OP, not one per group.**
//!
//! ⛔ WHY THAT WAS INVISIBLE, which is the part worth keeping: an inner QGI has `word2 == 0`, so a classifier
//! keyed on `unit_bits == 0` reads it as a Null header and re-serialises it byte-for-byte. The round trip stays
//! exact while the structure is misread — a check that only compares bytes cannot see it.

use crate::islands::AtThisArch;

/// ONE 16-BYTE INIT SLICE: four little-endian `u32` words. `InitPerSlice` / `SixteenByteStream`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Slice(pub [u32; Self::WORDS]);

impl Slice {
    /// How many bytes one slice occupies.
    pub const BYTES: usize = 16;
    /// How many `u32` words one slice holds.
    pub const WORDS: usize = 4;

    /// The slice's bytes, little-endian per word.
    pub const fn to_le_bytes(self) -> [u8; Self::BYTES] {
        let mut bytes = [0u8; Self::BYTES];
        let mut word = 0;
        while word < 4 {
            let value = self.0[word].to_le_bytes();
            let mut byte = 0;
            while byte < 4 {
                bytes[word * 4 + byte] = value[byte];
                byte += 1;
            }
            word += 1;
        }
        bytes
    }
}

/// ONE 128-BYTE FLIT: eight slices, indexed by unit SLICE COLUMN.
///
/// ⛔ A COLUMN IS A UNIT, WHICH IS WHY THIS IS AN ARRAY AND NOT A `Vec`. Column 0 is L0SU and column 7 is L3LU;
/// a flit with seven slices is not a short flit, it is a flit missing a unit. `[Slice; SLICES]` says so.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Flit(pub [Slice; Self::SLICES]);

impl Flit {
    /// Slices per flit — one per unit column.
    pub const SLICES: usize = 8;
    /// How many bytes one flit occupies.
    pub const BYTES: usize = Self::SLICES * Slice::BYTES;

    /// The flit's bytes, in slice-column order.
    pub const fn to_le_bytes(self) -> [u8; Self::BYTES] {
        let mut bytes = [0u8; Self::BYTES];
        let mut slice = 0;
        while slice < Self::SLICES {
            let value = self.0[slice].to_le_bytes();
            let mut byte = 0;
            while byte < Slice::BYTES {
                bytes[slice * Slice::BYTES + byte] = value[byte];
                byte += 1;
            }
            slice += 1;
        }
        bytes
    }
}

/// ⛔ THE GEOMETRY IS ARITHMETIC, NOT A LITERAL 128. A flit is eight sixteen-byte slices, and writing both the
/// product and its factors is how the two come to disagree.
const _: () = assert!(Flit::BYTES == 128);
const _: () = assert!(Slice::BYTES * Flit::SLICES == Flit::BYTES);

/// One flag's bit, or nothing. A `const fn` rather than a closure, which `const` context forbids.
const fn bit(set: bool, at: u32) -> u32 {
    if set { 1u32 << at } else { 0 }
}

/// THE QGI HEADER — one per packet, and its `myflits` is what chains the packets together.
///
/// `dip.cpp:126`. Every field is a bit position in word 0 except the flit count, which straddles words 0 and 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QgiHeader {
    /// Every init flit AFTER this header — so the next packet begins at `this + myflits + 1`.
    pub myflits: FlitCount,
    /// Micro-sequencer pause (bit 30).
    pub usm_pause: bool,
    /// Load-sequencer pause (bit 29).
    pub lsm_pause: bool,
    /// Host stall (bit 24).
    pub hoststall: bool,
    /// STZ data decision (bit 23).
    pub stz: bool,
    /// Sentinel control block (bit 22).
    pub sentinel_cb: bool,
}

/// HOW MANY FLITS FOLLOW THE HEADER — `myflits` (`dip.cpp:81`).
///
/// ⛔ THE VALUE IS `totalFlits_ - 1`, not the total. `dip.cpp:81` writes `uint32_t myflits = totalFlits_ - 1`, so
/// it counts every flit AFTER the header flit, and the next packet begins at `this + myflits + 1`.
///
/// ⛔⛔ IT IS 32 BITS, WRITTEN ACROSS TWO WORDS. `dip.cpp:85-91`:
///
/// ```text
/// word0 |= (myflits & 0xFFFFFF) << 8     // the low 24 bits, above the 0xdd tag
/// word1 |= (myflits >> 24) & 0xFF        // the top 8
/// ```
///
/// ⛔⛔ SO THE COUNT AND THE FLAGS GENUINELY OVERLAP IN IBM'S OWN CODE. Word 0's flags sit at bits 30
/// (`USM_pause`), 29 (`LSM_PAUSE_BITSHIFT`), 24 (`hoststall`), 23 (`STZ_data_decision_reqd`) and 22
/// (`SENTINEL_CB_BITSHIFT`) — `dip.cpp:25-26` and `:84-88` — while the shifted count occupies bits 8..31. A count
/// of 2^14 or more therefore lands on a flag bit. This is not a porting artefact to correct silently: it is what
/// the compiler writes, and the largest captured bundle (5599 flits, 13 bits) stays below the collision.
///
/// ⭐ SO WE REFUSE AT THE BOUNDARY AND SAY IT IS OURS. Accepting a colliding count would mean emitting a header
/// whose flags the runtime reads as set; writing it and hoping is what the C++ does. A build error names the real
/// condition instead — the program is too large for this header layout to describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FlitCount(u32);

impl FlitCount {
    /// The count is shifted this far up word 0, above the `0xdd` tag (`dip.cpp:87`).
    const SHIFT: u32 = 8;
    /// How many bits of it word 0 carries — `myflits & 0xFFFFFF` (`dip.cpp:87`).
    pub const BITS_IN_WORD0: u32 = 24;
    /// The lowest flag bit in word 0: `SENTINEL_CB_BITSHIFT` (`dip.cpp:26`). The count must end below it.
    const LOWEST_FLAG_BIT: u32 = 22;
    /// How many bits the count may use before it reaches that flag.
    pub const BITS: u32 = Self::LOWEST_FLAG_BIT - Self::SHIFT;
    /// The largest count that cannot collide with a flag.
    pub const MAX: u32 = (1 << Self::BITS) - 1;

    pub const fn of(flits: u32) -> Self {
        assert!(
            flits <= Self::MAX,
            "this program's flit count reaches the QGI header's flag bits. `dip.cpp:87` shifts the count 8 bits up \
             word 0 and masks 24 of it, while the flags sit at bits 22, 23, 24, 29 and 30 of the same word — so \
             the two overlap in the C++ as well, and the largest bundle anyone has captured (5599 flits) stays \
             below it. Refusing here rather than emitting a header whose flags the runtime would read as set"
        );
        Self(flits)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// HOW MANY FLITS THE FILE IS PADDED TO — the init allocation granularity, in flits.
///
/// `dip.cpp:70-75`: `gran = progInitAddrGranularity / bytesPerStick`, and the file is rounded up to a multiple of
/// it with ZERO flits (`dip.cpp:112-119`), which is also the size handed to the runtime (`:123`).
///
/// ⛔ AND IT IS AN ARCH FACT. `progInitAddrGranularity` is 128 on RCUDD1A and **256** on SEN1P5
/// (`sysdef.cpp:538-540`) against a 128-byte stick (`sysdef.cpp:181`) — so rounding is a no-op on one arch and
/// pads to an even number of flits on the other. A port that tested only RCUDD1A would never see it.
pub const PAD_TO_FLITS: usize = if cfg!(feature = "arch-rcudd1a") { 1 } else { 2 };

const _: () = assert!(
    PAD_TO_FLITS * Flit::BYTES
        == if cfg!(feature = "arch-rcudd1a") {
            128
        } else {
            256
        }
);

impl QgiHeader {
    /// The tag every QGI header's word 0 ends with (`0xDD`).
    const TAG: u32 = 0xDD;

    /// The four words of the header slice. Words 2 and 3 are zero.
    ///
    /// ⛔ AND WORD 2 BEING ZERO IS WHAT HID THE PACKET STRUCTURE. A classifier keyed on `unit_bits == 0` reads an
    /// inner QGI as a Null header, so a byte-exact round trip can still have misread where every packet begins.
    pub const fn words(self) -> [u32; 4] {
        // `dip.cpp:85-91`. Bounded by construction below the lowest flag bit, so the shift cannot reach one — see
        // [`FlitCount`], and note that the C++ itself masks 24 bits here and lets them collide.
        let count =
            (self.myflits.get() & ((1 << FlitCount::BITS_IN_WORD0) - 1)) << FlitCount::SHIFT;
        let word0 = bit(self.usm_pause, 30)
            | bit(self.lsm_pause, 29)
            | bit(self.hoststall, 24)
            | bit(self.stz, 23)
            | bit(self.sentinel_cb, 22)
            | count
            | Self::TAG;
        // ⭐ WORD 1 CARRIES THE COUNT'S TOP 8 BITS — `initslicepart(1) |= (myflits >> 24) & 0xFF` (`dip.cpp:91`).
        // Always zero for any count this crate will accept, and written from the same value rather than left out,
        // so the two halves cannot disagree.
        let word1 = (self.myflits.get() >> FlitCount::BITS_IN_WORD0) & 0xFF;
        [word0, word1, 0, 0]
    }

    /// The count this header states — the exact inverse of [`Self::words`].
    ///
    /// ⛔ AN INVERSE, NOT A SECOND READING. The packet chain is walked by this number, so a reader that masked a
    /// different width from the writer would desynchronise every packet after the first large one — and stay
    /// byte-exact on a round trip, which is how the misreading hid.
    pub const fn myflits_of(word0: u32) -> FlitCount {
        FlitCount::of((word0 >> FlitCount::SHIFT) & FlitCount::MAX)
    }

    /// The header's own flit: slice 0 is the header, the other seven columns are zero.
    pub const fn flit(self) -> Flit {
        let mut flit = Flit([Slice([0; 4]); Flit::SLICES]);
        flit.0[0] = Slice(self.words());
        flit
    }
}

/// ONE PACKET: its header and the flits it carries.
///
/// ⭐ ONE PER OP. `de8ab8a9` holds 18 packets for its group's 18 ops, which is the fact a "one packet per group"
/// reading gets wrong while still producing a file of the right length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packet {
    pub header: QgiHeader,
    pub flits: Vec<Flit>,
}

impl Packet {
    /// The header's flit count, DERIVED from the flits rather than stored beside them.
    ///
    /// ⛔ THE COUNT AND THE FLITS ARE ONE FACT. A `myflits` field a caller sets is a second place the length is
    /// decided, and the packet chain is walked BY that number — so a disagreement does not corrupt one packet, it
    /// desynchronises every packet after it.
    pub fn header_with_count(&self) -> QgiHeader {
        QgiHeader {
            myflits: FlitCount::of(
                u32::try_from(self.flits.len()).expect("a packet holds fewer than 2^32 flits"),
            ),
            ..self.header
        }
    }
}

/// Island 4: one group's packets, in the order they are written.
pub type Packets = AtThisArch<Packet>;
