// SPDX-License-Identifier: Apache-2.0
//! WHAT EACH UNIT'S REGISTER FILES HOLD.
//!
//! Ported from `/project_src/deeptools/sys-arch-spec/sysdef.cpp:287-423` — `regInfoPerUnit[component][type]`,
//! whose initialiser is `{maxNum, bitSize, bitSizeInitPacket, signedness, initializable}` (`sysdef.cpp:285-286`).
//! `isa.cpp`'s `numRegs`/`numJCRs`/`numMVRs` are `maxNum` read out of this table, so it is what bounds a register
//! field's encoding as well as what the allocator may hand out.
//!
//! The table is arch-conditional: 40 files on RCUDD1A, 38 on SEN1P5.

/// A COMPONENT — the nine units that have register files.
///
/// ⛔ NINE, NOT SIX. `L0LU` and `L0SU` have separate register files while sharing one opcode-value enum, so this is
/// a different partition from [`crate::values::OpUnit`] and the two must not be conflated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Component {
    Pt,
    Pe,
    Sfp,
    L0lu,
    L0su,
    Lxlu,
    Lxsu,
    L3lu,
    L3su,
}

impl Component {
    /// EVERY COMPONENT, in the order a packet's slice columns take them.
    ///
    /// ⛔ ONE LIST, so a pass that walks components cannot walk a different set from another. Bridge 3 laid out its
    /// blocks from a literal array while block formation walked its own.
    pub const ALL: [Self; 9] = [
        Self::Pt,
        Self::Pe,
        Self::Sfp,
        Self::L0lu,
        Self::L0su,
        Self::Lxlu,
        Self::Lxsu,
        Self::L3lu,
        Self::L3su,
    ];

    /// WHETHER THIS COMPONENT HAS A LOAD OR STORE OPCODE AT ALL.
    ///
    /// ⛔ THE ISA ANSWERS IT, NOT A ROLE. `defineOpcode` declares the whole LDST family on the `L0` and `LX` unit
    /// families (`isa.cpp:759-762`, `:887-890`) and the L3 has its own (`:1000-1010`, reached by
    /// `ConstructL3LoadInstr`); the PE, the SFP and the PT have none. So a transfer participant on a compute unit
    /// issues no instruction — the data arrives on the port that the memory unit's `consumertag` names.
    pub const fn issues_transfers(self) -> bool {
        match self {
            Self::L0lu | Self::L0su | Self::Lxlu | Self::Lxsu | Self::L3lu | Self::L3su => true,
            Self::Pt | Self::Pe | Self::Sfp => false,
        }
    }
}

/// A REGISTER FILE KIND — `RegType` (`sysdef.cpp:287-423` names all thirteen).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegType {
    /// The PT's accumulator file.
    Arf,
    /// L3 external address registers.
    Ear,
    /// L3 external bound registers.
    Ebr,
    /// L3 group tag registers.
    Gtr,
    /// The PT's input file.
    Irf,
    /// Jump condition registers.
    Jcr,
    /// L3 local address registers.
    Lar,
    /// L3 local bound registers.
    Lbr,
    /// The general local register file.
    Lrf,
    /// Move registers.
    Mvr,
    /// Special-purpose configuration registers.
    Spr,
    /// A unit's state register.
    State,
    /// The PT's weight file.
    Xrf,
}

impl RegType {
    /// THE ORDER'S ENDPOINTS, for a prefix range over a map keyed by one of these.
    ///
    /// ⭐ NOT REGISTER FILES — BOUNDS. `deeptools`'s `reginit` L3 map is keyed `(.., RegType, RegIndex)`, so asking for
    /// one (program, core, unit)'s entries is a `range` between these rather than a filter over every entry. They
    /// track the `derive(PartialOrd, Ord)` order, which is declaration order, so they are the first and last
    /// variants above.
    pub const FIRST: Self = Self::Arf;
    /// The last variant in declaration order — see [`Self::FIRST`].
    pub const LAST: Self = Self::Xrf;

    /// EVERY REGISTER FILE KIND, in declaration order — the thirteen `sysdef.cpp:287-423` names.
    ///
    /// ⛔ ONE LIST, for the same reason [`Component::ALL`] is one: a pass that walks the files must not walk a
    /// different set from the pass that checks it. The bounds assertion below reads this rather than a second
    /// copy of it.
    pub const ALL: [Self; 13] = [
        Self::Arf,
        Self::Ear,
        Self::Ebr,
        Self::Gtr,
        Self::Irf,
        Self::Jcr,
        Self::Lar,
        Self::Lbr,
        Self::Lrf,
        Self::Mvr,
        Self::Spr,
        Self::State,
        Self::Xrf,
    ];
}

/// ⛔ THE BOUNDS MUST BE THE ACTUAL EXTREMES OF THE DERIVED ORDER, or a range silently drops entries. Asserted
/// against every variant rather than trusted, so adding a variant outside them breaks the build.
const _: () = {
    let all = RegType::ALL;
    let mut i = 0;
    while i < all.len() {
        assert!(all[i] as u8 >= RegType::FIRST as u8);
        assert!(all[i] as u8 <= RegType::LAST as u8);
        i += 1;
    }
};

/// HOW MANY BITS ONE REGISTER HOLDS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BitWidth(u16);

impl BitWidth {
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// HOW WIDE THIS FILE'S ENTRIES ARE IN THE INIT PACKET, OR THAT IT IS NOT IN ONE.
///
/// ⛔ THE C++ WRITES `-1` FOR "NOT IN THE INIT PACKET" AND THAT IS NOT A WIDTH. `JCR`, `STATE`, `ARF`, `XRF` and
/// `IRF` all carry it, and every one of them is also `initializable: false` — so a port that kept the `-1` as a
/// number would let bridge 3 compute a packet region for a file that has none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitPacketWidth {
    /// `bitSizeInitPacket = -1`: this file is not written by the init packet.
    NotInPacket,
    /// The width one entry occupies in the packet.
    Bits(BitWidth),
}

/// HOW A REGISTER'S VALUE IS READ — `Isa::Signedness`.
///
/// ⛔ `UNDEFINED` IS A STATE THE TABLE REALLY WRITES, for the 128-bit data files whose contents are not a scalar.
/// It is not a missing value to fill in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signedness {
    Undefined,
    Signed,
    Unsigned,
    ModuloUnsigned,
}

/// WHETHER THE INIT PACKET MAY WRITE THIS FILE.
///
/// ⛔ A DEPTH IS NOT A LICENCE. The PT's XRF is 64 registers deep on RCUDD1A and `initializable: false`: the
/// allocator may name them, the packet may not fill them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Initializable {
    No,
    Yes,
}

/// ONE REGISTER FILE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegInfo {
    /// `maxNum` — how many registers, which is what `RegIndex::of_file` is bounded by.
    pub depth: FileDepth,
    /// `bitSize` — how wide one register is.
    pub bits: BitWidth,
    /// `bitSizeInitPacket`.
    pub in_packet: InitPacketWidth,
    pub sign: Signedness,
    pub initializable: Initializable,
}

/// HOW MANY INSTRUCTIONS ONE UNIT'S IBUFF HOLDS — `maxIBuffEntriesPerUnit` (`sysdef.cpp:451-500`).
///
/// ⛔⛔ THIS IS THE HARDWARE BUFFER, NOT THE HEADER FIELD. `IbuffFlitCount::MAX` is 127 because the header's flit
/// count is seven bits (`initpacket.cpp:224`) — 508 instructions for a 32-bit unit — while the unit itself holds
/// only what this table says. A program longer than its buffer is written past the end of it, and nothing in the
/// packet says so: `progtailor.cpp:784-786` is where dxp checks it,
/// `DT_CHECK_MSG(progInsts.size() < sysDef.maxIBuffEntriesPerUnit.at(comp), "Progstitch exceeded max ibuff")`.
///
/// ⛔ THE L3's IS UNCONDITIONAL AND EVERYONE ELSE'S IS ARCH-DEPENDENT. `L3LU`/`L3SU` are 256 immediately after
/// `maxIBuffEntriesPerUnit.clear()` and BEFORE the `coreArch` branch (`:451-453`); the MPW2/MPW3 arm gives 64 to
/// the LX/compute units and 40 to the L0 (`:454-...`), and the `else` arm this crate builds for gives **128** to
/// all of them (`:495-500`), with the LXLU alone rising to 256 on SEN1P5 — the C++ writes that one as a ternary,
/// `maxIBuffEntriesPerUnit[LXLU] = coreArch == SEN1P5_ISA ? 256 : 128` (`:493-494`).
///
/// ⛔ THE LINE NUMBERS HERE ONCE READ `424-478`/`425-426`/`463-475` AND WERE STALE — in the current mirror those
/// are `regInfoPerUnit` rows and the MPW arm. Every VALUE was right; only the lines had moved. Re-locate by the
/// assignment text, never by the number.
pub const fn max_ibuff_entries(comp: Component) -> u16 {
    match comp {
        // `sysdef.cpp:451-453` — set before the `coreArch` branch, so no arch changes them.
        Component::L3lu | Component::L3su => 256,
        // `maxIBuffEntriesPerUnit[LXLU] = coreArch == SEN1P5_ISA ? 256 : 128` (`sysdef.cpp:493-494`) — the one
        // unit whose capacity the arch changes, which is why this is a match on the feature rather than a number.
        Component::Lxlu => match cfg!(feature = "arch-sen1p5") {
            true => 256,
            false => 128,
        },
        // `sysdef.cpp:495-500` — 128 each on the `else` arm this crate builds for.
        Component::Lxsu
        | Component::L0lu
        | Component::L0su
        | Component::Sfp
        | Component::Pe
        | Component::Pt => 128,
    }
}

/// HOW MANY REGISTERS ONE FILE OF ONE UNIT HOLDS — `maxNum`, the number [`RegIndex::of_file`] is bounded by.
///
/// ⛔ IT LIVES HERE BECAUSE THIS IS THE TABLE THAT KNOWS IT — `maxNum` of `regInfoPerUnit`
/// (`sysdef.cpp:312-478`), read by [`depth_of`]. A depth is a fact about the MACHINE, not about any one
/// island's value, and holding it beside a register index put the ISA's own table one module away from the
/// number it states.
///
/// ⛔ A DISTINCT TYPE FROM AN INDEX AND FROM A COUNT OF ANYTHING ELSE, because it is the one number the allocator
/// must not confuse with the field's width. `regfile::depth` is its only source, so a depth cannot be invented at
/// a call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FileDepth(u8);

impl FileDepth {
    /// Minted by `regfile::depth` alone — the vendored table is the only thing that knows a file's size.
    pub(crate) const fn of(registers: u8) -> Self {
        Self(registers)
    }

    /// A BOUND STATED BY A PACKET FORMAT RATHER THAN BY THE REGISTER TABLE.
    ///
    /// ⛔ THE ONE LEGITIMATE EXCEPTION, AND IT IS NAMED SO IT STAYS ONE. An init packet may address FEWER
    /// registers than the unit has, because the header encodes the index in a fixed width — the LX/L0 LRF nibble
    /// of `dip.h:49` — and that ceiling is a fact about the PACKET, not about the machine. Such a caller is not
    /// inventing a depth it failed to look up; it is naming a different quantity, and [`Self::of`] stays private
    /// so the two cannot be confused.
    pub const fn of_packet_bound(registers: u8) -> Self {
        Self(registers)
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

/// WHETHER A UNIT HAS A FILE AT ALL.
///
/// ⛔ ABSENT IS NOT DEPTH ZERO, and it is not an `Option` either: "the SFP has no EAR" is a fact about the machine,
/// and a unit with a zero-deep file would be a different machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presence {
    /// This unit does not have this kind of register file.
    Absent,
    Present(RegInfo),
}

/// A file's depth, when it has one.
///
/// ⛔⛔ THIS IS NOT `RegIndex::COUNT`. That is 32 — what an instruction FIELD can encode — and using it as a depth
/// hands out registers a unit does not have. The two coincide for the SFP's LRF on SEN1P5 and differ on RCUDD1A,
/// which is why they are separate types rather than separated by a comparison.
pub const fn depth_of(comp: Component, reg: RegType) -> Presence {
    info_of(comp, reg)
}

const fn present(
    depth: u8,
    bits: u16,
    in_packet: InitPacketWidth,
    sign: Signedness,
    initializable: Initializable,
) -> Presence {
    Presence::Present(RegInfo {
        depth: FileDepth::of(depth),
        bits: BitWidth(bits),
        in_packet,
        sign,
        initializable,
    })
}

/// The width an init packet gives one entry of this file.
const fn packet(bits: u16) -> InitPacketWidth {
    InitPacketWidth::Bits(BitWidth(bits))
}

/// EVERY (component, file) PAIR THE MACHINE HAS ON RCUDD1A — 40 present, 77 absent.
///
/// ⛔ THE TWO ARCHES ARE TWO LISTS, NOT ONE LIST WITH AN ARCH COLUMN. SEN1P5 removes the L3's LBR outright, so a
/// pair that is `Present` here is `Absent` there — which a column beside a shared row cannot say.
#[cfg(feature = "arch-rcudd1a")]
pub const fn info_of(comp: Component, reg: RegType) -> Presence {
    match (comp, reg) {
        (Component::L3lu, RegType::Ear) => {
            present(16, 21, packet(32), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3lu, RegType::Lar) => present(
            16,
            14,
            packet(16),
            Signedness::ModuloUnsigned,
            Initializable::Yes,
        ),
        (Component::L3lu, RegType::Lbr) => {
            present(8, 14, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3lu, RegType::Ebr) => {
            present(8, 30, packet(32), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3lu, RegType::Gtr) => {
            present(8, 14, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3lu, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::L3lu, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3su, RegType::Ear) => {
            present(16, 21, packet(32), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3su, RegType::Lar) => present(
            16,
            14,
            packet(16),
            Signedness::ModuloUnsigned,
            Initializable::Yes,
        ),
        (Component::L3su, RegType::Lbr) => {
            present(8, 14, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3su, RegType::Ebr) => {
            present(8, 30, packet(32), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3su, RegType::Gtr) => {
            present(8, 14, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3su, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::L3su, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::Lxlu, RegType::Lrf) => present(
            16,
            21,
            packet(24),
            Signedness::ModuloUnsigned,
            Initializable::Yes,
        ),
        (Component::Lxlu, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::Lxlu, RegType::Mvr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::Lxlu, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::Lxsu, RegType::Lrf) => present(
            16,
            21,
            packet(24),
            Signedness::ModuloUnsigned,
            Initializable::Yes,
        ),
        (Component::Lxsu, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::Lxsu, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L0lu, RegType::Lrf) => present(
            16,
            10,
            packet(16),
            Signedness::ModuloUnsigned,
            Initializable::Yes,
        ),
        (Component::L0lu, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::L0lu, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L0su, RegType::Lrf) => present(
            16,
            10,
            packet(16),
            Signedness::ModuloUnsigned,
            Initializable::Yes,
        ),
        (Component::L0su, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::L0su, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::Sfp, RegType::Lrf) => present(
            16,
            128,
            packet(128),
            Signedness::Undefined,
            Initializable::Yes,
        ),
        (Component::Sfp, RegType::State) => present(
            1,
            128,
            InitPacketWidth::NotInPacket,
            Signedness::Undefined,
            Initializable::No,
        ),
        (Component::Sfp, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::Sfp, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::Pe, RegType::Lrf) => present(
            16,
            128,
            packet(128),
            Signedness::Undefined,
            Initializable::Yes,
        ),
        (Component::Pe, RegType::State) => present(
            1,
            128,
            InitPacketWidth::NotInPacket,
            Signedness::Undefined,
            Initializable::No,
        ),
        (Component::Pe, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::Pe, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::Pt, RegType::Arf) => present(
            4,
            128,
            InitPacketWidth::NotInPacket,
            Signedness::Undefined,
            Initializable::No,
        ),
        (Component::Pt, RegType::Xrf) => present(
            64,
            128,
            InitPacketWidth::NotInPacket,
            Signedness::Undefined,
            Initializable::No,
        ),
        (Component::Pt, RegType::Irf) => present(
            2,
            128,
            InitPacketWidth::NotInPacket,
            Signedness::Undefined,
            Initializable::No,
        ),
        (Component::Pt, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::Pt, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::Pt, RegType::Ear)
        | (Component::Pt, RegType::Ebr)
        | (Component::Pt, RegType::Gtr)
        | (Component::Pt, RegType::Lar)
        | (Component::Pt, RegType::Lbr)
        | (Component::Pt, RegType::Lrf)
        | (Component::Pt, RegType::Mvr)
        | (Component::Pt, RegType::State)
        | (Component::Pe, RegType::Arf)
        | (Component::Pe, RegType::Ear)
        | (Component::Pe, RegType::Ebr)
        | (Component::Pe, RegType::Gtr)
        | (Component::Pe, RegType::Irf)
        | (Component::Pe, RegType::Lar)
        | (Component::Pe, RegType::Lbr)
        | (Component::Pe, RegType::Mvr)
        | (Component::Pe, RegType::Xrf)
        | (Component::Sfp, RegType::Arf)
        | (Component::Sfp, RegType::Ear)
        | (Component::Sfp, RegType::Ebr)
        | (Component::Sfp, RegType::Gtr)
        | (Component::Sfp, RegType::Irf)
        | (Component::Sfp, RegType::Lar)
        | (Component::Sfp, RegType::Lbr)
        | (Component::Sfp, RegType::Mvr)
        | (Component::Sfp, RegType::Xrf)
        | (Component::L0lu, RegType::Arf)
        | (Component::L0lu, RegType::Ear)
        | (Component::L0lu, RegType::Ebr)
        | (Component::L0lu, RegType::Gtr)
        | (Component::L0lu, RegType::Irf)
        | (Component::L0lu, RegType::Lar)
        | (Component::L0lu, RegType::Lbr)
        | (Component::L0lu, RegType::Mvr)
        | (Component::L0lu, RegType::State)
        | (Component::L0lu, RegType::Xrf)
        | (Component::L0su, RegType::Arf)
        | (Component::L0su, RegType::Ear)
        | (Component::L0su, RegType::Ebr)
        | (Component::L0su, RegType::Gtr)
        | (Component::L0su, RegType::Irf)
        | (Component::L0su, RegType::Lar)
        | (Component::L0su, RegType::Lbr)
        | (Component::L0su, RegType::Mvr)
        | (Component::L0su, RegType::State)
        | (Component::L0su, RegType::Xrf)
        | (Component::Lxlu, RegType::Arf)
        | (Component::Lxlu, RegType::Ear)
        | (Component::Lxlu, RegType::Ebr)
        | (Component::Lxlu, RegType::Gtr)
        | (Component::Lxlu, RegType::Irf)
        | (Component::Lxlu, RegType::Lar)
        | (Component::Lxlu, RegType::Lbr)
        | (Component::Lxlu, RegType::State)
        | (Component::Lxlu, RegType::Xrf)
        | (Component::Lxsu, RegType::Arf)
        | (Component::Lxsu, RegType::Ear)
        | (Component::Lxsu, RegType::Ebr)
        | (Component::Lxsu, RegType::Gtr)
        | (Component::Lxsu, RegType::Irf)
        | (Component::Lxsu, RegType::Lar)
        | (Component::Lxsu, RegType::Lbr)
        | (Component::Lxsu, RegType::Mvr)
        | (Component::Lxsu, RegType::State)
        | (Component::Lxsu, RegType::Xrf)
        | (Component::L3lu, RegType::Arf)
        | (Component::L3lu, RegType::Irf)
        | (Component::L3lu, RegType::Lrf)
        | (Component::L3lu, RegType::Mvr)
        | (Component::L3lu, RegType::State)
        | (Component::L3lu, RegType::Xrf)
        | (Component::L3su, RegType::Arf)
        | (Component::L3su, RegType::Irf)
        | (Component::L3su, RegType::Lrf)
        | (Component::L3su, RegType::Mvr)
        | (Component::L3su, RegType::State)
        | (Component::L3su, RegType::Xrf) => Presence::Absent,
    }
}

/// EVERY (component, file) PAIR THE MACHINE HAS ON SEN1P5 — 38 present, 79 absent.
#[cfg(not(feature = "arch-rcudd1a"))]
pub const fn info_of(comp: Component, reg: RegType) -> Presence {
    match (comp, reg) {
        (Component::L3lu, RegType::Ear) => {
            present(16, 21, packet(32), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3lu, RegType::Lar) => present(
            16,
            15,
            packet(16),
            Signedness::ModuloUnsigned,
            Initializable::Yes,
        ),
        (Component::L3lu, RegType::Ebr) => {
            present(16, 32, packet(32), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3lu, RegType::Gtr) => {
            present(8, 14, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3lu, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::L3lu, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3su, RegType::Ear) => {
            present(16, 21, packet(32), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3su, RegType::Lar) => present(
            16,
            15,
            packet(16),
            Signedness::ModuloUnsigned,
            Initializable::Yes,
        ),
        (Component::L3su, RegType::Ebr) => {
            present(16, 32, packet(32), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3su, RegType::Gtr) => {
            present(8, 14, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L3su, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::L3su, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::Lxlu, RegType::Lrf) => present(
            16,
            21,
            packet(24),
            Signedness::ModuloUnsigned,
            Initializable::Yes,
        ),
        (Component::Lxlu, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::Lxlu, RegType::Mvr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::Lxlu, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::Lxsu, RegType::Lrf) => present(
            16,
            21,
            packet(24),
            Signedness::ModuloUnsigned,
            Initializable::Yes,
        ),
        (Component::Lxsu, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::Lxsu, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L0lu, RegType::Lrf) => present(
            16,
            12,
            packet(16),
            Signedness::ModuloUnsigned,
            Initializable::Yes,
        ),
        (Component::L0lu, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::L0lu, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::L0su, RegType::Lrf) => present(
            16,
            12,
            packet(16),
            Signedness::ModuloUnsigned,
            Initializable::Yes,
        ),
        (Component::L0su, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::L0su, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::Sfp, RegType::Lrf) => present(
            32,
            128,
            packet(128),
            Signedness::Undefined,
            Initializable::Yes,
        ),
        (Component::Sfp, RegType::State) => present(
            4,
            128,
            InitPacketWidth::NotInPacket,
            Signedness::Undefined,
            Initializable::No,
        ),
        (Component::Sfp, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::Sfp, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::Pe, RegType::Lrf) => present(
            32,
            128,
            packet(128),
            Signedness::Undefined,
            Initializable::Yes,
        ),
        (Component::Pe, RegType::State) => present(
            4,
            128,
            InitPacketWidth::NotInPacket,
            Signedness::Undefined,
            Initializable::No,
        ),
        (Component::Pe, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::Pe, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::Pt, RegType::Arf) => present(
            4,
            128,
            InitPacketWidth::NotInPacket,
            Signedness::Undefined,
            Initializable::No,
        ),
        (Component::Pt, RegType::Xrf) => present(
            128,
            128,
            InitPacketWidth::NotInPacket,
            Signedness::Undefined,
            Initializable::No,
        ),
        (Component::Pt, RegType::Irf) => present(
            2,
            128,
            InitPacketWidth::NotInPacket,
            Signedness::Undefined,
            Initializable::No,
        ),
        (Component::Pt, RegType::Jcr) => present(
            16,
            16,
            InitPacketWidth::NotInPacket,
            Signedness::Signed,
            Initializable::No,
        ),
        (Component::Pt, RegType::Spr) => {
            present(8, 16, packet(16), Signedness::Unsigned, Initializable::Yes)
        }
        (Component::Pt, RegType::Ear)
        | (Component::Pt, RegType::Ebr)
        | (Component::Pt, RegType::Gtr)
        | (Component::Pt, RegType::Lar)
        | (Component::Pt, RegType::Lbr)
        | (Component::Pt, RegType::Lrf)
        | (Component::Pt, RegType::Mvr)
        | (Component::Pt, RegType::State)
        | (Component::Pe, RegType::Arf)
        | (Component::Pe, RegType::Ear)
        | (Component::Pe, RegType::Ebr)
        | (Component::Pe, RegType::Gtr)
        | (Component::Pe, RegType::Irf)
        | (Component::Pe, RegType::Lar)
        | (Component::Pe, RegType::Lbr)
        | (Component::Pe, RegType::Mvr)
        | (Component::Pe, RegType::Xrf)
        | (Component::Sfp, RegType::Arf)
        | (Component::Sfp, RegType::Ear)
        | (Component::Sfp, RegType::Ebr)
        | (Component::Sfp, RegType::Gtr)
        | (Component::Sfp, RegType::Irf)
        | (Component::Sfp, RegType::Lar)
        | (Component::Sfp, RegType::Lbr)
        | (Component::Sfp, RegType::Mvr)
        | (Component::Sfp, RegType::Xrf)
        | (Component::L0lu, RegType::Arf)
        | (Component::L0lu, RegType::Ear)
        | (Component::L0lu, RegType::Ebr)
        | (Component::L0lu, RegType::Gtr)
        | (Component::L0lu, RegType::Irf)
        | (Component::L0lu, RegType::Lar)
        | (Component::L0lu, RegType::Lbr)
        | (Component::L0lu, RegType::Mvr)
        | (Component::L0lu, RegType::State)
        | (Component::L0lu, RegType::Xrf)
        | (Component::L0su, RegType::Arf)
        | (Component::L0su, RegType::Ear)
        | (Component::L0su, RegType::Ebr)
        | (Component::L0su, RegType::Gtr)
        | (Component::L0su, RegType::Irf)
        | (Component::L0su, RegType::Lar)
        | (Component::L0su, RegType::Lbr)
        | (Component::L0su, RegType::Mvr)
        | (Component::L0su, RegType::State)
        | (Component::L0su, RegType::Xrf)
        | (Component::Lxlu, RegType::Arf)
        | (Component::Lxlu, RegType::Ear)
        | (Component::Lxlu, RegType::Ebr)
        | (Component::Lxlu, RegType::Gtr)
        | (Component::Lxlu, RegType::Irf)
        | (Component::Lxlu, RegType::Lar)
        | (Component::Lxlu, RegType::Lbr)
        | (Component::Lxlu, RegType::State)
        | (Component::Lxlu, RegType::Xrf)
        | (Component::Lxsu, RegType::Arf)
        | (Component::Lxsu, RegType::Ear)
        | (Component::Lxsu, RegType::Ebr)
        | (Component::Lxsu, RegType::Gtr)
        | (Component::Lxsu, RegType::Irf)
        | (Component::Lxsu, RegType::Lar)
        | (Component::Lxsu, RegType::Lbr)
        | (Component::Lxsu, RegType::Mvr)
        | (Component::Lxsu, RegType::State)
        | (Component::Lxsu, RegType::Xrf)
        | (Component::L3lu, RegType::Arf)
        | (Component::L3lu, RegType::Irf)
        | (Component::L3lu, RegType::Lbr)
        | (Component::L3lu, RegType::Lrf)
        | (Component::L3lu, RegType::Mvr)
        | (Component::L3lu, RegType::State)
        | (Component::L3lu, RegType::Xrf)
        | (Component::L3su, RegType::Arf)
        | (Component::L3su, RegType::Irf)
        | (Component::L3su, RegType::Lbr)
        | (Component::L3su, RegType::Lrf)
        | (Component::L3su, RegType::Mvr)
        | (Component::L3su, RegType::State)
        | (Component::L3su, RegType::Xrf) => Presence::Absent,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// THE DEPTHS THE ENCODER AND THE ALLOCATOR BOTH READ
// ─────────────────────────────────────────────────────────────────────────────

/// A file a unit does not have is absent, which is a different fact from a depth of zero.
const _: () = assert!(matches!(
    info_of(Component::Sfp, RegType::Ear),
    Presence::Absent
));
const _: () = assert!(matches!(
    info_of(Component::Pt, RegType::Lrf),
    Presence::Absent
));
const _: () = assert!(matches!(
    info_of(Component::L0lu, RegType::Xrf),
    Presence::Absent
));

/// ⛔ THE SFP's LRF IS THE NUMBER THAT COST A LIVE FAILURE: 16 deep on RCUDD1A, 32 on SEN1P5, while a register FIELD
/// encodes 32 on both. Reading the field's width as the depth is correct on one arch and hands out registers
/// 16..=31 that do not exist on the other.
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(matches!(
    info_of(Component::Sfp, RegType::Lrf),
    Presence::Present(RegInfo { depth, .. }) if depth.get() == 16
));
#[cfg(not(feature = "arch-rcudd1a"))]
const _: () = assert!(matches!(
    info_of(Component::Sfp, RegType::Lrf),
    Presence::Present(RegInfo { depth, .. }) if depth.get() == 32
));

/// ⛔⛔ AND THE PT's XRF IS DEEPER THAN A REGISTER FIELD IS WIDE — 64 on RCUDD1A, 128 on SEN1P5 against a 32-entry
/// field. So "no register file is deeper than the field can encode" is FALSE, and any code asserting it panics the
/// moment the XRF is named. The XRF is addressed by the weight-load path rather than by a register field.
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(matches!(
    info_of(Component::Pt, RegType::Xrf),
    Presence::Present(RegInfo { depth, .. }) if depth.get() == 64
));
#[cfg(not(feature = "arch-rcudd1a"))]
const _: () = assert!(matches!(
    info_of(Component::Pt, RegType::Xrf),
    Presence::Present(RegInfo { depth, .. }) if depth.get() == 128
));

/// A depth is not a licence: the XRF is deep and not initializable.
const _: () = assert!(matches!(
    info_of(Component::Pt, RegType::Xrf),
    Presence::Present(RegInfo {
        initializable: Initializable::No,
        ..
    })
));
/// The LRF is both.
const _: () = assert!(matches!(
    info_of(Component::Pe, RegType::Lrf),
    Presence::Present(RegInfo {
        initializable: Initializable::Yes,
        ..
    })
));

/// ⛔ `bitSizeInitPacket = -1` IS "NOT IN THE PACKET", and every file that carries it is also not initializable —
/// so the two agree, and a port that read the `-1` as a width would give a packet region to a file with none.
const _: () = assert!(matches!(
    info_of(Component::Pt, RegType::Jcr),
    Presence::Present(RegInfo {
        in_packet: InitPacketWidth::NotInPacket,
        initializable: Initializable::No,
        ..
    })
));

/// The JCR is 16 deep and SIGNED on every unit that has one, which is what makes a jump condition a signed compare.
const _: () = assert!(matches!(
    info_of(Component::L0lu, RegType::Jcr),
    Presence::Present(RegInfo {
        sign: Signedness::Signed,
        depth,
        ..
    }) if depth.get() == 16
));

/// ⛔ THE L3's LAR IS `MODULO_UNSIGNED`, WHICH IS A THIRD READING AND NOT A SPELLING OF UNSIGNED. Its bit size also
/// differs by arch — 14 on RCUDD1A, 15 on SEN1P5 — while its depth does not.
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(matches!(
    info_of(Component::L3lu, RegType::Lar),
    Presence::Present(RegInfo { bits, sign: Signedness::ModuloUnsigned, .. }) if bits.get() == 14
));
#[cfg(not(feature = "arch-rcudd1a"))]
const _: () = assert!(matches!(
    info_of(Component::L3lu, RegType::Lar),
    Presence::Present(RegInfo { bits, sign: Signedness::ModuloUnsigned, .. }) if bits.get() == 15
));

/// ⛔ SEN1P5 REMOVES THE L3's LBR ENTIRELY (`sysdef.cpp`'s "NO LBR for sentient1.5"), so a file present on one arch
/// is absent on the other — which is why the table is arch-conditional rather than one list with an arch column.
#[cfg(feature = "arch-rcudd1a")]
const _: () = assert!(matches!(
    info_of(Component::L3lu, RegType::Lbr),
    Presence::Present(_)
));
#[cfg(not(feature = "arch-rcudd1a"))]
const _: () = assert!(matches!(
    info_of(Component::L3lu, RegType::Lbr),
    Presence::Absent
));

/// The data files are 128 bits wide and their signedness is `UNDEFINED`, which the table really writes.
const _: () = assert!(matches!(
    info_of(Component::Sfp, RegType::Lrf),
    Presence::Present(RegInfo {
        bits,
        sign: Signedness::Undefined,
        ..
    }) if bits.get() == 128
));
