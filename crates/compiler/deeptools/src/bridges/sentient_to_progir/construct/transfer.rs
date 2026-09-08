//! THE TRANSFER INSTRUCTIONS — the L3 and non-L3 loads and stores, the burst normalisation and
//! the load/store fusion.
//!
//! ⛔ TWO BURST DERIVATIONS EXIST IN OUR TREE AND dxp HAS ONE. Port the reference's.
//! ⛔ BOTH STORE-SIDE SYNCS WERE FOUND INVERTED ON AN EBR-MATCHED OP, and that writes zeros.
//!
//! 10 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e003_normalizeBurstSize` | 0 | 13 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2562` |
//! | `e054_ConstructL3LoadInstr` | 1 | 95 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2576` |
//! | `e055_ConstructL3StoreInstr` | 1 | 150 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2672` |
//! | `e056_ConstructZRAssignInstr` | 1 | 14 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2828` |
//! | `e057_ConstructL3LoadAndStoreInstr` | 1 | 179 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2843` |
//! | `e058_ConstructLoadComputeInstr` | 1 | 99 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3287` |
//! | `e059_ConstructLRFCopyInstr` | 1 | 49 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3509` |
//! | `e089_ConstructLoadInstr` | 2 | 201 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3024` |
//! | `e090_ConstructLoadInstr` | 2 | 60 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3226` |
//! | `e091_ConstructStoreInstr` | 2 | 121 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3387` |

use crate::arch::{Arch, Bounded, Elements, IsaGen};
use crate::bridges::sentient_to_progir::construct::reg_init::add_to_regs_to_init;
use crate::bridges::sentient_to_progir::construct::{descriptive, int};
use crate::bridges::sentient_to_progir::lower::labels_and_regs::address_scale;
use crate::bridges::sentient_to_progir::state::{RegsToInit, UnitKey};
use crate::bridges::sentient_to_progir::uniform::instr::{
    MapMode, MappedEntry, OperandMapRefusal, PerUnitOperand, UniformInstrInfo,
    add_entry_to_operand_map,
};
use crate::bridges::sentient_to_progir::utils::{LoadConsumer, proper_consumer};
use crate::formats::DataFormat;
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::islands::progir::ty::Operand;
use crate::islands::progir::{OpCode, OperandField};
use crate::islands::sentient::dialects::sentient::{
    Reg, RegIndex, RegType as SenRegType, RoutingDirection,
};
use crate::units::{Core, DfirUnit};
use sys_arch_spec::regfile::Component;

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// THE UNIT'S BURST SIZE, WHERE IT HAS ONE — `getMaxBurstSize` (`DccExtContext.cpp:338-348`).
///
/// ⛔ `-1` FOR EVERY COMPUTE UNIT, which is the reference's spelling for "no burst size at all"
/// rather than a size. As an `Option` no arithmetic can reach it.
#[must_use]
pub const fn max_burst_size<A: Arch>(comp: Component) -> Option<Elements> {
    match comp {
        Component::L3lu | Component::L3su => Some(Elements(A::L3_BURST as u64)),
        Component::Lxlu | Component::Lxsu => Some(Elements(A::LX_BURST as u64)),
        Component::L0lu | Component::L0su => Some(Elements(A::L0_BURST as u64)),
        Component::Pt | Component::Pe | Component::Sfp => None,
    }
}

/// Replaces: e003_normalizeBurstSize
///
/// The burst size as the ISA field spells it: the maximum wraps to 0 and 0 means one beat.
///
/// ⛔ THE REFERENCE'S RANGE CHECK IS A TAUTOLOGY — `(burst_size >= 0 || burst_size <= max)`
/// (`:2565-2566`) is an OR, so it holds for every `int`, and no burst size has ever been refused
/// there. [`Elements`] is unsigned, so the half it meant to test cannot be written down.
///
/// ⛔ A COMPUTE UNIT HAS NO MAXIMUM, so the wrap-to-zero arm cannot fire — matching the reference,
/// where `max_threshold` is `-1` and `burst_size == -1` is unreachable for a real size.
#[must_use]
pub const fn normalize_burst_size<A: Arch>(burst_size: Elements, comp: Component) -> Elements {
    if let Some(max) = max_burst_size::<A>(comp) {
        if burst_size.0 == max.0 {
            return Elements(0);
        }
    }
    if burst_size.0 == 0 {
        Elements(1)
    } else {
        burst_size
    }
}

// crustify:todo: e089_ConstructLoadInstr
// crustify:todo: e090_ConstructLoadInstr
// crustify:todo: e091_ConstructStoreInstr

/// A REGISTER'S INDEX AS AN INSTRUCTION FIELD — ⛔ `-1` WHERE IT IS UNASSIGNED, which is the sentinel
/// `getValueRegIndex` answers and what the reference then writes into `src0`/`src1`.
const fn reg_field(reg: Reg) -> Operand {
    int(match reg.index {
        Some(index) => index.get() as i64,
        None => -1,
    })
}

/// ONE `node` PER UNIT — what `OperandMap(…, Mode::unit_id)` fills in for a `uniform.query_map` peer
/// (`:2660-2663`, `:2807-2810`): the core id the map answers for each unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeMap {
    /// The entry a unit the map does not name falls back to — see [`PerUnitOperand`].
    pub first: (UnitKey, Core),
    /// The rest, in insertion order.
    pub rest: Vec<(UnitKey, Core)>,
}

impl NodeMap {
    /// The `node` field, one value per unit.
    #[must_use]
    pub fn operand(&self) -> PerUnitOperand {
        let node = |core: Core| int(i64::from(core.get()));
        PerUnitOperand {
            first: (self.first.0, node(self.first.1)),
            rest: self
                .rest
                .iter()
                .map(|(unit, core)| (*unit, node(*core)))
                .collect(),
        }
    }
}

/// WHICH PEER A TRANSFER NAMES — the four sources of the `node` field (`:2641-2665`, `:2799-2812`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Peer {
    /// An `ear` locale — the peer changes at run time, so the field carries the register index with
    /// ⛔ THE 8TH BIT SET as the flag that says so (`:2650-2654`).
    Dynamic(RegIndex),
    /// A `dataflow.get_unit` peer — its core id.
    Core(Core),
    /// A `uniform.query_map` peer — ⛔ AN OPERAND MAP, not a common field.
    Mapped(NodeMap),
    /// A `sentient.copy`/`sentient.if` peer — ⛔ NO `node` FIELD AT ALL, which is what falling through
    /// both `dyn_cast`s leaves.
    Unnamed,
}

/// `DYNAMIC_NODE_FLAG` (`:2652`) — the bit that says the low bits are a register, not a core.
const DYNAMIC_NODE_FLAG: u32 = 1 << 7;

/// The `node` field for one peer — nothing at all for an unnamed one.
fn set_node(instr: &mut UniformInstrInfo, peer: &Peer) {
    match peer {
        Peer::Dynamic(index) => {
            instr.set_common_field(
                OperandField::Node,
                int(i64::from(DYNAMIC_NODE_FLAG | index.get())),
            );
        }
        Peer::Core(core) => {
            instr.set_common_field(OperandField::Node, int(i64::from(core.get())));
        }
        // ⛔ PUSHED, NOT SORTED IN: `node` is the only operand map any transfer sets, so this map has
        // at most one entry and field order is whatever this puts there.
        Peer::Mapped(map) => instr.operand_map.push((OperandField::Node, map.operand())),
        Peer::Unnamed => {}
    }
}

/// WHERE A `load_and_send` SENDS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadTarget {
    /// A `gtr` locale — a multicast group.
    ///
    /// ⛔⛔ THE DIRECTION **IS** THE `node` FIELD HERE (`:2643-2645`), so it is required and the
    /// reference's `DT_CHECK(load_op.getDir())` cannot fire.
    Multicast {
        /// The GTR holding the group.
        group: RegIndex,
        /// Which way round the ring.
        dir: RoutingDirection,
    },
    /// One peer, and the direction SEN1P5's `drm` carries where there is one.
    Peer {
        /// Who receives.
        peer: Peer,
        /// `$dir`.
        dir: Option<RoutingDirection>,
    },
}

impl LoadTarget {
    /// `$dir` — what SEN1P5 puts in `drm` (`:2632-2636`).
    #[must_use]
    pub const fn dir(&self) -> Option<RoutingDirection> {
        match self {
            LoadTarget::Multicast { dir, .. } => Some(*dir),
            LoadTarget::Peer { dir, .. } => *dir,
        }
    }
}

/// ONE `sentient.load_and_send` WITH ITS REGISTERS ALREADY RESOLVED — the walk from a `Value` to a
/// register index is the reference's mechanism, not its lowering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L3Load {
    /// `mutable_addr` — the LAR, which is `src0`.
    pub mutable_addr: Reg,
    /// `immutable_addr` — the LBR. ⛔ ON SEN1P5 IT MUST BE THE CONSTANT 0 and there is no field.
    pub immutable_addr: Reg,
    /// `result`.
    pub result: Reg,
    /// Where the data goes.
    pub target: LoadTarget,
    /// `burst_size`, before normalisation.
    pub burst: Elements,
    /// `isUpdateMode` — the `U` in the opcode.
    pub update: bool,
    /// `dbgName`.
    pub dbg_name: Option<String>,
}

/// Replaces: e054_ConstructL3LoadInstr
///
/// Read the L3 at LAR(+LBR) and put the data on the wire.
///
/// ⛔ THE UNIT IS THE **STORE** HALF AND THE OPCODE IS `ST` — the reference DT_CHECKs `comp == L3SU`
/// (`:2580`), so the component is a constant here rather than an argument.
/// ⛔ `full_reg_init` FOLDS THE REFERENCE'S THREE CONDITIONS (`:2593`): prog stitching, the
/// `-force-full-reg-init` flag, and collecting code-quality stats.
/// ⭐ THE TRAILING `llvm_unreachable` (`:2669`) IS DEAD: a multicast and a dynamic peer are one
/// locale, so [`LoadTarget`]'s two arms are the whole space.
#[must_use]
pub fn construct_l3_load_instr<A: Arch>(
    load: &L3Load,
    units: &[UnitKey],
    full_reg_init: bool,
    regs_to_init: &mut RegsToInit,
) -> UniformInstrInfo {
    let sen1p5 = matches!(A::GEN, IsaGen::Sen1p5);
    if full_reg_init {
        if let LoadTarget::Multicast { group, .. } = load.target {
            let gtr = Reg {
                locale: SenRegType::Gtr,
                index: Some(group),
            };
            add_to_regs_to_init(units, &[gtr], regs_to_init);
        }
        if sen1p5 {
            add_to_regs_to_init(units, &[load.mutable_addr], regs_to_init);
        } else {
            add_to_regs_to_init(
                units,
                &[load.mutable_addr, load.immutable_addr, load.result],
                regs_to_init,
            );
        }
    }
    let mut instr = UniformInstrInfo::of(match (&load.target, load.update) {
        (LoadTarget::Multicast { .. }, false) => OpCode::STG,
        (LoadTarget::Multicast { .. }, true) => OpCode::STGU,
        (LoadTarget::Peer { .. }, false) => OpCode::ST,
        (LoadTarget::Peer { .. }, true) => OpCode::STU,
    });
    if let Some(name) = &load.dbg_name {
        instr = instr.with_common_comment(name);
    }
    instr.set_common_field(OperandField::Src0, reg_field(load.mutable_addr));
    if sen1p5 {
        // On SEN1P5 the routing direction is the DRM, and the LBR is gone (`:2617-2636`).
        if let Some(dir) = load.target.dir() {
            instr.set_common_field(OperandField::Drm, int(i64::from(dir.encoding())));
        }
    } else {
        instr.set_common_field(OperandField::Src1, reg_field(load.immutable_addr));
    }
    instr.set_common_field(
        OperandField::Burst,
        int(normalize_burst_size::<A>(load.burst, Component::L3su).0 as i64),
    );
    match &load.target {
        LoadTarget::Multicast { group, dir } => {
            instr.set_common_field(OperandField::Group, int(i64::from(group.get())));
            instr.set_common_field(OperandField::Node, int(i64::from(dir.encoding())));
        }
        LoadTarget::Peer { peer, .. } => {
            instr.set_common_field(OperandField::Group, int(0));
            set_node(&mut instr, peer);
        }
    }
    instr
}

/// THE CONSTANT AN `LDZ` BROADCASTS — ⛔ CARRIED AT THE VALUE'S OWN WIDTH, which is what makes
/// `Int64ToInt<n>BinToInt`'s "no bits above the width" DT_CHECK and the 4-bit splat's *"LDZ imms are
/// unsigned"* hold by construction (`util/sendefs/numeric_convert.cpp:429-445`).
///
/// ⛔⛔ `int16` AND `fp4` HAVE NO ROW in `getDataFormatFromSentientPrecisionAttr`
/// (`Dialect/Sentient/Utils.cpp:171-193` — int16 is commented out, *"no hw support"*), so a 16-bit LDZ
/// is fp16 and a 4-bit one is int4. Both other combinations are an `llvm_unreachable` there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdzConst {
    /// `int4` — splatted to fill the eight-bit imm field.
    Int4(Bounded<16>),
    /// `int8`.
    Int8(u8),
    /// `fp8`.
    Fp8(u8),
    /// `fp16` — ⛔ THE VALUE COMES FROM THE ZR, not the imm field.
    Fp16(u16),
}

impl LdzConst {
    /// What the imm field interprets — `Int64ToInt<n>BinToInt`, whose mask this type already is.
    #[must_use]
    pub const fn interpreted(self) -> i64 {
        match self {
            LdzConst::Int4(value) => value.get() as i64,
            LdzConst::Int8(value) | LdzConst::Fp8(value) => value as i64,
            LdzConst::Fp16(value) => value as i64,
        }
    }

    /// `getDataFormatFromSentientPrecisionAttr("<int|fp><bitwidth>")` (`:2782-2795`).
    #[must_use]
    pub const fn format(self) -> DataFormat {
        match self {
            LdzConst::Int4(_) => DataFormat::Senint4,
            LdzConst::Int8(_) => DataFormat::Senint8,
            LdzConst::Fp8(_) => DataFormat::Sen143Fp8,
            LdzConst::Fp16(_) => DataFormat::Sen169Fp16,
        }
    }
}

/// WHERE AN L3 STORE'S DATA COMES FROM.
///
/// ⛔⛔ THIS ENUM **IS** THE DT_CHECK *"multicast not supported for constant producers"* (`:2689`): a
/// constant carries no group, no node and no peer, and the wire carries no immediate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum L3StoreSource {
    /// A `sentient.constant` producer — the `Z` in the opcode.
    Constant(LdzConst),
    /// The wire.
    Wire {
        /// Who sends.
        peer: Peer,
        /// The GTR holding the multicast group, where there is one — ⛔ `gtr` BY THE REFERENCE'S OWN
        /// DT_CHECK on the locale (`:2683-2686`).
        multicast: Option<RegIndex>,
    },
}

/// ONE `sentient.receive_and_store` WITH ITS REGISTERS ALREADY RESOLVED — see [`L3Load`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L3Store {
    /// `mutable_addr` — the LAR, which is `src0`.
    pub mutable_addr: Reg,
    /// `immutable_addr` — the LBR. ⛔ ON SEN1P5 IT MUST BE THE CONSTANT 0 and there is no field.
    pub immutable_addr: Reg,
    /// `result`.
    pub result: Reg,
    /// Where the data comes from.
    pub source: L3StoreSource,
    /// `burst_size`, before normalisation.
    pub burst: Elements,
    /// `isUpdateMode` — the `U` in the opcode.
    pub update: bool,
    /// `dbgName`.
    pub dbg_name: Option<String>,
}

/// Replaces: e055_ConstructL3StoreInstr
///
/// Take a wire's data — or one immediate — and write it into the L3 at LAR(+LBR).
///
/// ⛔ THE UNIT IS THE **LOAD** HALF AND THE OPCODE IS `LD` — `comp == L3LU` is DT_CHECKed (`:2676`).
/// ⛔ A `$dst` IS REFUSED HERE (`:2687`): an L0 scale destination is not an L3 unit's business, so
/// [`L3Store`] has no such field.
/// ⛔ AN `LDZ` SETS NEITHER `node` NOR `group` — the constant arm returns before both.
#[must_use]
pub fn construct_l3_store_instr<A: Arch>(
    store: &L3Store,
    units: &[UnitKey],
    full_reg_init: bool,
    regs_to_init: &mut RegsToInit,
) -> UniformInstrInfo {
    let sen1p5 = matches!(A::GEN, IsaGen::Sen1p5);
    if full_reg_init {
        if let L3StoreSource::Wire {
            multicast: Some(group),
            ..
        } = store.source
        {
            let gtr = Reg {
                locale: SenRegType::Gtr,
                index: Some(group),
            };
            add_to_regs_to_init(units, &[gtr], regs_to_init);
        }
        if sen1p5 {
            add_to_regs_to_init(units, &[store.mutable_addr], regs_to_init);
        } else {
            add_to_regs_to_init(
                units,
                &[store.mutable_addr, store.immutable_addr, store.result],
                regs_to_init,
            );
        }
    }
    let mut instr = UniformInstrInfo::of(match (&store.source, store.update) {
        (L3StoreSource::Constant(_), false) => OpCode::LDZ,
        (L3StoreSource::Constant(_), true) => OpCode::LDZU,
        (
            L3StoreSource::Wire {
                multicast: None, ..
            },
            false,
        ) => OpCode::LD,
        (
            L3StoreSource::Wire {
                multicast: None, ..
            },
            true,
        ) => OpCode::LDU,
        (
            L3StoreSource::Wire {
                multicast: Some(_), ..
            },
            false,
        ) => OpCode::LDG,
        (
            L3StoreSource::Wire {
                multicast: Some(_), ..
            },
            true,
        ) => OpCode::LDGU,
    });
    if let Some(name) = &store.dbg_name {
        instr = instr.with_common_comment(name);
    }
    instr.set_common_field(OperandField::Src0, reg_field(store.mutable_addr));
    if !sen1p5 {
        instr.set_common_field(OperandField::Src1, reg_field(store.immutable_addr));
    }
    instr.set_common_field(
        OperandField::Burst,
        int(normalize_burst_size::<A>(store.burst, Component::L3lu).0 as i64),
    );
    match &store.source {
        L3StoreSource::Constant(value) => {
            let interpreted = value.interpreted();
            let (mode, imm) = if interpreted == 0 {
                ("zero2lx", Some(0))
            } else {
                match value {
                    LdzConst::Int8(_) | LdzConst::Fp8(_) => ("imm2lx", Some(interpreted)),
                    // ⛔ A NON-ZERO 16-BIT LDZ TAKES NO IMM: the value comes from the ZR (`:2761-2765`).
                    LdzConst::Fp16(_) => ("zr2lx", None),
                    // Four bits splatted to fill `min_ldz_imm_field_bits` = 8 (`:2766-2779`).
                    LdzConst::Int4(_) => ("imm2lx", Some(interpreted | (interpreted << 4))),
                }
            };
            instr.set_common_field(OperandField::Mode, descriptive(mode));
            if let Some(imm) = imm {
                instr.set_common_field(OperandField::Imm, int(imm));
            }
            // The LDZ also tells the senulator how to read what it wrote (`:2781-2795`).
            instr.set_common_field(
                OperandField::DatatypeVirtual,
                int(1i64 << value.format().encoding()),
            );
        }
        L3StoreSource::Wire { peer, multicast } => {
            set_node(&mut instr, peer);
            instr.set_common_field(
                OperandField::Group,
                int(multicast.map_or(0, |group| i64::from(group.get()))),
            );
        }
    }
    instr
}

/// Replaces: e056_ConstructZRAssignInstr
///
/// Put one 16-bit immediate in the L3 load unit's ZR, which a `zr2lx` LDZ then broadcasts.
///
/// ⛔ THE UNIT IS DT_CHECKED `L3LU` (`:2833`), so the component is a constant here.
/// ⛔ SIXTEEN BITS BY THE TYPE — `Int64ToInt16BinToInt` DT_CHECKs anything above them.
#[must_use]
pub fn construct_zr_assign_instr(imm: u16) -> UniformInstrInfo {
    let mut instr = UniformInstrInfo::of(OpCode::LDZimm16);
    instr.set_common_field(OperandField::Imm, int(i64::from(imm)));
    instr.set_common_field(OperandField::Mode, descriptive("imm2zr"));
    instr
}

/// WHICH HALF OF A FUSED L3 TRANSFER THIS UNIT IS — the reference's `comp`, DT_CHECKed to the two
/// (`:2846`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L3Half {
    /// `L3LU` — an `LD…`.
    Load {
        /// The GTR holding the multicast group, where a `sentient.copy` names one — ⛔ `gtr` BY THE
        /// REFERENCE'S OWN DT_CHECK on the locale (`:2851-2854`).
        multicast: Option<RegIndex>,
    },
    /// `L3SU` — an `ST…`.
    ///
    /// ⛔⛔ NO MULTICAST ARM HERE, AND THAT IS A GUARD: the reference appends `G` on either half
    /// (`:2299`), but the vendored ISA has no `STGM`/`STGMU`/`STIGM`/`STIGMU`, so
    /// `Isa::to_instopcode` could not answer for one. A store multicast cannot be written down, and
    /// with it go this half's `gtr` reg-init and its non-zero `group`.
    Store {
        /// The destination is the QGI, which makes this an `STZ` with `ibr:0`.
        dst_is_qgi: bool,
    },
}

/// WHAT `L3GatherScatterChecker` FOUND (`Transform/Sentient/Analyses/Utils.cpp:608-633`, an analysis
/// outside this campaign's 130 — so it is an input here, not a call).
///
/// ⛔ FOUR MUTUALLY EXCLUSIVE STATES, which is exactly what its two DT_CHECKs assert: neither
/// gather-and-scatter nor ibr-read-and-ibr-write can be written down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L3Indirection {
    /// Neither — `isValid()` is false.
    Direct,
    /// `L3IBR -> LX`: read `ear + ibr[jcr]`, write the LX.
    Gather,
    /// `LX -> L3IBR`: read the LX, write `ear + ibr[jcr]`.
    Scatter,
    /// Fill the IBR — an `STZ` with `ibr:1` on the store half, and its own field layout on the load one.
    IbrWrite,
}

impl L3Indirection {
    /// `isIBRRead()` — a gather or a scatter (`Utils.hpp:193`).
    const fn ibr_read(self) -> bool {
        matches!(self, L3Indirection::Gather | L3Indirection::Scatter)
    }

    /// `isValid()` — any of the three (`Utils.hpp:201`), which is also the `I` in the opcode.
    const fn valid(self) -> bool {
        !matches!(self, L3Indirection::Direct)
    }
}

/// ONE `sentient.load_and_store` WITH ITS REGISTERS ALREADY RESOLVED — see [`L3Load`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L3LoadAndStore {
    /// Which half is being emitted.
    pub half: L3Half,
    /// What the gather/scatter checker said about the op.
    pub indirection: L3Indirection,
    /// `src_mutable_addr` — a `lar` or an `ear`, and ⛔ WHICH ONE DECIDES THE WHOLE FIELD LAYOUT.
    pub src_mutable_addr: Reg,
    /// `src_immutable_addr` — the matching `lbr`, `ebr` or `jcr`.
    pub src_immutable_addr: Reg,
    /// `dst_mutable_addr`.
    pub dst_mutable_addr: Reg,
    /// `dst_immutable_addr`.
    pub dst_immutable_addr: Reg,
    /// Both results.
    pub results: [Reg; 2],
    /// `isLXUnit(src)` — ⛔ ON SEN1P5 AN LX ADDRESS HAS NO LBR, and the reference DT_CHECKs that its
    /// immutable address is the constant 0 (`:2243-2251`).
    pub src_is_lx: bool,
    /// `isLXUnit(dst)`.
    pub dst_is_lx: bool,
    /// `$dir` — what SEN1P5's `drm` carries in place of the LBR.
    pub dir: Option<RoutingDirection>,
    /// `burst_size`, before normalisation.
    pub burst: Elements,
    /// `src_inc != 0` — ⛔ ONE INCREMENT FOR BOTH SIDES BY THE ISA, which the reference DT_CHECKs
    /// (`:2311-2313`), so this is the pair's single shared answer.
    pub update: bool,
    /// `dbgName`.
    pub dbg_name: Option<String>,
}

/// Replaces: e057_ConstructL3LoadAndStoreInstr
///
/// The fused L3 read-and-write: one half's instruction, its four address registers, and the IBR
/// indirection they index through.
///
/// ⛔ `STZ` TAKES NO `U`, NO `burst` AND NO `group` — the update suffix and both fields sit inside the
/// `else` (`:2295-2312`, `:2385-2390`).
/// ⛔ THE `readibr` GUARD IS ASYMMETRIC: the LAR-side arm tests `comp == L3LU` (`:2340`), the EAR-side
/// one does not (`:2371`), so an EAR-side STORE still writes `readibr:1`.
/// ⛔ `src1` IS SKIPPED, NOT ZEROED, wherever the LBR is gone — and the reg-init list drops it too.
#[must_use]
pub fn construct_l3_load_and_store_instr<A: Arch>(
    ls: &L3LoadAndStore,
    units: &[UnitKey],
    full_reg_init: bool,
    regs_to_init: &mut RegsToInit,
) -> UniformInstrInfo {
    let sen1p5 = matches!(A::GEN, IsaGen::Sen1p5);
    let skip_src_lbr = sen1p5 && ls.src_is_lx;
    let skip_dst_lbr = sen1p5 && ls.dst_is_lx;
    let store = matches!(ls.half, L3Half::Store { .. });
    let multicast = match ls.half {
        L3Half::Load { multicast } => multicast,
        L3Half::Store { .. } => None,
    };
    if full_reg_init {
        if let Some(group) = multicast {
            let gtr = Reg {
                locale: SenRegType::Gtr,
                index: Some(group),
            };
            add_to_regs_to_init(units, &[gtr], regs_to_init);
        }
        if skip_src_lbr {
            add_to_regs_to_init(
                units,
                &[
                    ls.src_mutable_addr,
                    ls.dst_mutable_addr,
                    ls.dst_immutable_addr,
                ],
                regs_to_init,
            );
        } else if skip_dst_lbr {
            add_to_regs_to_init(
                units,
                &[
                    ls.src_mutable_addr,
                    ls.src_immutable_addr,
                    ls.dst_mutable_addr,
                ],
                regs_to_init,
            );
        } else {
            add_to_regs_to_init(
                units,
                &[
                    ls.src_mutable_addr,
                    ls.src_immutable_addr,
                    ls.dst_mutable_addr,
                    ls.dst_immutable_addr,
                    ls.results[0],
                    ls.results[1],
                ],
                regs_to_init,
            );
        }
    }
    // `is_stz` AND THE `ibr` BIT IT SETS ARE ONE FACT (`:2286-2290`, `:2333-2335`): an IBR write
    // pushes `ibr:1`, an LX->QGI transfer `ibr:0`, and the QGI wins where both hold.
    let stz_ibr = match ls.half {
        L3Half::Store { dst_is_qgi: true } => Some(0),
        L3Half::Store { dst_is_qgi: false } => match ls.indirection {
            L3Indirection::IbrWrite => Some(1),
            _ => None,
        },
        L3Half::Load { .. } => None,
    };
    let mut instr = UniformInstrInfo::of(
        match (stz_ibr, ls.half, ls.indirection.valid(), ls.update) {
            (Some(_), ..) => OpCode::STZ,
            (None, L3Half::Load { multicast: None }, false, false) => OpCode::LDM,
            (None, L3Half::Load { multicast: None }, false, true) => OpCode::LDMU,
            (None, L3Half::Load { multicast: None }, true, false) => OpCode::LDIM,
            (None, L3Half::Load { multicast: None }, true, true) => OpCode::LDIMU,
            (None, L3Half::Load { multicast: Some(_) }, false, false) => OpCode::LDGM,
            (None, L3Half::Load { multicast: Some(_) }, false, true) => OpCode::LDGMU,
            (None, L3Half::Load { multicast: Some(_) }, true, false) => OpCode::LDIGM,
            (None, L3Half::Load { multicast: Some(_) }, true, true) => OpCode::LDIGMU,
            (None, L3Half::Store { .. }, false, false) => OpCode::STM,
            (None, L3Half::Store { .. }, false, true) => OpCode::STMU,
            (None, L3Half::Store { .. }, true, false) => OpCode::STIM,
            (None, L3Half::Store { .. }, true, true) => OpCode::STIMU,
        },
    );
    if let Some(name) = &ls.dbg_name {
        instr = instr.with_common_comment(name);
    }
    if !store && matches!(ls.indirection, L3Indirection::IbrWrite) {
        // The LAR and LBR are unused, but the senulator treats the LAR as updated (`:2320-2331`).
        instr.set_common_field(OperandField::Readibr, int(0));
        instr.set_common_field(OperandField::Src0, reg_field(ls.dst_mutable_addr));
        instr.set_common_field(OperandField::Src2, reg_field(ls.src_mutable_addr));
        instr.set_common_field(OperandField::Src3, reg_field(ls.src_immutable_addr));
    } else if let Some(ibr) = stz_ibr {
        instr.set_common_field(OperandField::Ibr, int(ibr));
        instr.set_common_field(OperandField::Src0, reg_field(ls.src_mutable_addr));
        if !skip_src_lbr {
            instr.set_common_field(OperandField::Src1, reg_field(ls.src_immutable_addr));
        }
    } else {
        // The LAR side is `src0`+`src1` and the EAR side `src2`+`src3`, whichever end holds which
        // (`:2337-2381`) — the reference's two arms differ in nothing else.
        let lar_is_src = matches!(ls.src_mutable_addr.locale, SenRegType::Lar);
        let (lar, lbr, ear, ebr, skip_lbr) = if lar_is_src {
            (
                ls.src_mutable_addr,
                ls.src_immutable_addr,
                ls.dst_mutable_addr,
                ls.dst_immutable_addr,
                skip_src_lbr,
            )
        } else {
            (
                ls.dst_mutable_addr,
                ls.dst_immutable_addr,
                ls.src_mutable_addr,
                ls.src_immutable_addr,
                skip_dst_lbr,
            )
        };
        if ls.indirection.ibr_read() && (!store || !lar_is_src) {
            instr.set_common_field(OperandField::Readibr, int(1));
        }
        instr.set_common_field(OperandField::Src0, reg_field(lar));
        if skip_lbr {
            // For SEN1P5 the routing direction is the DRM (`:2347-2352`).
            if let Some(dir) = ls.dir {
                instr.set_common_field(OperandField::Drm, int(i64::from(dir.encoding())));
            }
        } else {
            instr.set_common_field(OperandField::Src1, reg_field(lbr));
        }
        instr.set_common_field(OperandField::Src2, reg_field(ear));
        instr.set_common_field(OperandField::Src3, reg_field(ebr));
    }
    if stz_ibr.is_none() {
        let comp = if store {
            Component::L3su
        } else {
            Component::L3lu
        };
        instr.set_common_field(
            OperandField::Burst,
            int(normalize_burst_size::<A>(ls.burst, comp).0 as i64),
        );
        if !ls.indirection.valid() || !store {
            instr.set_common_field(
                OperandField::Group,
                int(multicast.map_or(0, |group| i64::from(group.get()))),
            );
        }
    }
    instr
}

/// `shuffle_mode` — the four an `LDCVTI` spells (`:2464-2478`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadComputeShuffle {
    /// `noshuffle` — a plain 128-bit beat.
    NoShuffle,
    /// `splat2b`.
    Splat2b,
    /// `splat4b`.
    Splat4b,
    /// `splat16b`.
    Splat16b,
}

impl LoadComputeShuffle {
    /// The `ldtype` spelling — ⛔ `noshuffle` IS SPELLED `128b`, not by its own name.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            LoadComputeShuffle::NoShuffle => "128b",
            LoadComputeShuffle::Splat2b => "2bsplat",
            LoadComputeShuffle::Splat4b => "4bsplat",
            LoadComputeShuffle::Splat16b => "16bsplat",
        }
    }
}

/// WHO AN `LDCVTI` SENDS TO — the two `dyn_cast`s of its consumer (`:2480-2492`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadComputeConsumer<'a> {
    /// A `dataflow.get_unit` consumer — one common `consumertag`.
    Unit(LoadConsumer),
    /// A `uniform.query_map` consumer — one tag per unit.
    Mapped(&'a [MappedEntry]),
}

/// ONE `sentient.load_compute_and_send` WITH ITS REGISTERS ALREADY RESOLVED — see [`L3Load`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadCompute<'a> {
    /// `mutable_addr` — ⛔ `lrf` BY DT_CHECK (`:2405-2408`), and it is NOT a field of the instruction.
    pub mutable_addr: Reg,
    /// `immutable_addr` — `imm` by the same DT_CHECK; its value becomes `imm`.
    pub immutable_addr: Reg,
    /// `result` — which is `src0`.
    pub result: Reg,
    /// The constant `immutable_addr` holds, in source elements.
    pub immutable_value: i64,
    /// `element_index` — ⛔ FOUR BY THE TYPE (`:2447-2450`).
    pub element_index: Bounded<4>,
    /// `scale_index` — two by the type (`:2457`).
    pub scale_index: Bounded<2>,
    /// `shuffle_mode`.
    pub shuffle: LoadComputeShuffle,
    /// `consumer`.
    pub consumer: LoadComputeConsumer<'a>,
    /// `isUpdateMode` — the `U` in the opcode.
    pub update: bool,
    /// `dbgName`.
    pub dbg_name: Option<String>,
}

/// `src_element_size` — ⛔ FOUR BY DT_CHECK (`:2447-2449`), together with a 16-bit destination: *"current
/// support is only for 4 bit/256 elems to 16 bit/64 elems"*. So it is stated here, not passed.
const LDCVTI_SRC_BITS: u32 = 4;

/// Replaces: e058_ConstructLoadComputeInstr
///
/// Read 4-bit elements out of the LX, convert them to 16-bit ones and send them to one consumer.
///
/// ⛔ SEN1P5 AND `LXLU` ONLY, both DT_CHECKed (`:2398`), so neither is an argument.
/// ⛔ THE `imm` IS IN **DESTINATION** UNITS: the source's 4 bits over 8 halve it (`:2436-2441`), which
/// is why IBM's own 256-element load writes `imm:128`.
/// ⛔ AND A MAPPED CONSUMER **ASSIGNS** THE WHOLE OPERAND MAP (`UniformInstrAndBlock.hpp:133`) rather
/// than adding to it.
pub fn construct_load_compute_instr<A: Arch>(
    load: &LoadCompute<'_>,
    units: &[UnitKey],
    full_reg_init: bool,
    regs_to_init: &mut RegsToInit,
) -> (UniformInstrInfo, Vec<OperandMapRefusal>) {
    if full_reg_init {
        add_to_regs_to_init(
            units,
            &[load.mutable_addr, load.immutable_addr, load.result],
            regs_to_init,
        );
    }
    let mut instr = UniformInstrInfo::of(if load.update {
        OpCode::LDCVTIU
    } else {
        OpCode::LDCVTI
    });
    if let Some(name) = &load.dbg_name {
        instr = instr.with_common_comment(name);
    }
    let scale = address_scale::<A>(Component::Lxlu).get();
    let imm = load.immutable_value as f64 * f64::from(LDCVTI_SRC_BITS) / 8.0 / f64::from(scale);
    instr.set_common_field(OperandField::Imm, int(imm as i64));
    instr.set_common_field(OperandField::Src0, reg_field(load.result));
    instr.set_common_field(
        OperandField::Elemidx,
        int(i64::from(load.element_index.get())),
    );
    instr.set_common_field(
        OperandField::Scaleidx,
        int(i64::from(load.scale_index.get())),
    );
    instr.set_common_field(OperandField::Ldtype, descriptive(load.shuffle.spelling()));
    let mut refused = Vec::new();
    match &load.consumer {
        LoadComputeConsumer::Unit(consumer) => {
            instr.set_common_field(OperandField::Consumertag, proper_consumer::<A>(*consumer));
        }
        LoadComputeConsumer::Mapped(entries) => {
            let mapped = add_entry_to_operand_map(entries, MapMode::UnitName, 1.0);
            refused = mapped.refused;
            instr.operand_map = mapped
                .value
                .into_iter()
                .map(|per_unit| (OperandField::Consumertag, per_unit))
                .collect();
        }
    }
    (instr, refused)
}

/// ONE `sentient.receive_and_extract_scalar` WITH ITS REGISTERS ALREADY RESOLVED.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LrfCopy {
    /// `result` — the LRF the scalar lands in, which is `src0`.
    pub result: Reg,
    /// `unit` — who sent the vector.
    pub producer: DfirUnit,
    /// `position` — which lane of it.
    pub position: i64,
    /// `dbgName`.
    pub dbg_name: Option<String>,
}

/// WHAT AN LRF COPY COULD NOT BE GIVEN — the two `signalPassFailure`s (`:2523`, `:2535`), as offenders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LrfCopyRefusal {
    /// A producer outside `{SFP, PE, LXLU}`.
    UnsupportedUnit(GenericComp),
    /// Any lane but 0.
    UnsupportedPosition(i64),
}

/// Replaces: e059_ConstructLRFCopyInstr
///
/// Copy one lane of an arriving vector into the LX store unit's LRF.
///
/// ⛔ THE UNIT IS DT_CHECKED `LXSU` (`:2500`), so the component is a constant here.
/// ⛔ THE REFERENCE CARRIES ON PAST BOTH FAILURES and still writes a `producertag` for the producer it
/// just rejected; only the three admitted units have a spelling here, so an unsupported one leaves the
/// field unset rather than naming something the ISA does not accept.
pub fn construct_lrf_copy_instr(
    copy: &LrfCopy,
    units: &[UnitKey],
    full_reg_init: bool,
    regs_to_init: &mut RegsToInit,
) -> (UniformInstrInfo, Vec<LrfCopyRefusal>) {
    if full_reg_init {
        add_to_regs_to_init(units, &[copy.result], regs_to_init);
    }
    let mut refused = Vec::new();
    let producer = match copy.producer.generic() {
        GenericComp::Sfp => Some("sfp"),
        GenericComp::Pe => Some("pe"),
        GenericComp::Lxlu => Some("lxlu"),
        other => {
            refused.push(LrfCopyRefusal::UnsupportedUnit(other));
            None
        }
    };
    if copy.position != 0 {
        refused.push(LrfCopyRefusal::UnsupportedPosition(copy.position));
    }
    let mut instr = UniformInstrInfo::of(OpCode::LRFCOPY);
    if let Some(name) = &copy.dbg_name {
        instr = instr.with_common_comment(name);
    }
    instr.set_common_field(OperandField::Src0, reg_field(copy.result));
    if let Some(producer) = producer {
        instr.set_common_field(OperandField::Producertag, descriptive(producer));
    }
    (instr, refused)
}

#[cfg(test)]
mod unit_tests {
    use super::{
        Component, Elements, L3Half, L3Indirection, L3Load, L3LoadAndStore, L3Store, L3StoreSource,
        LdzConst, LoadCompute, LoadComputeConsumer, LoadComputeShuffle, LoadTarget, LrfCopy,
        LrfCopyRefusal, Peer, construct_l3_load_and_store_instr, construct_l3_load_instr,
        construct_l3_store_instr, construct_load_compute_instr, construct_lrf_copy_instr,
        construct_zr_assign_instr, normalize_burst_size,
    };
    use crate::arch::{Arch, Bounded, Dd2, Sen1p5, Target};
    use crate::bridges::sentient_to_progir::construct::{descriptive, int};
    use crate::bridges::sentient_to_progir::state::RegsToInit;
    use crate::bridges::sentient_to_progir::utils::{ConsumerUnit, LoadConsumer};
    use crate::islands::dataflow_ir::ty::GenericComp;
    use crate::islands::progir::{OpCode, OperandField};
    use crate::islands::sentient::dialects::sentient::{
        Reg, RegIndex, RegType as SenRegType, RoutingDirection,
    };
    use crate::units::{Core, DfirUnit};

    /// The LAR/LBR pair and the unassigned result every L3 transfer in IBM's tests carries.
    fn addrs() -> (Reg, Reg, Reg) {
        let at0 = |locale| Reg {
            locale,
            index: Some(RegIndex::at::<0>()),
        };
        (
            at0(SenRegType::Lar),
            at0(SenRegType::Lbr),
            Reg {
                locale: SenRegType::Unknown,
                index: None,
            },
        )
    }

    /// ⭐⭐ IBM'S OWN TWO CHECK LINES, AND THEY SHARE ONE FIELD: `node` IS THE CONSUMER'S CORE ON A
    /// UNICAST AND THE ROUTING DIRECTION ON A MULTICAST.
    ///
    /// `L3_STU :: burst:2 group:0 node:20 src0:0 src1:0` (`L3/core2core-unicast-e2e.mlir:12`) and
    /// `L3_STGU :: be:be burst:2 group:0 node:1 src0:0 src1:0` with `dir = CounterClockwise`
    /// (`L3/core2core-multicast-simple-e2e.mlir:10`) — ⛔ AND `group:0` IS THE **GTR INDEX**, not the
    /// group id, on the multicast too.
    #[test]
    fn the_node_field_is_a_core_on_a_unicast_and_a_direction_on_a_multicast() {
        let (mutable_addr, immutable_addr, result) = addrs();
        let load = |target| L3Load {
            mutable_addr,
            immutable_addr,
            result,
            target,
            burst: Elements(2),
            update: true,
            dbg_name: None,
        };
        let unicast = construct_l3_load_instr::<Dd2>(
            &load(LoadTarget::Peer {
                peer: Peer::Core(Core::checked(20).expect("this arch has core 20")),
                dir: None,
            }),
            &[],
            false,
            &mut RegsToInit::new(),
        );
        assert_eq!(unicast.opcode, OpCode::STU);
        assert_eq!(
            unicast.common_fields,
            vec![
                (OperandField::Burst, int(2)),
                (OperandField::Group, int(0)),
                (OperandField::Node, int(20)),
                (OperandField::Src0, int(0)),
                (OperandField::Src1, int(0)),
            ]
        );
        let multicast = construct_l3_load_instr::<Dd2>(
            &load(LoadTarget::Multicast {
                group: RegIndex::at::<0>(),
                dir: RoutingDirection::CounterClockwise,
            }),
            &[],
            false,
            &mut RegsToInit::new(),
        );
        assert_eq!(multicast.opcode, OpCode::STGU);
        assert_eq!(
            multicast.common_fields,
            vec![
                (OperandField::Burst, int(2)),
                (OperandField::Group, int(0)),
                (OperandField::Node, int(1)),
                (OperandField::Src0, int(0)),
                (OperandField::Src1, int(0)),
            ]
        );
    }

    /// ⭐⭐ IBM'S THREE `LDZ` LINES, WHICH ARE THREE DIFFERENT MODES OF ONE OPCODE.
    ///
    /// `L3_LDZ :: burst:1 datatype_virtual:1 mode:zr2lx src0:0 src1:0`,
    /// `L3_LDZ :: burst:1 datatype_virtual:256 imm:128 mode:imm2lx src0:0 src1:0` and
    /// `L3_LDZ :: burst:1 datatype_virtual:256 imm:0 mode:zero2lx src0:0 src1:0`
    /// (`L3/ldz.mlir:10,36,58`) — ⛔ A 16-BIT CONSTANT CARRIES **NO** `imm`, and `datatype_virtual`
    /// is `1 << 8` for int8, which a discriminant taken from Rust's own ordering would make 128.
    #[test]
    fn each_ldz_mode_is_the_constants_own_width() {
        let (mutable_addr, immutable_addr, result) = addrs();
        let ldz = |value| {
            construct_l3_store_instr::<Dd2>(
                &L3Store {
                    mutable_addr,
                    immutable_addr,
                    result,
                    source: L3StoreSource::Constant(value),
                    burst: Elements(1),
                    update: false,
                    dbg_name: None,
                },
                &[],
                false,
                &mut RegsToInit::new(),
            )
        };
        let from_the_zr = ldz(LdzConst::Fp16(178));
        assert_eq!(from_the_zr.opcode, OpCode::LDZ);
        assert_eq!(
            from_the_zr.common_fields,
            vec![
                (OperandField::Burst, int(1)),
                (OperandField::DatatypeVirtual, int(1)),
                (OperandField::Mode, descriptive("zr2lx")),
                (OperandField::Src0, int(0)),
                (OperandField::Src1, int(0)),
            ]
        );
        assert_eq!(
            ldz(LdzConst::Int8(128)).common_fields,
            vec![
                (OperandField::Burst, int(1)),
                (OperandField::DatatypeVirtual, int(256)),
                (OperandField::Imm, int(128)),
                (OperandField::Mode, descriptive("imm2lx")),
                (OperandField::Src0, int(0)),
                (OperandField::Src1, int(0)),
            ]
        );
        assert_eq!(
            ldz(LdzConst::Int8(0)).common_fields,
            vec![
                (OperandField::Burst, int(1)),
                (OperandField::DatatypeVirtual, int(256)),
                (OperandField::Imm, int(0)),
                (OperandField::Mode, descriptive("zero2lx")),
                (OperandField::Src0, int(0)),
                (OperandField::Src1, int(0)),
            ]
        );
    }

    /// ⭐ `L3_LDZimm16 :: imm:178 mode:imm2zr` (`L3/ldz.mlir:9`).
    #[test]
    fn the_zr_assignment_is_one_immediate_and_its_mode() {
        let instr = construct_zr_assign_instr(178);
        assert_eq!(instr.opcode, OpCode::LDZimm16);
        assert_eq!(
            instr.common_fields,
            vec![
                (OperandField::Imm, int(178)),
                (OperandField::Mode, descriptive("imm2zr")),
            ]
        );
    }

    /// THE THREE ARMS AND THE UNIT THAT HAS NO MAXIMUM.
    ///
    /// ⛔ THE WRAP IS PER COMPONENT: 32 is the L3's maximum and wraps to 0, but on the LX the same 32
    /// is an ordinary size that passes through — one number, two answers.
    #[test]
    fn the_maximum_wraps_to_zero_and_zero_means_one() {
        let l3 = |n| normalize_burst_size::<Target>(Elements(n), Component::L3lu);
        assert_eq!(l3(u64::from(Target::L3_BURST)), Elements(0));
        assert_eq!(l3(0), Elements(1));
        assert_eq!(l3(7), Elements(7));
        assert_eq!(
            normalize_burst_size::<Target>(Elements(u64::from(Target::L3_BURST)), Component::Lxlu),
            Elements(u64::from(Target::L3_BURST))
        );
        // A compute unit has no maximum, so only the zero arm applies.
        assert_eq!(
            normalize_burst_size::<Target>(Elements(0), Component::Pt),
            Elements(1)
        );
    }

    /// ⭐⭐ IBM'S THREE FUSED-TRANSFER LINES, WHICH ARE THREE FIELD LAYOUTS OF ONE OP.
    ///
    /// `L3_LDGM :: burst:1 group:0 src0:0 src1:0 src2:0 src3:0`,
    /// `L3_LDIGM :: burst:1 group:0 readibr:0 src0:1 src2:0 src3:2` and
    /// `L3_LDIM :: burst:1 group:0 readibr:1 src0:1 src1:0 src2:0 src3:0`
    /// (`symbolic_ebr.mlir:44,47,56`) — ⛔ AN IBR WRITE HAS **NO** `src1` AT ALL, and it writes
    /// `readibr:0` where the ordinary gather writes `readibr:1`.
    #[test]
    fn the_fused_transfers_field_layout_follows_the_indirection() {
        // ⛔ THE TABLE, NOT A `checked(u32)`: `RegIndex::at` is const-generic on purpose and the
        // runtime constructor was deliberately removed (`sentient.rs:1461-1478`).
        let reg = |locale, index: usize| Reg {
            locale,
            index: Some(RegIndex::ALL[index]),
        };
        let unassigned = Reg {
            locale: SenRegType::Unknown,
            index: None,
        };
        let fused = |half, indirection, src_immutable_addr, dst_mutable_addr| L3LoadAndStore {
            half,
            indirection,
            src_mutable_addr: reg(SenRegType::Ear, 0),
            src_immutable_addr,
            dst_mutable_addr,
            dst_immutable_addr: reg(SenRegType::Lbr, 0),
            results: [unassigned, unassigned],
            src_is_lx: false,
            dst_is_lx: false,
            dir: None,
            burst: Elements(1),
            update: false,
            dbg_name: None,
        };
        let emit = |ls: &L3LoadAndStore| {
            construct_l3_load_and_store_instr::<Dd2>(ls, &[], false, &mut RegsToInit::new())
        };
        let multicast = L3Half::Load {
            multicast: Some(RegIndex::at::<0>()),
        };
        let direct = emit(&fused(
            multicast,
            L3Indirection::Direct,
            reg(SenRegType::Ebr, 0),
            reg(SenRegType::Lar, 0),
        ));
        assert_eq!(direct.opcode, OpCode::LDGM);
        assert_eq!(
            direct.common_fields,
            vec![
                (OperandField::Burst, int(1)),
                (OperandField::Group, int(0)),
                (OperandField::Src0, int(0)),
                (OperandField::Src1, int(0)),
                (OperandField::Src2, int(0)),
                (OperandField::Src3, int(0)),
            ]
        );
        let ibr_write = emit(&fused(
            multicast,
            L3Indirection::IbrWrite,
            reg(SenRegType::Ebr, 2),
            reg(SenRegType::Lar, 1),
        ));
        assert_eq!(ibr_write.opcode, OpCode::LDIGM);
        assert_eq!(
            ibr_write.common_fields,
            vec![
                (OperandField::Burst, int(1)),
                (OperandField::Group, int(0)),
                (OperandField::Readibr, int(0)),
                (OperandField::Src0, int(1)),
                (OperandField::Src2, int(0)),
                (OperandField::Src3, int(2)),
            ]
        );
        let gather = emit(&fused(
            L3Half::Load { multicast: None },
            L3Indirection::Gather,
            reg(SenRegType::Jcr, 0),
            reg(SenRegType::Lar, 1),
        ));
        assert_eq!(gather.opcode, OpCode::LDIM);
        assert_eq!(
            gather.common_fields,
            vec![
                (OperandField::Burst, int(1)),
                (OperandField::Group, int(0)),
                (OperandField::Readibr, int(1)),
                (OperandField::Src0, int(1)),
                (OperandField::Src1, int(0)),
                (OperandField::Src2, int(0)),
                (OperandField::Src3, int(0)),
            ]
        );
    }

    /// ⭐ `LX_LDCVTIU :: consumertag:sfp elemidx:1 imm:128 ldtype:2bsplat scaleidx:1 src0:0`
    /// (`sen1p5/ldcvti.mlir:8`) — ⛔ THE OP'S `immutable_addr` IS **256**, and four source bits over
    /// eight halve it into destination units.
    #[test]
    fn the_load_compute_halves_its_immediate_into_destination_units() {
        let lrf0 = Reg {
            locale: SenRegType::Lrf,
            index: Some(RegIndex::at::<0>()),
        };
        let (instr, refused) = construct_load_compute_instr::<Sen1p5>(
            &LoadCompute {
                mutable_addr: lrf0,
                immutable_addr: Reg {
                    locale: SenRegType::Imm,
                    index: None,
                },
                result: lrf0,
                immutable_value: 256,
                element_index: Bounded::at::<1>(),
                scale_index: Bounded::at::<1>(),
                shuffle: LoadComputeShuffle::Splat2b,
                consumer: LoadComputeConsumer::Unit(LoadConsumer::Unit(ConsumerUnit::Sfp)),
                update: true,
                dbg_name: None,
            },
            &[],
            false,
            &mut RegsToInit::new(),
        );
        assert_eq!(instr.opcode, OpCode::LDCVTIU);
        assert_eq!(
            instr.common_fields,
            vec![
                (OperandField::Consumertag, descriptive("sfp")),
                (OperandField::Elemidx, int(1)),
                (OperandField::Imm, int(128)),
                (OperandField::Ldtype, descriptive("2bsplat")),
                (OperandField::Scaleidx, int(1)),
                (OperandField::Src0, int(0)),
            ]
        );
        assert!(refused.is_empty());
    }

    /// ⭐ `LX_LRFCOPY :: producertag:pe src0:0` (`lx_indirect_loads_stores.mlir:64`) — ⛔ AND THE
    /// NEGATIVE: a producer outside the three, or any lane but 0, comes back as an OFFENDER, and the
    /// rejected producer gets no `producertag` at all.
    #[test]
    fn the_lrf_copy_names_its_producer_and_returns_the_offenders() {
        let copy = |producer, position| LrfCopy {
            result: Reg {
                locale: SenRegType::Lrf,
                index: Some(RegIndex::at::<0>()),
            },
            producer,
            position,
            dbg_name: None,
        };
        let emit = |c: &LrfCopy| construct_lrf_copy_instr(c, &[], false, &mut RegsToInit::new());
        let (instr, refused) = emit(&copy(DfirUnit::Pe, 0));
        assert_eq!(instr.opcode, OpCode::LRFCOPY);
        assert_eq!(
            instr.common_fields,
            vec![
                (OperandField::Producertag, descriptive("pe")),
                (OperandField::Src0, int(0)),
            ]
        );
        assert!(refused.is_empty());
        let (unsupported, refused) = emit(&copy(DfirUnit::L0su, 1));
        assert_eq!(
            unsupported.common_fields,
            vec![(OperandField::Src0, int(0))]
        );
        assert_eq!(
            refused,
            vec![
                LrfCopyRefusal::UnsupportedUnit(GenericComp::L0su),
                LrfCopyRefusal::UnsupportedPosition(1),
            ]
        );
    }
}
