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
use crate::bridges::sentient_to_progir::state::{RegsToInit, UnitKey};
use crate::bridges::sentient_to_progir::uniform::instr::{PerUnitOperand, UniformInstrInfo};
use crate::formats::DataFormat;
use crate::islands::progir::ty::Operand;
use crate::islands::progir::{OpCode, OperandField};
use crate::islands::sentient::dialects::sentient::{
    Reg, RegIndex, RegType as SenRegType, RoutingDirection,
};
use crate::units::Core;
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

// crustify:todo: e057_ConstructL3LoadAndStoreInstr
// crustify:todo: e058_ConstructLoadComputeInstr
// crustify:todo: e059_ConstructLRFCopyInstr
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

#[cfg(test)]
mod unit_tests {
    use super::{
        Component, Elements, L3Load, L3Store, L3StoreSource, LdzConst, LoadTarget, Peer,
        construct_l3_load_instr, construct_l3_store_instr, construct_zr_assign_instr,
        normalize_burst_size,
    };
    use crate::arch::{Arch, Dd2, Target};
    use crate::bridges::sentient_to_progir::construct::{descriptive, int};
    use crate::bridges::sentient_to_progir::state::RegsToInit;
    use crate::islands::progir::{OpCode, OperandField};
    use crate::islands::sentient::dialects::sentient::{
        Reg, RegIndex, RegType as SenRegType, RoutingDirection,
    };
    use crate::units::Core;

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
}
