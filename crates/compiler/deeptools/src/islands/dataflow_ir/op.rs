//! THE DATAFLOWIR OPS. One variant per operation the emitted programs contain, and no more.
//!
//! The set is the union of `Dataflow.td`'s own ops and the standard-dialect ops a real program uses,
//! read off IBM's `dcc/test/PT/xrfbmm_int8_fwd.mlir` — a complete int8 BMM in about sixty lines.
//!
//! ⛔ WHAT IS NOT HERE IS THE POINT. No register, no port, no result forwarding, no unroll factor,
//! no precision per operand. Those are `sentient.*`, and dcc's 76 passes derive them
//! (`dbo/docs/pass_pipeline.md` D1-D76). An op here that named a register would be one rung down the
//! ladder from where this crate stands.

use crate::generated::{OpaqueFunc, ParamKey, ParamValue, RegName, SyncSignal};
use crate::islands::dataflow_ir::ty::{AffineMap, IntegerSet, MemRef, Vector};

/// WHERE ONE OF AN OPAQUE'S REGISTERS LIVES — the value its name binds to.
///
/// ⭐⭐ AN ADDRESS, NOT A NAME AND NOT A BLANK. `insertReg(regName, startAddress + n, ...)`
/// (`ddc/ddcv1.cpp:3369-3391`) binds each register name to its allocation's start address, and
/// `ConstructProgIRHelper.cpp:3999-4014` substitutes that value straight into an instruction's
/// operand field.
///
/// ⛔ AN EMPTY STRING PASSED THE CHECK AND SUBSTITUTED NOTHING. `DataflowToSentient.cpp:1986-1997`
/// only asserts the value IS a `StringAttr`, so `String::new()` — which is what this field held —
/// satisfied it and then wrote an empty operand into the instruction. A newtype over the address
/// makes that unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RegAddr(pub u32);

/// AN SSA VALUE, minted by the builder and never spelled by hand.
///
/// ⛔ A NEWTYPE OVER THE NUMBER, so a value cannot be confused with an extent, an address or a loop
/// bound — all of which are also small integers in this IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Val(pub u32);

/// WHICH REGISTER FILE a `get_local_unit` names.
///
/// ⭐ THESE ARE THE UNIT-PREFIXED SPELLINGS, and the prefix matters: `arch_enums.h:62-66` keeps a
/// generic `LRFREG` "for DSC-level and arch-level compatibility" but says new IR operations use
/// `PE_LRFREG`, `SFP_LRFREG`, `PT_LRFREG`. The vendored DataflowIR writes `pt_lrfreg` and `ptxrf`
/// (`dcc/test/PT/xrfbmm_int8_fwd.mlir:56,60`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LocalUnit {
    /// `pe_lrfreg`.
    PeLrf,
    /// `sfp_lrfreg`.
    SfpLrf,
    /// `pt_lrfreg`.
    PtLrf,
    /// `ptxrf` — the PT's transposed register file, where a matmul's kernel block lands.
    PtXrf,
    /// `ptarf` — the PT's accumulator register file.
    PtArf,
    /// `l0scale` — the L0's scale region.
    ///
    /// ⛔ A LOCAL UNIT OF A PT ROW, AND ONLY FROM SEN1P5. Each PT row's arm ends with
    /// `if (coreArch >= SEN1P5_ISA) component_to_handler_[L0_SCALE] = createGetLocalUnitOp(..)`
    /// (`DSC2ToDataflowIRUtils.hpp:180-183`), which matches RCUDD1A having no scale region at all
    /// (`Arch::L0_SCALE_CAPACITY` is zero there).
    L0Scale,
    // ⛔ `sfpstate` AND `pestate` ARE NOT HERE, AND THAT IS NOT AN OMISSION. Both are bound with
    // `createGetUnitOp`, not `createGetLocalUnitOp` (`DSC2ToDataflowIRUtils.hpp:369-370, 350-351`) —
    // they are units in their own right that a transfer addresses, not register files a unit owns.
    // An earlier version listed `SfpState` here, which would have emitted a `get_local_unit` where
    // the backend expects a `get_unit`.
}

impl LocalUnit {
    /// The `name=` the op carries.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::PeLrf => "pe_lrfreg",
            Self::SfpLrf => "sfp_lrfreg",
            Self::PtLrf => "pt_lrfreg",
            Self::PtXrf => "ptxrf",
            Self::PtArf => "ptarf",
            Self::L0Scale => "l0scale",
        }
    }
}

/// THE NUMERIC PRECISION OF A UNIT'S PROGRAM — `dataflow.program_unit`'s `precision` attribute.
///
/// ⛔ IT SELECTS THE MAC OPCODE. "This precision attribute is used to identify the MAC op code used
/// in the units" (`DSC2ToDataflowIR.hpp:137-143`), and it is produced by
/// `stringifyComputePrecision` (`:54-71`) — so it is the COMPUTE's type, not the tensor's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// `int8` — `IMA8`.
    Int8,
    /// `int4` — `IMA4`.
    Int4,
    /// `fp4` — `FMA4`.
    Fp4,
    /// `fp8` — `FMA8`.
    Fp8,
    /// `fp16` — `FMA16`.
    Fp16,
    /// `fp32` — `FMA32`, and also `FNMS`, which `stringifyComputePrecision` maps here too (`:67-68`).
    Fp32,
}

impl Precision {
    /// The string the attribute carries.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Fp4 => "fp4",
            Self::Fp8 => "fp8",
            Self::Fp16 => "fp16",
            Self::Fp32 => "fp32",
        }
    }
}

/// A `composite_load_and_store`'s operands and attributes.
///
/// See [`Op::CompositeLoadAndStore`] for what the op means and why it is the thing that gets a
/// weight out of the HBM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeTransfer {
    /// The view read from.
    pub src: Val,
    /// Its subscript.
    pub src_indices: Vec<Index>,
    /// Its type.
    pub src_ty: MemRef,
    /// The view written to.
    pub dst: Val,
    /// Its subscript.
    pub dst_indices: Vec<Index>,
    /// Its type.
    pub dst_ty: MemRef,
    /// The block argument carrying the vector loaded at each time step.
    pub load_iv: Val,
    /// That vector's type — ONE hardware vector, never the whole transfer.
    pub load_iv_ty: Vector,
    /// Which elements form each loaded vector.
    pub load_set: IntegerSet,
    /// How those elements are packed.
    pub load_order: AffineMap,
    /// Which elements form each stored vector.
    pub store_set: IntegerSet,
    /// How those are packed.
    pub store_order: AffineMap,
    /// The time steps the transfer takes — a single pinned step when it fits in one vector.
    pub time_set: IntegerSet,
    /// The order among them.
    pub time_order: AffineMap,
    /// The source offset at each time step, one result per source dimension.
    pub load_time_addr_map: AffineMap,
    /// The destination offset at each time step, one result per destination dimension.
    pub store_time_addr_map: AffineMap,
    /// The region, entered once per time step.
    pub body: Vec<Op>,
}

/// ONE INDEX OF A LOAD OR STORE: an induction variable, an applied map, or a literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Index {
    /// An SSA value — a loop's induction variable or an `affine.apply` result.
    Val(Val),
    /// A literal, as `%lrf_memory[4, 0]` writes it.
    Const(i64),
    /// ⭐⭐ A SUM OF STRIDED INDUCTION VARIABLES, WRITTEN INLINE — `%arg9 + %arg8 * 8`.
    ///
    /// ⛔⛔ INLINE, NOT AN `affine.apply`. One view dimension is walked by EVERY enclosing loop that
    /// strides its axis, so an index is a sum, not a single variable. IBM writes those sums straight
    /// into the index list — `agen.vector_store %48, %49[0, %arg9 + %arg8 * 8, 0]`
    /// (`dcc/test/PT/bf16-pt.mlir:161`) — and `bf16-pt.mlir` contains **zero** `affine.apply`.
    /// Emitting one per index instead made `dbo-opt` refuse outright: "'affine.apply' op Expanded
    /// affine.apply operation does not resolve to a constant".
    ///
    /// ⭐ A STRIDE OF ONE PRINTS BARE. `%arg9 * 1` is the same address written longer, and the
    /// vendored files never write it.
    ///
    /// ⭐⭐ AND A CONSTANT ADDEND, WHICH IS HOW A FAN-OUT'S SLICES DIFFER. One `ddl.data_transfer`
    /// to a row-expanding `unit="pt"` becomes one send PER ROW, and the eight PT rows are a systolic
    /// accumulation chain (`bmm.ddl:256-260`: row 0 seeds with `%zero_const`, rows 1-7 add
    /// `%pt_src02_north`) each MACing from its OWN XRF — so they need eight DIFFERENT slices, not
    /// one broadcast. The i-th slice sits `i * stride/count` along the axis the innermost enclosing
    /// loop strides.
    ///
    /// ⛔ INLINE, NOT AN `affine.apply` — same reason as the terms above.
    Strided(Vec<(Val, i64)>, i64),
}

/// WHICH COMPARISON — `VectorChainElementWiseCompareOperator` (`VectorChainEnums.td`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// `compare_eq`.
    Eq,
    /// `compare_neq`.
    Neq,
    /// `compare_lt`.
    Lt,
    /// `compare_le`.
    Le,
    /// `compare_gt`.
    Gt,
    /// `compare_ge`.
    Ge,
}

impl CompareOp {
    /// THE ENUMERATOR'S VALUE, which is what the attribute carries.
    ///
    /// ⛔⛔ THE MNEMONIC IS THE ENUM'S NAME, NOT THE ATTRIBUTE'S, and both wrong guesses were caught
    /// by dbo-opt rather than by reading. `#vectorchain<compare_op compare_gt>` gives "unknown
    /// attribute `compare_op` in dialect `vectorchain`"; a plain `4 : i32` gives "failed to satisfy
    /// constraint". IBM's own IR writes
    /// `#vectorchain<element_wise_compare_operator compare_gt>` — the mnemonic comes from
    /// `VectorChainElementWiseCompareOperator`, while `compare_op` is only the operand's name.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Eq => "compare_eq",
            Self::Neq => "compare_neq",
            Self::Lt => "compare_lt",
            Self::Le => "compare_le",
            Self::Gt => "compare_gt",
            Self::Ge => "compare_ge",
        }
    }
}

/// WHICH BINARY OPERATION — `VectorChainBinaryOperator` (`VectorChainEnums.td`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// `add`.
    Add,
    /// `sub`.
    Sub,
    /// `mul`.
    Mul,
    /// `mul_div2`.
    MulDiv2,
    /// `min`.
    Min,
    /// `max`.
    Max,
    /// `abs_min`.
    AbsMin,
    /// `abs_max`.
    AbsMax,
    /// `and0`.
    And,
    /// `or0`.
    Or,
    /// `xnor`.
    Xnor,
    /// `and_not`.
    AndNot,
}

impl BinaryOp {
    /// The enumerator's spelling, written as `#vectorchain<binary_operator add>` — the mnemonic is
    /// the enum's name, as it is for [`CompareOp::spelling`].
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::And => "and0",
            Self::Or => "or0",
            Self::Xnor => "xnor",
            Self::AndNot => "and_not",
            Self::Min => "min",
            Self::Max => "max",
            Self::AbsMin => "abs_min",
            Self::AbsMax => "abs_max",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Sub => "sub",
            Self::MulDiv2 => "mul_div2",
        }
    }
}

/// WHICH TRANSCENDENTAL ESTIMATE — one `vectorchain` op each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimateKind {
    /// `exp_estimate`.
    Exp,
    /// `rec_estimate`.
    Rec,
    /// `ln_estimate`.
    Ln,
    /// `rsqrt_estimate`.
    Rsqrt,
    /// `sigmoid_estimate`.
    Sigmoid,
    /// `tanh_estimate`.
    Tanh,
}

impl EstimateKind {
    /// The op's mnemonic.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Exp => "exp_estimate",
            Self::Rec => "rec_estimate",
            Self::Ln => "ln_estimate",
            Self::Rsqrt => "rsqrt_estimate",
            Self::Sigmoid => "sigmoid_estimate",
            Self::Tanh => "tanh_estimate",
        }
    }

    /// The attribute mnemonic its `version` is written under, where it takes one.
    ///
    /// ⛔ THE EXPONENTIAL HAS ITS OWN ENUM. IBM's IR writes `#vectorchain<exp_estimate a>` for exp
    /// and `#vectorchain<estimate_versions slope>` for the rest — `VectorChainExpEstimate` and
    /// `VectorChainEstimateVersions` are two enums, so one mnemonic would be wrong for one of them.
    #[must_use]
    pub const fn version_mnemonic(self) -> &'static str {
        match self {
            Self::Exp => "exp_estimate",
            Self::Rec | Self::Ln | Self::Rsqrt | Self::Sigmoid | Self::Tanh => "estimate_versions",
        }
    }
}

/// WHICH VERSION OF AN ESTIMATE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimateVersion {
    /// `a` — the exponential's first form.
    A,
    /// `b` — its second.
    B,
    /// `slope`.
    Slope,
    /// `offset`.
    Offset,
}

impl EstimateVersion {
    /// The enumerator's spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::A => "a",
            Self::B => "b",
            Self::Slope => "slope",
            Self::Offset => "offset",
        }
    }
}

/// WHICH CONNECTIVE - `ddl.condition_and` / `_or` / `_not`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicKind {
    /// `arith.andi`.
    And,
    /// `arith.ori`.
    Or,
    /// `arith.xori %c, true` - MLIR has no `not`, so a negation is an xor with true.
    Not,
}

/// A LOOP BOUND — a literal, or a value the program computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    /// `affine.for %i = 0 to 8`.
    Const(i64),
    /// `affine.for %i = 0 to %extent`, where the extent is an `arith.constant` the program declared.
    Val(Val),
}

/// A LANE MASK'S DEFINITION — how many lanes are live, and the i1 vector it is stated over.
///
/// ⭐ THE TWO TRAVEL TOGETHER because a prefix means nothing without the width it is a prefix OF.
/// This is what `vectorchain.create_affine_mask` carries, and [`LaneMask::binds`] is the ONLY way
/// to obtain a [`Predicate`] — so a mask's type at its USE is the same value that was written at
/// its DEFINITION.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneMask {
    lanes: u64,
    ty: Vector,
}

impl LaneMask {
    /// A continuous live prefix of `lanes`, over an i1 vector of type `ty`.
    #[must_use]
    pub const fn prefix_of(lanes: u64, ty: Vector) -> Self {
        Self { lanes, ty }
    }

    /// How many lanes are live.
    #[must_use]
    pub const fn live(self) -> u64 {
        self.lanes
    }

    /// The i1 vector type this mask is stated over.
    #[must_use]
    pub const fn ty(self) -> Vector {
        self.ty
    }

    /// THE VALUE A `create_affine_mask` BOUND, CARRYING THIS MASK'S OWN TYPE.
    #[must_use]
    pub const fn binds(self, result: Val) -> Predicate {
        Predicate {
            val: result,
            ty: self.ty,
        }
    }
}

/// A MASK VALUE AND THE TYPE IT WAS DEFINED AT.
///
/// ⛔⛔ THE TYPE TRAVELS WITH THE VALUE, AND THAT IS THE WHOLE POINT. dbo-opt refused granite-2b's
/// `group_7`: *"use of value '%42' expects different type than prior uses: 'vector<64xi1>' vs
/// 'i1'"*. `%42` was minted by `arith.constant true` — an `i1` — while the use site printed its
/// type by RECOMPUTING `vector<{lanes}xi1>` from the operands' lane count. Two records of one
/// fact, and they disagreed.
///
/// ⭐ THE FIELDS ARE PRIVATE AND THERE IS NO PUBLIC CONSTRUCTOR. A `Predicate` comes only from
/// [`LaneMask::binds`], so it cannot be built around a value whose definition says something else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Predicate {
    val: Val,
    ty: Vector,
}

impl Predicate {
    /// The masked value.
    #[must_use]
    pub const fn val(self) -> Val {
        self.val
    }

    /// The type it was DEFINED at — never recomputed at the use.
    #[must_use]
    pub const fn ty(self) -> Vector {
        self.ty
    }
}

/// ONE DATAFLOWIR OPERATION.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `arith.constant N : index` — the extents a loop counts to.
    Constant {
        /// The value it binds.
        result: Val,
        /// The literal.
        value: i64,
    },

    /// `dataflow.get_unit {core, corelet, name, type} : index`.
    ///
    /// ⛔⛔ `type` IS LOAD-BEARING AND `name` IS NOT — THEY ARE NOT THE SAME STRING. Only
    /// `StrAttr:$name` and `StrAttr:$type` are declared arguments (`Dataflow.td`, `get_unit`);
    /// `core` and `corelet` ride through as discardable attributes on `attr-dict`. Downstream
    /// identity is taken from `getType()`, the `type` attribute
    /// (`DataflowToSentient.cpp:119-130`), and that string is then fed to
    /// `symbolizeSentientLoadConsumer(..).value()` — which **aborts** on a spelling outside the
    /// sixteen-member `SentientLoadConsumer` set (`SentientTypes.td:556-593`). So `type` is a
    /// censused token, never free text.
    ///
    /// ⭐ THE `name` FOLLOWS THE SCHEDULER'S CONVENTION, `C{core}-{tag}[-CL{corelet}]`, which is
    /// what IBM's own DataflowIR carries (`/tmp/ktir_ref/export/debug/dfir.mlir:45-63`) and what
    /// `UnitMaterializer.cpp:53-159` writes. The ddc translator instead passes
    /// `senComponentsToString.at(comp)` as BOTH name and type
    /// (`DSC2ToDataflowIRUtils.hpp:69-73`, and `dcc/test/Conversion/DataflowToSentient/opaque.mlir`
    /// shows `{name = "pe", type = "pe"}`) — two producers, two conventions. No consumer reading
    /// `name` was found, so this matches the producer whose output the entry point we entered by
    /// was built to accept.
    ///
    /// ⛔ AND THE UNIT IS A [`DfirUnit`], NOT A TEMPLATE [`Unit`]. DataflowIR binds units the `.ddl`
    /// vocabulary has no `unit=` spelling for — `lx`, `l3lu`, `l3su` — and names each PT ROW rather
    /// than a span.
    GetUnit {
        /// The handle it binds.
        result: Val,
        /// WHERE THE UNIT LIVES, which decides both attributes and the name.
        ///
        /// ⛔ THIS WAS `core: u32` PLUS `corelet: Option<u32>` AND THAT PAIR COULD NOT SPELL A
        /// GLOBAL UNIT — `core` was mandatory, so the HBM, which carries neither attribute, had no
        /// representation. See [`crate::units::Residency`].
        residency: crate::units::Residency,
        /// `type=`.
        unit: crate::units::DfirUnit,
    },

    /// `dataflow.get_local_unit %unit {name} : index` — a register file of a unit already held.
    GetLocalUnit {
        /// The handle it binds.
        result: Val,
        /// The unit it belongs to.
        of: Val,
        /// Which file.
        which: LocalUnit,
    },

    /// `dataflow.get_logical_memory_view %unit, %start {layout_map} : index, index, memref<..>`.
    ///
    /// ⛔ `start` IS IN ELEMENTS (`Dataflow.td:250`).
    GetLogicalMemoryView {
        /// The view it binds.
        result: Val,
        /// The memory unit viewed.
        from: Val,
        /// The start address, in elements.
        start: Val,
        /// How the view's indices map onto the linear region.
        layout: AffineMap,
        /// The view's type.
        ty: MemRef,
    },

    /// `dataflow.program_unit %unit {precision} : { .. }` — one unit's whole program.
    ProgramUnit {
        /// The units this program runs on. More than one where the same program is bound across
        /// program time steps (`Dataflow.td:99-104`).
        units: Vec<Val>,
        /// `precision=`, which selects the MAC opcode. Absent on a unit that computes nothing.
        precision: Option<Precision>,
        /// The body.
        body: Vec<Op>,
    },

    /// `vectorchain.<kind>_estimate %in {version} : tin, tout` — the SFP's transcendental estimates.
    ///
    /// ⛔ WHICH ONE IS THE COMPUTE'S `mode=`, NOT ITS COMPUTETYPE. A `FEST` dispatches on mode 0-9
    /// into exp(a), exp(b), rec, ln, rsqrt, sigmoid(slope|offset) and tanh(slope|offset)
    /// (`SNComputeLowering.cpp:1316-1372`).
    Estimate {
        /// The vector it binds.
        result: Val,
        /// What is estimated.
        input: Val,
        /// Which estimate.
        kind: EstimateKind,
        /// Which version, where the op takes one. `rec` and `ln` do not.
        version: Option<EstimateVersion>,
        /// The input's type.
        input_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.scan_with_gap %in {reduction_op, gap, eval_order} : tin, tout` — a reduction.
    ///
    /// ⛔ WHICH REDUCTION IS THE COMPUTE'S `mode=`: 1 add, 8 max, 10 abs_max, 12 min, 14 abs_min
    /// (`SNComputeLowering.cpp:1453-1467`). The gap is 8 and the order left-to-right, both fixed
    /// there (`:1468-1470`).
    ScanWithGap {
        /// The vector it binds.
        result: Val,
        /// What is reduced.
        input: Val,
        /// Which reduction.
        reduction_op: BinaryOp,
        /// The input's type.
        input_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `arith.constant true` — the `i1` a negation xors against.
    ///
    /// (E) A SEPARATE OP BECAUSE THE TYPE DIFFERS. `Op::Constant` prints `: index`, and feeding that
    /// to `arith.xori .. : i1` is "use of value expects different type than prior uses: 'i1' vs
    /// 'index'".
    True {
        /// The i1 it binds.
        result: Val,
    },

    /// `arith.cmpi eq, %iv, <bound> : index` — one loop-position predicate.
    ///
    /// (E) `first` IS `iv == lower bound` AND `last` IS `iv == upper bound - 1`
    /// (`SNControlFlowLowering.cpp:100-108`). Comparing against the trip count rather than one less
    /// than it makes `last` true on no trip at all.
    Compare {
        /// The i1 it binds.
        result: Val,
        /// The induction variable tested.
        iv: Val,
        /// What it is compared against.
        ///
        /// (E) AN SSA VALUE, NOT A LITERAL. `arith.cmpi` takes two operands of the same type;
        /// writing `arith.cmpi eq, %14, 0 : index` is "expected SSA operand". The bound is minted as
        /// an `arith.constant` first.
        against: Val,
    },

    /// `arith.andi` / `arith.ori` / `arith.xori %c, true` - the connectives of a predicate.
    Logic {
        /// The i1 it binds.
        result: Val,
        /// Which connective.
        kind: LogicKind,
        /// Its operands. `Not` takes exactly one.
        operands: Vec<Val>,
    },

    /// `scf.if %cond { .. } else { .. }` - an undecided branch, BOTH arms in one op.
    ///
    /// (E) ONE OP WITH TWO REGIONS, WHICH IS WHAT IBM EMITS. `dcc/test/PT/fp8-bmm.mlir:1072-1101`
    /// is `scf.if %923 { .. } else { .. }`, and the `else` region there holds a whole further
    /// condition chain — so nesting arms inside an `else` is the vendored shape, not an
    /// optimisation.
    ///
    /// ⛔⛔ THE NEGATED-SIBLING FORM CRASHED THE BACKEND. An earlier version emitted the two arms as
    /// separate `scf.if`s on a predicate and its negation, reasoning that mutual exclusion kept a
    /// value produced in one out of the other's scope — which MLIR's own region scoping already
    /// guarantees. The negation is an `arith.xori %cond, true`, and once the predicate was a
    /// COMPOUND one (`arith.andi` of two positions, which `ddl.condition_and` at `bmm.ddl:234`
    /// genuinely asks for) `dbo-opt` converted the shared `andi` into a `sentient.if`, destroyed it
    /// converting the guard, and died on the `xori` still holding it: "'sentient.if' op operation
    /// destroyed but still has uses". With one op and two regions there is no negation to dangle.
    If {
        /// The predicate.
        cond: Val,
        /// The `then` region.
        body: Vec<Op>,
        /// The `else` region. Empty prints no `else` at all, which is the one-armed branch.
        else_body: Vec<Op>,
    },

    /// `affine.for %i = <lo> to <hi> { .. }`.
    For {
        /// The induction variable it binds.
        iv: Val,
        /// The lower bound.
        lo: Bound,
        /// The upper bound.
        hi: Bound,
        /// The body.
        body: Vec<Op>,
    },

    /// `affine.apply affine_map<..>(%args)` — an index computed from induction variables.
    Apply {
        /// The index it binds.
        result: Val,
        /// The map.
        map: AffineMap,
        /// Its arguments, in order.
        args: Vec<Val>,
    },

    /// `agen.vector_load %view[..] {load_order, load_set} : memref<..>, vector<..>`.
    ///
    /// ⭐⭐ THIS IS THE FORM THE BACKEND'S OWN PRODUCER EMITS. The dataflow scheduler's DataflowIR
    /// contains ONLY `agen.vector_load`/`agen.vector_store` — zero affine ones — and `dbo-opt` is
    /// what consumes that. The `affine` pair below appears in `dcc/test/PT/*.mlir`, which is run
    /// through `dcc-opt --kEmitProgIR`: a different tool entering at a different stage.
    ///
    /// ⛔ WHICH IS WHY EMITTING THE AFFINE FORM CRASHED THE PIPELINE. `VectorChainToSentientPESFP`
    /// has a working `VectorStoreOpLowering` for `agen::VectorStoreOp` and a `StoreOpLowering` for
    /// `vector::StoreOp` whose `reuse_info_.getId(..).value()` is unguarded
    /// (`VectorChainToSentientPESFP.cpp:123`) — undertested, because the scheduler never produces a
    /// `vector.store` for it to see.
    AgenVectorLoad {
        /// The vector it binds.
        result: Val,
        /// The view read.
        view: Val,
        /// The indices.
        indices: Vec<Index>,
        /// The view's type. Its INNERMOST extent is the lane count.
        view_ty: MemRef,
        /// The vector's type.
        ty: Vector,
    },

    /// `agen.composite_load_and_store src:%s[..] dst:%d[..] time_symbols(), load_iv(%v:vector<..>)
    /// {..} { .. } : memref<..>, memref<..>`.
    ///
    /// ⭐⭐ THIS IS HOW A WEIGHT LEAVES THE HBM. A view is only an address; nothing crosses a
    /// datapath until a transfer says so. The device declares the route explicitly —
    /// `datapath %dram to %l3lu` then `datapath #L3LU_LX %l3lu to %lx`
    /// (`spyre_dd2_basic.mlir:82-83`) — and this op is what runs it. Emitting the compute against
    /// an LX view with no transfer into it is a program that reads memory nothing ever filled.
    ///
    /// ⛔⛔ AT MOST ONE HARDWARE VECTOR PER TIME STEP. "An AGEN composite transfer moves at most one
    /// hardware vector per time step, so a transfer wider than that has to walk the remaining
    /// elements over AGEN time dimensions instead of widening `load_iv`"
    /// (`DataTransferLowering.cpp:306-310`). The walk is what [`time_set`](Self::CompositeLoadAndStore::time_set)
    /// and the two `*_time_addr_map`s describe; see
    /// [`crate::bridges::subtile_to_dataflow_ir::transfer`], which computes them.
    ///
    /// ⛔ THE REGION IS ENTERED ONCE PER TIME STEP and its block argument carries the vector loaded
    /// at that step. A plain memory-to-memory move yields immediately; a transfer that also sends
    /// the value onward puts that in the body.
    /// ⛔ BOXED, because it carries four affine maps, four integer sets and two subscripts, and an
    /// enum is as large as its largest variant. Every other op in this IR is a handful of words.
    CompositeLoadAndStore(Box<CompositeTransfer>),

    /// `agen.yield` — the terminator of a composite transfer's region.
    AgenYield,

    /// `agen.vector_store %value, %view[..] {store_order, store_set} : memref<..>, vector<..>`.
    ///
    /// See [`Op::AgenVectorLoad`] for why this is the form the bridge emits.
    AgenVectorStore {
        /// The vector stored.
        value: Val,
        /// The view written.
        view: Val,
        /// The indices.
        indices: Vec<Index>,
        /// The view's type. Its INNERMOST extent is the lane count.
        view_ty: MemRef,
        /// The vector's type.
        ty: Vector,
    },

    /// `affine.vector_load %view[..] : memref<..>, vector<..>`.
    ///
    /// ⛔ THE `dcc-opt` FORM. The BRIDGE emits [`Op::AgenVectorLoad`]; see there.
    VectorLoad {
        /// The vector it binds.
        result: Val,
        /// The view read.
        view: Val,
        /// The indices.
        indices: Vec<Index>,
        /// The view's type.
        view_ty: MemRef,
        /// The vector's type.
        ty: Vector,
    },

    /// `affine.vector_store %value, %view[..] : memref<..>, vector<..>`.
    VectorStore {
        /// The vector written.
        value: Val,
        /// The view written to.
        view: Val,
        /// The indices.
        indices: Vec<Index>,
        /// The view's type.
        view_ty: MemRef,
        /// The vector's type.
        ty: Vector,
    },

    /// `dataflow.send %to, %data : vector<..>`.
    ///
    /// ⛔ `to` IS THE FIRST HOP, NOT THE DESTINATION, where the transfer states a via:
    /// `to_unit = dst.via_.empty() ? dst.loc_.unit_ : dst.via_.front()`
    /// (`SNTransferLowering.cpp:2731-2732`).
    Send {
        /// The unit sent to.
        to: Val,
        /// The data.
        data: Val,
        /// Its type.
        ty: Vector,
    },

    /// `dataflow.receive %from : vector<..>`.
    Receive {
        /// The vector it binds.
        result: Val,
        /// The unit received from.
        from: Val,
        /// Its type.
        ty: Vector,
    },

    /// `dataflow.sync_send %unit : index`.
    SyncSend {
        /// The unit signalled.
        to: Val,
        /// Which signal, for the debug name.
        signal: SyncSignal,
    },

    /// `dataflow.sync_recv %unit : index` — blocking.
    SyncRecv {
        /// The unit waited on.
        from: Val,
        /// Which signal, for the debug name.
        signal: SyncSignal,
    },

    /// `dataflow.implicit_sync_on_streaming_buffer %view, %dst, %size : ..` — synchronisation at the
    /// granularity of a streaming buffer rather than per transfer.
    ImplicitSync {
        /// The buffer.
        view: Val,
        /// The unit synchronised with.
        dst: Val,
        /// The buffer size.
        size: Val,
        /// The buffer's type.
        view_ty: MemRef,
    },

    /// `dataflow.opaque {func_name, read_write_register_dictionary, read_only_register_dictionary,
    /// parameter_dictionary}` — one op standing for a whole `.smc` body, which dcc splices.
    ///
    /// ⛔⛔ THE TWO DICTIONARIES ARE CROSSED RELATIVE TO THEIR NAMES. `read_write_reg_map_` is filled
    /// from the body's INTERNAL registers and `read_only_reg_map_` from the caller-bound INPUT/OUTPUT
    /// ones (`ddcv1.cpp:3369-3391`).
    Opaque {
        /// `func_name=`.
        func: OpaqueFunc,
        /// `read_write_register_dictionary=` — the body's own scratch, from `internal_registers=`.
        read_write: Vec<(RegName, RegAddr)>,
        /// `read_only_register_dictionary=` — caller-bound, from `input_output_registers=`.
        read_only: Vec<(RegName, RegAddr)>,
        /// `parameter_dictionary=`, including the `prec` the op's own data format sets.
        params: Vec<(ParamKey, ParamValue)>,
    },

    /// `vectorchain.select %in {selection_map} : vector<..>, vector<..>` — a lane permutation.
    Select {
        /// The vector it binds.
        result: Val,
        /// The input.
        input: Val,
        /// Which lane each output lane reads.
        selection_map: AffineMap,
        /// The input's type.
        input_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.multiply %a, %b {reduction_map} : ..` — a product with a reduction.
    Multiply {
        /// The vector it binds.
        result: Val,
        /// The left operand.
        a: Val,
        /// The right operand.
        b: Val,
        /// Which product lanes fold into which result lane. Arch- and precision-dependent:
        /// `IMA8` on RCUDD1A is `(d0 mod 128) floordiv 2`, on SEN1P5 `d0 floordiv 16`
        /// (`SNComputeLowering.cpp:157-172`).
        reduction_map: AffineMap,
        /// The operands' type.
        operand_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.multiply_and_accumulate %a, %b, %acc {reduction_map} : ..`.
    MultiplyAccumulate {
        /// The vector it binds.
        result: Val,
        /// The left operand.
        a: Val,
        /// The right operand.
        b: Val,
        /// The accumulator read in.
        acc: Val,
        /// As [`Op::Multiply`].
        reduction_map: AffineMap,
        /// The operands' type.
        operand_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.element_wise_compare %op1, %op2 [%mask : t] {compare_op} : t1, t2, tres`.
    ///
    /// (E) A FMAX IS NOT ONE OP. `constructFMINorFMAXOperation` (`SNComputeLowering.cpp:1189-1272`)
    /// emits a COMPARE (`compare_gt` for FMAX, `compare_le` for FMIN) and then a SELECTION over its
    /// i1 result. An earlier `vectorchain.fmax` was rejected outright by dbo-opt: "custom op
    /// 'vectorchain.fmax' is unknown". The 23 real ops are in `VectorChain.td`.
    ElementWiseCompare {
        /// The i1 vector it binds.
        result: Val,
        /// Left operand.
        op1: Val,
        /// Right operand.
        op2: Val,
        /// The lane mask, IF THIS OP CARRIES ONE — `Optional<AnyVectorOfAnyRank>:$mask`
        /// (`VectorChain.td`), so `None` prints no bracket at all.
        mask: Option<Predicate>,
        /// Which comparison.
        compare_op: CompareOp,
        /// The operands' type.
        operand_ty: Vector,
        /// The i1 result's type.
        ty: Vector,
    },

    /// `vectorchain.element_wise_selection %cond ? %lhs : %rhs [%mask : t] : tc, tl, tr, tres`.
    ElementWiseSelection {
        /// The vector it binds.
        result: Val,
        /// The i1 vector chosen by, CARRYING ITS OWN TYPE.
        ///
        /// ⛔ THIS USED TO BE A `Val` BESIDE A `cond_ty: Vector` THE EMITTER FILLED IN — the same
        /// two-records shape that made the mask disagree with itself. A `Predicate` states it once.
        cond: Predicate,
        /// Taken where the condition holds.
        lhs: Val,
        /// Taken otherwise.
        rhs: Val,
        /// The lane mask, IF THIS OP CARRIES ONE. `Optional` in the dialect
        /// (`VectorChain.td:402`), and the condition is a SEPARATE operand from it.
        mask: Option<Predicate>,
        /// The operands' and result's type.
        ty: Vector,
    },

    /// `vectorchain.binary %op1, %op2 [%mask : t] {binary_op} : t1, t2, tres` — the elementwise family.
    Binary {
        /// The vector it binds.
        result: Val,
        /// Left operand.
        op1: Val,
        /// Right operand.
        op2: Val,
        /// The lane mask, IF THIS OP CARRIES ONE.
        ///
        /// ⛔⛔ `None` MEANS NO BRACKET, NOT AN ALL-TRUE CONSTANT. Every binary used to be handed
        /// an `arith.constant true` as a stand-in for "unmasked", which is both an `i1` where a
        /// `vector<Nxi1>` was printed (the granite-2b `group_7` refusal) and a mask that masks
        /// nothing. `Optional<AnyVectorOfAnyRank>:$mask` (`VectorChain.td:139`) — an unmasked
        /// binary simply omits the operand.
        mask: Option<Predicate>,
        /// Which operation.
        binary_op: BinaryOp,
        /// `op_specific_map=` — REQUIRED, and dbo-opt says so: "'vectorchain.binary' op requires
        /// attribute 'op_specific_map'". IBM's own IR writes the identity, `affine_map<(d0) -> (d0)>`,
        /// which is lane `i` of each operand into lane `i` of the result.
        op_specific_map: AffineMap,
        /// The operands' type.
        operand_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `arith.constant dense<V> : vector<NxT>` — an immediate operand of a compute.
    ///
    /// (E) NOT A SPLAT OP. `constructComputeInputOperandAndAddToList`
    /// (`SNComputeLowering.cpp:461-495`) builds a dense `arith.constant` of the RESULT vector type
    /// for the `ZERO` and `ONE` pseudo-units. `createSplatOperation` is one rung down, in
    /// `dcc/src/Conversion/VectorChainLowering/` — it turns a vectorchain op into sentient ops, which
    /// is not this stage's job. An earlier refusal here cited it, and cited the wrong layer.
    DenseConstant {
        /// The vector it binds.
        result: Val,
        /// Whether the value is one rather than zero — the only two the templates name.
        one: bool,
        /// Its type.
        ty: Vector,
    },

    /// `vectorchain.constant_bitstream {value = [0x0, 0x1]} : vector<Nxt>` — a literal, as wide as
    /// the constant the template declares.
    ///
    /// ⭐ THE VALUES ARE ELEMENT BIT PATTERNS, which is what `ddl.define_constant`'s `value=[0xFFFF]`
    /// already holds — the two forms line up exactly.
    ConstantBitstream {
        /// The vector it binds.
        result: Val,
        /// The element bit patterns.
        value: Vec<i64>,
        /// Its type — as many elements as `value` has.
        ty: Vector,
    },

    /// `vectorchain.shuffle input(%c) {indices = [..], repetition = N} : tin, tout` — the splat that
    /// widens a constant bitstream to the width its consumer reads.
    ///
    /// ⛔ `repetition` IS A QUOTIENT, NOT A COUNT SOMEONE PICKS:
    /// `getNumElements(result_type) / getNumElements(bitstream_vector_type)`
    /// (`SNTransferLowering.cpp:2505-2506`).
    Shuffle {
        /// The vector it binds.
        result: Val,
        /// What is widened.
        input: Val,
        /// One index per element of the input.
        indices: Vec<i32>,
        /// How many times the pattern repeats to fill the result.
        repetition: u32,
        /// The input's type.
        input_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.cast %v : tin, tout` — a precision conversion.
    ///
    /// ⛔ THE C++ EMITS ONE WHERE THE SOURCE AND DESTINATION FORMATS DIFFER: "Convert data if src
    /// precision and dst precision don't match" (`SNTransferLowering.cpp:2305-2323`), applied to the
    /// data AFTER the load or shuffle — which is why a shuffle's element type is the SOURCE's, and
    /// why `vectorchain.shuffle` refuses to change it: "input element type does not match output
    /// element type".
    Cast {
        /// The vector it binds.
        result: Val,
        /// What is converted.
        input: Val,
        /// The input's type.
        input_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.create_affine_mask {mask_set} : vector<Nxi1>` — the lane mask every elementwise
    /// op takes, from `getStaticContinuousMaskValue`.
    CreateAffineMask {
        /// The i1 vector it binds.
        result: Val,
        /// THE MASK ITSELF — its live prefix AND the type it is stated over, as one value.
        ///
        /// ⛔⛔ `lanes: u64` AND `ty: Vector` AS SEPARATE FIELDS WERE TWO RECORDS OF ONE FACT. The
        /// definition printed from `ty` while every USE recomputed `vector<{len}xi1>` from the
        /// operands — which is how `%42` came to be defined as `i1` and used as `vector<64xi1>`.
        /// [`LaneMask::binds`] hands the SAME `Vector` to the use site, so they cannot differ.
        mask: LaneMask,
    },
}
