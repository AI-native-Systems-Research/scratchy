//! `Dataflow.td` — UNITS, VIEWS, THE TRANSFERS BETWEEN UNITS, AND THE OPAQUE BODIES.
//!
//! The dialect declares fourteen operations; the ten here are the ones an emitted program contains.

use std::fmt::Write as _;

use crate::generated::{OpaqueFunc, ParamKey, ParamValue, RegName, SyncSignal};
use crate::islands::dataflow_ir::dialects::Val;
use crate::islands::dataflow_ir::link::{RecvEnd, SendEnd};
use crate::islands::dataflow_ir::print;
use crate::islands::dataflow_ir::ty::{
    AffineExpr, AffineMap, Constraint, IntegerSet, MemRef, Vector,
};

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
    /// `ptirf` — the PT's INPUT register file, where an int8 matmul's activations land
    /// (`dcc/test/PT/int8-genkg3-pt.mlir:123`).
    PtIrf,
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
            Self::PtIrf => "ptirf",
            Self::L0Scale => "l0scale",
        }
    }
}

/// A VALUE THAT ARRIVED ON THE WIRE — what a `dataflow.receive` binds.
///
/// # 🛑 A VALUE DEFINED INSIDE A PROGRAM UNIT DOES NOT LEAVE IT
///
/// ⛔⛔ MLIR's parser pushes a fresh DEFINITIONS scope for EVERY region, isolated or not:
/// `parseRegionBody` calls `pushSSANameScope(isIsolatedNameScope)` and that always runs
/// `isolatedNameScopes.back().pushSSANameScope()` (`Parser.cpp:2273-2276`, `:944-951`).
/// `IsolatedFromAbove` decides only whether the region can still SEE outward — it never makes the
/// region's own definitions escape. So a view a `dataflow.program_unit` takes is gone at its `}`,
/// and IBM's `lxlu` unit takes the LX view AGAIN rather than reuse the `l3lu`'s
/// (`/tmp/ktir_ref/export/debug/dfir.mlir:73-74` vs `:112-116`).
///
/// ⛔ THE EMITTER HANDED THE MOVER'S VIEWS STRAIGHT TO THE COMPUTE, and dbo-opt refused it twice
/// wearing two faces. First as *"use of undeclared SSA value name"* on `agen.vector_load %9`. Then,
/// once the leaked forward reference survived to the next program, as *"definition of SSA value
/// '%24#0' has type 'memref<2048x2048xf16>'"* — `Parser.cpp:1005-1011`, which fires only when the
/// name was already a forward-reference placeholder, here typed `memref<255x512xf16>` from the
/// PREVIOUS program's compute. One defect, two messages.
///
/// ⭐⭐ SO THE COMPUTE'S OPERANDS ARE WIRES, NOT VIEWS. Only [`Op::Receive`] mints one of these, so
/// a `Val` naming a view cannot reach a compute: the mover loads and sends, the compute receives.
///
/// ⛔ THE `Val` IS PRIVATE. A public field would be a way to launder the brand off at any call site
/// that happened to want a `Val`.
///
/// ⛔⛔⛔ AND IT IS ONE END OF A REUSE EDGE, SO IT IS SPENT ONCE — NOT `Copy`. The receive is the
/// PRODUCER and the compute that reads it is the CONSUMER, and that pairing is a fact the backend
/// checks: `OperandReuse::setReuseInformation` records `data_origins_` entries per USER
/// (`OperandReuse.cpp:17-38`), so a `dataflow.receive` earns an entry ONLY by being an operand of a
/// lowered compute, and `VectorChainToSentientPESFP.cpp:1343` then refuses any op that has none.
///
/// ⛔⛔ IT WAS `Copy`, AND THE TWO ENDS WERE TWO INDEPENDENT COUNTS. The receives were emitted one
/// per node operand; the compute was built over `split_first()` and `rest.first()`. On granite that
/// was SIX received and TWO consumed, and dbo-opt refused the four with "Dangling non-compute op
/// has no use | no OperandReuse entry: never an operand of a lowered compute". Spending the edge —
/// [`Received::operand`] takes `self` — makes a received vector nothing consumed a value still held
/// at the call site, which `#[must_use]` names at the line that made it.
#[must_use = "a received vector that no compute consumes is a `dataflow.receive` with no \
              OperandReuse entry — dbo-opt refuses it as a dangling non-compute op"]
#[derive(Debug, PartialEq, Eq)]
pub struct Received {
    val: Val,
    ty: Vector,
}

impl Received {
    /// EMIT THE `dataflow.receive`, AND BIND WHAT IT CARRIES.
    ///
    /// ⛔ THE ONLY CONSTRUCTOR, and it PUSHES the op rather than taking one to inspect. A
    /// `Received` therefore cannot exist without the receive that produces it, and the two cannot
    /// disagree about the value or its type. Nothing here can fail, so there is no arm to refuse in.
    pub fn receive(into: &mut Vec<super::Op>, result: Val, from: RecvEnd, ty: Vector) -> Received {
        into.push(super::Op::Dataflow(Op::Receive { result, from, ty }));
        Received { val: result, ty }
    }

    /// THE COMPUTE'S OPERAND, ONCE — CONSUMES THE EDGE.
    ///
    /// ⛔⛔ AND `#[must_use]` ALONE DOES NOT ENFORCE THAT. Rust has no linear types: a move-only
    /// value dropped out of a `Vec` triggers neither `must_use` nor any error, so a unit could mint
    /// six edges, spend two and drop four — measured, and dbo-opt refused it exactly as before.
    /// LINEARITY COMES FROM THE ARRAY: a `[Received; N]` destructured as `let [a, b] = edges` binds
    /// EVERY element or does not compile, so "minted and not consumed" is a pattern that cannot be
    /// written. That is why the edges travel as an array and never as a `Vec`.
    #[must_use]
    pub fn operand(self) -> Val {
        self.val
    }

    /// Its type — the wire's width, which the compute's operands are all at.
    #[must_use]
    pub const fn ty(&self) -> Vector {
        self.ty
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

    // ── THE FOUR SPELLINGS `stringifyComputePrecision` DOES NOT WRITE ────────────────────────────
    //
    // ⛔⛔ THE PRODUCER IS NOT THE ONLY SOURCE OF THIS ATTRIBUTE. `stringifyComputePrecision`
    // (`DSC2ToDataflowIR.hpp:54-71`) emits exactly the six above, and an island that stopped there
    // could not HOLD the DataflowIR the reference's own tests feed the PT lowering. A census of
    // `precision = "…"` on `dataflow.program_unit` across the authority tree's `dcc/test` gives:
    // `int8` 500, `fp16` 492, `mxfp8` 4, `mxfp4` 4, `bf16` 3, `fp8` 2, `fp32` 2, `int4` 1. Three of
    // those eight had no variant, so `computeUnitPrecision` (entry 094) could not be exercised on the
    // very files that pin its answer.
    /// `bf16`. ⚠️ NOT WRITTEN BY `stringifyComputePrecision` — 3 occurrences in `dcc/test`.
    Bf16,
    /// `mxfp4` — 4 occurrences, e.g. `dataflow.program_unit … {precision = "mxfp4"}` lowering to
    /// `ComputePrecision = #sentient<precision mxfp4>`
    /// (`dcc/test/Conversion/VectorChainToSentientPT/xrf_increments.mlir:374`).
    Mxfp4,
    /// `mxfp8` — 4 occurrences.
    Mxfp8,

    /// `fp80` — ⛔⛔ **AN ALIAS FOR `fp8`, AND THE ONLY REASON ENTRY 094 IS NOT THE IDENTITY.**
    ///
    /// ```cpp
    /// // Currently, we use fp80 type in MLIR to represent fp8.
    /// if (precision == "fp80") return "fp8";
    /// ```
    /// (`Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:37-38`)
    ///
    /// ⚠️ IT APPEARS **NOWHERE** in the authority tree's `dcc/test` — the remap is defensive. It is
    /// still the whole content of a scheduled function, so the island has to be able to state the
    /// input that reaches its one non-identity branch; a port that dropped it would be the identity
    /// function wearing a citation.
    Fp80,
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
            Self::Bf16 => "bf16",
            Self::Mxfp4 => "mxfp4",
            Self::Mxfp8 => "mxfp8",
            Self::Fp80 => "fp80",
        }
    }
}

/// ONE PAGE'S EXTENT IN ONE VIEW DIMENSION — the inclusive span `lo ..= hi`.
///
/// ⛔⛔ INCLUSIVE, BECAUSE THE SET IT PRINTS AS IS. A page's `idx_set` writes its upper side as
/// `-dk + <hi> >= 0` — `-d0 + 63 >= 0` is lanes `0 ..= 63`
/// (`dcc/test/Dialect/Dataflow/paged_mem_view.mlir:10`) — so `hi` is the last element and not the
/// count. See [`crate::islands::dataflow_ir::ty::BoundType`], whose `Ub` says the same thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageSpan {
    /// The first element of the page along this dimension.
    pub lo: i64,
    /// The last element, INCLUSIVE.
    pub hi: i64,
}

/// ONE PAGE'S `idx_set` — a hyper-rectangle, one span per view dimension.
///
/// # ⛔⛔ A RECTANGLE BY CONSTRUCTION, WHICH IS A VERIFIER ERROR AND TWO `DT_CHECK`s DELETED
///
/// `GetPagedLogicalMemoryViewOp::verify` refuses a page whose set is not hyper-rectangular —
/// *"idx_set should be hyper rectangular"* (`DataflowOps.cpp:291-293`, and
/// `dcc/test/Transform/TransformPagedMemView/paged_mem_view_diag_2.mlir:30` is the test that
/// provokes it) — `getPageValidity` `DT_CHECK`s the same property again
/// (`TransformPagedMemViewImpl.cpp:117-119`), and `calculateStartElementsForPage` `DT_CHECK`s the
/// consequence, *"expected constant lower bound"* (`:539`). A per-dimension span cannot express a
/// non-rectangle and always has a constant lower bound, so all three checks become the type.
///
/// ⭐ AND IT PRINTS THE SET THE REFERENCE PRINTS — see [`PageRect::as_integer_set`], whose
/// [`crate::islands::dataflow_ir::ty::IntegerSet::constant_bound`] answers exactly what
/// `getConstantBound(LB, dim)` answers on the flattened form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageRect {
    /// The spans, outermost dimension first — `d0` is `spans[0]`.
    pub spans: Vec<PageSpan>,
}

impl PageRect {
    /// THE `affine_set<..>` THIS RECTANGLE IS WRITTEN AS.
    ///
    /// A pinned dimension is one equality (`d2 == 0`); a spanning one is the PAIR `dk - lo >= 0` and
    /// `-dk + hi >= 0`, with the `- lo` omitted when `lo` is zero. That is IBM's own spelling:
    /// `(d0 >= 0, -d0 + 63 >= 0, d1 - 2 >= 0, -d1 + 3 >= 0, d2 == 0)`
    /// (`dcc/test/Dialect/Dataflow/paged_mem_view.mlir:11`, the round-tripped form of `#set1`).
    #[must_use]
    pub fn as_integer_set(&self) -> IntegerSet {
        let mut constraints = Vec::new();
        for (i, span) in self.spans.iter().enumerate() {
            let dim = AffineExpr::dim(u32::try_from(i).expect("a rank fits a u32"));
            // `dk - lo`, or the bare dimension when the page starts at zero.
            let from_lo = if span.lo == 0 {
                dim.clone()
            } else {
                dim.clone().plus(AffineExpr::Const(-span.lo))
            };
            if span.lo == span.hi {
                // A single element is pinned, which is how `buildIntegerSetFromSizes` writes a size
                // of one (`DataTransferLowering.cpp:40-69`) and how `d2 == 0` is written above.
                constraints.push(Constraint {
                    expr: from_lo,
                    is_equality: true,
                });
            } else {
                constraints.push(Constraint {
                    expr: from_lo,
                    is_equality: false,
                });
                constraints.push(Constraint {
                    expr: dim.times(-1).plus(AffineExpr::Const(span.hi)),
                    is_equality: false,
                });
            }
        }
        IntegerSet {
            dims: u32::try_from(self.spans.len()).expect("a rank fits a u32"),
            // ⛔ NO SYMBOLS: a span states its own bounds as literals, so there is nothing left to
            // substitute — the same `/*numSymbols=*/0` `buildIntegerSetFromSizes` passes.
            symbols: 0,
            constraints,
        }
    }
}

/// ONE PAGE OF A PAGED VIEW — its extent and where it starts.
///
/// # ⛔⛔ THE PAIRING IS THE TYPE, AND THAT IS THE OP'S FIRST VERIFIER ERROR DELETED
///
/// The reference carries the two halves apart — `idx_sets` is an `ArrayAttr`, `page_start_addrs` a
/// variadic operand list — and then has to check they agree: *"there should be a start address and
/// idx_set for every page"* (`DataflowOps.cpp:276-278`). Its printer asserts it a second time
/// (`:252-254`) before zipping them. One vector of pairs cannot disagree with itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    /// `idx_set = ..` — which elements of the view this page holds.
    pub idx_set: PageRect,
    /// `start_addr = ..` — where the page begins, in ELEMENTS, relative to the view's own start.
    ///
    /// ⚠️ THE VERIFIER ALSO REQUIRES THIS TO BE AN `arith.constant`: *"page start addresses should be
    /// arith::ConstantOp"* (`DataflowOps.cpp:282-285`). That is a fact about the op that defines the
    /// value, not about the value, so it stays with whoever mints it — the same shape
    /// [`super::arith::Op::Compare`]'s `rhs` documents.
    pub start_addr: Val,
}

/// A `dataflow.get_paged_logical_memory_view` — one view over SEVERAL disjoint pages.
///
/// # ⛔⛔ WITHOUT THIS OP `TransformPagedMemView` HAS NO INPUT
///
/// The whole pass exists to remove it: it picks the page a subscript can reach, guards the access
/// with the conditions that prove the page is the right one, and rewrites the access against a plain
/// [`Op::GetLogicalMemoryView`] over that page's start
/// (`dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:545-558`, entry
/// 124). A paged view is therefore the one thing every one of that file's 39 functions reads, and
/// nothing downstream of the pass ever sees one.
///
/// ⭐ ITS OPERANDS ARE `Index:$unit, Index:$start_addr, Variadic<Index>:$page_start_addrs` with
/// `AffineMapAttr:$layout_map` and `ArrayAttr:$idx_sets` (`Dataflow.td:267-299`) — so a page's start
/// address is an OPERAND and its extent is an ATTRIBUTE. [`Page`] holds them together.
///
/// ⛔ BOXED, for the reason [`super::agen::Op::CompositeLoadAndStore`] gives: it carries a page list,
/// an affine map and a memref, and an enum is as large as its largest variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedMemView {
    /// The view it binds.
    pub result: Val,
    /// `$unit` — the memory unit viewed.
    pub unit: Val,
    /// `$start_addr` — the view's own start, in elements. A page's start is relative to it: entry
    /// 124 emits `arith.addi %start_addr, %page_start_addr`.
    pub start_addr: Val,
    /// The pages, in the order the printer names them — `page0`, `page1`, ...
    pub pages: Vec<Page>,
    /// `layout_map` — how the view's indices map onto the linear region. The non-paged view entry
    /// 124 creates carries this same map.
    pub layout: AffineMap,
    /// The view's type, which the non-paged view inherits: `cast<MemRefType>(getResult().getType())`
    /// (`TransformPagedMemViewImpl.cpp:553`).
    pub ty: MemRef,
}

/// WHETHER A SIGNALLING UNIT DRAINS ITS ASYNCHRONOUS TRANSFERS FIRST —
/// `wait_immediately_for_async_transfers` (`Dataflow.td:191-200`).
///
/// ⛔⛔ THE `.td` DECLARES IT `OptionalAttr<BoolAttr>` AND EVERY LOWERING OF A SEND DEMANDS IT.
/// `lowerL3SyncOperationForAUnit` and `lowerL3SyncOperationForAGroupOfUnits` both open with
/// *"Unknown async transfers modes for sentient"* on a send that has none
/// (`DataflowToSentient.cpp:385-388`, `:677-680`), and the flag then decides the emitted sync's
/// `soft`. Making it mandatory is what makes that refusal unreachable — and it costs nothing:
/// all 292 `dataflow.sync_send` ops under `dcc/test` carry the attribute and none omits it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AsyncTransferWait {
    /// `= true` — wait for outstanding transfers before signalling. A HARD sync when lowered.
    Immediately,
    /// `= false` — signal without draining. A SOFT sync when lowered.
    Deferred,
}

impl AsyncTransferWait {
    /// The attribute's literal.
    #[must_use]
    pub const fn flag(self) -> bool {
        matches!(self, AsyncTransferWait::Immediately)
    }
}

/// ONE `dataflow` OPERATION.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
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
        /// `num_folds=`, or [`None`] for an op that carries no such attribute.
        ///
        /// ⛔ AN [`Option`] BECAUSE ONE WRITER IN THE TREE SETS IT AND NOTHING ELSE DOES:
        /// `UnitFilteringPass::cleanup` rebuilds a `get_unit` with
        /// `setAttr("num_folds", getI32IntegerAttr(new_num_folds))` (`UnitFiltering.cpp:296-297`),
        /// which is the only `setAttr` of it in `dcc/src`. Every other producer — the materializer
        /// among them — writes a `get_unit` without one, and the vendor's own hand-written PT input
        /// carries `num_folds = 1 : i32` (`dcc/test/PT/fp8-bmm-1p5.mlir:104`). A mandatory field
        /// would print the attribute on ops the reference prints bare.
        num_folds: Option<crate::units::NumFolds>,
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

    /// `dataflow.create_group (%a, %b : index, index) : index` — ONE HANDLE STANDING FOR A SET OF
    /// UNITS, which is what a collective sync addresses.
    ///
    /// ⛔⛔ ADDED FOR `separateBasedOnDestinationUnits` (entry 161), WHOSE FOURTH BUCKET IS THIS OP
    /// AND NOTHING ELSE. The reference sorts a sync's destinations into LX-corelet-0, LX-corelet-1,
    /// L3 and GROUP by `dyn_cast`-ing each destination's defining op to `GetUnitOp` **or**
    /// `CreateGroupOp` (`DataflowToSentient.cpp:768-783`). With no variant for the second the group
    /// list could never be non-empty, so the port would have been the same function with one arm
    /// deleted — and `lowerSyncLXL3ToLXL3`, its only caller, branches on exactly that list being
    /// the non-empty one (`:825-826`).
    ///
    /// ⭐ THE VENDOR WRITES IT SIX UNITS WIDE: `%group0 = dataflow.create_group (%lx_lu_unit0,
    /// %lx_lu_unit1, %lx_su_unit0, %lx_su_unit1, %l3_lu_unit, %l3_su_unit : index, index, index,
    /// index, index, index) : index` (`dcc/test/L3SU/sync-op-l3su.mlir:75`), and two units wide at
    /// `:81`.
    ///
    /// ⛔ THE OPERAND LIST IS OPTIONAL IN THE ASSEMBLY FORMAT — `` `(` ($unit_ids^ `:`
    /// type($unit_ids))? `)` `` (`Dataflow.td:155`) — so an empty group prints its parentheses with
    /// nothing between them, NOT `( : )`.
    CreateGroup {
        /// `$group_id` — the one handle the group is addressed by.
        result: Val,
        /// `$unit_ids` — the members, in the order they were named.
        unit_ids: Vec<Val>,
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

    /// `dataflow.get_paged_logical_memory_view %unit, %start {layout_map} {segments = page0: (..)}`
    /// — see [`PagedMemView`], which is the whole of it.
    GetPagedLogicalMemoryView(Box<PagedMemView>),

    /// `dataflow.program_unit %unit {precision} : { .. }` — one unit's whole program.
    ProgramUnit {
        /// The units this program runs on. More than one where the same program is bound across
        /// program time steps (`Dataflow.td:99-104`).
        units: Vec<Val>,
        /// `precision=`, which selects the MAC opcode. Absent on a unit that computes nothing.
        precision: Option<Precision>,
        /// The body.
        body: Vec<super::Op>,
    },

    /// `dataflow.send %to, %data : vector<..>`.
    ///
    /// ⛔ `to` IS THE FIRST HOP, NOT THE DESTINATION, where the transfer states a via:
    /// `to_unit = dst.via_.empty() ? dst.loc_.unit_ : dst.via_.front()`
    /// (`SNTransferLowering.cpp:2731-2732`).
    Send {
        /// The unit sent to — ONE END OF A [`crate::islands::dataflow_ir::link::Link`], not a unit handle a caller chose.
        to: SendEnd,
        /// The data.
        data: Val,
        /// Its type.
        ty: Vector,
    },

    /// `dataflow.receive %from : vector<..>`.
    Receive {
        /// The vector it binds.
        result: Val,
        /// The unit received from — THE OTHER END OF THE SAME [`crate::islands::dataflow_ir::link::Link`] the send spent.
        from: RecvEnd,
        /// Its type.
        ty: Vector,
    },

    /// `dataflow.sync_send %unit : index`.
    SyncSend {
        /// The unit signalled.
        to: Val,
        /// Which signal, for the debug name.
        signal: SyncSignal,
        /// `wait_immediately_for_async_transfers` — ⛔ MANDATORY HERE, `OptionalAttr<BoolAttr>` in
        /// the `.td`. See [`AsyncTransferWait`].
        wait: AsyncTransferWait,
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
    /// ⭐ THE FIELDS ARE THEIR OWN STRUCT because the rung above forwards ALL of them at once:
    /// `lowerOpaqueOperation` (`DataflowToSentient.cpp:1984-2012`) reads the three dictionaries, the
    /// func name and the `dbgName` and hands every one of them to `sentient::OpaqueOp::create`. A
    /// five-field inline variant makes that one call five arguments; see [`Opaque`].
    Opaque(Opaque),
}

/// WHAT ONE `dataflow.opaque` CARRIES — the whole `.smc` body's binding, in one value.
///
/// ⛔⛔ THE TWO DICTIONARIES ARE CROSSED RELATIVE TO THEIR NAMES. `read_write_reg_map_` is filled
/// from the body's INTERNAL registers and `read_only_reg_map_` from the caller-bound INPUT/OUTPUT
/// ones (`ddcv1.cpp:3369-3391`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Opaque {
    /// `func_name=`.
    pub func: OpaqueFunc,
    /// `read_write_register_dictionary=` — the body's own scratch, from `internal_registers=`.
    pub read_write: Vec<(RegName, RegAddr)>,
    /// `read_only_register_dictionary=` — caller-bound, from `input_output_registers=`.
    pub read_only: Vec<(RegName, RegAddr)>,
    /// `parameter_dictionary=`, including the `prec` the op's own data format sets.
    pub params: Vec<(ParamKey, ParamValue)>,
    /// `dbgName=` — ⛔ CARRIED BECAUSE THE RUNG ABOVE FORWARDS IT.
    ///
    /// `OptionalAttr<StrAttr>:$dbgName` (`Dataflow.td:342`), and `lowerOpaqueOperation` passes
    /// `getDbgNameAttr(opaque_op)` straight into the `sentient.opaque` it builds
    /// (`DataflowToSentient.cpp:2005-2010`). IBM's own input writes it —
    /// `dataflow.opaque {dbgName="opaque_op #1", func_name= "reciprocal", ...}`
    /// (`dcc/test/Conversion/DataflowToSentient/opaque.mlir:32`) — so without this field the port of
    /// that pass has nothing to forward and the lowered op loses the only name a debugger has for it.
    pub dbg_name: Option<String>,
}

/// ONE `dataflow` OP AS TEXT. The caller has already indented the opening line.
pub(crate) fn emit(out: &mut String, op: &Op, depth: usize) {
    match op {
        Op::GetUnit {
            result,
            residency,
            unit,
            num_folds,
        } => {
            use crate::units::Residency;

            // Attributes are written in MLIR's own key order, which is alphabetical: core, corelet,
            // name, type. Which of `core`/`corelet` appear is the residency's, and a missing
            // `corelet` is a fact rather than a hole — `ExtendUnitNameToCorelet` errors on one that
            // is absent where it needs it, and reads `0` as the name suffix `"0"`
            // (`DataflowToSentient.cpp:104-117`).
            let spelling = unit.spelling();
            let (attrs, name) = match residency {
                Residency::Global => (String::new(), spelling.to_owned()),
                Residency::Scratchpad { core } => (
                    format!("core = {} : i32, ", core.get()),
                    format!("C{}-{spelling}", core.get()),
                ),
                Residency::CoreWide { core } => (
                    format!("core = {} : i32, corelet = 0 : i32, ", core.get()),
                    format!("C{}-{spelling}", core.get()),
                ),
                Residency::Corelet { core, corelet } => (
                    format!(
                        "core = {} : i32, corelet = {} : i32, ",
                        core.get(),
                        corelet.get()
                    ),
                    format!("C{}-{spelling}-CL{}", core.get(), corelet.get()),
                ),
            };
            // ⛔ `num_folds` SITS BETWEEN `name` AND `type`, which is where alphabetical order
            // puts it: `{core = 0 : i32, corelet = 0 : i32, name = "ptrow0-CL0",
            // num_folds = 1 : i32, type = "ptrow0"}` (`dcc/test/PT/fp8-bmm-1p5.mlir:104`).
            let folds = match num_folds {
                Some(folds) => format!("num_folds = {} : i32, ", folds.0),
                None => String::new(),
            };
            let _ = writeln!(
                out,
                "{} = dataflow.get_unit {{{attrs}name = \"{name}\", {folds}type = \"{spelling}\"}} : index",
                print::val(*result),
            );
        }
        Op::GetLocalUnit { result, of, which } => {
            let _ = writeln!(
                out,
                "{} = dataflow.get_local_unit {} {{name = \"{}\"}} : index",
                print::val(*result),
                print::val(*of),
                which.spelling()
            );
        }
        Op::CreateGroup { result, unit_ids } => {
            // ⭐ THE TYPE LIST REPEATS `index` ONCE PER MEMBER, and the whole `operands : types`
            // clause is elided when the group is empty (`Dataflow.td:155`).
            let members = if unit_ids.is_empty() {
                String::new()
            } else {
                let types = std::iter::repeat_n("index", unit_ids.len())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{} : {types}", print::vals(unit_ids))
            };
            let _ = writeln!(
                out,
                "{} = dataflow.create_group ({members}) : index",
                print::val(*result)
            );
        }
        Op::GetLogicalMemoryView {
            result,
            from,
            start,
            layout,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = dataflow.get_logical_memory_view {}, {} {{layout_map = {}}} : index, index, {}",
                print::val(*result),
                print::val(*from),
                print::val(*start),
                print::affine_map(layout),
                print::memref(ty)
            );
        }
        // ⭐ THREE LINES, AS THE OP'S OWN PRINTER WRITES IT: the two operands and the attribute
        // dictionary, then `{segments =` on its own line, then one line per page
        // (`DataflowOps.cpp:240-269`). The trailing type list is the unit's, the start address's and
        // the result's — `index, index, memref<..>`.
        //
        // ⚠️ ONE CHARACTER FEWER THAN THE REFERENCE: it writes `"  {segments = "` and then a
        // newline (`:248-249`), so its line ends in a SPACE. FileCheck's substring match hides it,
        // which is why IBM's own autogenerated expectation reads `{segments =`
        // (`dcc/test/Dialect/Dataflow/paged_mem_view.mlir:20`), and the parser cannot see it either.
        // Emitting the space would make the test below un-writable from the file it quotes.
        Op::GetPagedLogicalMemoryView(view) => {
            let PagedMemView {
                result,
                unit,
                start_addr,
                pages,
                layout,
                ty,
            } = view.as_ref();
            let _ = writeln!(
                out,
                "{} = dataflow.get_paged_logical_memory_view {}, {} {{layout_map = {}}}",
                print::val(*result),
                print::val(*unit),
                print::val(*start_addr),
                print::affine_map(layout)
            );
            print::indent(out, depth);
            out.push_str("  {segments =");
            for (page_num, page) in pages.iter().enumerate() {
                if page_num > 0 {
                    out.push(',');
                }
                out.push('\n');
                print::indent(out, depth);
                let _ = write!(
                    out,
                    "    page{page_num}: (idx_set = {}, start_addr = {})",
                    print::integer_set(&page.idx_set.as_integer_set()),
                    print::val(page.start_addr)
                );
            }
            let _ = writeln!(out, "}} : index, index, {}", print::memref(ty));
        }
        Op::ProgramUnit {
            units,
            precision,
            body,
        } => {
            let precision = match precision {
                Some(p) => format!(" {{precision = \"{}\"}}", Precision::spelling(*p)),
                None => String::new(),
            };
            let _ = writeln!(
                out,
                "dataflow.program_unit {}{precision} : {{",
                print::vals(units)
            );
            for inner in body {
                print::emit(out, inner, depth + 1);
            }
            print::indent(out, depth);
            out.push_str("}\n");
        }
        Op::Send { to, data, ty } => {
            let _ = writeln!(
                out,
                "dataflow.send {}, {} : {}",
                print::val(to.val()),
                print::val(*data),
                print::vector(*ty)
            );
        }
        Op::Receive { result, from, ty } => {
            let _ = writeln!(
                out,
                "{} = dataflow.receive {} : {}",
                print::val(*result),
                print::val(from.val()),
                print::vector(*ty)
            );
        }
        Op::SyncSend { to, signal, wait } => {
            let _ = writeln!(
                out,
                // ⛔⛔ THE WAIT MODE IS OPTIONAL IN THE DIALECT AND MANDATORY TO LOWER, so it is a
                // FIELD and no longer a literal here — see [`AsyncTransferWait`]. It was hard-coded
                // `true` while nothing read it; entry 223 does, and a send whose flag the printer
                // invents would lower to a sync whose `soft` nobody chose.
                "dataflow.sync_send {} {{dbgName = \"{}\", \
                 wait_immediately_for_async_transfers = {}}} : index",
                print::val(*to),
                signal.spelling(),
                wait.flag()
            );
        }
        Op::SyncRecv { from, signal } => {
            let _ = writeln!(
                out,
                "dataflow.sync_recv {} {{dbgName = \"{}\"}} : index",
                print::val(*from),
                signal.spelling()
            );
        }
        Op::ImplicitSync {
            view,
            dst,
            size,
            view_ty,
        } => {
            let _ = writeln!(
                out,
                "dataflow.implicit_sync_on_streaming_buffer {}, {}, {} : {}, index, index",
                print::val(*view),
                print::val(*dst),
                print::val(*size),
                print::memref(view_ty)
            );
        }
        Op::Opaque(opaque) => {
            let Opaque {
                func,
                read_write,
                read_only,
                params,
                dbg_name,
            } = opaque;
            // ⛔ `dbgName` SORTS FIRST — `d` before `f`. A `DictionaryAttr` is key-ordered, so the
            // optional attribute is not a tail; it opens the dictionary when it is present.
            let named = dbg_name
                .as_ref()
                .map_or_else(String::new, |name| format!("dbgName = \"{name}\", "));
            let _ = writeln!(
                out,
                "dataflow.opaque {{{}func_name = \"{}\", parameter_dictionary = {}, \
                 read_only_register_dictionary = {}, read_write_register_dictionary = {}}}",
                named,
                func.spelling().to_lowercase(),
                dictionary(params, ParamKey::spelling, |v| ParamValue::spelling(v)
                    .to_owned()),
                // ⛔⛔ THE `R` IS LOAD-BEARING, NOT DECORATION. `insertReg` writes
                // `"R" + std::to_string(startAddress)` (`ddc/ddcv1.cpp:3350`) and the consumer
                // takes it back off by POSITION: `port_str = "lrf" + port_str.erase(0, 1)`
                // (`dcc/src/Dialect/Sentient/Utils.cpp:157`). A bare `"20"` becomes `lrf0` — the
                // wrong register, silently — and a bare `"8"` becomes `"lrf"`, which is no port at
                // all and aborts with "Invalid port".
                dictionary(read_only, RegName::spelling, |a: RegAddr| format!(
                    "R{}",
                    a.0
                )),
                dictionary(read_write, RegName::spelling, |a: RegAddr| format!(
                    "R{}",
                    a.0
                ))
            );
        }
    }
}

/// A `{key = "value", ..}` attribute dictionary, KEY-SORTED.
///
/// ⛔ SORTED BECAUSE MLIR SORTS. A `DictionaryAttr` is stored key-ordered, so a round trip through
/// the parser reorders anything else — and a printer whose output does not survive a round trip
/// cannot be checked against the vendored files.
/// ONE MLIR DICTIONARY ATTRIBUTE, sorted by key so the emission is byte-reproducible.
///
/// ⭐ THE KEYS AND VALUES ARRIVE AS TOKENS and are spelled here, at the one place text is the point.
/// Everything upstream holds enums, so a key that does not exist is a variant that does not exist.
fn dictionary<K: Copy, V: Copy>(
    entries: &[(K, V)],
    key: impl Fn(K) -> &'static str,
    value: impl Fn(V) -> String,
) -> String {
    let mut sorted: Vec<(&'static str, String)> =
        entries.iter().map(|(k, v)| (key(*k), value(*v))).collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    format!(
        "{{{}}}",
        sorted
            .iter()
            .map(|(key, value)| format!("{key} = \"{value}\""))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(test)]
mod tests {
    use crate::islands::dataflow_ir::dialects::dataflow::{
        Op, Page, PageRect, PageSpan, PagedMemView,
    };
    use crate::islands::dataflow_ir::dialects::{self, Val};
    use crate::islands::dataflow_ir::print::emit;
    use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, MemRef};
    use crate::units::{Core, Corelet, DfirUnit, Residency};

    /// A core this build's arch has.
    fn core(index: u32) -> Core {
        Core::checked(index).expect("cores 0 and 1 exist on every arch this crate builds for")
    }

    /// A corelet this build's arch has.
    fn corelet(index: u32) -> Corelet {
        Corelet::checked(index).expect("corelets 0 and 1 exist on every arch this crate builds for")
    }

    /// ⭐⭐ IBM'S OWN UNIT BLOCK, REPRODUCED — all four residency classes, carried as TEXT rather
    /// than as a shape.
    ///
    /// The nineteen lines are `/tmp/ktir_ref/export/debug/dfir.mlir:45-63`, the DataflowIR the
    /// stock KTIR pathway produced on a two-core two-corelet run. They are the whole point of
    /// [`Residency`]: `C0-l3lu` carries `corelet = 0`, `C0-lx` carries NO corelet, and `hbm`
    /// carries NEITHER attribute — three distinct attribute shapes, of which the previous
    /// `core: u32, corelet: Option<u32>` pair could spell two.
    ///
    /// ⛔ AN ASSERTION ON THE SHAPE WOULD NOT HAVE CAUGHT THE BUG THIS REPLACES. "The HBM line has
    /// no `core`" passes just as well on an emitter that also drops `core` from the LX. This
    /// compares IBM's bytes in IBM's order, so a wrong attribute set on any one of the four classes
    /// is a diff rather than a silent agreement.
    #[test]
    fn reproduces_ibms_unit_block() {
        let mut ops = Vec::new();
        let mut next = 0u32;
        let mut unit = |residency, unit| {
            let op = dialects::Op::Dataflow(Op::GetUnit {
                result: Val(next),
                residency,
                unit,
                num_folds: None,
            });
            next += 1;
            ops.push(op);
        };

        // Non-parallel compute, declared in the core group: `core` AND `corelet = 0`.
        for which in [DfirUnit::L3lu, DfirUnit::L3su] {
            for c in 0..2 {
                unit(Residency::CoreWide { core: core(c) }, which);
            }
        }
        // Parallel compute, declared in the corelet group. Corelet-major, as IBM emits it.
        for which in [DfirUnit::Lxlu, DfirUnit::Sfp, DfirUnit::Lxsu] {
            for cl in 0..2 {
                for c in 0..2 {
                    unit(
                        Residency::Corelet {
                            core: core(c),
                            corelet: corelet(cl),
                        },
                        which,
                    );
                }
            }
        }
        // Then the memories: the global root first, then one scratchpad per core.
        unit(Residency::Global, DfirUnit::Hbm);
        for c in 0..2 {
            unit(Residency::Scratchpad { core: core(c) }, DfirUnit::Lx);
        }

        let mut got = String::new();
        for op in &ops {
            emit(&mut got, op, 0);
        }

        let want = r#"%0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3lu", type = "l3lu"} : index
%1 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l3lu", type = "l3lu"} : index
%2 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3su", type = "l3su"} : index
%3 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l3su", type = "l3su"} : index
%4 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxlu-CL0", type = "lxlu"} : index
%5 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-lxlu-CL0", type = "lxlu"} : index
%6 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxlu-CL1", type = "lxlu"} : index
%7 = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-lxlu-CL1", type = "lxlu"} : index
%8 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfp-CL0", type = "sfp"} : index
%9 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-sfp-CL0", type = "sfp"} : index
%10 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-sfp-CL1", type = "sfp"} : index
%11 = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-sfp-CL1", type = "sfp"} : index
%12 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxsu-CL0", type = "lxsu"} : index
%13 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-lxsu-CL0", type = "lxsu"} : index
%14 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxsu-CL1", type = "lxsu"} : index
%15 = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-lxsu-CL1", type = "lxsu"} : index
%16 = dataflow.get_unit {name = "hbm", type = "hbm"} : index
%17 = dataflow.get_unit {core = 0 : i32, name = "C0-lx", type = "lx"} : index
%18 = dataflow.get_unit {core = 1 : i32, name = "C1-lx", type = "lx"} : index
"#;

        for (at, (want_line, got_line)) in want.lines().zip(got.lines()).enumerate() {
            assert_eq!(
                want_line, got_line,
                "unit line {at} diverges from IBM's own DataflowIR"
            );
        }
        assert_eq!(
            want.lines().count(),
            got.lines().count(),
            "emitted a different number of unit lines than IBM's block has"
        );
    }

    /// 🎯 IBM'S OWN PAGED VIEW, REPRODUCED — `dcc/test/Dialect/Dataflow/paged_mem_view.mlir`, whose
    /// only purpose is to round-trip this op.
    ///
    /// ⛔⛔ THE FOUR-LINE SHAPE IS THE OP'S OWN PRINTER AND NOTHING GENERIC. `printOptionalAttrDict`
    /// elides `idx_sets` and then a hand-written loop writes `page<N>: (idx_set = .., start_addr =
    /// ..)` one per line with the comma BEFORE the newline (`DataflowOps.cpp:239-267`). Every part of
    /// that is a choice: the two-space `{segments =` continuation, the four-space page indent, the
    /// three-type trailing list. A test that only checked "the pages are all there" would agree with
    /// an emitter that wrote them in a flat attribute dictionary, which `dcc-opt` cannot parse.
    ///
    /// ⭐⭐ AND IT PINS THE TWO SETS, WHICH IS WHERE [`PageRect`] EARNS ITS KEEP. `#set` starts at
    /// zero and prints `d0 >= 0`; `#set1` starts at 2 and prints `d1 - 2 >= 0` — a SUBTRACTION, the
    /// spelling MLIR uses for a negative addend and the reason [`crate::islands::dataflow_ir::print`]
    /// grew that arm. The pinned third dimension prints `d2 == 0` rather than a `>= 0` pair.
    ///
    /// ⚠️ ONE DEVIATION, AND IT IS THE ISLAND'S TYPE: IBM writes `memref<?x?x64xf16>`, and
    /// [`crate::islands::dataflow_ir::ty::MemRef`] holds extents rather than a dynamic marker (a `?`
    /// in an emitted view is a shape nobody downstream could size), so this asks for
    /// `memref<64x4x64xf16>` — the concrete extents the two pages cover.
    #[test]
    fn reproduces_ibms_paged_memory_view() {
        // `#map = affine_map<(d0, d1, d2) -> (d2 * 64 + d1 * 64 + d0)>`.
        let layout = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::dim(2)
                    .times(64)
                    .plus(AffineExpr::dim(1).times(64))
                    .plus(AffineExpr::dim(0)),
            ],
        };
        // 64 lanes of every page; page 0 holds rows 0..=1 and page 1 rows 2..=3; the third
        // dimension is pinned to element 0 in both.
        let page = |lo, hi, start_addr| Page {
            idx_set: PageRect {
                spans: vec![
                    PageSpan { lo: 0, hi: 63 },
                    PageSpan { lo, hi },
                    PageSpan { lo: 0, hi: 0 },
                ],
            },
            start_addr,
        };
        let op = dialects::Op::Dataflow(Op::GetPagedLogicalMemoryView(Box::new(PagedMemView {
            result: Val(5),
            unit: Val(1),
            start_addr: Val(2),
            pages: vec![page(0, 1, Val(3)), page(2, 3, Val(4))],
            layout,
            ty: MemRef {
                shape: vec![64, 4, 64],
                elem: ElemType::F16,
            },
        })));

        let mut got = String::new();
        emit(&mut got, &op, 0);

        // `paged_mem_view.mlir:19-22`, with the shape noted above and IBM's `%[[VAL_n]]` capture
        // names resolved to the values they stood for.
        let want = concat!(
            "%5 = dataflow.get_paged_logical_memory_view %1, %2 ",
            "{layout_map = affine_map<(d0, d1, d2) -> (d2 * 64 + d1 * 64 + d0)>}\n",
            "  {segments =\n",
            "    page0: (idx_set = affine_set<(d0, d1, d2) : ",
            "(d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 == 0)>, start_addr = %3),\n",
            "    page1: (idx_set = affine_set<(d0, d1, d2) : ",
            "(d0 >= 0, -d0 + 63 >= 0, d1 - 2 >= 0, -d1 + 3 >= 0, d2 == 0)>, start_addr = %4)",
            "} : index, index, memref<64x4x64xf16>\n",
        );
        assert_eq!(got, want);
    }
}
