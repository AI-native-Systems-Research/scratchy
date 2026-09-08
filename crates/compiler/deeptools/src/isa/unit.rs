// SPDX-License-Identifier: Apache-2.0
//! WHICH INSTRUCTION STREAM RUNS AN OP, AND WHERE A COMPUTE'S RESULT GOES.
//!
//! * [`Executor`] / [`UnitTarget`] / [`PtRowIdx`] / [`NUM_PT_ROWS`] — a unit is addressed BY ROW on the PT
//!   (`generatePTInitPacket`, `finalizeInitFlit`, `dip.cpp:2124-2142`), so "which stream" is a SPAN over rows
//!   and not one component. [`crate::islands::OnUnit`] is keyed on it, so every island reads it.
//! * [`TgtEncoding`] — a field's encoding, whose number comes from that field's own `encodeList` for the
//!   (component, instruction type) being emitted; the same variant encodes differently on the PE and the SFP
//!   (`ConstructProgIRHelper.cpp:1362-1366`).
//! * [`forward_via`] — the send-side port table, `getOperandFromSendOp` (`VectorOperands.cpp:110-135`).
//!
//! ⭐ ISA VOCABULARY, SO IT LIVES IN `isa`: it belongs to no single island, and every island needs it.

use crate::generated::Unit;

/// `tgtencoding=` — which port the result leaves by.
///
/// ⛔⛔ THESE TWO VARIANTS ARE THE TEMPLATE'S VOCABULARY AND NOT THE FIELD'S VALUE SET, and the field's set is PER
/// COMPONENT. `defineField`'s `tgtencoding` on the SFP states `{l0:0, lx:1, pe:2, pt:3}` — four destinations, with
/// `pe` at 2 — while the PE's own states `{lx:1, sfp:2}`, since "tgtencoding can't be itself"
/// (`ConstructProgIRHelper.cpp:1362-1366`). So `Sfp` encodes as 2 on the PE and 2 means `pe` on the SFP.
///
/// ⛔ WHICH MEANS THIS SLOT CANNOT BE ENCODED FROM ITS VARIANT ALONE. Like a source operand's port, the number must
/// come from the field's own `encodeList` for the (component, instruction type) being emitted — a helper mapping
/// `Lx`/`Sfp` to fixed numbers is correct on one component and silently wrong on the other.
///
/// ⛔ AND THE FIELD EXISTS ONLY FROM SEN1P5 ON. Up to RCUDD1A the destinations are separate fields (see
/// [`Target`]), so on the default arch a forward is not this pair at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TgtEncoding {
    /// `lx`.
    Lx,
    /// `sfp`.
    Sfp,
    /// `l0` — the SFP's link to its corelet's L0 store unit, field `tgtl0`.
    ///
    /// ⛔⛔ THE ONLY WAY A COMPUTE'S VALUE REACHES AN L0. An L0 store takes its stick off the SFP and nothing
    /// else, so a value the SFP holds gets there by the SFP naming the L0 in its own destination field —
    /// `output == "l0"` selects `tgtl0` (`ConstructProgIRHelper.cpp:1345-1347`). The other route into an L0 is an
    /// LX load stamped by `SETDSTMASK`, which is a transfer rather than a forward.
    ///
    /// ⭐ AND IT IS A SEPARATE FIELD FROM `tgtrf`, SO A COMPUTE DOES BOTH. Up to RCUDD1A every direction has its
    /// own field and `setSentientComputeOutputProgIROperands` sets one per output (`:1331-1360`) — writing a
    /// register and forwarding to the L0 are not alternatives.
    L0,
    /// `south` — the PT's own southward link, field `tgts`.
    ///
    /// ⭐ THE PT'S DIRECTIONS ARE ITS OWN BRANCH. `setSentientComputeOutputProgIROperands` handles `comp == PT`
    /// before the PE/SFP cases and maps `"east"` to `tgte` and `"south"` to `tgts`
    /// (`ConstructProgIRHelper.cpp:1319-1330`) — neither of which any PE/SFP output can name. `bmm.ddl:254` binds
    /// `%pt_dst00_south = ddl.unit(%ptsum) {unit="ptsouth"}`, so a matmul's PSUM leaves southward.
    ///
    /// ⛔ AND `tgts` NARROWS FROM SEN1P5 ON: `{no, src0, src2, result}` up to RCUDD1A and `{no, result}` after
    /// (`isa.cpp:325-328`). The value still comes from the field's own encodeList, so the narrowing is the table's
    /// business rather than this enum's.
    South,
    /// `pt` — a PE or SFP forwarding INTO the PT array, field `tgtpt`.
    ///
    /// ⭐ THE SENDER DECIDES, NOT THE DESTINATION. `getOperandFromSendOp` branches on `comp` FIRST
    /// (`VectorOperands.cpp:112-135`): from the PT, a PT-or-PE destination is `"south"`; from a PE or an SFP,
    /// a PT destination is `"pt"` and a PE destination is `"pe"`. One destination, two spellings, chosen by
    /// who is sending — which is why [`crate::isa::unit::forward_via`] takes both.
    Pt,
    /// `pe` — an SFP forwarding to the PE, field `tgtpe`. See [`TgtEncoding::Pt`] for why the sender decides.
    Pe,
}

/// One of the two units that run a whole compute by themselves.
///
/// Not "a unit that is not the PT": the PT is addressed BY ROW, so it is not one target but several. See
/// [`UnitTarget`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingleUnit {
    /// `sfp`.
    Sfp,
    /// `pe`.
    Pe,
}

/// Which row of the systolic array. `ptrow0`, and the ends of `ptrow1-3` / `ptrow1-7`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PtRowIdx(u8);

impl PtRowIdx {
    /// Only the generated tables mint these: a row index is a template's word, never a computed one.
    pub(crate) const fn from_row(row: u8) -> Self {
        Self(row)
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}


/// What a template's `unit=` names — ONE unit, or a RANGE OF PT ROWS.
///
/// ⛔ THIS IS THE DISTINCTION THE NUKED CRATE LOST. It asked `sen_component(unit) -> Option<SenComponents>`,
/// so `pt`, `ptrow1-3` and `ptrow1-7` all came back `None` and the caller DROPPED the statement — 24 real
/// `ddl.compute`s, 18 of them `unit="pt"`, which is the matmul. The `Option` had to answer three questions
/// with two answers: is this a unit, is it one unit, and if several then which. Three cases, three variants.
///
/// [`Self::AllPtRows`] stays a case of its own because the row count is ARCH-DEPENDENT (8 rows, 4 on sen1.5),
/// so `pt` is not a span this crate can write down — only the arch can.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitTarget {
    /// One unit, one instruction stream.
    Single(SingleUnit),
    /// `pt` — every row of the systolic array, however many the arch has.
    AllPtRows,
    /// `ptrow<first>-<last>`, inclusive; `ptrow<n>` is the span `n..=n`.
    PtRowSpan { first: PtRowIdx, last: PtRowIdx },
}

impl UnitTarget {
    /// The stream a template's `unit=` names.
    ///
    /// ⭐ `pt` BECOMES A SPAN OVER THE ARCH'S ROWS, which is the one place [`NUM_PT_ROWS`] is read: the DDL says
    /// "every row" and only the arch says how many that is.
    pub const fn executor(self) -> Executor {
        match self {
            Self::Single(SingleUnit::Sfp) => Executor::Whole(crate::isa::regfile::Component::Sfp),
            Self::Single(SingleUnit::Pe) => Executor::Whole(crate::isa::regfile::Component::Pe),
            Self::AllPtRows => Executor::PtRows {
                first: PtRowIdx::from_row(0),
                last: PtRowIdx::from_row(NUM_PT_ROWS - 1),
            },
            Self::PtRowSpan { first, last } => Executor::PtRows { first, last },
        }
    }
}

impl UnitTarget {
    /// WHICH COMPONENT RUNS THIS — the field table's key and the opcode value's, both of which are per component.
    ///
    /// ⭐ EVERY PT ROW IS THE PT. A row span is a range of rows of the ONE systolic array, so all three PT forms
    /// answer `Pt`: `regInfoPerUnit` has one entry for the component and `initIsa` one field table, and a row is
    /// not a unit with files of its own. Which rows an instruction addresses is a separate question, answered by
    /// this same value's other variants and read by bridge 3.
    pub const fn component(self) -> crate::isa::regfile::Component {
        use crate::isa::regfile::Component;
        match self {
            Self::Single(SingleUnit::Sfp) => Component::Sfp,
            Self::Single(SingleUnit::Pe) => Component::Pe,
            Self::AllPtRows => Component::Pt,
            Self::PtRowSpan { .. } => Component::Pt,
        }
    }
}

/// HOW MANY ROWS THE SYSTOLIC ARRAY HAS — `numPTRows = coreArch <= RCUDD1A_ISA ? 8 : 4`
/// (`sys-arch-spec/sysdef.cpp:199`).
///
/// ⛔ NOT [`crate::packet::block::PtRow::COUNT`], WHICH IS A DIFFERENT NUMBER'S TWIN. That one is how many rows can
/// be ADDRESSED — eight, the flit's slice count and the size of `dip.cpp`'s per-row arrays — while this is how many
/// the part HAS. They agree on RCUDD1A and differ on SEN1P5, so a `pt` span taken from the addressable bound would
/// emit four rows that do not exist.
pub const NUM_PT_ROWS: u8 = match cfg!(feature = "arch-sen1p5") {
    true => 4,
    false => 8,
};

/// HOW MANY `NOP`s A PROGRAM PADS WITH BEFORE ITS `RETURN` — `createFinalSenProg`'s PT pad
/// (`dcgbeCodegen.cpp:2617-2621`).
///
/// ⛔ ITS OWN TYPE BECAUSE ZERO IS AN ANSWER, not a missing one: every unit but the PT takes none, and an
/// `Option<u8>` here would make "no pad" and "not asked" the same state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PadCount(u8);

impl PadCount {
    pub const fn of(count: u8) -> Self {
        Self(count)
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}


/// WHICH INSTRUCTION STREAM AN OP BELONGS TO.
///
/// ⛔⛔ THE PT IS `numPTRows` STREAMS AND NOT ONE, which is what a bare [`crate::isa::regfile::Component`] cannot
/// say. `constructPTInitPacket` loops `for (int p = 0; p < sysDef.numPTRows; p++)` and names each row's component
/// `"pt_row" + p` (`dip.cpp:2043`, `:2124-2130`), then lays each row's flits by ROW rather than by unit column
/// (`finalizeInitFlit(p, …)`, `:2138`). So "which rows" is not a detail of the PT's one program — it is which
/// program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Executor {
    /// A component whose program is ONE stream — every unit but the PT.
    Whole(crate::isa::regfile::Component),
    /// A span of PT rows, inclusive. The instruction belongs to each row's own stream.
    PtRows { first: PtRowIdx, last: PtRowIdx },
}

impl Executor {
    /// WHICH COMPONENT RUNS THIS — the field table's key and the opcode value's, both of which are per component.
    pub const fn component(self) -> crate::isa::regfile::Component {
        match self {
            Self::Whole(comp) => comp,
            Self::PtRows { .. } => crate::isa::regfile::Component::Pt,
        }
    }

    /// A component's whole program, refusing the PT — which is a row span, not a component.
    pub const fn whole(comp: crate::isa::regfile::Component) -> Self {
        match comp {
            crate::isa::regfile::Component::Pt => panic!(
                "the PT is `numPTRows` instruction streams, one per row (`dip.cpp:2043`, `:2124-2130`), so it has \
                 no whole-component program — build the stream from the `unit=`'s row span"
            ),
            comp => Self::Whole(comp),
        }
    }

    /// Whether this stream includes a given PT row.
    pub const fn covers_pt_row(self, row: u8) -> bool {
        match self {
            Self::Whole(_) => false,
            Self::PtRows { first, last } => first.get() <= row && row <= last.get(),
        }
    }
}



/// WHICH COMPUTE IS SENDING — `comp` in `getOperandFromSendOp`, the table's FIRST test.
///
/// ⛔ ITS OWN TYPE, AND NOT [`Toward`]'s. Both name a place in the array and both are arguments of the same
/// table, so a transposed pair answers for the wrong direction and still compiles — see [`forward_via`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sender(pub crate::isa::regfile::Component);

/// WHERE THE STICK IS GOING — the `SendOp`'s unit. See [`Sender`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Toward(pub Unit);

/// WHICH DIRECTION A FORWARD IS — the send-side port table, `getOperandFromSendOp`
/// (`VectorOperands.cpp:110-135`).
///
/// ⭐ ONE TABLE, BECAUSE TWO PASSES ASK IT. Bridge 2 asks it of a BOUND output — a name whose `ddl.unit` had no
/// allocation, so the value leaves by a port — and bridge 1 asks it of a synthesized shuffle step, whose output unit
/// is STATED by the template rather than decided by the binding. A second copy would let the two disagree about which
/// direction `lxsu` is.
///
/// ⛔⛔⛔ THE SENDER DECIDES BEFORE THE DESTINATION DOES, AND THIS TOOK ONLY THE DESTINATION.
/// `getOperandFromSendOp` opens with `if (comp == PT)` and only then looks at where the stick is going
/// (`VectorOperands.cpp:112-135`):
///
/// ```text
/// if (comp == PT) { if (generic == PT || record->second == PE) link = "south"; }


/// WHICH DIRECTION A FORWARD IS — the send-side port table, `getOperandFromSendOp`
/// (`VectorOperands.cpp:110-135`).
///
/// ⭐ ONE TABLE, BECAUSE TWO PASSES ASK IT. Bridge 2 asks it of a BOUND output — a name whose `ddl.unit` had no
/// allocation, so the value leaves by a port — and bridge 1 asks it of a synthesized shuffle step, whose output unit
/// is STATED by the template rather than decided by the binding. A second copy would let the two disagree about which
/// direction `lxsu` is.
///
/// ⛔⛔⛔ THE SENDER DECIDES BEFORE THE DESTINATION DOES, AND THIS TOOK ONLY THE DESTINATION.
/// `getOperandFromSendOp` opens with `if (comp == PT)` and only then looks at where the stick is going
/// (`VectorOperands.cpp:112-135`):
///
/// ```text
/// if (comp == PT) { if (generic == PT || record->second == PE) link = "south"; }
/// else {  // PE/SFP
///   if      (generic == PT)                       link = "pt";
///   else if (record->second == PE)                link = "pe";
///   else if (record->second == SFP)               link = "sfp"  (+"ring" when comp == SFP);
///   else if (record->second == L0LU || == L0SU)   link = "l0";
///   else if (record->second == LXLU || == LXSU)   link = "lx";
/// }
/// ```
///
/// So a PE destination is `"south"` from the PT and `"pe"` from an SFP — ONE DESTINATION, TWO SPELLINGS.
/// Asking with the destination alone cannot tell them apart, and this refused `bmm.ddl:272`'s
/// `unit="ptrow7"` feed of the PE outright.
///
/// ⛔ [`Sender`] AND [`Toward`] ARE TWO TYPES because both name a place in the array and they are the two
/// arguments of one table — a transposed pair would answer for the wrong direction and still compile.
pub fn forward_via(from: Sender, toward: Toward) -> crate::isa::unit::TgtEncoding {
    use crate::isa::regfile::Component;
    use crate::isa::unit::TgtEncoding;

    // ⭐ THE PT'S BRANCH IS FIRST AND IT IS SHORT: everything it can reach across the array is `"south"`.
    if matches!(from.0, Component::Pt) {
        return match toward.0 {
            Unit::Pt | Unit::Ptnorth | Unit::Ptsouth | Unit::Ptrow3 | Unit::Ptrow7 | Unit::Pe => {
                TgtEncoding::South
            }
            // ⛔ `link` STAYS EMPTY AND THE C++ RAISES. *"Unsupported destination for PE/SFP FMA"*
            // (`VectorOperands.cpp:137-140`) — the PT's branch sets `link` for nothing else, so a PT
            // forwarding to an LX, an L0 or the SFP is a program the reference refuses to build.
            // 🔑 NAMED, NOT `_`: these are the destinations the PT's branch leaves `link` empty for.
            Unit::Lxlu
            | Unit::Lxsu
            | Unit::L0lu
            | Unit::L0su
            | Unit::Sfp
            | Unit::Sfpring
            | Unit::Constant => panic!(
                "a {:?} forwards to {:?}, which the PT's own branch does not spell. `if (comp == PT)` sets \
                 `link` only for a PT-or-PE destination (`VectorOperands.cpp:112-115`); for anything else \
                 the reference leaves `link` empty and raises \"Unsupported destination for PE/SFP FMA\" \
                 (`:137-140`)",
                from.0, toward.0
            ),
        };
    }
    match toward.0 {
        // `LXLU`/`LXSU` both give `lx` (`VectorOperands.cpp:130-131`).
        Unit::Lxlu | Unit::Lxsu => TgtEncoding::Lx,
        // `SFP` gives `sfp`; the `+= "ring"` case is the SFP forwarding to ITSELF, which is `tgtdatafifo` rather
        // than this pair (`ConstructProgIRHelper.cpp:1351-1357`).
        Unit::Sfp => TgtEncoding::Sfp,
        Unit::Sfpring => panic!(
            "an output forwarded to the SFP ring is `tgtdatafifo`, not a `tgtencoding` direction \
             (`ConstructProgIRHelper.cpp:1351-1357`), and only an FMA result may take it"
        ),
        // ⭐ FROM A PE OR AN SFP the array is `"pt"` and the PE is `"pe"` (`VectorOperands.cpp:117-120`).
        // ⛔ NOT `south` — that is the PT SENDER's spelling and it is answered above, before this match.
        // `setSentientComputeOutputProgIROperands` puts a `"south"` output in `tgts` and reaches that arm
        // only for `comp == PT` (`ConstructProgIRHelper.cpp:1319-1330`).
        // `senCompToGenericComp` folds every row spelling to the one component (`arch_enums.cpp:122-145`),
        // which is what `generic == PT` tests — so the row-qualified names answer the same way.
        Unit::Pt | Unit::Ptnorth | Unit::Ptsouth | Unit::Ptrow3 | Unit::Ptrow7 => TgtEncoding::Pt,
        Unit::Pe => TgtEncoding::Pe,
        // ⭐ `L0LU`/`L0SU` BOTH GIVE `l0` (`VectorOperands.cpp:110-135`), which `output == "l0"` puts in `tgtl0`
        // (`ConstructProgIRHelper.cpp:1345-1347`). Which of the two units the stick lands on is the L0's own
        // affair — the SFP has one link to the pair.
        Unit::L0lu | Unit::L0su => TgtEncoding::L0,
        Unit::Constant => panic!("a compute cannot write to the constant table"),
    }
}

/// The first version of these enums took its variants straight from the `.smc` bodies, which put `Prec`,
/// `Unroll` and `Be` in them — and every one of those is a PLACEHOLDER whose spelling is its own field's name:
///
/// | in a body | is | resolved from |
/// |---|---|---|
/// | `mode=prec` (195×) | a hole | the op's compute precision |
/// | `unroll=unroll` (209×) | a hole | the op's unroll factor |
/// | `be=be` (13×) | a hole | the op's big-endian flag |
/// | `imm=l0` | a hole | `ddl.opaque`'s `params={"l0"=…}` |
/// | `src0=in0`, `fwdencoding=out0` | a hole | the op's operands, via `params` |
/// | `mode=fp32`, `unroll=x1`, `mask=255` | a value | — |
///
/// An `.smc` body is a TEMPLATE, so bridge 1's job for a `ddl.opaque` is substitution, and island 2 holds what
/// substitution produced. `Precision` is `Fp16`/`Fp32` because that is what the field can carry after it:
/// `ConstructProgIRHelper.cpp:2040-2042` is `mode = (compute_precision == "none") ? "fp16" : compute_precision`
/// followed by a `DT_CHECK` that the result is one of exactly those two.
///
/// ⚠️ ONE VALUE IS UNEXPLAINED AND SAYING SO IS THE POINT. 38 vendored instructions state `mode=1`, across
/// `FCVT`, `REDUCE`, `SPLAT` and — beside `prec` and `fp32` on the same opcode — `FMA`, `FMUL`, `IME` and
/// others, so it is neither opcode-specific nor either legal precision. It is not in these enums because
/// guessing which precision `1` encodes is exactly the kind of invention that produces a plausible wrong byte.
/// The opaque splice has to answer it before it can emit those instructions, and it will fail loudly instead.
macro_rules! slot {
    ($stated:ident, $maybe:ident, $what:literal, [$($variant:ident = $spelling:literal),+ $(,)?]) => {
        #[doc = concat!("`", $what, "=`, which every instruction carrying this slot states.")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $stated {
            $(
                #[doc = concat!("`", $what, "=", $spelling, "`.")]
                $variant,
            )+
        }

        #[doc = concat!("`", $what, "=` for the opcodes that state it on SOME of their instructions only.")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $maybe {
            /// The instruction does not state this slot. A case, not an absence.
            Unstated,
            $(
                #[doc = concat!("`", $what, "=", $spelling, "`.")]
                $variant,
            )+
        }

        impl From<$stated> for $maybe {
            /// A STATED VALUE, IN THE FORM AN OPCODE THAT STATES THE SLOT ONLY SOMETIMES CARRIES.
            ///
            /// ⛔ ONE DIRECTION ONLY. Widening a value the instruction HAS into a slot that may be absent loses
            /// nothing; the reverse would have to answer what `Unstated` becomes, and there is no value it means —
            /// the field is not in the instruction at all.
            fn from(stated: $stated) -> Self {
                match stated {
                    $( $stated::$variant => Self::$variant, )+
                }
            }
        }
    };
}

slot!(Mask, MaskOrUnstated, "mask", [M1 = "1", M255 = "255"]);
slot!(
    Precision,
    PrecisionOrUnstated,
    "mode",
    [Fp16 = "fp16", Fp32 = "fp32"]
);
slot!(Be, BeOrUnstated, "be", [Off = "0", On = "1"]);
// ⛔⛔⛔ THE VARIANTS ARE THE **FIELD'S**, NOT THE UNION OF WHAT THE VENDORED BODIES SPELL.
// `isa.cpp`'s SFP branch declares `defineField(20, "unroll", 45, {{"x1",0},{"x2",1},{"x4",2},{"x8",3}})` — all
// four are legal values of the field. Only `x1`/`x2` appear in any `.smc` body, and this list used to stop there,
// which made `x4`/`x8` UNEXPRESSIBLE: a set derived from observed usage mistaken for the set of legal values, the
// same error as reading a `synctag` NAME table as a legality check.
//
// ⭐ AND THE REFERENCE EMITS `x8`. dxp's `sdsc_11` (`11_scalarmul_o1410`) `sfp`, Core 0 Corelet 0, carries
// `(3 << 45)` on both of its producing FMAs, where `3` is the ENCODING of `x8`. It matters because
// `SentientToTrace.cpp:1765-1800` expands one `unrollFactor` into that many `MacOp`s and advances the write
// pointer `i * write_incr` on each — UNCONDITIONALLY, `unrlfldsrc*` gating only the compute ports — so `x1`
// writes ONE output slot where `x8` writes EIGHT. A body that never spells `x8` is not evidence the hardware
// cannot take it; it is evidence no vendored body needed it.
slot!(
    Unroll,
    UnrollOrUnstated,
    "unroll",
    [U0 = "u0", X1 = "x1", X2 = "x2", X4 = "x4", X8 = "x8"]
);

/// `unrlfldsrc0=` / `unrlfldsrc1=` / `unrlfldsrc2=` / `unrlfldtgt=` — whether THIS OPERAND'S register advances
/// with the unroll.
///
/// ⛔ 975 STATEMENTS OF IT WERE BEING DROPPED, silently, by a generator that modelled eleven slot kinds while the
/// bodies state twenty-seven. It is not decoration: an unrolled instruction whose target does not advance writes
/// the same register on every iteration, so the unroll computes one value and discards the rest.
///
/// ⭐ ONE BIT, TWO SPELLINGS FOR EACH VALUE. `isa_init.rs:714-739` defines all four at bit 47..50 with
/// `Lit("yes", 1), Lit("no", 0)`, and the vendored bodies write BOTH `unrlfldtgt=yes` and `unrlfldtgt=1` — so the
/// symbolic and numeric forms are the same value, which is what the field table settles. An earlier note called
/// that "two spellings for one truth or two different facts" and left it open; it is the former, on evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnrollField {
    /// `no` / `0` — this operand holds still across the unroll.
    No,
    /// `yes` / `1` — this operand's register advances.
    Yes,
}

/// The same, for the opcodes that state it on only some of their instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnrollFieldOrUnstated {
    Unstated,
    Stated(UnrollField),
}

impl Unroll {
    /// HOW MANY REGISTER SLOTS AN UNROLL-FLAGGED OPERAND OF THIS INSTRUCTION OWNS — the same number as
    /// [`Self::factor`], asked by the ALLOCATOR rather than the encoder. `handleRegInput` adds the unroll
    /// counter to the base (`computeElement.cpp:3979-3983`), so a base under `x2` touches TWO consecutive
    /// registers, and dxp budgets exactly that: `internalRegs_.size() + (unroll - 1) *
    /// internalRegsWithUnroll_` sticks (`ddcv1.cpp:266-269`), laid out by `insertReg` as `t_0 = R(start)`,
    /// `t_1 = R(start + 1)` (`:3344-3358`).
    #[must_use]
    pub const fn slices(self) -> u8 {
        self.factor() as u8
    }

    /// THE NUMBER THE FIELD ENCODES — `x1` is 1 through `x4` is 4 (`isa.cpp:384`, `rsm_unroll`).
    ///
    /// ⛔ `u0` IS A HOLE, NOT A FACTOR. `unroll` is one of the four slots a `.smc` body leaves open for the
    /// op to fill, so a `U0` reaching an encoder is a splice that did not close — the same shape as
    /// `fwdencoding=out0`. It refuses rather than picking a number.
    #[must_use]
    pub const fn factor(self) -> u32 {
        match self {
            Self::X1 => 1,
            Self::X2 => 2,
            Self::X4 => 4,
            Self::X8 => 8,
            Self::U0 => panic!(
                "`unroll=u0` is a template HOLE the op fills, and it has reached a field that encodes a \
                 FACTOR (`isa.cpp:384` spells `x1`..`x4` as 1..4). A hole here means the splice did not close"
            ),
        }
    }
}

impl UnrollOrUnstated {
    /// See [`Unroll::slices`]. An instruction that does not state the slot runs once.
    #[must_use]
    pub const fn slices(self) -> u8 {
        match self {
            Self::Unstated | Self::X1 => 1,
            Self::X2 => 2,
            Self::X4 => 4,
            Self::X8 => 8,
            Self::U0 => panic!(
                "`unroll=u0` is a template HOLE the op fills, and the ALLOCATOR is being asked how many \
                 register slots it owns — a hole here means the splice did not close"
            ),
        }
    }
}

/// WHETHER A COMPUTETYPE HAS A SECOND OPCODE, AND WHICH — three facts, three variants.
///
/// ⛔ NOT AN `Option`. "This opcode serves one operation" and "the C++'s admitted set does not contain this
/// comparison" are different facts with different consequences: the first means the instruction has no `imm` field
/// to fill, the second means the field exists and what goes in it has not been read. An `Option` spells both
/// `None` and makes the second look handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondOpcode {
    /// This opcode serves ONE operation, so there is no variant to select.
    OneOperation,
    /// The variant `ConstructBinaryInstr` computes.
    Variant(OpVariant),
    /// The DDL names an operation the C++'s branch does not admit, so the value has not been read.
    Unread,
}

/// A SECOND OPCODE AS THE `imm` FIELD'S VALUE.
///
/// ⛔ THE TWO NON-VARIANT CASES ARE REFUSALS, NOT ZEROS. This is only called for an opcode whose instruction HAS an
/// `imm` field, so reaching it with `OneOperation` means the census and `second_opcode` disagree about whether the
/// opcode serves several operations — and `Unread` means the field exists and its value has not been read. A 0
/// would be `and` on a LOGICAL and `max` on an FMINMAX: a different operation, silently.
pub fn second_opcode_imm(second: SecondOpcode) -> crate::islands::progir::Imm {
    match second {
        SecondOpcode::Variant(variant) => crate::islands::progir::Imm::of(i64::from(variant.get())),
        SecondOpcode::OneOperation => panic!(
            "this instruction has an `imm` field and its computetype has no second opcode; \
             `ConstructBinaryInstr` writes the field only in a branch that computed one \
             (`ConstructProgIRHelper.cpp:1801-2053`), so one of the two readings is wrong"
        ),
        SecondOpcode::Unread => panic!(
            "this computetype's `imm` is a value this crate has not read: a comparison outside the C++'s admitted \
             FCMP set (`is_any_of(op, fcmp_eq, fcmp_neq, fcmp_le, fcmp_lt)`, \
             `ConstructProgIRHelper.cpp:1958-1962`), or a SPLAT whose pad flag comes from the shuffle's index \
             pattern (`VectorChainToSentientPESFP/Splat.cpp:159-181`)"
        ),
    }
}

/// WHICH VARIANT OF A SHARED OPCODE — the `int second_opcode` `ConstructBinaryInstr` puts in `imm`.
///
/// ⛔ A NEWTYPE, NOT AN INTEGER, because it shares `imm` with a template's data literal and the two must not
/// interchange: a `.smc` body's `imm=1` on a LOGICAL is the same field carrying the same 1 for a different reason,
/// and an `i64` at a call site reads as either.
///
/// ⭐ AND BOUNDED BY WHAT THE FIELD HOLDS. The widest set the C++ computes is PACK's `0..=27`
/// (`ConstructProgIRHelper.cpp:1898-1901`), which is its own explicit check, so a variant past it is a value no
/// branch produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpVariant(u8);

impl OpVariant {
    /// PACK's own bound, and the largest any branch computes (`:1899`).
    pub const MAX: u8 = 27;

    /// ⛔ `const`, SO AN OUT-OF-RANGE VARIANT IS A BUILD ERROR. Every call site is a literal from the C++.
    pub const fn of(variant: u8) -> Self {
        assert!(
            variant <= Self::MAX,
            "a second opcode lies in 0..=27, which is PACK's own check and the widest any branch computes"
        );
        Self(variant)
    }

    /// The value, for the encoder placing it in `imm`.
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// A COUNT OF STICKS IN ONE BURST — what an L3 transfer's `burst` field measures.
///
/// ⛔ NOT A BYTE COUNT AND NOT AN ADDRESS. The burst is how many 128-byte sticks one bus transaction moves, so it
/// interchanges with neither a register value nor a byte length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BurstSticks(u32);

impl BurstSticks {
    pub const fn of(sticks: u32) -> Self {
        Self(sticks)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// THE LONGEST BURST AN L3 TRANSFER MAY REQUEST — `l3BurstSize = 32` (`sysdef.cpp:190`).
///
/// An arch fact from the system definition, which is why it sits beside [`NUM_PT_ROWS`] rather than in a
/// bridge. What DIVIDES by it is a question for the pass that emits the transfer, and is not settled here:
/// the splitting-into-whole-bursts-plus-remainder that this constant used to claim was cited to
/// `pcfgDTToInstrL3` in the dcg BACK END, which consumes a PCFG and is therefore off our path.
pub const L3_BURST_SIZE: BurstSticks = BurstSticks::of(32);
