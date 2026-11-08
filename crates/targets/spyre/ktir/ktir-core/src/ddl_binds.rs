// SPDX-License-Identifier: Apache-2.0
//! THE DEVICE-FACTS TABLE — one const table, keyed by a TYPED device op,
//! stating every (op, dtype, layout) fact the ladder's lowerings read.
//!
//! The card does not have "an exp op"; it has a DDL `operation_bind` of an
//! opFuncName at a list of data formats, and an op asked for at a format the
//! bind does not list is an op the card was never asked to compute. That
//! fact — and the output-arrangement (layout) fact beside it — is a
//! TARGET-ABI FACT, so it lives here, in the dependency-free core both
//! sides depend on, as ONE table each row of which cites the vendor
//! template and line it was read from:
//!
//! * the TRITON SIDE (`triton-ktir`'s `legalize_types`) reads it at the
//!   RED-stop: an op the card binds at fp32 lowers AS f32 — nothing narrows,
//!   so there is nothing to refuse — exactly the decision the pass's old
//!   inline `FP32_CARD_BINDS` table made, now read through the typed key;
//! * the ARRANGEMENT PROPAGATION (`ktir-superdsc`'s
//!   `element_arrangement`-fed walk) reads each row's arrangement — the
//!   output layout an op carries when both operands are standard — so no
//!   staging rule lives in a match arm (the old `OP_OUTPUT_ARRANGEMENT`
//!   string-keyed table, absorbed here).
//!
//! This commit is the RE-HOMING only: every reader decides exactly what it
//! decided before, and the emitted bundles are byte-identical. The rows the
//! table also carries beyond today's readers — the fp16 census, the reduce
//! family, the converts' binds — are measured facts stated with their
//! citations for the readers that land next (#1142's island collapse and
//! door refusals, the MoE reduce consult): a row no reader reads changes no
//! decision, and a reader that lands reads a row that already exists. The
//! measured defect those readers close: `exp` bound fp32-only in the
//! pointwise family (`unary_parallel.ddl:39`) was collapsed to f16 by an
//! island collapse that never consulted any bind; dxp compiled it, and the
//! card computed a smooth WRONG function (card-gate-1112, run-6 follow-up:
//! max_ulp 6872–13536 against the host f16 chain). Nothing refused, because
//! nothing asked.
//!
//! ⛔ THE FACTS ARE A GENERATION'S, never "the device's" (user, 2026-11-07:
//! spyre gen1 now, gen2 soon). [`DeviceGen`] is the generation type; every
//! row belongs to the generation whose templates it cites (`Gen1` today),
//! and bringing gen2 up means filling in ITS table beside this one with
//! gen2's own citations — never an edit to a gen1 row, never a lowering
//! change. A lowering asks the generation it bakes for; a literal baked
//! into a lowering is a review failure.
//!
//! ⛔ THE OP IS KEYED BY TYPE, never by its string spelling: a typo'd or
//! renamed door key would silently miss every row, read as "no device op",
//! and refuse something the device has. [`DeviceOp`] is the key; the door's
//! string spelling is a PROPERTY of the enum ([`DeviceOp::op_func`]), not a
//! second key. Each side maps its own vocabulary onto `DeviceOp` at its own
//! door; the FACTS are stated once, here.
//!
//! ⛔ EVERY ROW IS A MEASURED FACT, read directly from the vendor templates
//! on the card pod (2026-11-07, `/opt/ibm/spyre/deeptools/share/ddc/`
//! `ddl_templates/`): the citation names the template and line. Nothing is
//! inferred from a composition or from another op's bind, and a `DeviceOp`
//! with no row is UNBOUND at every format — fail-closed, the refusal stands.

/// A data format a DDL `operation_bind` lists. The doors lower in these
/// only; every other `superdsc_opspec`-style format is refused by the door
/// before this table is asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DdlDf {
    /// `SEN169_FP16` — `%type_fp16`.
    Fp16,
    /// `IEEE_FP32` — `%type_fp32`.
    Fp32,
}

/// The element arrangement a value's buffer carries: the order of elements
/// within each 64-element block of the flat word stream. A TYPE, not a
/// string — the ABI fact the arrangement rows state, moved beside the table
/// so the fact and its row live in one place.
///
/// `Dl16ToFp32` is the MEASURED `dl16tofp32` output stagger (2026-10-26,
/// granite-2b fp8, probe-8): when both operands are addressed standard, the
/// convert writes value-exact but lane-permuted output — within each
/// 64-element block, the sixteen 4-lane groups permute by a 1-bit right
/// rotation of the 4-bit group index. The permutation itself (and the
/// staging over it) stays in `ktir-superdsc`'s `element_arrangement`, which
/// re-exports this enum.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ElementArrangement {
    /// Logical lane `i` at buffer position `i` — every buffer the card's ops
    /// address unless a table row says otherwise.
    Standard,
    /// Logical lane `4g + r` at buffer position `4·ror(g) + r` within each
    /// 64-element block — the measured `dl16tofp32` output stagger.
    Dl16ToFp32,
}

/// The device family an op is asked in — the layout half of the table's
/// key. The pointwise door, the reduce door and the convert door are three
/// different asks of the card; `max` at the reduce door and `maximum` at
/// the pointwise door are different rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceFamily {
    /// The pointwise doors (`elementwise_df`, `scalar_binary`).
    Pointwise,
    /// The reduce door (`reduce_df`).
    Reduce,
    /// The convert door (`convert`).
    Convert,
}

/// The device ops, spelled ONCE: the enum and its exhaustive list [`ALL`]
/// are generated from the same token list, so a variant cannot exist
/// without its list entry — adding one to the invocation extends `ALL` in
/// the same breath. This is the pin: no pattern to arm, no index to
/// renumber, no second list to forget.
macro_rules! device_ops {
    ($($variant:ident),* $(,)?) => {
        /// The variant count, taken from the list itself — no literal to
        /// go stale.
        pub const DEVICE_OP_COUNT: usize = <[&str]>::len(&[$(stringify!($variant)),*]);

        /// A device op, TYPED — the table's key. One variant per opFuncName
        /// the table states a fact for; the door's string spelling is
        /// [`DeviceOp::op_func`], a property, never a second key.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum DeviceOp {
            $($variant,)*
        }

        impl DeviceOp {
            /// Every variant, exactly once, in the spelled order — generated
            /// from the same list as the enum itself, so it cannot go stale.
            pub const ALL: [DeviceOp; DEVICE_OP_COUNT] = [$(DeviceOp::$variant),*];
        }
    };
}

device_ops!(
    // ── the pointwise family
    Add,
    Sub,
    Multiply,
    RealDiv,
    Maximum,
    Minimum,
    Equal,
    Gelu,
    Reciprocal,
    Tanh,
    Abs,
    Neg,
    Silu,
    Exp,
    Sigmoid,
    Sqrt,
    Rsqrt,
    Mish,
    Log,
    // ── the convert family
    AddI32ToI32,
    MulI32ToI32,
    Int32ToFp32,
    Fp32ToInt32,
    Dl16ToFp32,
    Fp32ToDl16,
    // ── the reduce family
    Sum,
    Mean,
    Max,
    Min,
    AbsMax,
);

impl DeviceOp {
    /// The door's own spelling — the `op_func_from_str` key the emitter
    /// spells (`"multiply"`, not the DDL's `"mul"`; `"gelu"`, not
    /// `"gelufwd"`). A property of the op, read at emission; never a lookup
    /// key.
    pub const fn op_func(self) -> &'static str {
        match self {
            DeviceOp::Add => "add",
            DeviceOp::Sub => "sub",
            DeviceOp::Multiply => "multiply",
            DeviceOp::RealDiv => "realdiv",
            DeviceOp::Maximum => "maximum",
            DeviceOp::Minimum => "minimum",
            DeviceOp::Equal => "equal",
            DeviceOp::Gelu => "gelu",
            DeviceOp::Reciprocal => "reciprocal",
            DeviceOp::Tanh => "tanh",
            DeviceOp::Abs => "abs",
            DeviceOp::Neg => "neg",
            DeviceOp::Silu => "silu",
            DeviceOp::Exp => "exp",
            DeviceOp::Sigmoid => "sigmoid",
            DeviceOp::Sqrt => "sqrt",
            DeviceOp::Rsqrt => "rsqrt",
            DeviceOp::Mish => "mish",
            DeviceOp::Log => "log",
            DeviceOp::AddI32ToI32 => "addi32toi32",
            DeviceOp::MulI32ToI32 => "muli32toi32",
            DeviceOp::Int32ToFp32 => "int32tofp32",
            DeviceOp::Fp32ToInt32 => "fp32toint32",
            DeviceOp::Dl16ToFp32 => "dl16tofp32",
            DeviceOp::Fp32ToDl16 => "fp32todl16",
            DeviceOp::Sum => "sum",
            DeviceOp::Mean => "mean",
            DeviceOp::Max => "max",
            DeviceOp::Min => "min",
            DeviceOp::AbsMax => "absmax",
        }
    }

    /// The device family this op is asked in.
    pub const fn family(self) -> DeviceFamily {
        match self {
            DeviceOp::Add
            | DeviceOp::Sub
            | DeviceOp::Multiply
            | DeviceOp::RealDiv
            | DeviceOp::Maximum
            | DeviceOp::Minimum
            | DeviceOp::Equal
            | DeviceOp::Gelu
            | DeviceOp::Reciprocal
            | DeviceOp::Tanh
            | DeviceOp::Abs
            | DeviceOp::Neg
            | DeviceOp::Silu
            | DeviceOp::Exp
            | DeviceOp::Sigmoid
            | DeviceOp::Sqrt
            | DeviceOp::Rsqrt
            | DeviceOp::Mish
            | DeviceOp::Log => DeviceFamily::Pointwise,
            DeviceOp::AddI32ToI32 | DeviceOp::MulI32ToI32 => DeviceFamily::Pointwise,
            DeviceOp::Int32ToFp32
            | DeviceOp::Fp32ToInt32
            | DeviceOp::Dl16ToFp32
            | DeviceOp::Fp32ToDl16 => DeviceFamily::Convert,
            DeviceOp::Sum | DeviceOp::Mean | DeviceOp::Max | DeviceOp::Min | DeviceOp::AbsMax => {
                DeviceFamily::Reduce
            }
        }
    }
}

/// One device fact: the op, the data formats the vendor DDL binds it at,
/// the arrangement its output carries when both operands are standard, and
/// the template line the fact was read from.
#[derive(Debug)]
pub struct DeviceOpRow {
    /// The typed device op — the key.
    pub op: DeviceOp,
    /// The formats the bind lists. An op asked at a format not here is
    /// refused by the reader, never silently degraded.
    pub dfs: &'static [DdlDf],
    /// The output arrangement the op carries at standard addressing.
    pub arrangement: ElementArrangement,
    /// The vendor template and line(s) the fact was read from.
    pub citation: &'static str,
}

/// A device GENERATION. Each generation states its own device facts — the
/// op/dtype/layout table AND the geometry constants below — so a lowering
/// reads them through the generation it bakes for, never as free constants.
/// `Gen1` is today's only implementor; `Gen2` will be a second impl with
/// its own citations, not an edit here.
///
/// A reader cannot drift from the table: the constants are read through
/// these associated consts, so there is no second spelling of a value to
/// get wrong. What is NOT claimed here is a pin against senlib itself —
/// scratchy does not build-depend on senlib-chip, and a module of
/// hand-refreshed senlib literals compared against this table tested the
/// typist, not the silicon ("two literals in one repo"); it is deleted.
/// The one cross-crate pin that compares two things the build actually
/// compiles is `scratchy-target-spyre`'s `runtime.rs`, which const-asserts
/// [`DeviceGen::FLIT_BYTES`] against `flex_rs::control_block_wire::Flits::BYTES`.
/// The follow-up is a direct dependency on senlib-chip's public half once
/// it is CI-fetchable, at which point a senlib change fails the build.
pub trait DeviceGen {
    /// This generation's device-facts table — the one statement of its
    /// card's (op, dtype, layout) facts, read by `legalize_types`
    /// (RED-stop) and the arrangement propagation today; the island
    /// collapse and pointwise-door consults that land next read the same
    /// rows.
    const DEVICE_OPS: &'static [DeviceOpRow];
    /// The card's compute cores — the count every work plan caps at (one
    /// entry per core on the gather's index stick; `MaxCores` in the work
    /// planner). The frontend's default core count is the SAME fact.
    const CORES: u32;
    /// Corelets per compute tile.
    const CORELETS: u32;
    /// Per-core LX scratchpad, in bytes.
    const LX_CAPACITY_BYTES: u64;
    /// How many write-done sids a card holds.
    const MAX_SIDS: u32;
    /// The widest single p2p message, in bytes — a longer send is a run of
    /// messages.
    const RDMA_MESSAGE_CAP_BYTES: u64;
    /// Segments of the executor's packed-segment model.
    const NUM_SEGMENTS: usize;
    /// The bytes one segment may hold.
    const MAX_SEGMENT_BYTES: u64;
    /// One flit, in bytes — the unit every CB address and length field
    /// counts in, and (by the device's own stick law) the width of every
    /// format's stick: elems per stick = `FLIT_BYTES / element size`.
    const FLIT_BYTES: u64;
    /// This generation's `dl16tofp32` lane permutation:
    /// `DL16TOFP32_PERMUTATION[i]` is the buffer position logical lane `i`
    /// of a 64-element block occupies in a staggered buffer. A const
    /// table, not a derivation — the staging over it lives in
    /// `ktir-superdsc`'s `element_arrangement`.
    const DL16TOFP32_PERMUTATION: [u8; 64];
}

/// Spyre generation 1 — the card this tree bakes today.
pub struct Gen1;

/// The generation this tree bakes for — the ONE name the bake's readers
/// hold. `Gen1` is spelled here alone; gen2 lands as a second [`DeviceGen`]
/// impl with its own table and constants, and this alias moves to it — a
/// one-line change in one place, not a grep.
pub type Baked = Gen1;

impl DeviceGen for Gen1 {
    /// GEN1'S TABLE. Every row cites the gen1 vendor templates (read on the
    /// card pod, 2026-11-07) and line; a `DeviceOp` with no row is unbound
    /// at every format — fail-closed.
    const DEVICE_OPS: &'static [DeviceOpRow] = &[
        // ── broadcast_ops.ddl — the two-tensor pointwise family, every row
        //    bound at BOTH fp16 and fp32.
        DeviceOpRow {
            op: DeviceOp::Add,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "broadcast_ops.ddl:25 add [%type_fp16, %type_fp32]",
        },
        DeviceOpRow {
            op: DeviceOp::Sub,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "broadcast_ops.ddl:26 sub [%type_fp16, %type_fp32]",
        },
        DeviceOpRow {
            op: DeviceOp::Multiply,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "broadcast_ops.ddl:27 mul [%type_fp16, %type_fp32]",
        },
        DeviceOpRow {
            op: DeviceOp::Equal,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "broadcast_ops.ddl:41 equal [%type_fp16, %type_fp32]",
        },
        DeviceOpRow {
            op: DeviceOp::RealDiv,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "broadcast_ops.ddl:44 realdiv [%type_fp16, %type_fp32]",
        },
        DeviceOpRow {
            op: DeviceOp::Maximum,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "broadcast_ops.ddl:45 maximum [%type_fp16, %type_fp32]",
        },
        DeviceOpRow {
            op: DeviceOp::Minimum,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "broadcast_ops.ddl:46 minimum [%type_fp16, %type_fp32]",
        },
        // ── unary_parallel.ddl — the one-input family, most rows bound at both
        //    formats (the fp32 rows ride an internal tensor).
        DeviceOpRow {
            op: DeviceOp::Gelu,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:26 gelufwd [%type_fp16] (FEST) / :27 \
                   gelufp32 [%type_fp32] with an interim tensor — TWO recipes",
        },
        DeviceOpRow {
            op: DeviceOp::Reciprocal,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:30 reciprocal [%type_fp16, %type_fp32]",
        },
        DeviceOpRow {
            op: DeviceOp::Tanh,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:34 tanh [%type_fp16] (FEST, two \
                   interims) / :35 tanhfp32 [%type_fp32] with an interim \
                   tensor — TWO recipes",
        },
        DeviceOpRow {
            op: DeviceOp::Abs,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:36 abs [%type_fp16, %type_fp32]",
        },
        DeviceOpRow {
            op: DeviceOp::Neg,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:37 neg [%type_fp16, %type_fp32]",
        },
        DeviceOpRow {
            op: DeviceOp::Silu,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:38 silu [%type_fp16, %type_fp32]",
        },
        // ⭐ THE ROW THIS TABLE EXISTS FOR: the pointwise family binds `exp` at
        // FP32 ALONE. The PIPELINE family binds an fp16 exp (`unary_pipeline.ddl:19`,
        // FEST-based, `exp.smc`: "THIS IMPLEMENTATION IS FP32 SPECIFIC (ICVT
        // changes imm)" on the parallel form), and that pipeline bind does not
        // serve the pointwise door: MEASURED on card (card-gate-1112 run-6
        // follow-up), an exp collapsed to f16 and emitted through this door
        // compiled clean and computed a smooth wrong function — max_ulp
        // 6872–13536 against the host f16 chain, E(0)≈e^0.5, log-slope→0.63. So
        // the fp16 pipeline bind is NOT an fp16 bind of this door's op, and the
        // row states the parallel family's alone.
        DeviceOpRow {
            op: DeviceOp::Exp,
            dfs: &[DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:39 exp [%type_fp32] — fp32 ONLY; the \
                   unary_pipeline.ddl:19 fp16 exp is a FEST form this door \
                   does not instantiate (measured wrong on card, cg1112)",
        },
        DeviceOpRow {
            op: DeviceOp::Sigmoid,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:42 sigmoid [%type_fp16, %type_fp32]",
        },
        DeviceOpRow {
            op: DeviceOp::Sqrt,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:43 sqrt [%type_fp16, %type_fp32]",
        },
        DeviceOpRow {
            op: DeviceOp::Rsqrt,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:44 rsqrt [%type_fp16, %type_fp32]",
        },
        // ── unary_pipeline.ddl — the FEST pipeline family. Mish is fp16
        //    ALONE because the template says so in its own words ("not
        //    possible in fp32 (FEST hardware limitation)"). Log binds BOTH:
        //    its fp16 is the pipeline bind, and an fp32 PARALLEL bind exists
        //    (unary_parallel.ddl:40). The row states the template fact; the
        //    POLICY — f32 log not admitted, main's decision, no card cell
        //    yet — lives at the reader that holds it (the RED-stop in
        //    legalize_types names it), never as a falsified bind here.
        DeviceOpRow {
            op: DeviceOp::Mish,
            dfs: &[DdlDf::Fp16],
            arrangement: ElementArrangement::Standard,
            citation: "unary_pipeline.ddl:21 mish [%type_fp16] — not possible in \
                   fp32 (FEST hardware limitation)",
        },
        DeviceOpRow {
            op: DeviceOp::Log,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_pipeline.ddl:22 log [%type_fp16] (PE+SFP \
                   pipeline) / unary_parallel.ddl:40 log [%type_fp32] \
                   (parallel)",
        },
        // ── the integer broadcast ops — the MoE lanes chain's own rows (the
        //    i32 mul/add). Bound in fp32 MODE over uint32 tensors, the value
        //    casts' reading: the type list names the MODE, the tensor roles
        //    carry the integer data.
        DeviceOpRow {
            op: DeviceOp::AddI32ToI32,
            dfs: &[DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "broadcast_ops.ddl:28 addi32toi32 [%type_fp32] \
                   (uint32 tensors in and out)",
        },
        DeviceOpRow {
            op: DeviceOp::MulI32ToI32,
            dfs: &[DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "broadcast_ops.ddl:30 muli32toi32 [%type_fp32] \
                   (uint32 tensors in and out)",
        },
        // ── the value casts, for the RED-stop's projection (`arith.sitofp` from
        //    an IEEE_INT32 input). Not pointwise-door ops; they ride the convert
        //    lowering. Bound fp32 alone, keyed on the DDL's own opFuncName.
        DeviceOpRow {
            op: DeviceOp::Int32ToFp32,
            dfs: &[DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:50 int32tofp32 [%type_fp32] (IEEE_INT32 in)",
        },
        DeviceOpRow {
            op: DeviceOp::Fp32ToInt32,
            dfs: &[DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:51 fp32toint32 [%type_fp32] (IEEE_INT32 out)",
        },
        // ── the dl16 converts. The stagger row: the WIDEN's output is measured
        //    lane-permuted at standard addressing (probe-8, granite-2b fp8,
        //    2026-10-26) — the arrangement fact the propagation reads, cited to
        //    the card measurement because the DDL does not state it. The
        //    NARROW-BACK (`fp32todl16`) is ALSO MEASURED — the inverse stagger,
        //    shift −1 (convert-probe on the card, 2026-11-07: element-exact at
        //    [256,2048] and [1,2048], legs 1 and 2) — but main's arrangement
        //    table stated NO row for it, so the propagation stages it Standard;
        //    this row keeps main's behavior (Standard) with the measured fact
        //    recorded in its citation, and moving the arrangement field to the
        //    measured inverse stagger is the Z4 solve's own PR, where the
        //    staging over it lands with it.
        DeviceOpRow {
            op: DeviceOp::Dl16ToFp32,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Dl16ToFp32,
            citation: "quantization_double_pad.ddl:58 dl16tofp32 — a \
                   TYPELESS bind ([]) states no type list; the dtypes are \
                   the CARD's (probe-8, 2026-10-26: fp16 in / fp32 out, the \
                   lane stagger measured there)",
        },
        DeviceOpRow {
            op: DeviceOp::Fp32ToDl16,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "quantization_double_pad.ddl:59 fp32todl16 — a \
                   TYPELESS bind ([]) states no type list; the dtypes are \
                   the CARD's (convert-probe, 2026-11-07: fp32 in / fp16 \
                   out, element-exact at [256,2048] and [1,2048]). The \
                   inverse stagger MEASURED — shift −1, \
                   convert-probe 2026-11-07, element-exact at [256,2048] and \
                   [1,2048]; main stages it Standard, so this row keeps main's \
                   behavior, and moving it is the Z4 solve's PR",
        },
        // ── summeanmaxexx2.ddl / summeanmaxexx2_fp32.ddl — the reduce family,
        //    every combiner bound at BOTH formats (the fp16 template's stick
        //    binds and the fp32 template's alike; the two files' rows are
        //    line-for-line twins). THE MoE ROW: the i32 max composition's middle
        //    leg — the fp32 max reduce over the widened lanes — is this table's
        //    `Max` row, not a hand-written arm.
        DeviceOpRow {
            op: DeviceOp::Sum,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "summeanmaxexx2.ddl:38 sum (stick) / :39 sumnonstick; \
                   summeanmaxexx2_fp32.ddl:39 / :40",
        },
        DeviceOpRow {
            op: DeviceOp::Mean,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "summeanmaxexx2.ddl:41 mean (stick) / :42 meannonstick; \
                   summeanmaxexx2_fp32.ddl:42 / :43",
        },
        DeviceOpRow {
            op: DeviceOp::Max,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "summeanmaxexx2.ddl:44 max (stick) / :45 maxnonstick; \
                   summeanmaxexx2_fp32.ddl:45 / :46",
        },
        DeviceOpRow {
            op: DeviceOp::Min,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "summeanmaxexx2.ddl:47 min (stick) / :48 minnonstick; \
                   summeanmaxexx2_fp32.ddl:48 / :49",
        },
        DeviceOpRow {
            op: DeviceOp::AbsMax,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "summeanmaxexx2.ddl:53 absmax (stick) / :54 absmaxnonstick; \
                   summeanmaxexx2_fp32.ddl:54 / :55",
        },
    ];

    /// GEN1'S GEOMETRY. Every constant cites the device's own statement —
    /// the vendor's deeptools headers (read on the card pod, 2026-11-08)
    /// and senlib-rs's snt1p0 table — and the spyre-target crate pins the
    /// numerics it can against senlib's own names, so a wrong copy here is
    /// a build failure there.
    ///
    /// deeptools paths below are on the card pod
    /// (`/opt/ibm/spyre/deeptools/include/...`); senlib-rs paths are
    /// `~/git/senlib-rs/crates/chip/src/generation/snt1p0.rs`.
    ///
    /// CORES = 32: senlib-rs `RCU_CORES: u32 = 32` (snt1p0.rs:225,
    /// "probe-pinned at 32 on snt1p0 silicon; chip geometry, not a
    /// per-card discovery") and the vendor `SenSystemDef`'s own default
    /// `core = 32` (sys-arch-spec/sysdef.h:169).
    const CORES: u32 = 32;

    /// CORELETS = 2: the vendor `SenSystemDef`'s own default
    /// `corelets = 2` (sys-arch-spec/sysdef.h:169).
    const CORELETS: u32 = 2;

    /// LX = 2 MiB per core: the vendor `SenSystemDef`'s own default
    /// `lxCap = 2 * 1024 * 1024` (sys-arch-spec/sysdef.h:169); confirmed
    /// against torch-spyre `scratchpad/allocator.py:305` and
    /// `sentient_dd2_sysconfig.json` (the citation the old literal site
    /// carried). No senlib name exists for the LX — senlib is the
    /// host/wire layer; the LX is device-internal.
    const LX_CAPACITY_BYTES: u64 = 2 * 1024 * 1024;

    /// SIDS = 64: senlib-rs `BASE_ADDR_TABLE_ENTRIES: usize = 64`
    /// (snt1p0.rs:246) — the same 64 its `Tid::new` refuses to exceed.
    const MAX_SIDS: u32 = 64;

    /// RDMA = 4 MiB per message: the vendor's per-message RDMA window
    /// (`P2P_RDMA_WINDOW_SIZE`, pf_runtime_scheduler.cpp:50; flex-rs's
    /// `scheduler::P2P_RDMA_WINDOW_SIZE`, which the target crate pins
    /// equal).
    const RDMA_MESSAGE_CAP_BYTES: u64 = 4 * 1024 * 1024;

    /// SEGMENTS = 7: the executor's ≤7-packed-segment model over the
    /// device's VA segments — senlib-rs states 8 VA segments for snt1p0
    /// (snt1p0.rs:545) and the executor model packs 7.
    const NUM_SEGMENTS: usize = 7;

    /// One segment = 16 GiB: senlib-rs `XLAT_LEN_MAX_FLITS: u64 = 1 << 27`
    /// (snt1p0.rs:535 — "a stored length of zero is the field maximum,
    /// 2^27 flits (16 GiB)") at 128 B per flit; flex-rs's
    /// `MAX_REGION_BYTES`, which the target crate pins equal.
    const MAX_SEGMENT_BYTES: u64 = 16 * 1024 * 1024 * 1024;

    /// FLIT = 128 B: senlib-rs `FLIT_SHIFT: u32 = 7` (snt1p0.rs:292) and
    /// the vendor's own `#define STICK_BYTESIZE 128`
    /// (util/sendefs/dataType.h:9) — the flit and the stick are the same
    /// width, which is the stick law the per-format stick widths derive
    /// from: `getNumElemsInStick = (STICK_BYTESIZE << 3) >> bitWidth`
    /// (dataType.h:22-24), i.e. elems per stick = FLIT_BYTES / element
    /// size, for EVERY format (the vendor's `dataFormatToStickSize` map —
    /// util/sendefs/sendefs.h:65-72 — holds exactly that quotient for all
    /// twelve of its rows, packed int2/int4 included).
    const FLIT_BYTES: u64 = 128;

    /// GEN1'S `dl16tofp32` LANE PERMUTATION — measured on the card
    /// (2026-10-26, granite-2b fp8, probe-8): within each 64-element block
    /// of the f32 word stream, the sixteen 4-lane groups permute by a
    /// 1-bit RIGHT rotation of the 4-bit group index — logical lane `4g+r`
    /// lands at buffer position `4·ror(g)+r`, `ror(g) = (g>>1) |
    /// ((g&1)<<3)`. The staging over this table stays in ktir-superdsc's
    /// `element_arrangement`; only the table itself moved here, so a
    /// generation is one impl with its own measured permutation.
    const DL16TOFP32_PERMUTATION: [u8; 64] = dl16tofp32_permutation();
}

/// The const constructor behind Gen1's
/// [`DL16TOFP32_PERMUTATION`](DeviceGen::DL16TOFP32_PERMUTATION) — evaluated
/// at compile time, so the table on disk IS the formula (the right rotation
/// of the 4-bit group index), stated once.
const fn dl16tofp32_permutation() -> [u8; 64] {
    let mut table = [0u8; 64];
    let mut g = 0;
    while g < 16 {
        // ror(g) = (g >> 1) | ((g & 1) << 3) — a 1-bit right rotation of the 4-bit group index.
        let rotated = (g >> 1) | ((g & 1) << 3);
        let mut r = 0;
        while r < 4 {
            table[g * 4 + r] = (rotated * 4 + r) as u8;
            r += 1;
        }
        g += 1;
    }
    table
}

/// The row stating `op`'s facts on generation `G`, or `None` when no row
/// names it — UNBOUND, fail-closed: the caller refuses, never degrades.
pub fn row<G: DeviceGen>(op: DeviceOp) -> Option<&'static DeviceOpRow> {
    G::DEVICE_OPS.iter().find(|r| r.op == op)
}

/// The formats `op` is bound at on `G`, or `None` when no row names it —
/// UNBOUND, fail-closed: the caller refuses, never degrades.
pub fn bound_dfs<G: DeviceGen>(op: DeviceOp) -> Option<&'static [DdlDf]> {
    row::<G>(op).map(|r| r.dfs)
}

/// Whether `op` is bound at `df` on `G`. An absent row is unbound at every
/// format: `false`, and the caller refuses by name.
pub fn binds<G: DeviceGen>(op: DeviceOp, df: DdlDf) -> bool {
    bound_dfs::<G>(op).is_some_and(|dfs| dfs.contains(&df))
}

/// The arrangement `op`'s output carries on `G` when both operands are
/// addressed standard, or `None` when no row names it. There is NO silent
/// `Standard` default here — a missing row is a fact the caller does not
/// hold, and the caller refuses by name rather than guessing (main's old
/// table defaulted silently; every convert op now states a row, so the
/// refusal is unreachable for every op this tree emits and the behavior is
/// unchanged).
pub fn output_arrangement<G: DeviceGen>(op: DeviceOp) -> Option<ElementArrangement> {
    row::<G>(op).map(|r| r.arrangement)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE LAW THIS TABLE EXISTS FOR: exp is bound fp32 alone in the
    /// pointwise family — the fact the RED-stop has always consulted (and
    /// the island-collapse reader that lands with #1142 exists to honor).
    #[test]
    fn exp_is_bound_fp32_alone() {
        assert!(binds::<Gen1>(DeviceOp::Exp, DdlDf::Fp32));
        assert!(!binds::<Gen1>(DeviceOp::Exp, DdlDf::Fp16));
    }

    /// The pipeline family, asserted as the TEMPLATES state it. Mish is
    /// fp16 alone (the template's own words); log binds BOTH — the pipeline
    /// fp16 and the parallel fp32. The fp32 log's ADMISSION is refused by
    /// POLICY at the reader that holds it (the RED-stop in legalize_types
    /// names it, pending a card cell) — the table is never falsified to
    /// carry that policy.
    #[test]
    fn mish_is_fp16_alone_by_template_and_log_binds_both_forms() {
        assert!(binds::<Gen1>(DeviceOp::Mish, DdlDf::Fp16));
        assert!(!binds::<Gen1>(DeviceOp::Mish, DdlDf::Fp32));
        assert!(binds::<Gen1>(DeviceOp::Log, DdlDf::Fp16));
        assert!(binds::<Gen1>(DeviceOp::Log, DdlDf::Fp32));
    }

    /// The two-tensor family and the one-input family collapse to f16 as
    /// they always have: every pointwise op the enum names is bound at both
    /// formats, except the measured exceptions — exp is fp32-only (measured
    /// wrong on card through the pipeline form, cg1112), mish is fp16-only
    /// (the template's own words), and the two integer broadcast ops run in
    /// fp32 MODE over uint32 tensors (the value casts' reading). Iterated
    /// over the enum's own exhaustive list, not a hand-picked slice.
    #[test]
    fn every_pointwise_op_the_door_ships_is_bound_except_the_measured_exceptions() {
        for op in DeviceOp::ALL {
            if op.family() != DeviceFamily::Pointwise {
                continue;
            }
            if matches!(
                op,
                DeviceOp::Exp | DeviceOp::Mish | DeviceOp::AddI32ToI32 | DeviceOp::MulI32ToI32
            ) {
                continue;
            }
            assert!(binds::<Gen1>(op, DdlDf::Fp16), "{op:?} must stay f16-bound");
            assert!(
                binds::<Gen1>(op, DdlDf::Fp32),
                "{op:?} must stay fp32-bound"
            );
        }
    }

    /// Every op the enum names states a row — iterated over the enum's own
    /// exhaustive list, so a variant added without a row fails HERE, not
    /// silently in a reader.
    #[test]
    fn every_op_in_the_enum_states_a_row() {
        for op in DeviceOp::ALL {
            assert!(
                Gen1::DEVICE_OPS.iter().any(|r| r.op == op),
                "{op:?} states no row"
            );
            assert!(
                output_arrangement::<Gen1>(op).is_some(),
                "{op:?} states no arrangement row"
            );
        }
    }

    /// ⛔ NO TWO ROWS NAME ONE OP — a duplicate would shadow the first and
    /// silently change a bind. With the typed key, this is also the
    /// misspelling guard: a row's key is an enum value, so a typo cannot
    /// compile, and this test pins that only one row exists per op.
    #[test]
    fn no_two_rows_name_one_op() {
        for (i, a) in Gen1::DEVICE_OPS.iter().enumerate() {
            for b in Gen1::DEVICE_OPS.iter().skip(i + 1) {
                assert!(a.op != b.op, "{a:?} and {b:?} share one op key");
            }
        }
    }

    /// Every row's op spells a non-empty door key, states its family, and
    /// carries a citation — the row's own shape, pinned.
    #[test]
    fn every_row_carries_a_spelling_a_family_and_a_citation() {
        for r in Gen1::DEVICE_OPS {
            assert!(!r.op.op_func().is_empty());
            assert!(!r.citation.is_empty());
            assert!(!r.dfs.is_empty(), "{:?} states no formats", r.op);
        }
    }

    /// The dl16 stagger row: the widen's output is the measured permuted
    /// arrangement; the narrow-back's is not (yet) — the propagation's own
    /// refusal guards it.
    #[test]
    fn the_dl16_stagger_row_is_the_widen_alone() {
        assert_eq!(
            output_arrangement::<Gen1>(DeviceOp::Dl16ToFp32),
            Some(ElementArrangement::Dl16ToFp32)
        );
        assert_eq!(
            output_arrangement::<Gen1>(DeviceOp::Fp32ToDl16),
            Some(ElementArrangement::Standard)
        );
        // No silent default: a row-less op answers `None` and the caller
        // refuses. Every DeviceOp variant states a row today, so `None` is
        // unreachable for every op the enum names — the Option is the API's
        // honesty for the first op that lands without one.
    }

    /// THE MoE ROW: the max reduce is bound at BOTH formats (the fp16 and
    /// fp32 reduce templates are line-for-line twins), and its output is
    /// standard — a reduction over a staggered operand is order-free, so
    /// the arrangement never escapes through it.
    #[test]
    fn the_reduce_family_is_bound_at_both_formats() {
        for op in DeviceOp::ALL {
            if op.family() != DeviceFamily::Reduce {
                continue;
            }
            assert!(binds::<Gen1>(op, DdlDf::Fp16), "{op:?} must stay f16-bound");
            assert!(
                binds::<Gen1>(op, DdlDf::Fp32),
                "{op:?} must stay fp32-bound"
            );
            assert_eq!(
                output_arrangement::<Gen1>(op),
                Some(ElementArrangement::Standard)
            );
        }
    }
}
