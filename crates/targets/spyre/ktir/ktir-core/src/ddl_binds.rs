// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

/// A device op, TYPED — the table's key. One variant per opFuncName the
/// table states a fact for; the door's string spelling is
/// [`DeviceOp::op_func`], a property, never a second key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceOp {
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
}

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

/// A device GENERATION. The marker trait: each generation states its own
/// device-facts table (and, in the modules that carry them, its own
/// constants — stick width, core count, LX budget, page geometry); a
/// lowering reads them through the generation it bakes for, never as free
/// constants. `Gen1` is today's only implementor; `Gen2` will be a second
/// impl with its own citations, not an edit here.
pub trait DeviceGen {
    /// This generation's device-facts table — the one statement of its
    /// card's (op, dtype, layout) facts, read by `legalize_types`
    /// (RED-stop) and the arrangement propagation today; the island
    /// collapse and pointwise-door consults that land next read the same
    /// rows.
    const DEVICE_OPS: &'static [DeviceOpRow];
}

/// Spyre generation 1 — the card this tree bakes today.
pub struct Gen1;

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
            citation: "unary_parallel.ddl:26-27 gelufwd [%type_fp16] / [%type_fp32]",
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
            citation: "unary_parallel.ddl:34-35 tanh [%type_fp16] / tanhfp32 [%type_fp32]",
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
            citation: "unary_parallel.ddl:41 sigmoid [%type_fp16, %type_fp32]",
        },
        DeviceOpRow {
            op: DeviceOp::Sqrt,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:42 sqrt [%type_fp16, %type_fp32]",
        },
        DeviceOpRow {
            op: DeviceOp::Rsqrt,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:43 rsqrt [%type_fp16, %type_fp32]",
        },
        // ── unary_pipeline.ddl — the FEST pipeline family, FP16 ALONE ("not
        //    possible in fp32 (FEST hardware limitation)" — the template's own
        //    words). These ops are bound where the collapse wants them and
        //    refused at fp32 by the RED-stop, exactly as before this table
        //    existed; stating them here keeps the refusal a table read instead
        //    of an absence.
        DeviceOpRow {
            op: DeviceOp::Mish,
            dfs: &[DdlDf::Fp16],
            arrangement: ElementArrangement::Standard,
            citation: "unary_pipeline.ddl:21 mish [%type_fp16] — not possible in \
                   fp32 (FEST hardware limitation)",
        },
        DeviceOpRow {
            op: DeviceOp::Log,
            dfs: &[DdlDf::Fp16],
            arrangement: ElementArrangement::Standard,
            citation: "unary_pipeline.ddl:22 log [%type_fp16]",
        },
        // ── the value casts, for the RED-stop's projection (`arith.sitofp` from
        //    an IEEE_INT32 input). Not pointwise-door ops; they ride the convert
        //    lowering. Bound fp32 alone, keyed on the DDL's own opFuncName.
        DeviceOpRow {
            op: DeviceOp::Int32ToFp32,
            dfs: &[DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:49 int32tofp32 [%type_fp32] (IEEE_INT32 in)",
        },
        DeviceOpRow {
            op: DeviceOp::Fp32ToInt32,
            dfs: &[DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "unary_parallel.ddl:50 fp32toint32 [%type_fp32] (IEEE_INT32 out)",
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
            citation: "quantization_double_pad.ddl:58 dl16tofp32 (fp16 in, fp32 \
                   out); the stagger MEASURED on card, probe-8 2026-10-26",
        },
        DeviceOpRow {
            op: DeviceOp::Fp32ToDl16,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "quantization_double_pad.ddl:59 fp32todl16 (fp32 in, fp16 \
                   out); the inverse stagger MEASURED on card — shift −1, \
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
            citation: "summeanmaxexx2.ddl:39 / summeanmaxexx2_fp32.ddl:39 sum",
        },
        DeviceOpRow {
            op: DeviceOp::Mean,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "summeanmaxexx2.ddl:41 / summeanmaxexx2_fp32.ddl:42 mean",
        },
        DeviceOpRow {
            op: DeviceOp::Max,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "summeanmaxexx2.ddl:44 / summeanmaxexx2_fp32.ddl:45 max",
        },
        DeviceOpRow {
            op: DeviceOp::Min,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "summeanmaxexx2.ddl:47 / summeanmaxexx2_fp32.ddl:48 min",
        },
        DeviceOpRow {
            op: DeviceOp::AbsMax,
            dfs: &[DdlDf::Fp16, DdlDf::Fp32],
            arrangement: ElementArrangement::Standard,
            citation: "summeanmaxexx2.ddl:53 / summeanmaxexx2_fp32.ddl:54 absmax",
        },
    ];
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

    /// The fp16-only pipeline family: log and mish collapse to f16 freely and
    /// are refused at fp32.
    #[test]
    fn the_fest_family_is_fp16_alone() {
        assert!(binds::<Gen1>(DeviceOp::Log, DdlDf::Fp16));
        assert!(!binds::<Gen1>(DeviceOp::Log, DdlDf::Fp32));
        assert!(binds::<Gen1>(DeviceOp::Mish, DdlDf::Fp16));
        assert!(!binds::<Gen1>(DeviceOp::Mish, DdlDf::Fp32));
    }

    /// The two-tensor family and the one-input family collapse to f16 as
    /// they always have: every one of them is f16-bound.
    #[test]
    fn every_pointwise_op_the_door_ships_is_f16_bound_except_exp() {
        for op in [
            DeviceOp::Silu,
            DeviceOp::Gelu,
            DeviceOp::Multiply,
            DeviceOp::Add,
            DeviceOp::Sub,
            DeviceOp::Neg,
            DeviceOp::Rsqrt,
            DeviceOp::Sqrt,
            DeviceOp::Abs,
            DeviceOp::Reciprocal,
            DeviceOp::Sigmoid,
            DeviceOp::Tanh,
            DeviceOp::RealDiv,
            DeviceOp::Maximum,
            DeviceOp::Minimum,
            DeviceOp::Equal,
        ] {
            assert!(binds::<Gen1>(op, DdlDf::Fp16), "{op:?} must stay f16-bound");
            assert!(
                binds::<Gen1>(op, DdlDf::Fp32),
                "{op:?} must stay fp32-bound"
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
        // No silent default: an op with no row answers `None`, and the caller
        // refuses rather than guessing `Standard`.
        assert_eq!(output_arrangement::<Gen1>(DeviceOp::Sum), None);
    }

    /// THE MoE ROW: the max reduce is bound at BOTH formats (the fp16 and
    /// fp32 reduce templates are line-for-line twins), and its output is
    /// standard — a reduction over a staggered operand is order-free, so
    /// the arrangement never escapes through it.
    #[test]
    fn the_reduce_family_is_bound_at_both_formats() {
        for op in [
            DeviceOp::Sum,
            DeviceOp::Mean,
            DeviceOp::Max,
            DeviceOp::Min,
            DeviceOp::AbsMax,
        ] {
            assert!(binds::<Gen1>(op, DdlDf::Fp16), "{op:?} must stay f16-bound");
            assert!(
                binds::<Gen1>(op, DdlDf::Fp32),
                "{op:?} must stay fp32-bound"
            );
            assert_eq!(
                output_arrangement::<Gen1>(op),
                Some(ElementArrangement::Standard)
            );
            assert_eq!(op.family(), DeviceFamily::Reduce);
        }
    }
}
