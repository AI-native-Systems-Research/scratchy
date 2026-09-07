//! UPSTREAM `symbol` — a scalar whose VALUE is not known until the schedule is instantiated.
//!
//! Not one of `dcc`'s own dialects: `Symbol.td` belongs to the dataflow scheduler
//! (`dataflow-scheduler/external/dataflow-scheduler-dialects/include/dataflow-scheduler/Dialect/Symbol/Symbol.td`),
//! and its ops arrive in the DataflowIR the scheduler hands to `dcc`.
//!
//! ⭐ IT IS NOT RARE. `symbol.create_symbol` appears **560** times across the authority tree's
//! `dcc/test`, which is why the lowerings test for it by name rather than falling through.

use std::fmt::Write as _;

use crate::islands::dataflow_ir::dialects::Val;
use crate::islands::dataflow_ir::print;

/// ONE `symbol` OPERATION.
///
/// ⚠️ ONE OF `Symbol.td`'S FOUR. `symbol.create_id`, `symbol.symbol_immutable_mapping` and
/// `symbol.query_map` are declared beside it and are absent here: no bridge-2 function this campaign
/// has reached names any of the three, and an op nothing reads would be a variant every total match
/// in this island has to answer for with nothing to say.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `symbol.create_symbol {SymbolId = N : i32} : index` — an `index` standing for a quantity the
    /// schedule fixes later.
    ///
    /// ⛔⛔ A LOOP BOUND DEFINED BY ONE IS A BOUND THE LOWERING TREATS AS **ZERO**, DELIBERATELY.
    /// `getForOpBound` (entry 091) reaches through a `divsi`/`subi` chain to the value that feeds it
    /// and answers 0 for this op, with the reference's own reason attached
    /// (`Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:278-284`):
    ///
    /// > *"We know that XRF read/write accesses don't involve loops with symbolic bounds. So, the
    /// > caller of this function which is computing the movement, it can be safe to treat as zero."*
    ///
    /// Without this op in the island that branch is unreachable and a symbolic bound would fall into
    /// the `else` that stops the compile — the opposite answer.
    ///
    /// ⭐ NO OPERANDS AND NO SIDE EFFECT (`NoMemoryEffect`, `Symbol.td:53`): the id is an attribute,
    /// so two symbols differ only by it.
    CreateSymbol {
        /// The `index` it binds.
        result: Val,
        /// `SymbolId` — read back by `getSymbolID()` (`Symbol.td:64`).
        ///
        /// ⭐ SIGNED, AND THE TREE USES BOTH SIGNS: `{SymbolId = 0 : i32}`
        /// (`dcc/test/LXLU/int8-kg3-lxlu-symbol.mlir:175`) and `{SymbolId = -43 : i64}`
        /// (`dcc/test/Conversion/SentientToProgIR/uniform-nop-incorrect-label.mlir:285`).
        symbol_id: i64,

        /// `maxValue` — THE LARGEST VALUE THE SCHEDULE MAY FIX THIS SYMBOL TO, when it says.
        ///
        /// # ⛔⛔ A DISCARDABLE ATTRIBUTE, AND THE ONE THING THAT MAKES A SYMBOLIC LOOP BOUND USABLE
        ///
        /// `Symbol.td` declares no attributes at all — the op's assembly format is a bare
        /// `attr-dict `:` type(results)` and `getSymbolID()` reaches into the dictionary by name
        /// (`Symbol.td:57-68`) — so `maxValue` is one the scheduler *adds*, and exactly one function
        /// in the whole authority tree reads it back:
        ///
        /// ```text
        /// } else if (auto ub_sym_op = dyn_cast<symbol::CreateSymbolOp>(ub_op)) {
        ///   if (ub_sym_op->hasAttr("maxValue"))
        ///     ub = cast<IntegerAttr>(ub_sym_op->getAttr("maxValue")).getInt();
        ///   else
        ///     return false;
        /// }
        /// ```
        /// (`Transform/Dataflow/MutableAddrSplitting.cpp:772-778`, bridge-2 entry 184
        /// [`get_loop_trip_count`](crate::bridges::dataflow_ir_to_sentient::tf_mutable_addr_splitting::get_loop_trip_count))
        ///
        /// ⭐ `Option`, BECAUSE `hasAttr` IS THE QUESTION THE READER ASKS. Absent is not zero: a
        /// symbol with no `maxValue` makes that function bail (returning **0**, since `return false`
        /// in an `int64_t` function), while `maxValue = 0` would be a bound of zero it accepts.
        ///
        /// ⭐⭐ AND THE VENDOR'S OWN ANSWER KEY DEPENDS ON IT.
        /// `constant_start_addr_1` writes an `scf.for` whose upper bound is
        /// `symbol.create_symbol {SymbolId = -1476 : i64, granularity = 8 : i64, maxValue = 8 : i64}`
        /// (`dcc/test/Transform/MutableAddrSplitting/mutable_addr_splitting_one_dim.mlir:251`) and
        /// expects the pass to partition against a trip count of **8**. Without this field the bound
        /// reads as absent, the trip count is 0, and the case cannot be built.
        ///
        /// ⚠️ `granularity` IS THE OTHER ATTRIBUTE ON THAT LINE AND IS DELIBERATELY ABSENT HERE.
        /// Nothing in the authority tree reads it — `grep -rn granularity dcc/src` finds no reader of
        /// a `create_symbol`'s — so it would be a field this island prints and no port consults, in
        /// the same way the three other `Symbol.td` ops are absent above. ⛔ The consequence is that
        /// the emitted text for that vendor line is one attribute short of theirs; recorded here
        /// rather than papered over, because a field is cheap to add the day a reader appears.
        max_value: Option<i64>,
    },
}

/// ONE `symbol` OP AS TEXT. The caller has already indented.
pub(crate) fn emit(out: &mut String, op: &Op) {
    match op {
        Op::CreateSymbol {
            result,
            symbol_id,
            max_value,
        } => {
            // `attr-dict `:` type(results)` (`Symbol.td:60-62`) — the result is always `index`.
            //
            // ⭐ `i32`, WHICH IS WHAT THE SCHEDULER WRITES AND WHAT THE REFERENCE'S OWN
            // `CHECK-SENT-IR` EXPECTS: `%[[VAL_38:.*]] = symbol.create_symbol {SymbolId = 0 : i32}`
            // (`dcc/test/LXLU/int8-kg3-lxlu-symbol.mlir:51`). `getSymbolID()` casts to a plain
            // `IntegerAttr`, so the width is not read back.
            //
            // ⭐ `maxValue` FOLLOWS `SymbolId`, BECAUSE MLIR SORTS AN ATTRIBUTE DICTIONARY BY NAME
            // and `S` precedes `m` in ASCII — which is the order the vendor's own line comes out in:
            // `{SymbolId = -1476 : i64, granularity = 8 : i64, maxValue = 8 : i64}`
            // (`dcc/test/Transform/MutableAddrSplitting/mutable_addr_splitting_one_dim.mlir:251`).
            //
            // ⭐ AT `i64`, WHICH IS THE WIDTH THAT LINE WRITES IT AT. `getAttr("maxValue")` is read
            // back through a plain `IntegerAttr`, so the width is not consulted either.
            let max = match max_value {
                Some(max) => format!(", maxValue = {max} : i64"),
                None => String::new(),
            };
            let _ = writeln!(
                out,
                "{} = symbol.create_symbol {{SymbolId = {symbol_id} : i32{max}}} : index",
                print::val(*result)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::islands::dataflow_ir::dialects::Val;
    use crate::islands::dataflow_ir::dialects::symbol::{Op, emit};

    /// ⭐⭐ IBM'S OWN LINE, REPRODUCED BYTE FOR BYTE.
    ///
    /// `dcc/test/LXLU/int8-kg3-lxlu-symbol.mlir:51` — the `CHECK-SENT-IR` expectation, with the
    /// capture standing in for the SSA name.
    #[test]
    fn a_symbol_prints_as_the_reference_writes_it() {
        let mut out = String::new();
        emit(
            &mut out,
            &Op::CreateSymbol {
                result: Val(38),
                symbol_id: 0,
                max_value: None,
            },
        );
        assert_eq!(
            out,
            "%38 = symbol.create_symbol {SymbolId = 0 : i32} : index\n"
        );
    }

    /// AND A NEGATIVE ID PRINTS ITS SIGN — `{SymbolId = -43 : i64}`
    /// (`dcc/test/Conversion/SentientToProgIR/uniform-nop-incorrect-label.mlir:285`), at this
    /// island's own width.
    #[test]
    fn a_negative_symbol_id_keeps_its_sign() {
        let mut out = String::new();
        emit(
            &mut out,
            &Op::CreateSymbol {
                result: Val(220),
                symbol_id: -43,
                max_value: None,
            },
        );
        assert_eq!(
            out,
            "%220 = symbol.create_symbol {SymbolId = -43 : i32} : index\n"
        );
    }

    /// AND A `maxValue` PRINTS AFTER THE ID — the bound
    /// [`get_loop_trip_count`](crate::bridges::dataflow_ir_to_sentient::tf_mutable_addr_splitting::get_loop_trip_count)
    /// reads.
    ///
    /// ⭐ THE VENDOR'S OWN SYMBOL, from
    /// `dcc/test/Transform/MutableAddrSplitting/mutable_addr_splitting_one_dim.mlir:251` — ⚠️ minus
    /// its `granularity`, which this island does not carry (see [`Op::CreateSymbol`]), and at this
    /// island's `i32` id width.
    #[test]
    fn a_max_value_prints_after_the_symbol_id() {
        let mut out = String::new();
        emit(
            &mut out,
            &Op::CreateSymbol {
                result: Val(719),
                symbol_id: -1476,
                max_value: Some(8),
            },
        );
        assert_eq!(
            out,
            "%719 = symbol.create_symbol {SymbolId = -1476 : i32, maxValue = 8 : i64} : index\n"
        );
    }
}
