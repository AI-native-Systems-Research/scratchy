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
    },
}

/// ONE `symbol` OP AS TEXT. The caller has already indented.
pub(crate) fn emit(out: &mut String, op: &Op) {
    match op {
        Op::CreateSymbol { result, symbol_id } => {
            // `attr-dict `:` type(results)` (`Symbol.td:60-62`) — the result is always `index`.
            //
            // ⭐ `i32`, WHICH IS WHAT THE SCHEDULER WRITES AND WHAT THE REFERENCE'S OWN
            // `CHECK-SENT-IR` EXPECTS: `%[[VAL_38:.*]] = symbol.create_symbol {SymbolId = 0 : i32}`
            // (`dcc/test/LXLU/int8-kg3-lxlu-symbol.mlir:51`). `getSymbolID()` casts to a plain
            // `IntegerAttr`, so the width is not read back.
            let _ = writeln!(
                out,
                "{} = symbol.create_symbol {{SymbolId = {symbol_id} : i32}} : index",
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
            },
        );
        assert_eq!(
            out,
            "%220 = symbol.create_symbol {SymbolId = -43 : i32} : index\n"
        );
    }
}
