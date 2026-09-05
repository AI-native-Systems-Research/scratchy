//! WHAT BRIDGE 1 ACTUALLY HAS TO WALK — the statement kinds the reachable schedules contain.
//!
//! ⭐ MEASURED, NOT ESTIMATED. "Implement the DDL statements" is 36 mnemonics if you count the
//! census and something much smaller if you count what the op-funcs a forward tape names actually
//! use. This prints the second number, per op-func and in total, so the walker is built against the
//! surface that exists rather than the one the enum suggests.

use std::collections::BTreeMap;

use deeptools::arch::{Arch, Target};
use deeptools::generated::{DataType, OpFunc};

fn main() {
    // The op-funcs a dense decoder-only forward names, as the tape's SubOps map onto them. Rope,
    // rmsnorm and attention arrive already decomposed by the tape's own rewrites, so they appear
    // here as their pieces rather than as fused ops.
    let reachable = [
        OpFunc::Matmul,
        OpFunc::Batchmatmul,
        OpFunc::Add,
        OpFunc::Sub,
        OpFunc::Mul,
        OpFunc::Realdiv,
        OpFunc::Silu,
        OpFunc::Exp,
        OpFunc::Reciprocal,
        OpFunc::Rsqrt,
        OpFunc::Sqrt,
        OpFunc::Sigmoid,
        OpFunc::Tanh,
        OpFunc::Sum,
        OpFunc::Max,
        OpFunc::Mean,
        OpFunc::Identity,
        OpFunc::Maximum,
        OpFunc::Minimum,
        OpFunc::Restickifyophbm,
        OpFunc::InterslicetransposeFp16,
    ];

    let mut overall: BTreeMap<String, usize> = BTreeMap::new();
    let mut widest: Vec<(usize, OpFunc)> = Vec::new();

    for op_func in reachable {
        let program = op_func.program(Target::GEN, DataType::Sen169Fp16);
        let mut per_op: BTreeMap<String, usize> = BTreeMap::new();
        for stmt in program.stmts {
            *per_op.entry(format!("{:?}", stmt.kind)).or_default() += 1;
            *overall.entry(format!("{:?}", stmt.kind)).or_default() += 1;
        }
        widest.push((program.stmts.len(), op_func));
        println!(
            "{:<26} template {:?}  bind {:<22} {} stmts, {} kinds",
            format!("{op_func:?}"),
            program.template,
            program.bind,
            program.stmts.len(),
            per_op.len()
        );
    }

    println!("\n== every statement kind these schedules use, by total count ==");
    let mut rows: Vec<(&String, &usize)> = overall.iter().collect();
    rows.sort_by(|a, b| b.1.cmp(a.1));
    for (kind, count) in &rows {
        println!("  {count:>6}  {kind}");
    }
    println!("\n{} distinct statement kinds used by these schedules", overall.len());

    widest.sort_by_key(|(len, _)| std::cmp::Reverse(*len));
    println!("\n== widest schedules ==");
    for (len, op_func) in widest.iter().take(5) {
        println!("  {len:>6} stmts  {op_func:?}");
    }
}

