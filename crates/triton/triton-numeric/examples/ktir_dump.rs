// DIAGNOSTIC: dump the op list of a lowered configuration around a chosen SSA, to attribute an
// executor refusal to a producer chain. Prints each op's kind, result, operands and Dim
// attribute, so a "charging %N tile [r, c]" refusal can be walked backwards without guessing.

use triton_numeric::lower;

fn main() {
    let config = std::env::args().nth(1).expect("usage: ktir_dump <config> [ssa]");
    let ssa_want: Option<u32> = std::env::args().nth(2).and_then(|s| s.parse().ok());
    let l = lower(&config).expect("the config lowers");

    // Map SSA -> defining op, so producers can be named on demand.
    let ops: Vec<&ktir_core::ir::Operation> = l.func.ops_deep().into_iter().collect();
    println!("{}: {} deep ops, {} arguments", config, ops.len(), l.func.arguments.len());
    for (i, (name, ty)) in l.func.arguments.iter().enumerate() {
        println!("  arg{i}: %{name:?} {ty:?}");
    }
    let find = |s: ktir_core::ir::Ssa| ops.iter().position(|o| o.result == Some(s));
    let show = |o: &ktir_core::ir::Operation, prefix: &str| {
        let dim = o
            .attr(ktir_core::attrkey::AttrKey::Dim)
            .map(|a| format!("{a:?}"))
            .unwrap_or_default();
        let value = o
            .attr(ktir_core::attrkey::AttrKey::Value)
            .map(|a| format!("{a:?}"))
            .unwrap_or_default();
        let names: Vec<String> = o.operands.iter().map(|s| format!("%{:?}", s)).collect();
        println!(
            "{prefix}{:?} -> %{:?} = op({}) [{}] dim={} value={} ty={:?}",
            o.op_type,
            o.result.map(|r| r.0).unwrap_or(u32::MAX),
            o.operands.len(),
            names.join(", "),
            dim,
            value,
            o.result_type
        );
    };
    if let Some(w) = ssa_want {
        let s = ktir_core::ir::Ssa(w);
        // producers, two levels up
        let mut frontier = vec![s];
        for level in 0..3 {
            let mut next = Vec::new();
            for s in &frontier {
                if let Some(i) = find(*s) {
                    show(ops[i], &format!("L{level} "));
                    next.extend(ops[i].operands.iter().copied());
                } else if let Some((i, _)) =
                    l.func.arguments.iter().enumerate().find(|(_, (a, _))| a == s)
                {
                    println!("L{level} %{s:?} is argument {i}");
                }
            }
            frontier = next;
        }
        // consumers, two levels down
        let mut frontier = vec![s];
        for level in 0..3 {
            let mut next = Vec::new();
            for s in &frontier {
                for o in &ops {
                    if o.operands.contains(s) {
                        show(o, &format!("C{level} "));
                        next.push(o.result.unwrap_or(*s));
                    }
                }
            }
            frontier = next;
        }
    } else {
        for o in &ops {
            show(o, "  ");
        }
    }
}
