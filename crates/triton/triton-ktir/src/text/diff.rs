// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! THE STRUCTURAL DIFF: field by field, and with a control.
//!
//! # WHY NOT A TEXT DIFF
//!
//! MLIR regenerates SSA names on every print (`%acc_30` here, `%acc_26` there) and
//! its constant insertion order is internal. A text diff reports those as
//! differences and a reader learns to ignore the output, which is worse than no
//! diff. So the comparison is on the SSA GRAPH:
//!
//! * each op contributes its KIND, its ATTRIBUTES, its RESULT TYPES, and the
//!   CANONICAL IDS of its operands;
//! * a canonical id is the position of the operand's defining op in a numbering
//!   derived from program order -- so a RENAME is invisible and a REWIRE is not;
//! * ops are compared IN ORDER within each block, because program order is
//!   semantic in MLIR;
//! * with ONE exception, stated and bounded: a block's leading `arith.constant`s
//!   are compared as a MULTISET. A constant has no operands and no side effects, so
//!   its position among other constants is not semantic, and matching MLIR's
//!   internal hoist order is a claim this port cannot make from the outside.
//!
//! # THE CONTROL IS NOT OPTIONAL
//!
//! A diff can pass while comparing nothing -- a normaliser that erases the field
//! that differs, a walk that visits no ops, a comparison of two empty strings. So
//! [`Difference`] carries the FIELD NAME, and the golden tests assert that each of
//! a set of PLANTED differences is caught AND NAMED. See
//! `tests/golden_ktir.rs::planted_differences_are_each_caught_by_name`.

use std::collections::HashMap;

use crate::ir::*;

/// One structural difference, naming the field that differs.
#[derive(Clone, Debug, PartialEq)]
pub struct Difference {
    /// Where, as a readable path: `"@attn_fwd/scf.for[4]/linalg.matmul[7]"`.
    pub where_: String,
    /// WHICH FIELD -- the part that makes the control possible.
    pub field: &'static str,
    pub expected: String,
    pub actual: String,
}

impl std::fmt::Display for Difference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} differs\n    golden: {}\n    ours:   {}",
            self.where_, self.field, self.expected, self.actual
        )
    }
}

/// Compare two modules structurally. An empty result is agreement.
pub fn diff(golden: &Module, ours: &Module) -> Vec<Difference> {
    let mut out = Vec::new();
    let g = Numbering::of(golden);
    let o = Numbering::of(ours);
    diff_block(&g, &o, &golden.ops, &ours.ops, "", &mut out);
    // Module-level attributes are load-bearing: `spyre.canonical_verified` is the
    // emitter's whole trust decision.
    let ga = attr_summary(&golden.attrs);
    let oa = attr_summary(&ours.attrs);
    if ga != oa {
        out.push(Difference {
            where_: "module".into(),
            field: "module attributes",
            expected: ga,
            actual: oa,
        });
    }
    out
}

/// A canonical id per value: the index of its defining op in program order, plus
/// the result number. A block argument is named by its owner and position, so the
/// function's arg 0 is `arg0` in both modules regardless of its spelling.
pub struct Numbering {
    ids: HashMap<Ssa, String>,
}

impl Numbering {
    /// A CONSTANT IS NAMED BY ITS VALUE; every other op by its position among the
    /// NON-constant ops.
    ///
    /// This is the other half of treating constants as an unordered set. Numbering
    /// every op by its absolute position leaks constant ORDER into the id of every
    /// value downstream, so two modules that differ only in where the hoisted
    /// constants landed report an `operands` difference on every op that reads one --
    /// a rename presented as a rewire, which is exactly what the diff is supposed to
    /// see through. Naming a constant by `const(value:type)` makes the id independent
    /// of position, and makes two occurrences of the same constant interchangeable,
    /// which they are.
    fn of(module: &Module) -> Numbering {
        let mut ids = HashMap::new();
        // Function arguments first, so they are stable across the two modules.
        for op in module.ops_deep() {
            if op.kind == OpKind::TtFunc {
                if let Some(r) = op.regions.first() {
                    for (i, (v, _)) in r.args.iter().enumerate() {
                        ids.insert(*v, format!("arg{i}"));
                    }
                }
            }
        }
        let mut n = 0usize;
        for op in module.ops_deep() {
            if is_orderless_constant(op) {
                let key = format!(
                    "const({}:{})",
                    op.attr(&AttrKey::Value)
                        .map(super::print::print_attr)
                        .unwrap_or_else(|| "?".into()),
                    op.result_type().map(super::print::print_type).unwrap_or_else(|| "?".into())
                );
                for r in &op.results {
                    ids.insert(*r, key.clone());
                }
                continue;
            }
            for (k, r) in op.results.iter().enumerate() {
                ids.insert(*r, format!("v{n}#{k}"));
            }
            // Region block arguments (an scf.for's IV and iter_args, a combiner's
            // pair) are named by their owner's position.
            for (ri, region) in op.regions.iter().enumerate() {
                for (ai, (v, _)) in region.args.iter().enumerate() {
                    ids.entry(*v).or_insert_with(|| format!("v{n}r{ri}a{ai}"));
                }
            }
            n += 1;
        }
        Numbering { ids }
    }

    fn id(&self, v: Ssa) -> String {
        self.ids.get(&v).cloned().unwrap_or_else(|| "<undef>".to_string())
    }
}

/// Is this op a leading constant, whose position among other constants is not
/// semantic?
fn is_orderless_constant(op: &Op) -> bool {
    op.kind == OpKind::ArithConstant && op.regions.is_empty()
}

fn diff_block(
    g: &Numbering,
    o: &Numbering,
    golden: &[Op],
    ours: &[Op],
    path: &str,
    out: &mut Vec<Difference>,
) {
    // Split each block into its orderless constants and the ordered remainder.
    let (gc, gr): (Vec<&Op>, Vec<&Op>) = golden.iter().partition(|x| is_orderless_constant(x));
    let (oc, or_): (Vec<&Op>, Vec<&Op>) = ours.iter().partition(|x| is_orderless_constant(x));

    // Constants: a MULTISET comparison, which is the one exception and it is bounded.
    let mut gk: Vec<String> = gc.iter().map(|x| fingerprint(g, x)).collect();
    let mut ok: Vec<String> = oc.iter().map(|x| fingerprint(o, x)).collect();
    gk.sort();
    ok.sort();
    if gk != ok {
        let only_golden: Vec<&String> = gk.iter().filter(|x| !ok.contains(x)).collect();
        let only_ours: Vec<&String> = ok.iter().filter(|x| !gk.contains(x)).collect();
        out.push(Difference {
            where_: format!("{path}/<constants>"),
            field: "constant set",
            expected: format!("{} constants; missing here: {only_golden:?}", gk.len()),
            actual: format!("{} constants; extra here: {only_ours:?}", ok.len()),
        });
    }

    // Everything else, IN ORDER.
    if gr.len() != or_.len() {
        out.push(Difference {
            where_: path.to_string(),
            field: "op count",
            expected: format!(
                "{} non-constant ops: {:?}",
                gr.len(),
                gr.iter().map(|x| x.kind.spelling()).collect::<Vec<_>>()
            ),
            actual: format!(
                "{} non-constant ops: {:?}",
                or_.len(),
                or_.iter().map(|x| x.kind.spelling()).collect::<Vec<_>>()
            ),
        });
    }
    for (i, (ga, oa)) in gr.iter().zip(or_.iter()).enumerate() {
        diff_op(g, o, ga, oa, &format!("{path}/{}[{i}]", ga.kind.spelling()), out);
    }
}

fn diff_op(
    g: &Numbering,
    o: &Numbering,
    golden: &Op,
    ours: &Op,
    path: &str,
    out: &mut Vec<Difference>,
) {
    if golden.kind != ours.kind {
        out.push(Difference {
            where_: path.into(),
            field: "op kind",
            expected: golden.kind.spelling().into(),
            actual: ours.kind.spelling().into(),
        });
        return; // nothing below is comparable
    }
    // Operands, by CANONICAL ID -- a rename is invisible, a rewire is not.
    let gops: Vec<String> = golden.operands.iter().map(|v| g.id(*v)).collect();
    let oops: Vec<String> = ours.operands.iter().map(|v| o.id(*v)).collect();
    if gops != oops {
        out.push(Difference {
            where_: path.into(),
            field: "operands",
            expected: format!("{gops:?}"),
            actual: format!("{oops:?}"),
        });
    }
    let gt: Vec<String> = golden.result_types.iter().map(super::print::print_type).collect();
    let ot: Vec<String> = ours.result_types.iter().map(super::print::print_type).collect();
    if gt != ot {
        out.push(Difference {
            where_: path.into(),
            field: "result types",
            expected: format!("{gt:?}"),
            actual: format!("{ot:?}"),
        });
    }
    let ga = attr_summary(&golden.attrs);
    let oa = attr_summary(&ours.attrs);
    if ga != oa {
        out.push(Difference {
            where_: path.into(),
            field: "attributes",
            expected: ga,
            actual: oa,
        });
    }
    if golden.regions.len() != ours.regions.len() {
        out.push(Difference {
            where_: path.into(),
            field: "region count",
            expected: golden.regions.len().to_string(),
            actual: ours.regions.len().to_string(),
        });
        return;
    }
    for (ri, (grr, orr)) in golden.regions.iter().zip(&ours.regions).enumerate() {
        // Block argument TYPES are load-bearing: LegalizeTypes retypes a combiner's
        // pair, and a port that misses that leaves an f32 combiner.
        let gargs: Vec<String> = grr.args.iter().map(|(_, t)| super::print::print_type(t)).collect();
        let oargs: Vec<String> = orr.args.iter().map(|(_, t)| super::print::print_type(t)).collect();
        if gargs != oargs {
            out.push(Difference {
                where_: format!("{path}/region{ri}"),
                field: "block argument types",
                expected: format!("{gargs:?}"),
                actual: format!("{oargs:?}"),
            });
        }
        diff_block(g, o, &grr.ops, &orr.ops, &format!("{path}/region{ri}"), out);
    }
}

/// A stable, order-insensitive rendering of an attribute list.
///
/// Sorted, because MLIR's attribute order is a dictionary's and carries no meaning;
/// the VALUES do, so they are printed in full.
fn attr_summary(attrs: &[(AttrKey, Attr)]) -> String {
    let mut parts: Vec<String> = attrs
        .iter()
        .map(|(k, v)| format!("{}={}", k.spelling(), super::print::print_attr(v)))
        .collect();
    parts.sort();
    format!("{{{}}}", parts.join(", "))
}

/// An op's whole structural identity, as one string. Used for the constant
/// multiset and available to a caller that wants to compare op sets directly.
pub fn fingerprint(n: &Numbering, op: &Op) -> String {
    format!(
        "{}({})->{}{}",
        op.kind.spelling(),
        op.operands.iter().map(|v| n.id(*v)).collect::<Vec<_>>().join(","),
        op.result_types
            .iter()
            .map(super::print::print_type)
            .collect::<Vec<_>>()
            .join(","),
        attr_summary(&op.attrs)
    )
}

/// A CENSUS DIFF: how many of each op kind, compared. A wrong count IS the bug, and
/// this catches it in one line even when the structural walk has a lot to say.
pub fn census_diff(golden: &Module, ours: &Module) -> Vec<Difference> {
    let g: HashMap<String, usize> = golden.census().into_iter().collect();
    let o: HashMap<String, usize> = ours.census().into_iter().collect();
    let mut keys: Vec<&String> = g.keys().chain(o.keys()).collect();
    keys.sort();
    keys.dedup();
    let mut out = Vec::new();
    for k in keys {
        let gv = g.get(k).copied().unwrap_or(0);
        let ov = o.get(k).copied().unwrap_or(0);
        if gv != ov {
            out.push(Difference {
                where_: format!("census/{k}"),
                field: "op count",
                expected: gv.to_string(),
                actual: ov.to_string(),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse;

    const BASE: &str = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %s = arith.constant 1.275630e-01 : f16
    %sp = tensor.splat %s : tensor<64xf16>
    %a = arith.addf %sp, %sp : tensor<64xf16>
    tt.return
  }
}
";

    #[test]
    fn a_module_agrees_with_itself() {
        let a = parse::parse(BASE).unwrap();
        let b = parse::parse(BASE).unwrap();
        assert!(diff(&a, &b).is_empty(), "{:?}", diff(&a, &b));
        assert!(census_diff(&a, &b).is_empty());
    }

    /// A RENAME must be invisible: MLIR regenerates names on every print, so a diff
    /// that reports them is a diff nobody reads.
    #[test]
    fn renaming_every_value_changes_nothing() {
        let renamed = BASE
            .replace("%c0", "%zz0")
            .replace("%c1", "%zz1")
            .replace("%sp", "%qq")
            .replace("%s ", "%tt ")
            .replace("%s :", "%tt :")
            .replace("%a ", "%bb ");
        let a = parse::parse(BASE).unwrap();
        let b = parse::parse(&renamed).unwrap();
        let d = diff(&a, &b);
        assert!(d.is_empty(), "a rename is not a difference: {d:?}");
    }

    /// THE CONTROL. Each planted change must be caught, and caught BY FIELD NAME --
    /// without this the diff could pass while comparing nothing.
    #[test]
    fn each_planted_difference_is_caught_by_name() {
        let cases: Vec<(&str, &str, &str)> = vec![
            (
                "op kind",
                "%a = arith.addf %sp, %sp : tensor<64xf16>",
                "%a = arith.mulf %sp, %sp : tensor<64xf16>",
            ),
            (
                "result types",
                "%a = arith.addf %sp, %sp : tensor<64xf16>",
                "%a = arith.addf %sp, %sp : tensor<64xf32>",
            ),
            (
                "operands",
                "%a = arith.addf %sp, %sp : tensor<64xf16>",
                "%a = arith.addf %sp, %s : tensor<64xf16>",
            ),
            (
                "constant set",
                "%c1 = arith.constant 1 : index",
                "%c1 = arith.constant 7 : index",
            ),
            (
                "op count",
                "%a = arith.addf %sp, %sp : tensor<64xf16>",
                "%a = arith.addf %sp, %sp : tensor<64xf16>\n    %a2 = arith.addf %a, %a : tensor<64xf16>",
            ),
        ];
        let golden = parse::parse(BASE).unwrap();
        for (field, from, to) in cases {
            let mutated = BASE.replace(from, to);
            assert_ne!(mutated, BASE, "the planted change for `{field}` did not apply");
            let ours = parse::parse(&mutated).unwrap();
            let d = diff(&golden, &ours);
            assert!(
                d.iter().any(|x| x.field == field),
                "planting a `{field}` difference was NOT caught; got {d:?}"
            );
        }
    }

    #[test]
    fn an_attribute_change_is_caught() {
        let golden = parse::parse(BASE).unwrap();
        let mutated = BASE.replace(
            "%sp = tensor.splat %s : tensor<64xf16>",
            "%sp = tensor.splat %s {axis = 1 : i32} : tensor<64xf16>",
        );
        let ours = parse::parse(&mutated).unwrap();
        let d = diff(&golden, &ours);
        assert!(d.iter().any(|x| x.field == "attributes"), "got {d:?}");
    }

    #[test]
    fn a_retyped_combiner_block_argument_is_caught() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %x = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
    %m = \"tt.reduce\"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f16, %b: f16):
      %c = arith.maxnumf %a, %b : f16
      tt.reduce.return %c : f16
    }) : (tensor<64x64xf16>) -> tensor<64xf16>
    tt.return
  }
}
";
        let golden = parse::parse(src).unwrap();
        let ours = parse::parse(&src.replace("^bb0(%a: f16, %b: f16)", "^bb0(%a: f32, %b: f32)"))
            .unwrap();
        let d = diff(&golden, &ours);
        assert!(
            d.iter().any(|x| x.field == "block argument types"),
            "an f32 combiner must be caught: {d:?}"
        );
    }

    #[test]
    fn the_census_diff_names_the_op_whose_count_is_wrong() {
        let golden = parse::parse(BASE).unwrap();
        let ours = parse::parse(&BASE.replace(
            "%a = arith.addf %sp, %sp : tensor<64xf16>",
            "%a = arith.addf %sp, %sp : tensor<64xf16>\n    %a2 = arith.mulf %a, %a : tensor<64xf16>",
        ))
        .unwrap();
        let d = census_diff(&golden, &ours);
        assert!(d.iter().any(|x| x.where_ == "census/arith.mulf"), "got {d:?}");
    }
}
