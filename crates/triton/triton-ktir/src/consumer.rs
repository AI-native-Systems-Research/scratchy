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

//! THE CONSUMER CONTRACT: what scratchy's `ktir_to_superdsc` can actually accept.
//!
//! The KTIR → SuperDSC leg is no longer ours. scratchy's
//! `crates/targets/spyre/src/ktir_to_superdsc/` walks
//! `ktir_core::ir::{IRFunction, Operation}` directly -- no text, no printer, no
//! parser -- and our text-reading `triton-superdsc-lower` is superseded. So the
//! question for THIS crate stops being "does our KTIR match the C++'s" and becomes
//! "can our KTIR be EXPRESSED in their type and WALKED by their lowering".
//!
//! Those are different questions, and the second one currently answers NO. This
//! module says so as DATA rather than as a comment, so the answer changes when the
//! facts do instead of when someone remembers to reread a paragraph.
//!
//! # WHERE THE VOCABULARY BELOW COMES FROM
//!
//! Read out of their tree, with the file and the reason, not inferred:
//!
//! * `ktir/ktir-core/src/opkind.rs` -- the `OpKind` enum. It has **388 lines and
//!   ZERO `tt.*` variants**: the vocabulary is `ktdp`/`arith`/`math`/`linalg`/`scf`/
//!   `func`. A `tt.reduce` is not "unsupported" in their type; it is
//!   UNREPRESENTABLE.
//! * `ktir/ktir-core/src/attrkey.rs` -- the `AttrKey` enum. It has no
//!   `AccessTileOrder` and no `AccessTileSet`, but it DOES have `CoordinateOrder` and
//!   `Permutation`, which is what gap 4's fix can use.
//! * `ktir/ktir-core/src/ir.rs:173` -- `regions: &'a [&'a [Operation<'a>]]`. A
//!   region is a bare op list: **there are no block arguments**. An `scf.for`'s
//!   induction variable and loop-carried values are declared by the
//!   `AttrKey::IterVar` / `AttrKey::IterArgs` attributes instead.
//! * `src/ktir_to_superdsc/lower.rs` and `opmap.rs` -- the walker. [`WALKED_OPS`] is
//!   the union of `lower.rs`'s dispatch arms and `opmap.rs`'s table, and
//!   [`READ_ATTRS`] the exact set of `AttrKey`s it reads (7 of them).
//!
//! # THE RUNTIME-BROADCAST FORM, ANSWERED FROM THEIR TREE
//!
//! `TensorSplat` is scalar-and-constant-only on their path (their walker's
//! `tensor_splat` mints a caller-filled buffer from a float scalar), so it cannot express
//! a RUNTIME broadcast of a reduced value back across lanes -- which attention's online
//! softmax and RMSNorm both need. The question was how scratchy expresses that today.
//!
//! **THE FORM EXISTS, AND IT IS THE NAMED `linalg.broadcast`.** From their producer,
//! `crates/targets/spyre/src/lower_subtile_tape_to_superdsc.rs:11457`:
//!
//! ```text
//!   %init = tensor.empty          {shape = dims, dtype = f16}
//!   %v    = linalg.broadcast (%x, %init) {dimensions = [dim]}
//! ```
//!
//! Not `tensor.splat`, and not a `linalg.generic`. Their RMSNorm uses exactly it, at the
//! place the reduction result has to reach the lanes again --
//! `let invb = self.broadcast(inv_e, dims.clone(), 1);` -- fanning the reduced `1/rms`
//! across the row. Their two `LinalgReduce` emission sites (11352, 11527) both feed that
//! shape.
//!
//! **AND THEIR WALKER CANNOT CONSUME IT YET.** `grep -n 'LinalgBroadcast\|TensorEmpty'`
//! over `ktir_to_superdsc/lower.rs` and `opmap.rs` is **zero hits in both**. So their own
//! producer's RMSNorm KTIR is not walkable by their own KTIR -> SuperDSC lowering today --
//! which their module doc says in its own words ("what this does NOT yet handle ... left
//! for the next slice").
//!
//! THAT SETTLES THE DIRECTION OF GAP 6 BELOW, without us designing anything: their walker
//! has to grow `LinalgBroadcast` + `TensorEmpty` arms regardless, because their own
//! producer already emits them. Once it does, option 2 (we emit the named ops) becomes
//! viable for scratchy -- at the cost of the deeptools path, which probe p07 says rejects
//! the named broadcast. The choice is still the owner's; what is no longer open is whether
//! a form exists.
//!
//! # GAP 6, AND IT BLOCKS THE HANDOVER: THE TWO CONSUMERS WANT OPPOSITE FORMS
//!
//! This is the one that stops `ToSchedulerKTIR`'s output being handed to their walker,
//! and it is not something more code on our side fixes. There are TWO downstream
//! consumers and they disagree about the same two ops:
//!
//! | op | deeptools' scheduler | scratchy's `ktir_to_superdsc` |
//! |----|----------------------|-------------------------------|
//! | a reduction | `linalg.generic` + reduction iterator | `linalg.reduce` + `AttrKey::ReduceFn` |
//! | a broadcast | `linalg.generic`, yield-only body | -- |
//! | named `linalg.broadcast` | **REJECTED** (probe p07) | (has a `LinalgBroadcast` variant) |
//! | `linalg.generic` | required | **NO DISPATCH ARM AT ALL** |
//!
//! MEASURED, not inferred: `grep -c LinalgGeneric` over their `lower.rs` and `opmap.rs`
//! is **0 and 0**. Their walker's `compute_or_refuse` falls through to
//! `err("KTIR operation ... has NO SuperDSC mapping")`, so a `linalg.generic` is refused
//! by name rather than mishandled -- which is the right behaviour, and it is also why
//! this cannot be papered over.
//!
//! So the form `ToSchedulerKTIR` produces -- the form the C++'s own `3_sched.mlir`
//! contains, and the form probe p07 established for the scheduler -- is exactly the form
//! their lowering has no arm for. Emitting `ktir_core` values from it today produces a
//! module their `lower` refuses on its first reduction.
//!
//! THREE WAYS OUT, and choosing is an owner decision because each one costs something
//! different:
//!
//! 1. **Their walker gains a `LinalgGeneric` arm** that recognises the two shapes we
//!    already emit (a reduction iterator plus a one-op body; an all-parallel map plus a
//!    yield-only body). Cheapest for us, and it keeps ONE KTIR serving both consumers.
//! 2. **We emit the named `linalg.reduce`/`linalg.broadcast`.** But probe p07 says the
//!    deeptools scheduler REJECTS the named broadcast, so this makes our KTIR
//!    scratchy-only and forks the two paths. That is a bigger decision than it looks.
//! 3. **Two emission modes, chosen by consumer.** Honest, and it doubles the surface
//!    that needs a golden.
//!
//! Nothing here is a defect in either implementation: both are internally consistent and
//! each documents its own rule. It is an interface that was never negotiated, and it is
//! better named than discovered.
//!
//! # A FIFTH THING, AND IT IS A QUESTION FOR THE OWNER RATHER THAN A GAP
//!
//! **Their vocabulary has no `ktdf` at all**, so the `ktdf.corelet_plan` and its two
//! `ktdf.corelet`s that `PlanCorelets` exists to produce are simply not read. Either
//! they derive the corelet split themselves -- their emitter IS the scheduler -- and
//! `PlanCorelets` is a stage that stops at the C++ boundary, or the plan is
//! load-bearing for them and their walker needs it. Our port reproduces `PlanCorelets`
//! faithfully either way, including the `vector_add` refusal, so nothing is lost by
//! settling this later; but it should be settled rather than assumed, because "we emit
//! it and they ignore it" is how a plan silently stops constraining anything.
//!
//! # THE FOUR GAPS, EACH WITH ITS CONSEQUENCE
//!
//! 1. **`ToSchedulerKTIR` IS A PREREQUISITE, NOT AN INDEPENDENT ITEM.** Our
//!    `make_ktir` output carries `tt.func`, `tt.return`, `tt.reduce`,
//!    `tt.expand_dims` and `tt.broadcast`. None exists in `ktir_core::OpKind`. The
//!    pass whose job the audit script describes as "tt.* out of the body; the grid
//!    loop folded" is exactly the one that removes them. So their consumer's input
//!    is the **post-`ToSchedulerKTIR` stage**, and until that pass is ported there
//!    is nothing to hand them. This reorders the remaining work: the three unported
//!    passes are not the larger half by volume, they are the gate.
//!
//! 2. **THE REDUCE COMBINER IS A TYPED ATTRIBUTE, NOT A REGION.**
//!    `lower.rs:872` reads `AttrKey::ReduceFn -> Attr::Op(OpKind)` and refuses a
//!    `linalg.reduce` without it; `Combiner::from_op_kind` accepts only max and sum.
//!    Our `tt.reduce` carries a region body with an `arith.maxnumf` /
//!    `arith.addf` inside. Converting is mechanical -- lift the combiner op out of
//!    the region into the attribute -- and is `ToSchedulerKTIR`'s job anyway, since
//!    that pass is what turns `tt.reduce` into the `linalg` form. Attention needs
//!    max AND sum, both of which they have.
//!
//! 3. **ONE `scf.for`, CONSTANT BOUNDS, ONE CARRIED VALUE** (`lower.rs:1000-1050`).
//!    Two of our loops fail this today and for different reasons:
//!    * non-causal attention's KV loop carries **FIVE** iter_args (`acc`, `l_i`,
//!      `m_i`, `offsetk_y`, `offsetv_y`), and their walker refuses more than one by
//!      name. `CarriedValuesToMemory` is the pass that removes loop-carried values
//!      entirely ("the loop carries nothing; the trees emitted"), so this gap closes
//!      with that port rather than needing anything on their side. Note their limit is
//!      not arbitrary either: it matches the one construction their own emitter builds.
//!    * causal attention's off-band loop has a **non-constant upper bound**, because
//!      the trip count is position-dependent. The C++ ALREADY refuses this at the
//!      same stage -- `attention_flash_causal/4_groups.err` says "its bounds are not
//!      all constant, positive-step integers, so the step count is not known here".
//!      So this is a genuine, pre-existing hole in BOTH implementations, not a port
//!      defect, and it is better named than discovered.
//!
//! 4. **A TRANSPOSING ACCESS TILE IS SILENTLY DROPPED, NOT REFUSED.** This is the
//!    one that should be fixed on THEIR side and it is the most dangerous of the
//!    four. Their `construct_access_tile` (`lower.rs:532`) takes the tile extents
//!    from `AttrKey::Shape` and reads NOTHING about traversal order. **CORRECTION to an
//!    earlier note in this file: their `AttrKey` DOES have `CoordinateOrder` and
//!    `Permutation`.** So the fix is their walker reading an attribute THAT ALREADY
//!    EXISTS, not a type extension -- which makes it a smaller change than first
//!    recorded, and removes the excuse for leaving it. Their module doc does say
//!    "does NOT yet handle: transposing access tiles", but a doc comment is not a
//!    refusal: the op still lowers, as an UNTRANSPOSED read of a tile whose declared
//!    shape is the transposed one. Our whole `tt.trans` fold exists to make
//!    attention's `q·Kᵀ` free, and it is expressed exactly as
//!    `access_tile_order = (d0, d1) -> (d1, d0)`. Handing that to their walker today
//!    produces a program that reads the wrong elements with no diagnostic anywhere,
//!    which is the cardinal sin in this tree. Their side needs either to read
//!    `CoordinateOrder`/`Permutation` or to refuse by name. NOT PATCHED HERE: their
//!    repo is read-only to us, so this is recorded for the owner to land or to ask us
//!    for.

use crate::ir::{Attr, AttrKey, Module, OpKind};

/// Every `OpKind` scratchy's `ktir_to_superdsc` can walk, by spelling.
///
/// TWO SOURCES, and reading only the first is a mistake I made and had to measure my
/// way out of:
///
/// * `lower.rs`'s `OpKind::` match arms -- the structural ops (`scf.for`, `ktdp.*`,
///   the contraction, the reduce, `tensor.splat`, scalar `arith` integer folding);
/// * `opmap.rs`'s TABLE -- every elementwise op, which never appears as an
///   `OpKind::` arm in `lower.rs` at all because `compute_or_refuse` dispatches
///   through `opmap::lookup`. Taking only the match arms reported `arith.mulf`,
///   `math.exp2` and 24 others as unwalkable when they are the best-supported ops
///   they have.
pub const WALKED_OPS: &[&str] = &[
    "arith.addi",
    "arith.ceildivsi",
    "arith.ceildivui",
    "arith.cmpf",
    "arith.cmpi",
    "arith.constant",
    "arith.divsi",
    "arith.divui",
    "arith.floordivsi",
    "arith.muli",
    "arith.remsi",
    "arith.remui",
    "arith.subi",
    "func.return",
    "ktdp.construct_access_tile",
    "ktdp.construct_memory_view",
    "ktdp.get_compute_tile_id",
    "ktdp.load",
    "ktdp.store",
    "linalg.batch_matmul",
    "linalg.matmul",
    "linalg.reduce",
    "linalg.yield",
    "scf.for",
    "scf.yield",
    "tensor.splat",
    // --- from opmap.rs's table (dispatched via `opmap::lookup`, never a match arm)
    "arith.addf",
    "arith.divf",
    "arith.extf",
    "arith.maxnumf",
    "arith.minnumf",
    "arith.mulf",
    "arith.negf",
    "arith.select",
    "arith.subf",
    "arith.truncf",
    "math.abs",
    "math.exp",
    // `math.exp2` maps to `Mapping::Exp2Decompose` (opmap.rs:179): scale by ln2 then
    // `exp`, because the vocabulary has no base-2 exponential.
    "math.exp2",
    "math.floor",
    "math.log2",
    "math.rsqrt",
    "math.sqrt",
    "math.tanh",
];

/// The seven `AttrKey`s their walker reads. Anything else we attach is ignored --
/// which is why gap 4 above is silent rather than loud.
pub const READ_ATTRS: &[&str] =
    &["iter_args", "iter_var", "predicate", "reduce_fn", "sizes", "strides", "value"];

/// Ops scratchy's own PRODUCER emits that their own WALKER has no arm for.
///
/// Measured, both directions: `lower_subtile_tape_to_superdsc.rs` builds these, and
/// `ktir_to_superdsc/{lower,opmap}.rs` matches none of them. Kept as data because it is
/// the evidence that the walker is an incomplete slice against their own producer, not
/// that our KTIR is unusual -- and because the RUNTIME BROADCAST every softmax needs is
/// in this list.
pub const THEIR_PRODUCER_EMITS_BUT_WALKER_LACKS: &[&str] =
    &["linalg.broadcast", "tensor.empty", "tensor.extract", "linalg.transpose"];

/// One reason a module cannot be handed to their lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Incompatibility {
    /// The op spelling, or the construct.
    pub what: String,
    /// Which of the four gaps this is.
    pub gap: &'static str,
    pub detail: String,
}

impl std::fmt::Display for Incompatibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.gap, self.what, self.detail)
    }
}

/// Check a module against the consumer contract.
///
/// An empty result means their lowering could walk it. This is deliberately a
/// REPORT rather than a `bool`: the point is to say which of the four gaps is in the
/// way for a given fixture, so the remaining work is ordered by evidence.
///
/// NOTE what this does NOT check: that their walker then SUCCEEDS. It checks
/// representability and the loop/reduce shape contracts, which are the parts
/// knowable from our side. Their own refusals (an op with no `opmap` entry, a tile
/// running past its view) are theirs to report.
pub fn check(module: &Module) -> Vec<Incompatibility> {
    let mut out = Vec::new();

    for op in module.ops_deep() {
        let spelling = op.kind.spelling().to_string();

        // Gap 1: representability. A `tt.*` op has no variant in ktir_core::OpKind.
        if !WALKED_OPS.contains(&spelling.as_str()) {
            let gap = if spelling.starts_with("tt.") {
                "gap1-tt-not-in-vocabulary"
            } else {
                "gap1-not-walked"
            };
            out.push(Incompatibility {
                what: spelling.clone(),
                gap,
                detail: if gap == "gap1-tt-not-in-vocabulary" {
                    "no `tt.*` variant exists in ktir_core::OpKind, so this op cannot be \
                     built at all; ToSchedulerKTIR is what removes it"
                        .to_string()
                } else {
                    "their lower.rs dispatch has no arm for this op".to_string()
                },
            });
        }

        // Gap 6: `linalg.generic` is REPRESENTABLE in their enum and has NO dispatch arm
        // in their walker, so it is refused by name. This is the form ToSchedulerKTIR
        // emits for both a reduction and a broadcast.
        if op.kind == OpKind::LinalgGeneric {
            let is_reduction = matches!(op.attr(&AttrKey::IteratorTypes), Some(Attr::StrList(s))
                if s.iter().any(|x| x == "reduction"));
            out.push(Incompatibility {
                what: spelling.clone(),
                gap: "gap6-linalg-generic-has-no-walker-arm",
                detail: format!(
                    "a {} linalg.generic. `grep -c LinalgGeneric` over their lower.rs and \
                     opmap.rs is 0 and 0, so compute_or_refuse falls through to \"has NO \
                     SuperDSC mapping\". Their walker wants linalg.reduce + \
                     AttrKey::ReduceFn; probe p07 says the deeptools scheduler wants the \
                     generic and REJECTS the named broadcast. An owner decision, not a \
                     port fix",
                    if is_reduction { "reduction" } else { "broadcast/elementwise" }
                ),
            });
        }

        // Gap 2: a reduce must carry its combiner as an attribute.
        if matches!(op.kind, OpKind::TtReduce | OpKind::LinalgReduce)
            && op.attr(&AttrKey::Other("reduce_fn".into())).is_none()
        {
            let combiner = op
                .regions
                .first()
                .and_then(|r| r.ops.first())
                .map(|c| c.kind.spelling().to_string())
                .unwrap_or_else(|| "<none>".into());
            out.push(Incompatibility {
                what: spelling.clone(),
                gap: "gap2-reduce-combiner-in-a-region",
                detail: format!(
                    "the combiner is a region body (`{combiner}`); their lower.rs:872 \
                     requires AttrKey::ReduceFn -> Attr::Op(..) and refuses without it"
                ),
            });
        }

        // Gap 3: the scf.for contract.
        if op.kind == OpKind::ScfFor {
            // THE GRID LOOP IS ITS OWN CASE, and conflating it with the K-loop
            // over-reports. It carries nothing, so their `attr_ssas(IterArgs)` finds no
            // attribute and refuses -- but `ToSchedulerKTIR` FOLDS this loop away
            // ("the grid loop folded"), so it is gap 1's prerequisite showing up again
            // rather than a fourth independent problem.
            let is_grid = crate::passes::distribute_work::is_per_core_work_loop(module, op);
            let iter_args = op.operands.len().saturating_sub(3);
            if is_grid {
                out.push(Incompatibility {
                    what: spelling.clone(),
                    gap: "gap1-grid-loop-not-folded",
                    detail: "the per-core grid loop carries nothing, so their \
                             `attr_ssas(AttrKey::IterArgs)` finds no attribute and \
                             refuses. ToSchedulerKTIR folds this loop away; it is not a \
                             loop their walker should ever see"
                        .to_string(),
                });
            } else if iter_args != 1 {
                out.push(Incompatibility {
                    what: spelling.clone(),
                    gap: "gap3-loop-carried-count",
                    detail: format!(
                        "carries {iter_args} loop-carried value(s); their lower.rs \
                         supports exactly one. CarriedValuesToMemory removes them"
                    ),
                });
            }
            for (i, which) in ["lower bound", "upper bound", "step"].iter().enumerate() {
                let v = op.operands[i];
                // Their `KtdpGetComputeTileId` binds to `Aff::con(opts.grid_point)`
                // (lower.rs:471), so a bound that traces to the landmark IS constant on
                // their path. Counting it as non-constant was my own error, and it
                // pointed at the wrong pass.
                let resolves = crate::passes::dot_to_linalg::const_int(module, v).is_some()
                    || module
                        .def_of(v)
                        .map(|d| d.kind == OpKind::KtdpGetComputeTileId)
                        .unwrap_or(false);
                if !resolves {
                    out.push(Incompatibility {
                        what: spelling.clone(),
                        gap: "gap3-non-constant-bound",
                        detail: format!(
                            "its {which} is not a compile-time constant; their lower.rs \
                             refuses, and the C++ CarriedValuesToMemory refuses the same \
                             loop for the same reason"
                        ),
                    });
                }
            }
        }

        // Gap 4: a transposing access tile is dropped in silence.
        if op.kind == OpKind::KtdpConstructAccessTile {
            let order = op.attr(&AttrKey::AccessTileOrder).and_then(|a| match a {
                crate::ir::Attr::AffineMap(s) => Some(s.clone()),
                _ => None,
            });
            let rank = op.result_type().map(|t| t.rank()).unwrap_or(0);
            let identity = {
                let d: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
                format!("({}) -> ({})", d.join(", "), d.join(", "))
            };
            if let Some(o) = order {
                if o != identity {
                    out.push(Incompatibility {
                        what: spelling.clone(),
                        gap: "gap4-transpose-dropped-silently",
                        detail: format!(
                            "access_tile_order is `{o}`, a TRANSPOSING read. Their \
                             construct_access_tile (lower.rs:532) takes the extents from \
                             AttrKey::Shape and reads no order attribute -- and their \
                             AttrKey has none -- so this lowers as an UNTRANSPOSED read \
                             with no diagnostic. Their side needs the attribute or a \
                             refusal naming it"
                        ),
                    });
                }
            }
        }
    }

    // A region with block arguments has to be re-expressed: their regions are bare
    // op lists and carry the IV / carried values as attributes instead.
    for op in module.ops_deep() {
        for (i, r) in op.regions.iter().enumerate() {
            if !r.args.is_empty() && op.kind != OpKind::TtFunc {
                out.push(Incompatibility {
                    what: format!("{}/region{i}", op.kind.spelling()),
                    gap: "gap1-regions-have-no-block-arguments",
                    detail: format!(
                        "{} block argument(s); ktir_core::ir.rs:173 makes a region a bare \
                         op list, so these must become AttrKey::IterVar / IterArgs",
                        r.args.len()
                    ),
                });
            }
        }
    }

    out
}

/// The gaps a module hits, deduplicated and counted -- the one-line form.
pub fn summary(module: &Module) -> Vec<(String, usize)> {
    let mut counts: std::collections::HashMap<String, usize> = Default::default();
    for i in check(module) {
        *counts.entry(i.gap.to_string()).or_insert(0) += 1;
    }
    let mut v: Vec<(String, usize)> = counts.into_iter().collect();
    v.sort();
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse;

    #[test]
    fn a_tt_op_is_reported_as_unrepresentable_not_merely_unsupported() {
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
        let m = parse::parse(src).unwrap();
        let gaps = summary(&m);
        let names: Vec<&String> = gaps.iter().map(|(g, _)| g).collect();
        assert!(
            names.iter().any(|g| g.as_str() == "gap1-tt-not-in-vocabulary"),
            "a tt.* op must be reported as unrepresentable: {gaps:?}"
        );
        assert!(
            names.iter().any(|g| g.as_str() == "gap2-reduce-combiner-in-a-region"),
            "and its region-body combiner named separately: {gaps:?}"
        );
    }

    #[test]
    fn a_transposing_access_tile_is_reported_because_their_side_drops_it_in_silence() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %v = ktdp.construct_memory_view %q, sizes: [256, 128], strides: [128, 1] : memref<256x128xf16>
    %t = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = affine_map<(d0, d1) -> (d1, d0)>} : memref<256x128xf16> -> !ktdp.access_tile<128x64xindex>
    %l = ktdp.load %t : <128x64xindex> -> tensor<128x64xf16>
    tt.return
  }
}
";
        let m = parse::parse(src).unwrap();
        let found = check(&m);
        assert!(
            found.iter().any(|i| i.gap == "gap4-transpose-dropped-silently"),
            "the K^T read must be flagged: {found:?}"
        );
    }

    #[test]
    fn an_identity_access_tile_order_is_not_a_gap() {
        // THE CONTROL for gap 4: only a TRANSPOSING order is a problem, so the check
        // must not fire on every access tile.
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %v = ktdp.construct_memory_view %q, sizes: [256, 128], strides: [128, 1] : memref<256x128xf16>
    %t = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>} : memref<256x128xf16> -> !ktdp.access_tile<64x128xindex>
    %l = ktdp.load %t : <64x128xindex> -> tensor<64x128xf16>
    tt.return
  }
}
";
        let m = parse::parse(src).unwrap();
        let found = check(&m);
        assert!(
            !found.iter().any(|i| i.gap == "gap4-transpose-dropped-silently"),
            "an identity order is fine: {found:?}"
        );
    }

    #[test]
    fn the_walked_vocabulary_has_no_tt_ops_which_is_the_whole_point() {
        assert!(
            !WALKED_OPS.iter().any(|s| s.starts_with("tt.")),
            "if a tt.* op appears here, ktir_core's OpKind grew one and gap 1 has changed"
        );
        // And the attribute set is the seven their walker reads.
        assert_eq!(READ_ATTRS.len(), 7);
        assert!(!READ_ATTRS.contains(&"access_tile_order"), "gap 4's premise");
        // Gap 4's premise is that their WALKER reads no order attribute -- not that
        // their type lacks one. `CoordinateOrder` and `Permutation` both exist in their
        // AttrKey, which is why the fix is small.
        assert!(!READ_ATTRS.contains(&"coordinate_order"));
        assert!(!READ_ATTRS.contains(&"permutation"));
    }
}
