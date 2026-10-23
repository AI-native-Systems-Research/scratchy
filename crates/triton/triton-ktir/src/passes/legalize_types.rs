// SPDX-License-Identifier: Apache-2.0

//! Ported from `LegalizeTypes.cpp` (SP-E2-03). Collapse Triton's f32
//! stability-widening ISLAND -- and only that island.
//!
//! Triton auto-widens f16 reductions for numeric stability: `arith.extf` on the
//! inputs, the combiner in f32, `arith.truncf` back. Spyre's compute units run f16
//! natively, so the widening is churn the emitter would have to model. Three steps:
//!
//! 1. Remove `arith.extf` (f16 -> f32): the cast's result is replaced by the f16
//!    input it widened. Every op downstream now consumes an f16 operand while its
//!    own result is still typed f32.
//! 2. Collapse the island: retype an op's f32 result to f16 ONLY IF at least one
//!    operand has ALREADY become f16. Retyping propagates forward to a fixed point
//!    until the island terminates at the `truncf`.
//! 3. Remove `arith.truncf` (now f16 -> f16, a no-op).
//!
//! ⭐ THE ONE ISLAND THIS PASS MUST NOT TOUCH: an f32 chain that EXITS through
//! `arith.fptosi` instead of a `truncf`. That chain is not widening churn -- widening
//! before an integer cast exists for exactly one reason, integer exactness past f16's
//! range, and the device has the recipe (the KSPLIT pair's `dl16tofp32` → same-stick
//! f32 → `fp32toint32`). [`fptosi_island`] computes that chain's backward slice and
//! every step consults it; degrading it was a measured green-run mis-numeric (the
//! sampler's `b*64 + l` answered 40032 for 40017 -- see the pass tests).
//!
//! WHY THE ISLAND GATE, and not a blanket module-wide retype: a blanket retype (a)
//! makes verifier-invalid IR wherever an f32 result is tied to an un-retyped
//! companion, and (b) SILENTLY MIS-NUMERICS a genuine-f32 kernel -- `ktdp.load`-ing
//! a `memref<f32>` as `tensor<f16>` verifies and reinterprets the f32 bytes as f16.
//!
//! RED-STOP, and it is real rather than comment-only: after the walk, any op result
//! still typed compute-domain f32 is GENUINE f32 that this pass did not, and must
//! not silently, legalize. Silent precision degradation is forbidden; legalizing
//! genuine f32 is an owner decision. (A PROTECTED result -- one that reaches an
//! `arith.fptosi` -- is exempt: it is legal f32 by construction, and a value that
//! leaks OFF the island into an unprotected consumer is refused here BY NAME rather
//! than degraded.)

use crate::ir::*;
use crate::passes::walk::{self, OpPath};
use crate::{Refusal, Result};
use std::collections::{HashMap, HashSet};

const PASS: &str = "LegalizeTypes";

pub fn run(module: &mut Module) -> Result<()> {
    // THE FPTOSI ISLAND FIRST, while every extf still stands: the protected set is a
    // backward slice over VALUES, and step 1 below erases the casts whose results it
    // must be able to see through.
    let protected = fptosi_island(module);
    step_1_remove_extf(module, &protected);
    step_2_collapse_island(module, &protected);
    step_2b_island_constants(module, &protected);
    step_3_remove_truncf(module);
    red_stop_on_genuine_f32(module, &protected)
}

/// ⭐ THE f32 VALUES THAT REACH AN `arith.fptosi` — the exactness island, PROTECTED.
///
/// Triton's widening churn ROUND-TRIPS: `extf ... truncf`, back to the f16 it widened.
/// An f32 chain that instead EXITS through `arith.fptosi` is not churn and never was:
/// widening before an integer cast has exactly one purpose — integer exactness past
/// f16's range (the sampler's `b*64 + l` at vocab width, KSPLIT's fp32 merge), and the
/// device has the recipe (the KSPLIT pair's own `dl16tofp32` → same-stick f32 →
/// `fp32toint32`). Degrading that chain to f16 is the SILENT MIS-NUMERICS this pass's
/// own header forbids: `40017 = 625*64 + 17` is not f16-representable (ULP 32 up
/// there), and a collapsed combine answers 40032 on a green run — measured, pinned by
/// the argmax numeric leg before this protection existed.
///
/// The slice stays INSIDE f32: it seeds from every fptosi's f32 operand and walks
/// definitions keeping only f32-typed operands, so the f16 input a protected extf
/// widened FROM is not swept in (it is legal for the collapse to touch everything
/// around the island). A protected value consumed by an op the collapse still wants to
/// retype makes that op SKIP the retype (step 2), which leaves a genuine-f32 result the
/// RED-stop then refuses BY NAME — the leak case is fail-closed through the existing
/// gate, not a new one.
fn fptosi_island(module: &Module) -> HashSet<Ssa> {
    let index = module.def_index();
    let mut protected: HashSet<Ssa> = HashSet::new();
    let mut work: Vec<Ssa> = module
        .ops_deep()
        .into_iter()
        // `arith.fptosi` rides `Other` (it has no KTIR-specific semantics below the
        // emitter), the same spelling `to_ktir_emit` dispatches on.
        .filter(|o| matches!(&o.kind, OpKind::Other(s) if s == "arith.fptosi"))
        .flat_map(|o| o.operands.iter().copied())
        .filter(|v| index.type_of(*v).and_then(|t| t.elem()) == Some(DType::F32))
        .collect();
    while let Some(v) = work.pop() {
        if !protected.insert(v) {
            continue;
        }
        let Some(def) = index.def_of(v) else {
            continue;
        };
        for u in &def.operands {
            if index.type_of(*u).and_then(|t| t.elem()) == Some(DType::F32) {
                work.push(*u);
            }
        }
    }
    protected
}

/// Is `op` on the descriptor path / in the KTDP dialect, and so out of scope?
/// The dialect check catches every `ktdp.*`; the explicit `tt.*` descriptor ops
/// cover the pre-lowering form.
fn is_descriptor_or_ktdp_op(op: &Op) -> bool {
    if op.kind.spelling().starts_with("ktdp.") {
        return true;
    }
    matches!(
        op.kind,
        OpKind::TtDescriptorLoad
            | OpKind::TtDescriptorStore
            | OpKind::TtMakeTensorDescriptor
            | OpKind::TtDescriptorGather
            | OpKind::TtDescriptorScatter
    )
}

/// --- Step 1: remove `arith.extf` (f16 -> f32).
fn step_1_remove_extf(module: &mut Module, protected: &HashSet<Ssa>) {
    // ONE `find` with the WHOLE predicate -- collected for EVERY extf, not one per
    // loop trip: the `while find` shape re-walked the whole module (and re-ran the
    // whole-module `type_of`) per cast, which is quadratic at kernel scale. The
    // victims are erased DEEPEST-AND-LAST FIRST (`walk::erase`'s own order), so
    // every surviving path stays valid and each rewrite sees the same module the
    // per-trip scan did: `replace_all_uses` only touches operands, never paths.
    let index = module.def_index();
    let victims: Vec<OpPath> = walk::paths(module)
        .into_iter()
        .filter(|p| {
            let op = walk::at(module, p).expect("path");
            if op.kind != OpKind::ArithExtf {
                return false;
            }
            // A protected extf is the HEAD of an exactness island (its result reaches
            // an `arith.fptosi`); removing it would splice f16 into the island.
            if protected.contains(&op.results[0]) {
                return false;
            }
            let in_elem = op
                .operands
                .first()
                .and_then(|v| index.type_of(*v))
                .and_then(|t| t.elem());
            let out_elem = op.result_type().and_then(|t| t.elem());
            in_elem == Some(DType::F16) && out_elem == Some(DType::F32)
        })
        .collect();
    // ONE `replace_all_uses_many` + ONE `erase` for every victim -- the
    // per-victim loop paid a whole-module walk per cast, quadratic at kernel
    // scale. `from` (an f32 result) can never be a `to` (an f16 operand), so
    // the pairs are disjoint by type; the flatten loop is kept anyway because
    // `replace_all_uses_many` gives each operand ONE lookup and a chain left
    // unflattened would strand a use on a value the erase deletes.
    let mut map: HashMap<Ssa, Ssa> = HashMap::with_capacity(victims.len());
    for path in &victims {
        let op = walk::at(module, path).expect("path");
        map.insert(op.results[0], op.operands[0]);
    }
    walk::replace_all_uses_many(module, &walk::flatten_rewire_map(&map));
    walk::erase(module, &victims);
}

/// --- Step 2: the gated f32 -> f16 retype, to a forward fixed point.
fn step_2_collapse_island(module: &mut Module, protected: &HashSet<Ssa>) {
    loop {
        let mut changed = false;
        // Collect the retypes first, then apply, so the walk is not mutated
        // underneath itself. The operand-type reads go through a ONE-SHOT
        // definition index: `type_of` is a whole-module scan, and this loop calls
        // it per operand over per op -- quadratic at kernel scale (measured: the
        // scan was ~100% of this pass's time on the unrolled attention kernel).
        // Types only change in the APPLY phase below, so a snapshot taken at the
        // top of each round sees exactly what the old per-call scan saw.
        let index = module.def_index();
        let mut retype: Vec<OpPath> = Vec::new();
        for path in walk::paths(module) {
            let op = walk::at(module, &path).expect("path");
            if is_descriptor_or_ktdp_op(op) {
                continue; // out of scope: descriptor path / KTDP transfer boundary
            }
            // Skip ops whose f32 result is address/transfer-boundary bound.
            if op.result_types.iter().any(|t| t.is_out_of_scope_f32()) {
                continue;
            }
            // A PROTECTED OPERAND pins this op out of the collapse: if the op is in
            // the island its own result is protected too (the slice is transitive);
            // if it is NOT, then it consumes exactness-f32 and exits elsewhere --
            // skipping the retype leaves its f32 for the RED-stop to refuse by name
            // rather than silently degrading. Either way, no retype here.
            //
            // ⛔ AND A PROTECTED **RESULT** PINS ITS OWN OP OUT TOO -- the ISLAND HEAD.
            // The closure walks backward over f32 values, so every island member but
            // the head is reached through a protected operand; the head (an `arith.extf`
            // f16→f32 whose widen the island exists to hold) has an f16 OPERAND and a
            // protected RESULT. The operand test alone does not see it, so without this
            // second test the collapse retypes the head's result to f16 while leaving
            // the op standing: a "widen" typed f16→f16, an op whose stated meaning
            // (dl16tofp32, the device recipe the island is) is erased from its own type
            // while its body survives. The numbers survive (the consumer promotes), but
            // the DOOR reads that type to pick the convert -- an f16-typed extf cannot
            // be lowered as the f32 widen it is. The head keeps its widen type.
            if op.operands.iter().any(|v| protected.contains(v))
                || op.results.iter().any(|r| protected.contains(r))
            {
                continue;
            }
            let operand_is_f16 = op
                .operands
                .iter()
                .any(|v| index.type_of(*v).and_then(|t| t.elem()) == Some(DType::F16));
            let has_compute_f32 = op.result_types.iter().any(|t| t.is_compute_f32());
            if !operand_is_f16 || !has_compute_f32 {
                continue;
            }
            retype.push(path);
        }
        for path in retype {
            let op = walk::at_mut(module, &path).expect("path");
            // Fix an arith.constant's value attribute first, so the attr type stays
            // in sync with the result type set below.
            if op.kind == OpKind::ArithConstant
                && let Some(a) = op.attr(&AttrKey::Value).cloned()
                && let Some(f) = a.as_float()
                && f.width == 32
            {
                let re = f.to_f16();
                op.set_attr(
                    AttrKey::Value,
                    match a {
                        Attr::SplatFloat(_) => Attr::SplatFloat(re),
                        _ => Attr::Float(re),
                    },
                );
            }
            for t in op.result_types.iter_mut() {
                if t.is_compute_f32() {
                    *t = t.with_elem(DType::F16);
                    changed = true;
                }
            }
            // `tt.reduce` combiner block-arg types: fix so the combiner runs on f16.
            // Only reached when the reduce op is itself in the island.
            if op.kind == OpKind::TtReduce {
                for r in op.regions.iter_mut() {
                    for (_, t) in r.args.iter_mut() {
                        if t.is_compute_f32() {
                            *t = t.with_elem(DType::F16);
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
}

/// ISLAND CONSTANTS. A splat f32 `arith.constant` added in the widened domain
/// (a bias, or the flash kernel's `qk_scale`) is part of the island even though it
/// has NO f16 operand -- it is identified by being consumed by an island op that
/// has already been retyped. Genuine-f32 constants not feeding the island are left
/// alone.
fn step_2b_island_constants(module: &mut Module, protected: &HashSet<Ssa>) {
    let mut victims: Vec<OpPath> = Vec::new();
    let index = module.def_index();
    for path in walk::paths(module) {
        let op = walk::at(module, &path).expect("path");
        if op.kind != OpKind::ArithConstant {
            continue;
        }
        if !op.result_types.iter().any(|t| t.is_compute_f32()) {
            continue;
        }
        let Some(res) = op.result() else { continue };
        // A PROTECTED constant is INSIDE the exactness island (its value reaches an
        // `arith.fptosi`); re-rounding it to f16 would degrade the very exactness
        // the island exists to hold. It is not island churn, so it is not a victim.
        if protected.contains(&res) {
            continue;
        }
        // ONE snapshot for the whole scan: the per-constant `ops_deep().any()`
        // plus per-operand `type_of` were whole-module scans -- quadratic at
        // kernel scale. Read-only phase, so the snapshot cannot go stale.
        let feeds_island = module.ops_deep().into_iter().any(|user| {
            user.operands.contains(&res)
                && !is_descriptor_or_ktdp_op(user)
                && user
                    .operands
                    .iter()
                    .any(|v| index.type_of(*v).and_then(|t| t.elem()) == Some(DType::F16))
        });
        if feeds_island {
            victims.push(path);
        }
    }
    for path in victims {
        let op = walk::at_mut(module, &path).expect("path");
        if let Some(a) = op.attr(&AttrKey::Value).cloned()
            && let Some(f) = a.as_float()
            && f.width == 32
        {
            let re = f.to_f16();
            op.set_attr(
                AttrKey::Value,
                match a {
                    Attr::SplatFloat(_) => Attr::SplatFloat(re),
                    _ => Attr::Float(re),
                },
            );
        }
        for t in op.result_types.iter_mut() {
            if t.is_compute_f32() {
                *t = t.with_elem(DType::F16);
            }
        }
    }
    // A retyped constant can widen the island further, so re-run the fixed point.
    step_2_collapse_island(module, protected);
}

/// --- Step 3: remove the now-redundant `arith.truncf` (f16 -> f16).
fn step_3_remove_truncf(module: &mut Module) {
    // Collected-then-erased, the same shape as step 1: a `while find` here
    // re-walked the module per cast.
    let index = module.def_index();
    let victims: Vec<OpPath> = walk::paths(module)
        .into_iter()
        .filter(|p| {
            let op = walk::at(module, p).expect("path");
            if op.kind != OpKind::ArithTruncf {
                return false;
            }
            let in_elem = op
                .operands
                .first()
                .and_then(|v| index.type_of(*v))
                .and_then(|t| t.elem());
            let out_elem = op.result_type().and_then(|t| t.elem());
            in_elem == Some(DType::F16) && out_elem == Some(DType::F16)
        })
        .collect();
    // The same ONE-walk shape as step 1. Chains are REAL here: truncf f16->f16
    // feeding another truncf's operand is exactly the shape a folded island can
    // leave, so the flatten loop is load-bearing, not defensive.
    let mut map: HashMap<Ssa, Ssa> = HashMap::with_capacity(victims.len());
    for path in &victims {
        let op = walk::at(module, path).expect("path");
        map.insert(op.results[0], op.operands[0]);
    }
    walk::replace_all_uses_many(module, &walk::flatten_rewire_map(&map));
    walk::erase(module, &victims);
}

/// --- RED-stop: refuse to leave genuine f32 in the compute domain.
fn red_stop_on_genuine_f32(module: &Module, protected: &HashSet<Ssa>) -> Result<()> {
    for op in module.ops_deep() {
        if is_descriptor_or_ktdp_op(op) {
            continue;
        }
        for (i, t) in op.result_types.iter().enumerate() {
            if t.is_out_of_scope_f32() {
                continue; // legitimately out-of-scope ptr/tensordesc/memref f32
            }
            // A PROTECTED result is the exactness island terminating at an
            // `arith.fptosi` -- legal f32 by construction (see [`fptosi_island`]),
            // the KSPLIT-precedented device recipe, not a degradation candidate.
            if op.results.get(i).is_some_and(|res| protected.contains(res)) {
                continue;
            }
            if t.is_compute_f32() {
                return Err(Refusal::new(
                    PASS,
                    format!(
                        "genuine f32 compute remains after collapsing the \
                         stability-widening island -- this op's f32 result is not a \
                         transient extf/truncf artifact. Refusing to silently degrade it \
                         to f16 (SP-E2-03 RED-stop: f32->f16 of genuine compute is an \
                         owner decision, not a pass code-around). The op is '{}'.",
                        op.kind.spelling()
                    ),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse;

    /// The flash body's widening island, reduced to its shape: extf, an f32
    /// reduce, an f32 multiply by a splat, truncf.
    #[test]
    fn the_widening_island_collapses_and_the_splat_is_re_rounded_to_f16() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %scale = arith.constant dense<0.127517432> : tensor<64xf32>
    %qk = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
    %input = arith.extf %qk : tensor<64x64xf16> to tensor<64x64xf32>
    %m = \"tt.reduce\"(%input) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %c = arith.maxnumf %a, %b : f32
      tt.reduce.return %c : f32
    }) : (tensor<64x64xf32>) -> tensor<64xf32>
    %ms = arith.mulf %m, %scale : tensor<64xf32>
    %mt = arith.truncf %ms : tensor<64xf32> to tensor<64xf16>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).expect("the island must collapse with no residual f32");
        let c = m.census();
        let get = |n: &str| c.iter().find(|(k, _)| k == n).map(|(_, v)| *v).unwrap_or(0);
        assert_eq!(get("arith.extf"), 0, "step 1 removes every widening cast");
        assert_eq!(
            get("arith.truncf"),
            0,
            "step 3 removes every narrowing cast"
        );

        // Every remaining compute type is f16, and the reduce's combiner too.
        for op in m.ops_deep() {
            for t in &op.result_types {
                assert!(
                    !t.is_compute_f32(),
                    "{} kept an f32 result",
                    op.kind.spelling()
                );
            }
            for r in &op.regions {
                for (_, t) in &r.args {
                    assert!(!t.is_compute_f32(), "a combiner block arg stayed f32");
                }
            }
        }
        // The scale constant re-rounded to the f16 the golden prints.
        let scale = m
            .ops_deep()
            .into_iter()
            .find(|o| {
                o.kind == OpKind::ArithConstant
                    && o.result_type()
                        .map(|t| t.dims() == Some(&[64][..]))
                        .unwrap_or(false)
                    && o.attr(&AttrKey::Value)
                        .and_then(|a| a.as_float())
                        .map(|f| !f.is_zero())
                        == Some(true)
            })
            .expect("the scale constant survives");
        let f = scale.attr(&AttrKey::Value).unwrap().as_float().unwrap();
        assert_eq!((f.width, f.bits), (16, 0x3015), "1.275630e-01 as f16");
    }

    #[test]
    fn genuine_f32_compute_red_stops_by_name() {
        // No extf anywhere, so there is no island: this f32 add is GENUINE and must
        // be refused rather than silently degraded.
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %a = arith.constant dense<1.000000e+00> : tensor<64xf32>
    %b = arith.constant dense<2.000000e+00> : tensor<64xf32>
    %c = arith.addf %a, %b : tensor<64xf32>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        let e = run(&mut m).unwrap_err();
        assert!(e.message.contains("genuine f32 compute remains"), "got {e}");
        assert!(e.message.contains("owner decision"), "got {e}");
    }

    #[test]
    fn a_memref_of_f32_is_never_retyped() {
        // The silent-misnumerics bug the island gate exists to prevent: retyping a
        // memref<f32> reinterprets its bytes as f16 and still verifies.
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f32>) attributes {noinline = false} {
    %h = arith.constant dense<1.000000e+00> : tensor<64xf16>
    %e = arith.extf %h : tensor<64xf16> to tensor<64xf32>
    %v = ktdp.construct_memory_view %q, sizes: [64], strides: [1] : memref<64xf32>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).expect("the memref is out of scope, not a residual");
        let view = m
            .ops_deep()
            .into_iter()
            .find(|o| o.kind == OpKind::KtdpConstructMemoryView)
            .unwrap();
        assert_eq!(
            view.result_type().and_then(|t| t.elem()),
            Some(DType::F32),
            "the memref keeps f32 -- retyping it would reinterpret the bytes"
        );
    }

    /// The sampler's exactness island, reduced to its shape: two extfs, an f32
    /// multiply by an f32 constant, an f32 add, an `arith.fptosi` out to i32. This
    /// is the chain whose collapse answered 40032 for 40017 on a green numeric run
    /// (`40017 = 625*64 + 17` is not f16-representable; ULP is 32 up there) -- the
    /// measured defect `fptosi_island` exists to close.
    #[test]
    fn the_fptosi_island_survives_while_the_widening_churn_still_collapses() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %b = arith.constant dense<6.25e+02> : tensor<1x64xf16>
    %c64 = arith.constant dense<6.400000e+01> : tensor<1x64xf32>
    %l = arith.constant dense<1.700000e+01> : tensor<1x64xf16>
    %bf = arith.extf %b : tensor<1x64xf16> to tensor<1x64xf32>
    %lf = arith.extf %l : tensor<1x64xf16> to tensor<1x64xf32>
    %hi = arith.mulf %bf, %c64 : tensor<1x64xf32>
    %idx = arith.addf %hi, %lf : tensor<1x64xf32>
    %i = arith.fptosi %idx : tensor<1x64xf32> to tensor<1x64xi32>
    %qk = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
    %scale = arith.constant dense<1.275630e-01> : tensor<64xf32>
    %input = arith.extf %qk : tensor<64x64xf16> to tensor<64x64xf32>
    %m = \"tt.reduce\"(%input) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %bb: f32):
      %c = arith.maxnumf %a, %bb : f32
      tt.reduce.return %c : f32
    }) : (tensor<64x64xf32>) -> tensor<64xf32>
    %ms = arith.mulf %m, %scale : tensor<64xf32>
    %mt = arith.truncf %ms : tensor<64xf32> to tensor<64xf16>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).expect("the island is legal; the churn still collapses");
        let c = m.census();
        let get = |n: &str| c.iter().find(|(k, _)| k == n).map(|(_, v)| *v).unwrap_or(0);
        // The TWO island extfs stand; the churn's one is gone; the churn's truncf
        // is gone. The counts are exact, not floors: two islands, one churn.
        assert_eq!(get("arith.extf"), 2, "the fptosi island's extfs survive");
        assert_eq!(get("arith.truncf"), 0, "the churn's truncf still collapses");
        // ⛔ AND THE ISLAND HEAD KEEPS ITS WIDEN TYPE: both surviving extfs are typed
        // f16→f32, not f16→f16. The collapse's operand test cannot see the head (its
        // operand is f16, only its RESULT is protected), so without the result test the
        // head's type is silently rewritten to f16 while the op stands -- a "widen" that
        // no longer widens. The door reads that type to pick `dl16tofp32`; this pins it.
        let extf_f32 = m
            .ops_deep()
            .into_iter()
            .filter(|o| {
                o.kind == OpKind::ArithExtf
                    && o.result_type().and_then(|t| t.elem()) == Some(DType::F32)
            })
            .count();
        assert_eq!(
            extf_f32, 2,
            "the island's extf heads keep their f32 widen type"
        );
        // The island's compute stays F32 -- mulf, addf, and the f32 constant -- and
        // the fptosi's operand is still the f32 add.
        let f32_count = m
            .ops_deep()
            .into_iter()
            .filter(|o| {
                matches!(o.kind, OpKind::ArithMulf | OpKind::ArithAddf)
                    && o.result_type().and_then(|t| t.elem()) == Some(DType::F32)
            })
            .count();
        assert_eq!(f32_count, 2, "the island's mulf and addf keep f32");
        // The island's f32 constant (`dense<6.4e1>`, [1,64]) is not re-rounded: it
        // is distinguished from the f16 [1,64] constants by its element type, and
        // finding exactly ONE such survivor is the assertion (a re-rounded %c64
        // leaves zero).
        let island_consts = m
            .ops_deep()
            .into_iter()
            .filter(|o| {
                o.kind == OpKind::ArithConstant
                    && o.result_type().and_then(|t| t.elem()) == Some(DType::F32)
                    && o.result_type().and_then(|t| t.dims().map(|d| d.to_vec()))
                        == Some(vec![1, 64])
            })
            .count();
        assert_eq!(
            island_consts, 1,
            "the island's f32 constant survives unre-rounded"
        );
        // And the churn half collapsed exactly as before: its reduce (`tt.reduce`
        // at this stage -- `LinalgReduce` is the post-`to_ktir` spelling) and its
        // multiply are f16 now.
        let churn_f16 = m
            .ops_deep()
            .into_iter()
            .filter(|o| {
                matches!(o.kind, OpKind::TtReduce | OpKind::ArithMulf)
                    && o.result_type().and_then(|t| t.elem()) == Some(DType::F16)
            })
            .count();
        assert_eq!(
            churn_f16, 2,
            "the churn's reduce and multiply collapsed to f16"
        );
    }

    /// THE LEAK: a protected f32 value consumed by an op whose result does NOT
    /// reach the fptosi. Retyping that consumer would silently degrade a value the
    /// kernel stated as exactness-f32; skipping the retype leaves it genuine-f32
    /// for the RED-stop to refuse BY NAME -- fail-closed through the existing gate.
    #[test]
    fn a_leak_off_the_fptosi_island_red_stops_rather_than_degrading() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %b = arith.constant dense<6.25e+02> : tensor<1x64xf16>
    %h = arith.constant dense<1.000000e+00> : tensor<1x64xf16>
    %bf = arith.extf %b : tensor<1x64xf16> to tensor<1x64xf32>
    %i = arith.fptosi %bf : tensor<1x64xf32> to tensor<1x64xi32>
    %leak = arith.addf %bf, %h : tensor<1x64xf32>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        let e = run(&mut m).unwrap_err();
        assert!(e.message.contains("genuine f32 compute remains"), "got {e}");
        assert!(
            e.message.contains("arith.addf"),
            "the refusal names the leaking op: {e}"
        );
    }
}
