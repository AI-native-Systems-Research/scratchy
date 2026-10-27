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
//! sampler's `b*64 + l` answered 40032 for 40017 -- see the pass tests). A COMPARISON
//! such an island READS is part of the island (its f32 operands decide which lane the
//! one-hot lands on), so the sweep covers those too -- see [`fptosi_island`].
//!
//! WHY THE ISLAND GATE, and not a blanket module-wide retype: a blanket retype (a)
//! makes verifier-invalid IR wherever an f32 result is tied to an un-retyped
//! companion, and (b) SILENTLY MIS-NUMERICS a genuine-f32 kernel -- `ktdp.load`-ing
//! a `memref<f32>` as `tensor<f16>` verifies and reinterprets the f32 bytes as f16.
//!
//! RED-STOP, and it is real rather than comment-only: after the walk, any op result
//! still typed compute-domain f32 is GENUINE f32 that this pass did not, and must
//! not silently, legalize. Silent precision degradation is forbidden. (A PROTECTED
//! result -- one that reaches an `arith.fptosi` -- is exempt: it is legal f32 by
//! construction, and a value that leaks OFF the island into an unprotected consumer
//! is refused here BY NAME rather than degraded.)
//!
//! ⭐ THE FP32-BINDING GATE (auditor ruling 2026-10-26): the pass exists to refuse
//! SILENT f32→f16 narrowing. An op the CARD BINDS AT FP32 lowers AS f32 -- nothing
//! narrows, so there is nothing to refuse: the RED-stop consults [`FP32_CARD_BINDS`]
//! and lets a bound op's f32 result stand. An op with NO fp32 bind (`math.log` --
//! the vendor templates bind it to fp16 alone) keeps today's refusal by name:
//! emitting it would mean silently degrading it to f16, or failing at dxp. The
//! bindings are TARGET-ABI FACTS -- a const table the pass takes as input, each row
//! cited to the vendor DDL template and line it was read from -- never match arms,
//! never an inference: an opcode absent from the table is UNBOUND at fp32, and the
//! refusal stands (fail-closed).

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
///
/// ⭐⭐ A COMPARISON THE ISLAND READS IS PART OF THE ISLAND. The equality-selection
/// idiom (`(x == m).to(f32)`, then a product chain, then the `fptosi`) reads its i1
/// off an `arith.cmpf` whose OPERANDS are the values being selected among. The
/// comparison itself is never retyped (its result is i1, not compute f32), but its
/// f32 operands' PRECISION decides which lane the one-hot lands on — the island's own
/// exactness. Collapsing them to f16 re-rounds near-ties into ties and silently moves
/// the selected lane, the same mis-numeric the island exists to prevent, one step
/// earlier. So the walk also sweeps, to a fixed point with the value walk: every
/// comparison whose i1 result an op with a protected result READS has its f32
/// operands protected. An `f16` comparison is untouched by this (the walk keeps only
/// f32 operands), so the flash kernel's masked selects collapse exactly as before.
fn fptosi_island(module: &Module) -> HashSet<Ssa> {
    let index = module.def_index();
    // ⭐ A REGION-INTERIOR RESULT IS PART OF ITS ENCLOSING OP'S RESULT: a `tt.reduce`'s
    // combiner `maxnumf` COMPUTES the reduce's result, an `scf.for`'s body computes its
    // iter_args. The def-index hands back HEADS (no regions), so this map states the
    // containment once, up front: each result of a region-carrying op → every interior
    // result beneath it. Protecting the reduce therefore protects its combiner — the
    // RED-stop would otherwise refuse the combiner's f32 as "genuine" the moment the
    // reduce itself is island-protected and stays f32 (the collapsed case retypes the
    // region args in step 2, so a collapsed reduce never reaches that refusal).
    let interior: HashMap<Ssa, Vec<Ssa>> = {
        fn collect(ops: &[Op], into: &mut Vec<Ssa>) {
            for o in ops {
                into.extend(o.results.iter().copied());
                for r in &o.regions {
                    collect(&r.ops, into);
                }
            }
        }
        let mut m: HashMap<Ssa, Vec<Ssa>> = HashMap::new();
        for op in module.ops_deep() {
            if op.regions.is_empty() {
                continue;
            }
            let mut inner: Vec<Ssa> = Vec::new();
            for r in &op.regions {
                collect(&r.ops, &mut inner);
            }
            for res in &op.results {
                m.insert(*res, inner.clone());
            }
        }
        m
    };
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
    loop {
        // The VALUE walk: protect `v`, then keep walking `v`'s definition's f32
        // operands — the slice stays inside f32 by construction.
        while let Some(v) = work.pop() {
            if !protected.insert(v) {
                continue;
            }
            // The interior results ride along — PUSHED, not inserted, so the pop's own
            // insert-and-walk runs for each (their operands are region block args,
            // which have no definition and dead-end harmlessly, or outer values the
            // sweep should take anyway).
            if let Some(inner) = interior.get(&v) {
                work.extend(inner.iter().copied());
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
        // The COMPARISON walk: an op with a protected result that READS an i1 takes
        // that i1's comparison's f32 operands into the island (and back into the
        // value walk above, so the sweep covers everything they in turn read).
        // `arith.cmpf` rides `Other` for the same reason `fptosi` does.
        let mut grew = false;
        for op in module.ops_deep() {
            if !op.results.iter().any(|r| protected.contains(r)) {
                continue;
            }
            for u in &op.operands {
                if index.type_of(*u).and_then(|t| t.elem()) != Some(DType::I1) {
                    continue;
                }
                let Some(cmp) = index.def_of(*u) else {
                    continue;
                };
                if !matches!(&cmp.kind, OpKind::Other(s) if s == "arith.cmpf") {
                    continue;
                }
                for c in &cmp.operands {
                    if index.type_of(*c).and_then(|t| t.elem()) == Some(DType::F32)
                        && !protected.contains(c)
                    {
                        // ⭐ PUSH, NEVER INSERT: the value walk above marks a value
                        // walked by INSERTING it as it pops — its `!insert →
                        // continue` guard is the walk-once dedup. Inserting HERE
                        // would pre-mark the value and the pop would skip the def
                        // walk entirely: the sweep would protect the comparison's
                        // operands while never walking anything they read (measured
                        // exactly that way — the pass stabilized one round early and
                        // the score chain stayed unprotected).
                        work.push(*c);
                        grew = true;
                    }
                }
            }
        }
        if !grew {
            break;
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
///
/// ⭐ THE FP32-BINDING GATE (auditor ruling 2026-10-26): the pass exists to refuse
/// SILENT f32→f16 narrowing. An op the CARD BINDS AT FP32 lowers AS f32 -- nothing
/// narrows, so there is nothing to refuse: the RED-stop consults [`FP32_CARD_BINDS`]
/// and lets a bound op's f32 result stand. An op with NO fp32 bind (`math.log` --
/// the vendor templates bind it to fp16 alone) keeps today's refusal by name:
/// emitting it would mean silently degrading it to f16, or failing at dxp. The
/// bindings are TARGET-ABI FACTS -- a const table the pass takes as input, each row
/// cited to the vendor DDL template and line it was read from -- never match arms,
/// never an inference: an opcode absent from the table is UNBOUND at fp32, and the
/// refusal stands (fail-closed).
fn red_stop_on_genuine_f32(module: &Module, protected: &HashSet<Ssa>) -> Result<()> {
    // The operand-element-type half of the binding key: `type_of` covers every
    // def and block arg, so the gate can ask what an f32 result was fed.
    let index = module.def_index();
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
                // ⭐ THE FP32-BINDING GATE: an op the card binds at fp32 lowers
                // AS f32 — nothing narrows, so there is nothing to refuse. The
                // binding is the const table's own measured fact, cited to the
                // vendor DDL template and line, and keyed on the FULL
                // (opcode, operand element type, result element type) triple —
                // `arith.sitofp` from an i1 is NOT the DDL's int32tofp32
                // (IEEE_INT32 input), so a same-spelling different-source op is
                // UNBOUND and the refusal below stands (fail-closed).
                let in_elems: Vec<DType> = op
                    .operands
                    .iter()
                    .filter_map(|u| index.type_of(*u).and_then(|ty| ty.elem()))
                    .collect();
                if FP32_CARD_BINDS.iter().any(|(opcode, din, dout, _)| {
                    *opcode == op.kind.spelling()
                        && in_elems.contains(din)
                        && Some(*dout) == t.elem()
                }) {
                    continue;
                }
                return Err(Refusal::new(
                    PASS,
                    format!(
                        "genuine f32 compute remains after collapsing the \
                         stability-widening island -- this op's f32 result is not a \
                         transient extf/truncf artifact, and the card binds it to no \
                         fp32 op (see FP32_CARD_BINDS; math.log is fp16-only in the \
                         vendor templates). Refusing to silently degrade it to f16 \
                         (SP-E2-03 RED-stop: f32 of an fp32-unbound op is an owner \
                         decision, not a pass code-around). The op is '{}'.",
                        op.kind.spelling()
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// ⭐ THE CARD'S OWN FP32 BINDINGS — the vendor DDLs' op-to-dtype binds, stated
/// ONCE as a const table the RED-stop takes as input (the auditor ruling named
/// above): an op bound at fp32 lowers AS f32 — no narrowing happens, nothing to
/// refuse — while an op with NO row here keeps the refusal, because emitting it
/// would silently degrade it to f16 or fail at dxp.
///
/// ⛔ EVERY ROW IS A MEASURED FACT, read directly from the vendor templates on
/// the card pod (2026-10-26): the row's citation names the template and line.
/// Nothing here is inferred from a composition or from another op's bind, and an
/// opcode absent from the table is treated as UNBOUND at fp32 — fail-closed, the
/// refusal stands. The opcodes are the KTIR spellings `from_ttir`'s census
/// admits; only opcodes that can carry an f32 RESULT at RED-stop time need rows
/// (an unprotected `arith.extf` is already gone by step 1, an `arith.truncf`
/// result is f16, an `arith.fptosi` result is i32 — none of them consult this
/// table).
///
/// ⛔ THE KEY IS THE FULL (opcode, OPERAND element type, RESULT element type)
/// TRIPLE, never the spelling alone: the DDL binds `int32tofp32` from an
/// IEEE_INT32 input, so an `arith.sitofp` reading an i1 is a DIFFERENT op with
/// the SAME spelling — unbound, refused. A row matches only when the op's result
/// element type is the row's and one of its operands carries the row's input
/// element type.
const FP32_CARD_BINDS: &[(&str, DType, DType, &str)] = &[
    // broadcast_ops.ddl — the two-tensor pointwise family, instantiated fp32.
    (
        "arith.addf",
        DType::F32,
        DType::F32,
        "broadcast_ops.ddl:25 add [%type_fp16, %type_fp32]",
    ),
    (
        "arith.subf",
        DType::F32,
        DType::F32,
        "broadcast_ops.ddl:26 sub [%type_fp16, %type_fp32]",
    ),
    (
        "arith.mulf",
        DType::F32,
        DType::F32,
        "broadcast_ops.ddl:27 mul [%type_fp16, %type_fp32]",
    ),
    (
        "arith.divf",
        DType::F32,
        DType::F32,
        "broadcast_ops.ddl:44 realdiv [%type_fp16, %type_fp32]",
    ),
    (
        "arith.maxnumf",
        DType::F32,
        DType::F32,
        "broadcast_ops.ddl:45 maximum [%type_fp16, %type_fp32]",
    ),
    (
        "arith.minnumf",
        DType::F32,
        DType::F32,
        "broadcast_ops.ddl:46 minimum [%type_fp16, %type_fp32]",
    ),
    // unary_parallel.ddl — the one-input family and the integer convert.
    (
        "math.absf",
        DType::F32,
        DType::F32,
        "unary_parallel.ddl:36 abs [%type_fp16, %type_fp32]",
    ),
    (
        "arith.negf",
        DType::F32,
        DType::F32,
        "unary_parallel.ddl:37 neg [%type_fp16, %type_fp32]",
    ),
    (
        "arith.sitofp",
        DType::I32,
        DType::F32,
        "unary_parallel.ddl:49 int32tofp32 [%type_fp32]",
    ),
    (
        "math.exp",
        DType::F32,
        DType::F32,
        "unary_parallel.ddl:39 exp [%type_fp32]",
    ),
    (
        "math.tanh",
        DType::F32,
        DType::F32,
        "unary_parallel.ddl:35 tanhfp32 [%type_fp32]",
    ),
    (
        "math.rsqrt",
        DType::F32,
        DType::F32,
        "unary_parallel.ddl:43 rsqrt [%type_fp16, %type_fp32]",
    ),
    // `tt.broadcast` MOVES DATA, so it is dtype-free: its result is its input's
    // dtype verbatim (no DDL op, no opfunc — the emitter copies the lanes), and
    // an f32 result is an f32 input arriving intact. It is the one non-DDL row:
    // the general table's MOVES-DATA class (auditor ruling 2026-10-26).
    (
        "tt.broadcast",
        DType::F32,
        DType::F32,
        "moves data — dtype-free, no DDL op",
    ),
    // NOTE, deliberately NO row for math.log: unary_pipeline.ddl:22 binds it to
    // [%type_fp16] ALONE — the one bind this gate exists to keep refusing.
];

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
        // No extf anywhere, so there is no island, and math.log has NO fp32 card
        // binding (unary_pipeline.ddl:22 binds it to fp16 alone): this f32 log is
        // GENUINE and UNBOUND, and must be refused rather than silently degraded.
        // (An fp32-BOUND op in the same position stays -- the fold test below.)
        let src = "\
module {
  tt.func public @k(%q: tensor<64xf32>) attributes {noinline = false} {
    %l = math.log %q : tensor<64xf32>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        let e = run(&mut m).unwrap_err();
        assert!(e.message.contains("genuine f32 compute remains"), "got {e}");
        assert!(e.message.contains("owner decision"), "got {e}");
        assert!(
            e.message.contains("math.log"),
            "the refusal names the op: {e}"
        );
    }

    /// The auditor's fp32-binding gate (ruling 2026-10-26), reduced to the
    /// sampler's own fold shape: an integer convert headed f32 chain (sitofp,
    /// absf, mulf, subf, minnumf, maxnumf, addf) terminating at an EXPLICIT
    /// author-written truncf. Every op in the chain is fp32-bound on the card
    /// (broadcast_ops.ddl:25-27/45-46, unary_parallel.ddl:36/49), so nothing
    /// narrows and the RED-stop lets every f32 result stand.
    #[test]
    fn an_f32_op_the_card_binds_at_fp32_stays_f32() {
        let src = "\
module {
  tt.func public @k(%r: tensor<64xi32>, %fm8: tensor<64xf32>, %ftop: tensor<64xf32>, %fzero: tensor<64xf32>, %fhalf: tensor<64xf32>, %fm22: tensor<64xf32>) attributes {noinline = false} {
    %rf = arith.sitofp %r : tensor<64xi32> to tensor<64xf32>
    %a = math.absf %rf : tensor<64xf32>
    %n = arith.mulf %a, %fm8 : tensor<64xf32>
    %t = arith.subf %ftop, %n : tensor<64xf32>
    %nm = arith.minnumf %n, %t : tensor<64xf32>
    %np = arith.maxnumf %nm, %fzero : tensor<64xf32>
    %s = arith.addf %np, %fhalf : tensor<64xf32>
    %u = arith.mulf %s, %fm22 : tensor<64xf32>
    %u16 = arith.truncf %u : tensor<64xf32> to tensor<64xf16>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).expect("every op in the fold is fp32-bound on the card");
        let c = m.census();
        let get = |n: &str| c.iter().find(|(k, _)| k == n).map(|(_, v)| *v).unwrap_or(0);
        // The whole fold stands at f32 -- no op was retyped, collapsed, or refused.
        assert_eq!(get("arith.sitofp"), 1, "the integer convert stays f32");
        assert_eq!(get("math.absf"), 1);
        assert_eq!(get("arith.mulf"), 2);
        assert_eq!(get("arith.subf"), 1);
        assert_eq!(get("arith.minnumf"), 1);
        assert_eq!(get("arith.maxnumf"), 1);
        assert_eq!(get("arith.addf"), 1);
        // The author's explicit narrowing SURVIVES: it is f32 -> f16, not the
        // f16 -> f16 no-op step 3 removes.
        assert_eq!(get("arith.truncf"), 1, "the explicit truncf survives");
        for op in m.ops_deep() {
            for t in &op.result_types {
                if op.kind.spelling() == "arith.truncf" {
                    assert_eq!(t.elem(), Some(DType::F16), "the truncf narrows to f16");
                } else if !matches!(op.kind, OpKind::ArithConstant) {
                    assert_eq!(
                        t.elem(),
                        Some(DType::F32),
                        "{} kept its f32 result",
                        op.kind.spelling()
                    );
                }
            }
        }
    }

    /// `tt.broadcast` MOVES DATA (dtype-free): an f32 broadcast result is the f32
    /// input arriving intact — the sampler's per-row operand alignment shape — so
    /// the RED-stop lets it stand rather than refusing it as genuine compute.
    #[test]
    fn an_f32_tt_broadcast_moves_data_and_stands() {
        let src = "\
module {
  tt.func public @k(%q: tensor<1x1xf32>) attributes {noinline = false} {
    %b = tt.broadcast %q : tensor<1x1xf32> -> tensor<1x64xf32>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).expect("tt.broadcast moves data — nothing narrows, nothing to refuse");
        let b = m
            .ops_deep()
            .into_iter()
            .find(|o| o.kind.spelling() == "tt.broadcast")
            .unwrap();
        assert_eq!(
            b.result_type().and_then(|t| t.elem()),
            Some(DType::F32),
            "the broadcast keeps its f32 result"
        );
    }

    /// The MOVES-DATA row exempts the BROADCAST, never its consumer: a broadcast
    /// whose f32 result feeds an fp32-UNBOUND op is refused AT that op — the gate
    /// stays fail-closed one hop past the exempted shape plumbing (the wall-7
    /// form: the fold's [1,1] constants broadcast into bound mulf/subf/addf/
    /// maxnumf; this is the negative image, the same broadcast into math.log).
    #[test]
    fn a_tt_broadcast_into_an_unbound_op_still_red_stops_naming_the_consumer() {
        let src = "\
module {
  tt.func public @k(%q: tensor<1x1xf32>) attributes {noinline = false} {
    %b = tt.broadcast %q : tensor<1x1xf32> -> tensor<1x64xf32>
    %l = math.log %b : tensor<1x64xf32>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        let err = run(&mut m).expect_err("math.log is fp16-only; the broadcast may not launder it");
        assert!(
            format!("{err}").contains("The op is 'math.log'."),
            "the refusal names the consumer, not the broadcast: {err}"
        );
    }

    /// The triple key's negative pin: `int32tofp32`'s bind names an i32 OPERAND
    /// domain, so a `sitofp` off an i1 finds no row and RED-stops — keying on
    /// the spelling alone would silently excuse it. The module states the ONLY
    /// shape that reaches the question: the sitofp's f32 result is LOAD-BEARING
    /// (it feeds an fp32-BOUND multiply, which cannot narrow, so the convert
    /// cannot collapse away), and no comparison precedes it (an equal-mask
    /// chain absorbs its retype into the Equal lowering, and a dead-end convert
    /// is churn the collapse owns — both legal, neither this pin's subject).
    #[test]
    fn a_sitofp_off_an_i1_is_not_int32tofp32_and_red_stops() {
        let src = "\
module {
  tt.func public @k(%b: tensor<1x64xi1>, %s: tensor<1x64xf32>) attributes {noinline = false} {
    %f = arith.sitofp %b : tensor<1x64xi1> to tensor<1x64xf32>
    %m = arith.mulf %f, %s : tensor<1x64xf32>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        let err = run(&mut m).expect_err("sitofp off i1 has no int32tofp32 bind");
        assert!(
            format!("{err}").contains("The op is 'arith.sitofp'."),
            "the refusal names the op: {err}"
        );
    }

    /// The triple key's positive control: the SAME opcode off an i32 operand IS
    /// the DDL's int32tofp32 and stays f32.
    #[test]
    fn a_sitofp_off_an_i32_is_int32tofp32_and_stays() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %i = arith.constant dense<5> : tensor<1x64xi32>
    %f = arith.sitofp %i : tensor<1x64xi32> to tensor<1x64xf32>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).expect("sitofp off i32 is int32tofp32, the bind's own domain");
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
    /// The leaking consumer is math.log, an op with NO fp32 card binding
    /// (unary_pipeline.ddl:22): under the auditor's fp32-binding gate a leak onto
    /// a BOUND op is legal (it lowers as f32 -- the fold test above), so the leak
    /// this test guards is the one onto an op the card cannot run at f32.
    #[test]
    fn a_leak_off_the_fptosi_island_red_stops_rather_than_degrading() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %b = arith.constant dense<6.25e+02> : tensor<1x64xf16>
    %bf = arith.extf %b : tensor<1x64xf16> to tensor<1x64xf32>
    %i = arith.fptosi %bf : tensor<1x64xf32> to tensor<1x64xi32>
    %leak = math.log %bf : tensor<1x64xf32>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        let e = run(&mut m).unwrap_err();
        assert!(e.message.contains("genuine f32 compute remains"), "got {e}");
        assert!(
            e.message.contains("math.log"),
            "the refusal names the leaking op: {e}"
        );
    }

    /// ⭐ A COMPARISON THE ISLAND READS IS PART OF THE ISLAND. The sampler's
    /// equality-selection, reduced to its shape: the score chain (an extf head, an
    /// f32 add against a staged f32 table) is COMPARED against that same table, the
    /// i1 mask is widened back to f32, and the mask-times-table product rides to
    /// the `fptosi`. The score is reached ONLY through the comparison -- no f32
    /// value path connects it to the fptosi -- so without the comparison sweep its
    /// extf head is removed as churn and the collapse degrades the very values the
    /// comparison selects among: near-ties re-round into ties and the one-hot moves
    /// a lane, silently. With the sweep the comparison's f32 operands are protected,
    /// the extf head keeps its widen type, and the selection stays f32.
    #[test]
    fn a_comparison_the_island_reads_protects_its_f32_operands() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %row = arith.constant dense<1.000000e+00> : tensor<1x64xf16>
    %thr = arith.constant dense<2.000000e+00> : tensor<1x64xf32>
    %x = arith.extf %row : tensor<1x64xf16> to tensor<1x64xf32>
    %s = arith.addf %x, %thr : tensor<1x64xf32>
    %eq = arith.cmpf oeq, %s, %thr : tensor<1x64xi1>
    %mask = arith.sitofp %eq : tensor<1x64xi1> to tensor<1x64xf32>
    %w = arith.mulf %mask, %thr : tensor<1x64xf32>
    %i = arith.fptosi %w : tensor<1x64xf32> to tensor<1x64xi32>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).expect("the comparison is island-read; its operands stay f32");
        // The score chain's extf head SURVIVES with its f32 widen type: without the
        // sweep it is removed as churn (its result is on no f32 value path to the
        // fptosi) and the score collapses to f16.
        let extf_f32 = m
            .ops_deep()
            .into_iter()
            .filter(|o| {
                o.kind == OpKind::ArithExtf
                    && o.result_type().and_then(|t| t.elem()) == Some(DType::F32)
            })
            .count();
        assert_eq!(extf_f32, 1, "the score head keeps its f32 widen type");
        // And the compared score itself stays f32 -- the add and the mask product
        // both keep the domain the kernel stated.
        let f32_kept = m
            .ops_deep()
            .into_iter()
            .filter(|o| {
                matches!(o.kind, OpKind::ArithAddf | OpKind::ArithMulf)
                    && o.result_type().and_then(|t| t.elem()) == Some(DType::F32)
            })
            .count();
        assert_eq!(f32_kept, 2, "the score add and the mask product stay f32");
        let mask_f32 = m
            .ops_deep()
            .into_iter()
            .filter(|o| {
                o.kind.spelling() == "arith.sitofp"
                    && o.result_type().and_then(|t| t.elem()) == Some(DType::F32)
            })
            .count();
        assert_eq!(mask_f32, 1, "the i1->f32 mask widen stays f32");
    }

    /// The NEGATIVE space of the comparison sweep: an f32 comparison whose result
    /// feeds NOTHING protected is still GENUINE f32 -- the owner decision stands.
    /// The sweep follows protected results only; it must not become a blanket
    /// "all f32 comparisons are legal" amnesty.
    #[test]
    fn an_f32_comparison_off_the_island_still_red_stops() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %a = arith.constant dense<1.000000e+00> : tensor<1x64xf32>
    %b = arith.constant dense<2.000000e+00> : tensor<1x64xf32>
    %eq = arith.cmpf oeq, %a, %b : tensor<1x64xi1>
    %mask = arith.sitofp %eq : tensor<1x64xi1> to tensor<1x64xf32>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        let e = run(&mut m).unwrap_err();
        assert!(e.message.contains("genuine f32 compute remains"), "got {e}");
    }
}
