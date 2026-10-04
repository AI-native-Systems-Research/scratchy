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

//! `make_ttir`: THE PASS PIPELINE BETWEEN THE TWO BRIDGES, in Rust, on values.
//!
//! # WHY THIS MODULE HAD TO EXIST BEFORE THE BRIDGES COULD BE WIRED
//!
//! `triton.compile` reaches the form bridge two consumes in TWO steps, not one:
//!
//! ```text
//!   ASTSource.make_ir(...)     code_generator.py walks the AST -> RAW ttir
//!   backend.make_ttir(...)     inliner, canonicalizer, ttir-combine,
//!                              reorder-broadcast, cse, symbol-dce
//! ```
//!
//! [`crate::codegen`] ports the FIRST step, and its oracle is `<name>.ttir_raw.mlir`.
//! `third_party/spyre/backend/compiler.py:115` is the second, and its output
//! (`<name>.ttir.mlir`) is what [`triton_ktir::make_ktir`] was ported against. The gap
//! between them is not cosmetic: **the raw form has EIGHT to TEN `tt.func`s** (every
//! `@triton.jit` helper and every generated `triton.language.standard.*`), and bridge
//! two's `Module::kernel()` refuses anything but exactly one. So "wire bridge one to
//! bridge two as values" is not a plumbing job with a converter in the middle -- the
//! inliner is load-bearing, and without it the value path cannot reach KTIR at all.
//!
//! # WHAT IS PORTED, EACH BECAUSE A GOLDEN DEMANDS IT
//!
//! Not MLIR's canonicalizer. The EFFECTS the pinned pipeline depends on, measured one
//! at a time against `tests/goldens/<name>.ttir.mlir`:
//!
//! 1. [`drop_unreachable_blocks`] -- `handle_returns` leaves a trailing block holding
//!    one `ub.poison` per result plus a second `tt.return`, and every generated helper
//!    has one. MLIR's inliner cannot splice a multi-block callee into the middle of a
//!    block without splitting it; the canonicalizer deletes the block instead, because
//!    it has no predecessors.
//! 2. [`inline_calls`] -- `tt.call` disappears and the callee's body is spliced in with
//!    its arguments substituted. Run to a fixed point: `tl.sum` calls
//!    `_sum_combine`, which is a second level.
//! 3. [`cse`] then [`dce`], to a fixed point -- the duplicate constants each descriptor
//!    materializes, and the whole `extsi`/`cmpi`/`andi` 32-bit-overflow check
//!    `semantic.mul` emits and nothing reads.
//! 4. [`hoist_constants`] -- constants to the top of the function's entry block.
//! 5. [`symbol_dce`] -- the private helpers, once nothing calls them.
//!
//! `ttir-combine` and `reorder-broadcast` are NOT ported, and that is a measurement
//! rather than an omission: `tests/make_ttir.rs` diffs all thirteen configurations
//! field by field and neither pass changes any of them. If a fixture ever needs one,
//! that test is what will say so.
//!
//! # WHAT THIS DELIBERATELY DOES NOT REPRODUCE, AND WHY IT CANNOT MATTER
//!
//! **LOCATIONS.** MLIR's inliner rewrites every inlined op's location to
//! `callsite(<callee loc> at <caller loc>)`, and its constant folder materializes the
//! hoisted constants afresh with `loc(unknown)` -- measured, in
//! `tests/goldens/rmsnorm.ttir.mlir`, where the seven hoisted constants carry
//! `#loc3 = loc(unknown)` although the raw ones each carried a `NameLoc`.
//! [`crate::ttir::Loc`] has no `CallSite` variant, so the first is not representable
//! here at all.
//!
//! THE REASON THIS IS SOUND, stated so it is checkable rather than asserted: bridge
//! two's IR (`triton_ktir::ir`) HAS NO LOCATION FIELD. Not a dropped one -- there is no
//! place to put one. So a location difference at this stage cannot reach KTIR, the
//! bundle, or the program. `tests/make_ttir.rs` therefore gates on structure and
//! reports location findings SEPARATELY, with their count, so "we ignored the locs"
//! stays a number on the record instead of a silence.

use std::collections::HashMap;

use crate::ttir::{Attr, Block, Func, Module, Op, Region, Type, ValueId, Visibility};
use crate::{Error, Result};

/// Run the `make_ttir` pipeline, in `make_ttir`'s order, on a ttir value.
///
/// FAIL CLOSED: a construct these passes cannot account for is an [`Error`] naming it,
/// never a partially optimized module. The one that can actually happen is a
/// `tt.call` to a symbol that is not in the module.
pub fn make_ttir(m: &mut Module) -> Result<()> {
    drop_unreachable_blocks(m)?;
    inline_calls(m)?;
    // A fold enables a CSE, a CSE enables a DCE, and a DCE can expose another fold.
    for _ in 0..64 {
        let mut changed = false;
        changed |= fold_identity_casts(m);
        changed |= fold_constant_arith(m);
        changed |= fold_int_identities(m);
        changed |= fold_splat_of_constant(m);
        changed |= sort_commutative_operands(m);
        changed |= promote_single_trip_loops(m);
        changed |= cse(m);
        changed |= dce(m);
        if !changed {
            break;
        }
    }
    hoist_constants(m);
    // HOISTING EXPOSES DUPLICATES, and it is not a hypothetical: `attention_flash_causal`
    // has two `scf.for` bodies that each fold `qk_scale` into its own
    // `dense<1.44269502> : tensor<64x64xf32>`. Inside the loops those are in different
    // scopes and CSE must not merge them; once both are in the entry block they are one
    // constant, and the golden has one. (`triton_ktir`'s canonicalize records the same
    // ordering requirement for the same reason.)
    while cse(m) | dce(m) {}
    symbol_dce(m);
    Ok(())
}

//===----------------------------------------------------------------------===//
// 3b. Integer identities
//===----------------------------------------------------------------------===//

/// `addi x, 0` / `subi x, 0` / `muli x, 1` / `divsi x, 1` -> `x`, and integer arithmetic
/// on two constants -> a constant.
///
/// MEASURED: `attention_flash`'s `offsetk_y = offset_kv_y + 0` is an `arith.addi` against
/// the zero constant in the raw and is simply absent from the golden -- the loop's
/// `iter_args` take `%offset_kv_y_11` directly. One op, but it sits immediately before the
/// `scf.for`, so without this fold the op sequence diverges at the loop and every finding
/// after it is suppressed.
pub fn fold_int_identities(m: &mut Module) -> bool {
    let ints = int_constants(m);
    let tys: Vec<Type> = m.values.iter().map(|v| v.ty.clone()).collect();
    let mut subst: HashMap<ValueId, ValueId> = HashMap::new();
    let mut changed = false;
    for f in &mut m.funcs {
        for b in &mut f.body.blocks {
            changed |= fold_int_ops(&mut b.ops, &ints, &tys, &mut subst);
        }
    }
    if !subst.is_empty() {
        substitute(m, &subst);
        changed = true;
    }
    changed
}

fn fold_int_ops(
    ops: &mut [Op],
    ints: &HashMap<ValueId, i128>,
    tys: &[Type],
    subst: &mut HashMap<ValueId, ValueId>,
) -> bool {
    let mut changed = false;
    for op in ops.iter_mut() {
        for r in &mut op.regions {
            for b in &mut r.blocks {
                changed |= fold_int_ops(&mut b.ops, ints, tys, subst);
            }
        }
        if op.results.len() != 1 || op.operands.len() != 2 {
            continue;
        }
        let a = op.operands[0];
        let b = op.operands[1];
        let (ka, kb) = (ints.get(&a).copied(), ints.get(&b).copied());

        // Both constant: compute. `divsi`/`remsi` by zero is left alone -- MLIR does not
        // fold it either, because the result is undefined rather than known.
        if let (Some(x), Some(y)) = (ka, kb) {
            let v = match op.name.as_str() {
                "arith.addi" => Some(x + y),
                "arith.subi" => Some(x - y),
                "arith.muli" => Some(x * y),
                "arith.divsi" if y != 0 => Some(x / y),
                "arith.remsi" if y != 0 => Some(x % y),
                _ => None,
            };
            if let Some(v) = v {
                let ty = tys[op.results[0].0 as usize].clone();
                if ty.is_int() && !ty.is_tensor() {
                    op.name = "arith.constant".to_string();
                    op.operands.clear();
                    op.attrs.clear();
                    op.attrs.insert("value".to_string(), Attr::Int(v, ty));
                    changed = true;
                    continue;
                }
            }
        }

        // One side an identity element.
        let keep = match op.name.as_str() {
            "arith.addi" => {
                if kb == Some(0) {
                    Some(a)
                } else if ka == Some(0) {
                    Some(b)
                } else {
                    None
                }
            }
            "arith.subi" if kb == Some(0) => Some(a),
            "arith.muli" => {
                if kb == Some(1) {
                    Some(a)
                } else if ka == Some(1) {
                    Some(b)
                } else {
                    None
                }
            }
            "arith.divsi" if kb == Some(1) => Some(a),
            _ => None,
        };
        if let Some(k) = keep {
            subst.insert(op.results[0], k);
        }
    }
    changed
}

//===----------------------------------------------------------------------===//
// 3a. The three folds the goldens demand
//===----------------------------------------------------------------------===//

/// `arith.bitcast %x : T to T` -> `%x`.
///
/// MEASURED: `codegen`'s `scf.for` bounds go through a bitcast that is an IDENTITY when
/// the bound is already i32, which every fixture's is. `arith::BitcastOp::fold` removes
/// it; leaving it in put three extra ops in each SwiGLU configuration (six in the tiled
/// one, which has two loops) and made the op sequence diverge at the loop.
pub fn fold_identity_casts(m: &mut Module) -> bool {
    let tys: Vec<Type> = m.values.iter().map(|v| v.ty.clone()).collect();
    let mut subst: HashMap<ValueId, ValueId> = HashMap::new();
    for f in &m.funcs {
        for b in &f.body.blocks {
            collect_identity_casts(&b.ops, &tys, &mut subst);
        }
    }
    if subst.is_empty() {
        return false;
    }
    substitute(m, &subst);
    // The casts are now unused; `dce` in the same fixed point removes them.
    true
}

fn collect_identity_casts(
    ops: &[Op],
    tys: &[Type],
    subst: &mut HashMap<ValueId, ValueId>,
) {
    for op in ops {
        let identity = matches!(
            op.name.as_str(),
            "arith.bitcast" | "arith.extsi" | "arith.extui" | "arith.trunci"
        ) && op.operands.len() == 1
            && op.results.len() == 1
            && tys[op.operands[0].0 as usize] == tys[op.results[0].0 as usize];
        if identity {
            subst.insert(op.results[0], op.operands[0]);
        }
        for r in &op.regions {
            for b in &r.blocks {
                collect_identity_casts(&b.ops, tys, subst);
            }
        }
    }
}

/// CONSTANT-FOLD float arithmetic whose operands are all constants.
///
/// # WHY THIS IS NOT OPTIONAL, MEASURED IN attention_flash
///
/// `attention_flash.py` initializes `m_i = tl.zeros([BLOCK_M], f16) - float("inf")` and
/// computes `qk_scale = sm_scale * 1.44269504` at f32. In the raw TTIR all four are ops;
/// in `tests/goldens/attention_flash_noncausal.ttir.mlir` they are three CONSTANTS
/// (`dense<0xFC00>` -- an f16 negative infinity -- and the two splats of `1.44269502`),
/// and the scalars that fed them are gone. That is seven ops of difference, and the
/// KTIR-stage passes do not fold floats, so leaving them here carries the difference all
/// the way to the bundle.
///
/// THE ARITHMETIC IS DONE AT THE RESULT'S TYPE, via [`crate::ttir::F64Bits::rounded`].
/// Doing it at f64 and storing the f64 is the trap the golden diff catches immediately:
/// `1.0 * 1.44269504` is `1.44269502` as an f32 and `1.4426950400000001` as a double, and
/// MLIR stores the constant AT ITS TYPE. `1.0/0.0` is left to IEEE, which is what
/// `APFloat` does too, so `0.0 - inf` folds to `-inf` rather than being refused.
///
/// The op is REWRITTEN IN PLACE into an `arith.constant`, keeping its result value, so no
/// operand anywhere needs rewiring. CSE in the same fixed point then merges it with an
/// equal constant and `hoist_constants` lifts it.
pub fn fold_constant_arith(m: &mut Module) -> bool {
    let consts = float_constants(m);
    let tys: Vec<Type> = m.values.iter().map(|v| v.ty.clone()).collect();
    let mut changed = false;
    for f in &mut m.funcs {
        for b in &mut f.body.blocks {
            changed |= fold_arith_ops(&mut b.ops, &consts, &tys);
        }
    }
    changed
}

/// `value -> (the float it is, whether it was a dense splat)`.
fn float_constants(m: &Module) -> HashMap<ValueId, (f64, bool)> {
    fn walk(ops: &[Op], out: &mut HashMap<ValueId, (f64, bool)>) {
        for op in ops {
            if op.name == "arith.constant" && op.results.len() == 1 {
                match op.attrs.get("value") {
                    Some(Attr::Float(b, _)) => {
                        out.insert(op.results[0], (b.get(), false));
                    }
                    Some(Attr::DenseSplat(inner, _)) => {
                        if let Attr::Float(b, _) = &**inner {
                            out.insert(op.results[0], (b.get(), true));
                        }
                    }
                    _ => {}
                }
            }
            for r in &op.regions {
                for b in &r.blocks {
                    walk(&b.ops, out);
                }
            }
        }
    }
    let mut out = HashMap::new();
    for f in &m.funcs {
        for b in &f.body.blocks {
            walk(&b.ops, &mut out);
        }
    }
    out
}

fn fold_arith_ops(
    ops: &mut [Op],
    consts: &HashMap<ValueId, (f64, bool)>,
    tys: &[Type],
) -> bool {
    let mut changed = false;
    for op in ops.iter_mut() {
        for r in &mut op.regions {
            for b in &mut r.blocks {
                changed |= fold_arith_ops(&mut b.ops, consts, tys);
            }
        }
        if op.results.len() != 1 {
            continue;
        }
        let args: Option<Vec<f64>> = op
            .operands
            .iter()
            .map(|v| consts.get(v).map(|(f, _)| *f))
            .collect();
        let Some(args) = args else { continue };
        let folded = match (op.name.as_str(), args.len()) {
            ("arith.addf", 2) => Some(args[0] + args[1]),
            ("arith.subf", 2) => Some(args[0] - args[1]),
            ("arith.mulf", 2) => Some(args[0] * args[1]),
            ("arith.divf", 2) => Some(args[0] / args[1]),
            ("arith.negf", 1) => Some(-args[0]),
            // `maxnumf`/`minnumf` are IEEE-754 maxNum/minNum: a NaN operand yields the
            // other one, which `f64::max`/`min` also do.
            ("arith.maxnumf", 2) => Some(args[0].max(args[1])),
            ("arith.minnumf", 2) => Some(args[0].min(args[1])),
            _ => None,
        };
        let Some(v) = folded else { continue };
        let res_ty = tys[op.results[0].0 as usize].clone();
        if !res_ty.is_floating() {
            continue;
        }
        let elem = res_ty.scalar().clone();
        let inner = Attr::Float(crate::ttir::F64Bits::rounded(v, &elem), elem);
        let attr = if res_ty.is_tensor() {
            Attr::DenseSplat(Box::new(inner), res_ty)
        } else {
            inner
        };
        op.name = "arith.constant".to_string();
        op.operands.clear();
        op.attrs.clear();
        op.attrs.insert("value".to_string(), attr);
        changed = true;
    }
    changed
}

/// `tt.splat` of a CONSTANT scalar -> a dense splat constant.
///
/// `Semantic::splat` already does this when the constant is known at build time (its
/// header records the measurement). It has to be done again HERE because the constant may
/// only become one after [`fold_constant_arith`]: `attention_flash`'s `qk_scale` is a
/// `mulf` at build time and a constant afterwards, and the golden has its two splats as
/// `dense<1.44269502>` constants with no `tt.splat` left.
pub fn fold_splat_of_constant(m: &mut Module) -> bool {
    let fconsts = float_constants(m);
    let iconsts = int_constants(m);
    let tys: Vec<Type> = m.values.iter().map(|v| v.ty.clone()).collect();
    let mut changed = false;
    for f in &mut m.funcs {
        for b in &mut f.body.blocks {
            changed |= fold_splats(&mut b.ops, &fconsts, &iconsts, &tys);
        }
    }
    changed
}

fn fold_splats(
    ops: &mut [Op],
    fconsts: &HashMap<ValueId, (f64, bool)>,
    iconsts: &HashMap<ValueId, i128>,
    tys: &[Type],
) -> bool {
    let mut changed = false;
    for op in ops.iter_mut() {
        for r in &mut op.regions {
            for b in &mut r.blocks {
                changed |= fold_splats(&mut b.ops, fconsts, iconsts, tys);
            }
        }
        if op.name != "tt.splat" || op.operands.len() != 1 || op.results.len() != 1 {
            continue;
        }
        let res_ty = tys[op.results[0].0 as usize].clone();
        if !res_ty.is_tensor() {
            continue;
        }
        let elem = res_ty.scalar().clone();
        let inner = match (fconsts.get(&op.operands[0]), iconsts.get(&op.operands[0])) {
            // A SCALAR constant only. Splatting an already-dense one is not a form the
            // front end builds, and guessing at it would be a fold with no oracle.
            (Some((v, false)), _) => Attr::Float(crate::ttir::F64Bits::rounded(*v, &elem), elem),
            (None, Some(i)) => Attr::Int(*i, elem),
            _ => continue,
        };
        op.name = "arith.constant".to_string();
        op.operands.clear();
        op.attrs.clear();
        op.attrs
            .insert("value".to_string(), Attr::DenseSplat(Box::new(inner), res_ty));
        changed = true;
    }
    changed
}

/// Operations MLIR marks `Commutative`, whose operands its folder therefore reorders.
///
/// `arith.subf`, `arith.divf` and the compares are deliberately absent: reordering those
/// changes the answer.
const COMMUTATIVE: &[&str] = &[
    "arith.addi",
    "arith.addf",
    "arith.muli",
    "arith.mulf",
    "arith.andi",
    "arith.ori",
    "arith.xori",
    "arith.maxnumf",
    "arith.minnumf",
    "arith.maxsi",
    "arith.maxui",
    "arith.minsi",
    "arith.minui",
];

/// Move CONSTANT operands of a commutative op to the right, stably.
///
/// This is `OpTrait::IsCommutative::foldTrait`, and it is not cosmetic: the golden for
/// every SwiGLU configuration has `arith.addf %e_6, %one` where this front end (following
/// `standard.py`'s `1 + tl.exp(-x)`) builds `addf %one, %e`. Operand LINKAGE is what the
/// structural diff compares, so an unsorted commutative op is a reported difference.
pub fn sort_commutative_operands(m: &mut Module) -> bool {
    let consts = constant_values(m);
    let mut changed = false;
    for f in &mut m.funcs {
        for b in &mut f.body.blocks {
            changed |= sort_ops(&mut b.ops, &consts);
        }
    }
    changed
}

fn sort_ops(ops: &mut [Op], consts: &std::collections::HashSet<ValueId>) -> bool {
    let mut changed = false;
    for op in ops.iter_mut() {
        for r in &mut op.regions {
            for b in &mut r.blocks {
                changed |= sort_ops(&mut b.ops, consts);
            }
        }
        if !COMMUTATIVE.contains(&op.name.as_str()) || op.operands.len() < 2 {
            continue;
        }
        let mut sorted: Vec<ValueId> = op.operands.iter().copied().filter(|v| !consts.contains(v)).collect();
        sorted.extend(op.operands.iter().copied().filter(|v| consts.contains(v)));
        if sorted != op.operands {
            op.operands = sorted;
            changed = true;
        }
    }
    changed
}

/// Every value defined by an `arith.constant`, anywhere in the module.
fn constant_values(m: &Module) -> std::collections::HashSet<ValueId> {
    fn walk(ops: &[Op], out: &mut std::collections::HashSet<ValueId>) {
        for op in ops {
            if op.name == "arith.constant" {
                out.extend(op.results.iter().copied());
            }
            for r in &op.regions {
                for b in &r.blocks {
                    walk(&b.ops, out);
                }
            }
        }
    }
    let mut out = std::collections::HashSet::new();
    for f in &m.funcs {
        for b in &f.body.blocks {
            walk(&b.ops, &mut out);
        }
    }
    out
}

/// Replace an `scf.for` whose CONSTANT bounds give a trip count of exactly one with its
/// body.
///
/// This is `scf`'s `SimplifyTrivialLoops`. MEASURED, and it is why two SwiGLU
/// configurations have 35 golden ops and the third has 37: `swiglu_mlp` and
/// `swiglu_mlp_granite` set `BLOCK_K == D_MODEL`, so the K loop runs once and MLIR
/// deletes it; `swiglu_mlp_tiledk` sets `BLOCK_K = 64` against `D_MODEL = 128`, so the
/// loop stays. A port without this pass reports the promoted loop as a missing op and
/// every op after it as a sequence divergence.
///
/// The trip count is MLIR's: `ceil((ub - lb) / step)`, computed only when all three are
/// integer `arith.constant`s. A dynamic bound is left alone.
pub fn promote_single_trip_loops(m: &mut Module) -> bool {
    let mut any = false;
    // One at a time: promoting a loop changes the op indices around it, and the body may
    // itself contain a promotable loop.
    for _ in 0..64 {
        let ints = int_constants(m);
        let Some(site) = find_promotable_for(m, &ints) else {
            return any;
        };
        let loop_op = op_at(m, &site).clone();
        let lb = loop_op.operands[0];
        let inits: Vec<ValueId> = loop_op.operands[3..].to_vec();
        let body = &loop_op.regions[0].blocks[0];

        // iv := lb, carried arg i := init i.
        let mut map: HashMap<ValueId, ValueId> = HashMap::new();
        map.insert(body.args[0], lb);
        for (a, i) in body.args[1..].iter().zip(inits.iter()) {
            map.insert(*a, *i);
        }
        let mut cloned: Vec<Op> = Vec::new();
        let mut yielded: Vec<ValueId> = Vec::new();
        let body_ops = body.ops.clone();
        for op in &body_ops {
            if op.name == "scf.yield" {
                yielded = op.operands.iter().map(|v| *map.get(v).unwrap_or(v)).collect();
                continue;
            }
            cloned.push(clone_op(m, op, &mut map));
        }
        let ops = ops_at_mut(m, &site);
        ops.splice(site.index..site.index + 1, cloned);
        let mut subst: HashMap<ValueId, ValueId> = HashMap::new();
        for (r, v) in loop_op.results.iter().zip(yielded.iter()) {
            subst.insert(*r, *v);
        }
        substitute(m, &subst);
        any = true;
    }
    any
}

/// `value -> the integer it is`, for every integer `arith.constant`.
fn int_constants(m: &Module) -> HashMap<ValueId, i128> {
    fn walk(ops: &[Op], out: &mut HashMap<ValueId, i128>) {
        for op in ops {
            if op.name == "arith.constant" && op.results.len() == 1 {
                if let Some(Attr::Int(v, _)) = op.attrs.get("value") {
                    out.insert(op.results[0], *v);
                }
            }
            for r in &op.regions {
                for b in &r.blocks {
                    walk(&b.ops, out);
                }
            }
        }
    }
    let mut out = HashMap::new();
    for f in &m.funcs {
        for b in &f.body.blocks {
            walk(&b.ops, &mut out);
        }
    }
    out
}

fn find_promotable_for(m: &Module, ints: &HashMap<ValueId, i128>) -> Option<Site> {
    for (fi, f) in m.funcs.iter().enumerate() {
        let b = f.body.blocks.first()?;
        if let Some(s) = find_for_in(&b.ops, fi, ints, &mut Vec::new()) {
            return Some(s);
        }
    }
    None
}

fn find_for_in(
    ops: &[Op],
    func: usize,
    ints: &HashMap<ValueId, i128>,
    path: &mut Vec<(usize, usize)>,
) -> Option<Site> {
    for (i, op) in ops.iter().enumerate() {
        // INNERMOST FIRST, so a promoted body cannot be re-scanned with stale indices.
        for (ri, r) in op.regions.iter().enumerate() {
            if let Some(b) = r.blocks.first() {
                path.push((i, ri));
                if let Some(s) = find_for_in(&b.ops, func, ints, path) {
                    return Some(s);
                }
                path.pop();
            }
        }
        if op.name != "scf.for" || op.operands.len() < 3 || op.regions.len() != 1 {
            continue;
        }
        let (Some(lb), Some(ub), Some(st)) = (
            ints.get(&op.operands[0]),
            ints.get(&op.operands[1]),
            ints.get(&op.operands[2]),
        ) else {
            continue;
        };
        if *st <= 0 {
            continue;
        }
        // MLIR's `constantTripCount`: ceil((ub - lb) / step), clamped at zero.
        let span = ub - lb;
        if span <= 0 {
            continue;
        }
        let trips = (span + st - 1) / st;
        if trips == 1 {
            let body = op.regions[0].blocks.first()?;
            if body.args.len() == op.operands.len() - 2 {
                return Some(Site { func, path: path.clone(), index: i });
            }
        }
    }
    None
}

//===----------------------------------------------------------------------===//
// 1. Unreachable blocks
//===----------------------------------------------------------------------===//

/// Delete every block after the first, in every region.
///
/// # This is safe HERE and the check that makes it safe is in the code, not the comment.
///
/// A block after the first is only unreachable if nothing branches to it. In general
/// MLIR that needs a CFG walk; in TTIR as [`crate::codegen`] emits it there are NO
/// branch operations at all -- control flow is `scf.for`'s region and nothing else --
/// so the only way into a later block is to be the entry. The trailing block
/// `handle_returns` creates is exactly the `^bb1: // no predecessors` in every golden.
///
/// So: an op with successors is REFUSED by name rather than assumed absent. If a `cf.br`
/// ever appears, this fails loudly instead of silently deleting reachable code.
pub fn drop_unreachable_blocks(m: &mut Module) -> Result<()> {
    for f in &mut m.funcs {
        drop_in_region(&mut f.body)?;
    }
    Ok(())
}

/// Op names that carry block successors. TTIR from this front end has none; the list
/// exists so that adding one is a refusal instead of a silent miscompile.
const BRANCHING: &[&str] = &["cf.br", "cf.cond_br", "cf.switch", "tt.br"];

fn drop_in_region(r: &mut Region) -> Result<()> {
    for b in &mut r.blocks {
        for op in &mut b.ops {
            if BRANCHING.contains(&op.name.as_str()) {
                return Err(Error::new(
                    format!(
                        "`{}` has block successors, so dropping the later blocks of a region \
                         is not sound. make_ttir's unreachable-block removal is written for \
                         TTIR with no CFG branches (control flow is scf.for's region); a \
                         real CFG needs a reachability walk.",
                        op.name
                    ),
                    0,
                    0,
                ));
            }
            for sub in &mut op.regions {
                drop_in_region(sub)?;
            }
        }
    }
    r.blocks.truncate(1);
    Ok(())
}

//===----------------------------------------------------------------------===//
// 2. The inliner
//===----------------------------------------------------------------------===//

/// Replace every `tt.call` with the callee's body, to a fixed point.
///
/// The callee's values are CLONED into fresh arena slots, because a helper called twice
/// (`triton.language.standard.zeros` is called three times in `attention_flash`) must
/// not share values between the two expansions. The callee's arguments are mapped onto
/// the call's operands, and the call's results are rewritten module-wide to whatever the
/// callee's `tt.return` yielded.
pub fn inline_calls(m: &mut Module) -> Result<()> {
    // A call inside an inlined body is a second round; `attention_flash`'s `tl.max`
    // reaches `_elementwise_max` that way. Bounded, so a recursive @triton.jit is a
    // refusal rather than a hang.
    for _ in 0..64 {
        if !inline_one_round(m)? {
            return Ok(());
        }
    }
    Err(Error::new(
        "make_ttir's inliner did not reach a fixed point in 64 rounds -- a @triton.jit \
         function that (indirectly) calls itself cannot be inlined, and Triton's own \
         inliner would not terminate on it either",
        0,
        0,
    ))
}

/// One call site, inlined. `Ok(false)` means there were none left.
fn inline_one_round(m: &mut Module) -> Result<bool> {
    // Find one call: which function, which block path, which op index.
    let Some(site) = find_call(m) else {
        return Ok(false);
    };

    let call = op_at(m, &site).clone();
    let callee_name = match call.attrs.get("callee") {
        Some(Attr::Str(s)) => s.clone(),
        _ => {
            return Err(Error::new(
                "a tt.call with no `callee` string attribute -- the front end always sets \
                 one, so this is a malformed module rather than an unsupported construct",
                0,
                0,
            ))
        }
    };
    let callee = match m.funcs.iter().find(|f| f.name == callee_name) {
        Some(f) => f.clone(),
        None => {
            return Err(Error::new(
                format!(
                    "tt.call to `@{callee_name}`, which is not in the module -- the inliner \
                     cannot proceed and leaving the call would hand bridge two a symbol it \
                     refuses"
                ),
                0,
                0,
            ))
        }
    };

    // ---- build the substitution: callee value -> caller value -------------------
    let entry = callee.body.blocks.first().ok_or_else(|| {
        Error::new(format!("`@{callee_name}` has an empty body"), 0, 0)
    })?;
    if entry.args.len() != call.operands.len() {
        return Err(Error::new(
            format!(
                "tt.call to `@{callee_name}` passes {} operand(s) but the callee takes {}",
                call.operands.len(),
                entry.args.len()
            ),
            0,
            0,
        ));
    }
    let mut map: HashMap<ValueId, ValueId> = HashMap::new();
    for (a, o) in entry.args.iter().zip(call.operands.iter()) {
        map.insert(*a, *o);
    }

    // Clone the body, minting a fresh arena slot per callee-defined value.
    let mut cloned: Vec<Op> = Vec::new();
    let mut returned: Option<Vec<ValueId>> = None;
    for op in &entry.ops {
        if op.name == "tt.return" {
            // The reachable terminator. Its operands are the call's results.
            returned = Some(
                op.operands
                    .iter()
                    .map(|v| *map.get(v).unwrap_or(v))
                    .collect(),
            );
            continue;
        }
        cloned.push(clone_op(m, op, &mut map));
    }
    let returned = returned.ok_or_else(|| {
        Error::new(
            format!("`@{callee_name}`'s entry block has no tt.return terminator"),
            0,
            0,
        )
    })?;
    if returned.len() != call.results.len() {
        return Err(Error::new(
            format!(
                "`@{callee_name}` returns {} value(s) but its tt.call has {} result(s)",
                returned.len(),
                call.results.len()
            ),
            0,
            0,
        ));
    }

    // ---- splice, then rewrite the call's results away --------------------------
    let n = cloned.len();
    let ops = ops_at_mut(m, &site);
    ops.splice(site.index..site.index + 1, cloned);
    let mut subst: HashMap<ValueId, ValueId> = HashMap::new();
    for (r, v) in call.results.iter().zip(returned.iter()) {
        subst.insert(*r, *v);
    }
    substitute(m, &subst);
    let _ = n;
    Ok(true)
}

/// Where a `tt.call` is: the function, the chain of (op index, region index) to get
/// into nested regions, and the index within the innermost block.
#[derive(Clone, Debug)]
struct Site {
    func: usize,
    /// `(op index, region index)` for each level of nesting, outermost first.
    path: Vec<(usize, usize)>,
    index: usize,
}

fn find_call(m: &Module) -> Option<Site> {
    for (fi, f) in m.funcs.iter().enumerate() {
        let b = f.body.blocks.first()?;
        if let Some(s) = find_call_in(&b.ops, fi, &mut Vec::new()) {
            return Some(s);
        }
    }
    None
}

fn find_call_in(ops: &[Op], func: usize, path: &mut Vec<(usize, usize)>) -> Option<Site> {
    for (i, op) in ops.iter().enumerate() {
        if op.name == "tt.call" {
            return Some(Site { func, path: path.clone(), index: i });
        }
        for (ri, r) in op.regions.iter().enumerate() {
            if let Some(b) = r.blocks.first() {
                path.push((i, ri));
                if let Some(s) = find_call_in(&b.ops, func, path) {
                    return Some(s);
                }
                path.pop();
            }
        }
    }
    None
}

fn ops_at_mut<'m>(m: &'m mut Module, s: &Site) -> &'m mut Vec<Op> {
    let mut ops = &mut m.funcs[s.func].body.blocks[0].ops;
    for (oi, ri) in &s.path {
        ops = &mut ops[*oi].regions[*ri].blocks[0].ops;
    }
    ops
}

fn op_at<'m>(m: &'m Module, s: &Site) -> &'m Op {
    let mut ops = &m.funcs[s.func].body.blocks[0].ops;
    for (oi, ri) in &s.path {
        ops = &ops[*oi].regions[*ri].blocks[0].ops;
    }
    &ops[s.index]
}

/// Deep-clone `op`, minting a fresh arena value for every value it DEFINES (results and
/// nested block arguments) and mapping every value it USES through `map`.
fn clone_op(m: &mut Module, op: &Op, map: &mut HashMap<ValueId, ValueId>) -> Op {
    let mut out = Op::new(op.name.clone(), op.loc.clone());
    out.attrs = op.attrs.clone();
    out.operands = op.operands.iter().map(|v| *map.get(v).unwrap_or(v)).collect();
    for r in &op.results {
        let info = m.values[r.0 as usize].clone();
        let fresh = ValueId(m.values.len() as u32);
        m.values.push(info);
        map.insert(*r, fresh);
        out.results.push(fresh);
    }
    for r in &op.regions {
        let mut nr = Region::default();
        for b in &r.blocks {
            let mut nb = Block::default();
            for a in &b.args {
                let info = m.values[a.0 as usize].clone();
                let fresh = ValueId(m.values.len() as u32);
                m.values.push(info);
                map.insert(*a, fresh);
                nb.args.push(fresh);
            }
            for inner in &b.ops {
                let c = clone_op(m, inner, map);
                nb.ops.push(c);
            }
            nr.blocks.push(nb);
        }
        out.regions.push(nr);
    }
    out
}

/// Rewrite every operand (and every `tt.return` operand) in the module through `subst`.
fn substitute(m: &mut Module, subst: &HashMap<ValueId, ValueId>) {
    if subst.is_empty() {
        return;
    }
    for f in &mut m.funcs {
        for b in &mut f.body.blocks {
            subst_ops(&mut b.ops, subst);
        }
    }
}

fn subst_ops(ops: &mut [Op], subst: &HashMap<ValueId, ValueId>) {
    for op in ops {
        for o in &mut op.operands {
            // Chase, so a chain built over several rounds resolves.
            let mut cur = *o;
            let mut guard = 0;
            while let Some(next) = subst.get(&cur) {
                if *next == cur || guard > 64 {
                    break;
                }
                cur = *next;
                guard += 1;
            }
            *o = cur;
        }
        for r in &mut op.regions {
            for b in &mut r.blocks {
                subst_ops(&mut b.ops, subst);
            }
        }
    }
}

//===----------------------------------------------------------------------===//
// 3. CSE and DCE
//===----------------------------------------------------------------------===//

/// Operations with NO side effects, so CSE may merge them and DCE may delete them.
///
/// # A LIST, not a predicate on the name's prefix, and the reason is `tt.descriptor_load`.
///
/// `arith.*` and `math.*` are uniformly pure, but `tt.*` is not: `tt.descriptor_load`
/// and `tt.load` READ memory, which makes them neither CSE-able (two loads of the same
/// descriptor at the same offset are not the same value if something wrote between
/// them) nor trivially dead. MLIR's `wouldOpBeTriviallyDead` says the same thing via
/// `MemoryEffectOpInterface`; there is no interface here, so the set is written out.
///
/// Anything NOT in this list is kept and never merged. That is the fail-closed
/// direction: an op wrongly kept shows up as a golden difference, an op wrongly deleted
/// is a silent miscompile.
fn is_pure(op: &Op) -> bool {
    if op.name.starts_with("arith.") || op.name.starts_with("math.") {
        return true;
    }
    matches!(
        op.name.as_str(),
        "tt.splat"
            | "tt.broadcast"
            | "tt.expand_dims"
            | "tt.make_range"
            | "tt.trans"
            | "tt.addptr"
            | "tt.dot"
            | "tt.get_program_id"
            | "tt.get_num_programs"
            | "tt.make_tensor_descriptor"
            | "tt.reduce"
            | "ub.poison"
            | "tensor.splat"
            | "tensor.empty"
    )
}

/// An operation's memory effect, aggregated over its regions.
///
/// `Read` exists because of `tt.descriptor_load`, whose `desc` operand is declared
/// `Arg<TT_TensorDescType, "", [MemRead<GlobalMemory>]>`
/// (`include/triton/Dialect/Triton/IR/TritonOps.td:1234`). That makes it neither pure nor
/// opaque, and MLIR's CSE treats that middle case specially -- see [`cse`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Eff {
    Pure,
    Read,
    Other,
}

/// Operations that only READ memory.
fn is_read_only(op: &Op) -> bool {
    matches!(
        op.name.as_str(),
        "tt.descriptor_load" | "tt.load" | "tt.descriptor_gather"
    )
}

/// An op's effect, taking the WORST of its own and everything in its regions.
///
/// This models MLIR's `RecursiveMemoryEffects`, which `scf.for` carries: a loop whose body
/// only reads has only a read effect, and that is precisely what lets the second
/// `desc_cos` load in `decoder_two_layers` be CSE'd against the first ACROSS two `scf.for`s.
fn effect(op: &Op) -> Eff {
    // TERMINATORS AND REGION CONTAINERS HAVE NO EFFECT OF THEIR OWN. They are listed
    // separately from `is_pure` because that set means "mergeable and deletable", which a
    // terminator is not. Getting this wrong is not subtle in its consequences and it was
    // measured: with `tt.reduce.return` treated as an unknown writer, every `tt.reduce` in
    // `decoder_two_layers` aggregated to `Eff::Other` and cleared the read map, so the
    // second layer's `desc_cos`/`desc_sin`/`desc_mask` loads did not merge.
    let no_effect_of_its_own = matches!(
        op.name.as_str(),
        "scf.for" | "scf.yield" | "tt.reduce.return" | "linalg.yield" | "tt.return"
    );
    let own = if no_effect_of_its_own || is_pure(op) {
        Eff::Pure
    } else if is_read_only(op) {
        Eff::Read
    } else {
        Eff::Other
    };
    let mut worst = own;
    for r in &op.regions {
        for b in &r.blocks {
            for inner in &b.ops {
                worst = worst.max(effect(inner));
            }
        }
    }
    worst
}

/// Common subexpression elimination, dominance-scoped, with MLIR's read-only rule.
///
/// # PURE OPS: the scope stack is the dominance check
///
/// An op inside an `scf.for` body may reuse a definition from the enclosing block (that
/// definition dominates it), but an op in the enclosing block must NOT reuse one from
/// inside the loop. Entering a region pushes a scope; leaving pops it.
///
/// # READ-ONLY OPS: same block, and no writer in between
///
/// `mlir/lib/Transforms/CSE.cpp` does not stop at memory-effect-free ops. It has an
/// explicit arm for an op with `onlyHasEffect<MemoryEffects::Read>()`, which is merged
/// against an earlier one when `existing->getBlock() == op->getBlock()` and
/// `!hasOtherSideEffectingOpInBetween(existing, op)`.
///
/// THIS IS NOT AN OPTIMIZATION WE CHOSE, IT IS A DIFFERENCE THE GOLDEN FORCED.
/// `decoder_two_layers` calls one block function twice, so after inlining both layers'
/// RoPE loads `desc_cos[0, 0]` and `desc_sin[0, 0]`, and both attention stages load
/// `desc_mask[0, 0]`. The golden has ONE of each. Without this arm ours has six loads
/// where the golden has three, and the op sequence diverges 165 ops in.
///
/// The two conditions are modelled exactly: reuse is looked up in the CURRENT scope only
/// (same block), and the read map is CLEARED by any op whose [`effect`] is [`Eff::Other`]
/// (the single `tt.descriptor_store` at the end of every kernel, and anything containing
/// one).
///
/// A region-carrying op is never itself merged. Two `tt.reduce`s with identical bodies are
/// equal in MLIR's eyes, but proving it needs a structural region comparison and no
/// fixture has two.
pub fn cse(m: &mut Module) -> bool {
    let mut subst: HashMap<ValueId, ValueId> = HashMap::new();
    for f in &mut m.funcs {
        for b in &mut f.body.blocks {
            let mut scopes: Vec<Scope> = vec![Scope::default()];
            cse_ops(&mut b.ops, &mut scopes, &mut subst, &m.values);
        }
    }
    if subst.is_empty() {
        return false;
    }
    substitute(m, &subst);
    true
}

/// One block's worth of CSE state. `pure` is visible to nested scopes; `read` is not, and
/// is cleared by a writer.
#[derive(Default)]
struct Scope {
    pure: HashMap<String, Vec<ValueId>>,
    read: HashMap<String, Vec<ValueId>>,
}

fn cse_ops(
    ops: &mut Vec<Op>,
    scopes: &mut Vec<Scope>,
    subst: &mut HashMap<ValueId, ValueId>,
    values: &[crate::ttir::ValueInfo],
) {
    let mut keep: Vec<Op> = Vec::with_capacity(ops.len());
    for mut op in ops.drain(..) {
        // Operands may already have been rewritten by an earlier merge in this pass.
        for o in &mut op.operands {
            if let Some(r) = subst.get(o) {
                *o = *r;
            }
        }
        if effect(&op) == Eff::Other {
            // A writer: nothing read before it may be reused after it.
            scopes.last_mut().expect("a scope is always open").read.clear();
        }
        if !op.regions.is_empty() {
            scopes.push(Scope::default());
            for r in &mut op.regions {
                for b in &mut r.blocks {
                    cse_ops(&mut b.ops, scopes, subst, values);
                }
            }
            scopes.pop();
            keep.push(op);
            continue;
        }
        if op.results.is_empty() {
            keep.push(op);
            continue;
        }
        let read_only = is_read_only(&op);
        if !is_pure(&op) && !read_only {
            keep.push(op);
            continue;
        }
        let key = cse_key(&op, values);
        let existing = if read_only {
            // SAME BLOCK ONLY -- an enclosing block's load does not dominate in MLIR's
            // read-only arm, which compares `getBlock()` directly.
            scopes.last().and_then(|s| s.read.get(&key)).cloned()
        } else {
            scopes.iter().rev().find_map(|s| s.pure.get(&key)).cloned()
        };
        match existing {
            Some(prev) => {
                for (r, p) in op.results.iter().zip(prev.iter()) {
                    subst.insert(*r, *p);
                }
            }
            None => {
                let last = scopes.last_mut().expect("a scope is always open");
                let slot = if read_only { &mut last.read } else { &mut last.pure };
                slot.insert(key, op.results.clone());
                keep.push(op);
            }
        }
    }
    *ops = keep;
}

/// The identity of a pure op: name, operands, attributes, result types.
///
/// Result types are IN the key because `arith.constant 1 : i64` and
/// `arith.constant 1 : i32` differ only there, and merging them would retype a value.
fn cse_key(op: &Op, values: &[crate::ttir::ValueInfo]) -> String {
    let mut s = String::new();
    s.push_str(&op.name);
    s.push('(');
    for o in &op.operands {
        s.push_str(&format!("%{},", o.0));
    }
    s.push_str(") {");
    for (k, v) in &op.attrs {
        s.push_str(&format!("{k}={},", attr_key(v)));
    }
    s.push_str("} -> ");
    for r in &op.results {
        s.push_str(&format!("{},", values[r.0 as usize].ty));
    }
    s
}

fn attr_key(a: &Attr) -> String {
    match a {
        Attr::Int(v, t) => format!("i{v}:{t}"),
        Attr::Float(b, t) => format!("f{:#x}:{t}", b.0),
        Attr::DenseSplat(inner, t) => format!("dense<{}>:{t}", attr_key(inner)),
        Attr::Bool(b) => format!("b{b}"),
        Attr::Str(s) => format!("s{s:?}"),
        Attr::Unit => "unit".to_string(),
        Attr::Pred(p) => format!("p{p}"),
        Attr::Axis(x) => format!("ax{x}"),
        Attr::Array(items) => {
            let inner: Vec<String> = items.iter().map(attr_key).collect();
            format!("[{}]", inner.join(","))
        }
        Attr::Type(t) => format!("t{t}"),
    }
}

/// Delete pure ops whose results nothing reads, to a fixed point.
///
/// Liveness walks INTO regions: a value used only inside an `scf.for` body is LIVE.
/// (`triton-ktir`'s port of the same idea records that treating it as dead drops every
/// attention bundle's key pointer.)
pub fn dce(m: &mut Module) -> bool {
    let mut any = false;
    loop {
        let mut used: std::collections::HashSet<ValueId> = std::collections::HashSet::new();
        for f in &m.funcs {
            for b in &f.body.blocks {
                collect_used(&b.ops, &mut used);
            }
        }
        let mut changed = false;
        for f in &mut m.funcs {
            for b in &mut f.body.blocks {
                changed |= dce_ops(&mut b.ops, &used);
            }
        }
        any |= changed;
        if !changed {
            return any;
        }
    }
}

fn collect_used(ops: &[Op], used: &mut std::collections::HashSet<ValueId>) {
    for op in ops {
        for o in &op.operands {
            used.insert(*o);
        }
        for r in &op.regions {
            for b in &r.blocks {
                collect_used(&b.ops, used);
            }
        }
    }
}

fn dce_ops(ops: &mut Vec<Op>, used: &std::collections::HashSet<ValueId>) -> bool {
    let mut changed = false;
    for op in ops.iter_mut() {
        for r in &mut op.regions {
            for b in &mut r.blocks {
                changed |= dce_ops(&mut b.ops, used);
            }
        }
    }
    let before = ops.len();
    ops.retain(|op| {
        if !is_pure(op) || op.results.is_empty() {
            return true;
        }
        // A region-carrying pure op is dead only if it is pure THROUGHOUT; `tt.reduce`
        // over a pure body is, and that is the only one here.
        op.results.iter().any(|r| used.contains(r))
    });
    changed || ops.len() != before
}

//===----------------------------------------------------------------------===//
// 4. Constant hoisting
//===----------------------------------------------------------------------===//

/// Move every `arith.constant` to the top of its function's ENTRY block.
///
/// Out of `scf.for` bodies too: MLIR's `OperationFolder` hoists into the nearest
/// region that is isolated from above, which for TTIR is the `tt.func` body -- a loop
/// region is not. Safe unconditionally because a constant has no operands, so nothing
/// it depends on can be left behind.
///
/// # THE ORDER IS A REVERSAL, AND IT IS DERIVED FROM MLIR'S RULE RATHER THAN FITTED
///
/// `OperationFolder::insertKnownConstant` moves each constant it meets to the block
/// FRONT -- except when the constant is already at the front, or the op before it is
/// another constant the folder already placed. Walking in program order under that rule
/// makes the hoisted run come out REVERSED, with any run of constants that was ALREADY
/// leading left in its original order behind them:
///
/// ```text
///   [C1, C2, X, C3, C4]   ->   [C4, C3, C1, C2, X]
/// ```
///
/// MEASURED, in `tests/goldens/vector_add.ttir.mlir` (two survivors, `64 : i32` first in
/// the raw and `1 : i64` second, printed `c1_i64` then `c64_i32`) and in
/// `rmsnorm.ttir.mlir` (seven survivors, the two dense splats last in the raw and first
/// in the post). Reproducing it is what lets the op SEQUENCE be compared in order, which
/// is a strictly stronger check than the multiset comparison
/// `triton_ktir::text::diff` falls back on at the next stage.
pub fn hoist_constants(m: &mut Module) {
    for f in &mut m.funcs {
        let Some(entry) = f.body.blocks.first_mut() else { continue };
        // The run of constants ALREADY at the front keeps its order and stays behind the
        // moved ones.
        let leading = entry
            .ops
            .iter()
            .take_while(|o| o.name == "arith.constant" && o.regions.is_empty())
            .count();
        let mut tail: Vec<Op> = entry.ops.split_off(leading);
        let head: Vec<Op> = std::mem::take(&mut entry.ops);
        let mut moved: Vec<Op> = Vec::new();
        take_constants(&mut tail, &mut moved);
        moved.reverse();
        entry.ops = moved;
        entry.ops.extend(head);
        entry.ops.extend(tail);
    }
}

fn take_constants(ops: &mut Vec<Op>, out: &mut Vec<Op>) {
    let mut keep: Vec<Op> = Vec::with_capacity(ops.len());
    for mut op in ops.drain(..) {
        for r in &mut op.regions {
            for b in &mut r.blocks {
                take_constants(&mut b.ops, out);
            }
        }
        if op.name == "arith.constant" && op.regions.is_empty() {
            out.push(op);
        } else {
            keep.push(op);
        }
    }
    *ops = keep;
}

//===----------------------------------------------------------------------===//
// 5. symbol-dce
//===----------------------------------------------------------------------===//

/// Delete private functions nothing calls, to a fixed point.
///
/// Public functions are roots: they are the kernel, reachable from outside the module.
pub fn symbol_dce(m: &mut Module) {
    loop {
        let mut called: std::collections::HashSet<String> = std::collections::HashSet::new();
        for f in &m.funcs {
            for b in &f.body.blocks {
                collect_callees(&b.ops, &mut called);
            }
        }
        let before = m.funcs.len();
        m.funcs.retain(|f| {
            f.visibility == Visibility::Public || called.contains(&f.name)
        });
        if m.funcs.len() == before {
            return;
        }
    }
}

fn collect_callees(ops: &[Op], out: &mut std::collections::HashSet<String>) {
    for op in ops {
        if op.name == "tt.call" {
            if let Some(Attr::Str(s)) = op.attrs.get("callee") {
                out.insert(s.clone());
            }
        }
        for r in &op.regions {
            for b in &r.blocks {
                collect_callees(&b.ops, out);
            }
        }
    }
}

//===----------------------------------------------------------------------===//
// Reporting
//===----------------------------------------------------------------------===//

/// What the pipeline did, so a test can assert on it rather than on a printed module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub funcs_before: usize,
    pub funcs_after: usize,
    pub ops_before: usize,
    pub ops_after: usize,
    pub calls_before: usize,
    pub calls_after: usize,
}

/// Count ops (regions included) and `tt.call`s across every function.
pub fn measure(m: &Module) -> (usize, usize) {
    fn walk(ops: &[Op], n: &mut usize, calls: &mut usize) {
        for op in ops {
            *n += 1;
            if op.name == "tt.call" {
                *calls += 1;
            }
            for r in &op.regions {
                for b in &r.blocks {
                    walk(&b.ops, n, calls);
                }
            }
        }
    }
    let mut n = 0;
    let mut calls = 0;
    for f in &m.funcs {
        for b in &f.body.blocks {
            walk(&b.ops, &mut n, &mut calls);
        }
    }
    (n, calls)
}

/// [`make_ttir`], with a before/after census. The census is the check: `calls_after`
/// must be zero and `funcs_after` must be one, and a test asserts both.
pub fn make_ttir_measured(m: &mut Module) -> Result<Stats> {
    let (ops_before, calls_before) = measure(m);
    let funcs_before = m.funcs.len();
    make_ttir(m)?;
    let (ops_after, calls_after) = measure(m);
    Ok(Stats {
        funcs_before,
        funcs_after: m.funcs.len(),
        ops_before,
        ops_after,
        calls_before,
        calls_after,
    })
}

// Keep the unused-import warnings honest: these are part of the module's vocabulary
// even where only some passes name them.
const _: Option<fn(&Func, &Type)> = None;
