// SPDX-License-Identifier: Apache-2.0

//! THE GRID-CORNER UNIFORM SPELLING.
//!
//! A 2-D `ktdp.construct_access_tile` whose corner is computed from the grid position
//! is the one shape the whole-function door cannot address: `region_for_operand`
//! (ktir-superdsc's `whole_function.rs`) resolves a 2-D window only when both corners
//! are constants, and refuses everything else by name --
//!
//! ```text
//!   A loop-carried or computed corner needs hoisting or unrolling before the
//!   whole-function door can address it.
//! ```
//!
//! The unroll it names is `unroll_grid_positions`'s clone protocol, and it is the right
//! answer for a corner that genuinely differs per position in a way no single form
//! states. But there is one computed corner a single form DOES state: the per-core
//! uniform, `stride · core_id + base` -- the same form `fold_grid_work_loop` ships when
//! it folds or unrolls the work loop ("every trip's position is
//! `ktdp.get_compute_tile_id` plus a constant -- the uniform form"), and the same form
//! `mul_scale_small`'s shipped corner already spells
//! (`index_cast(muli(index_cast(landmark), 64))`). The door's work division places core
//! `c` at work item `c`, so a corner that provably computes `c · stride` with `stride`
//! the tile's row extent is REDUNDANT with that division; what is missing is the
//! proof, not the mechanism. This pass states the proof in a form the door can check.
//!
//! # HOW THE PROOF WORKS -- EVALUATION, NOT PATTERN MATCHING
//!
//! The corner is evaluated at every position the launch runs (`ctid` in `[0, n)`), and
//! rewritten only when every value lands on one line `stride · ctid + base`. No op
//! chain is recognized: attention's [4,4] Q/O corner `off_hz · 256 + start_m · 64` is
//! built from `divui`/`remui` bridges a delinearisation minted, and it still evaluates
//! to `ctid · 64` because the head stride (256) is exactly `gx · BLOCK_M` -- found by
//! evaluation, not by knowing what attention is. The K/V corner of the same kernel
//! (`(off_hz / 2) · 256 + t · 64`) does NOT land on a line (8 cores share each KV
//! window), stays computed, and the fire gate still unrolls -- the designed outcome for
//! the head-grouped family, not a miss.
//!
//! # THE TWO READERS, AND THE SHARED CONTRACT
//!
//! `ktir-superdsc` is a serde-only leaf that cannot depend on this crate, so the
//! uniform family is stated TWICE: here, over this crate's ops, and there, over
//! `ktir-core`'s. [`uniform_corners`] is this side's statement -- an affine-form
//! propagation (`stride · ctid + base` per SSA, the same single-pass shape
//! `index_constants` uses for constants) -- and the door carries its twin. The contract
//! between them is exact: SOUND (only ops closed under affine forms propagate -- a
//! cast is transparent, an add of a constant or another uniform composes, mul by a
//! constant scales; div, rem, and a product of two position terms are dropped, so
//! nothing is claimed the twin cannot also claim) and COMPLETE for every form this
//! pass mints and for the two forms `fold_grid_work_loop` ships (the landmark itself,
//! and `landmark + t · num_cores`). The bridge test in `triton-ktir-superdsc` runs one
//! module through both and asserts they agree.
//!
//! # WHAT IS LEFT STANDING
//!
//! The pass rewrites the CORNER USE, not the corner's own chain: the tile's operand
//! switches to the minted uniform value, and every other consumer of the old chain
//! keeps it. A chain nothing consumes anymore is dead weight the later folds and the
//! clone protocol's per-copy constants collapse; erasing it here would be a second pass
//! (DCE) with its own proof obligations, and none of this pass's readers can see a dead
//! op. A corner that is constant, already uniform, non-affine over the domain, of
//! negative or zero stride, or of a type other than index/i32 is left standing.

use std::collections::HashMap;

use crate::Result;
use crate::ir::{Attr, AttrKey, DType, DefIndex, IrType, Module, Op, OpKind, Ssa};
use crate::passes::dot_to_linalg::const_int_index;

/// A corner's affine form over the launch position: `stride · ctid + base`.
///
/// This is the shared contract between this crate's recognizer and the door's
/// `ktir-core` twin -- see the module doc for why there are two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Uniform {
    pub stride: i64,
    pub base: i64,
}

/// The i32 domain the corner chains compute in. The delinearisation's bridges and the
/// frontend's corner arithmetic are index/i32, and a value that leaves that range is
/// not one this vocabulary produces -- it is refused rather than wrapped, the same law
/// `fold_index_arithmetic` applies to the same ops.
const I32_LO: i64 = i32::MIN as i64;
const I32_HI: i64 = i32::MAX as i64;

fn in_i32(x: i64) -> Option<i64> {
    if (I32_LO..=I32_HI).contains(&x) {
        Some(x)
    } else {
        None
    }
}

/// The value `v` computes when the launch position is `ctid`, or `None` when the chain
/// leaves the delinearisation vocabulary or the i32 domain.
///
/// The vocabulary is exactly what `distribute_work`'s axis bridges and the frontend's
/// corner arithmetic build -- integer constants, index casts, the add/mul/div/rem ops
/// (the dialect's integer set has no `subi`; subtraction arrives as `addi` of a negated
/// constant) -- plus the landmark itself. Anything else (a load, an argument, a float)
/// is not a corner fact this pass can state, and is left standing rather than guessed
/// at.
fn eval_at(index: &DefIndex, landmark: Ssa, v: Ssa, ctid: i64) -> Option<i64> {
    if v == landmark {
        return in_i32(ctid);
    }
    let head = index.def_of(v)?;
    let a = head.operands.first().copied();
    let b = head.operands.get(1).copied();
    match head.kind {
        OpKind::ArithConstant => match head.attr(&AttrKey::Value) {
            Some(Attr::Int(i)) => in_i32(*i),
            _ => None,
        },
        // An index cast is value-transparent at the widths this vocabulary uses: the
        // corner chains cast index <-> i32, and both hold every value the domain guard
        // admits, so the cast changes the type and not the number.
        OpKind::ArithIndexCast => eval_at(index, landmark, a?, ctid),
        OpKind::ArithMuli => eval_at(index, landmark, a?, ctid)?
            .checked_mul(eval_at(index, landmark, b?, ctid)?)
            .and_then(in_i32),
        OpKind::ArithAddi => eval_at(index, landmark, a?, ctid)?
            .checked_add(eval_at(index, landmark, b?, ctid)?)
            .and_then(in_i32),
        // Truncating division and remainder -- the C semantics the ops have. Division
        // by zero is left standing rather than turned into a claim.
        OpKind::ArithDivsi => {
            let (x, y) = (
                eval_at(index, landmark, a?, ctid)?,
                eval_at(index, landmark, b?, ctid)?,
            );
            if y == 0 {
                None
            } else {
                x.checked_div(y).and_then(in_i32)
            }
        }
        OpKind::ArithRemsi => {
            let (x, y) = (
                eval_at(index, landmark, a?, ctid)?,
                eval_at(index, landmark, b?, ctid)?,
            );
            if y == 0 {
                None
            } else {
                x.checked_rem(y).and_then(in_i32)
            }
        }
        // Unsigned semantics: stated only for a non-negative dividend and a positive
        // divisor, where they agree with the signed ones. A negative operand is left
        // standing rather than guessed at -- `fold_index_arithmetic`'s law for the same
        // ops.
        OpKind::ArithDivui => {
            let (x, y) = (
                eval_at(index, landmark, a?, ctid)?,
                eval_at(index, landmark, b?, ctid)?,
            );
            if y <= 0 || x < 0 { None } else { in_i32(x / y) }
        }
        OpKind::ArithRemui => {
            let (x, y) = (
                eval_at(index, landmark, a?, ctid)?,
                eval_at(index, landmark, b?, ctid)?,
            );
            if y <= 0 || x < 0 { None } else { in_i32(x % y) }
        }
        _ => None,
    }
}

/// `corner(ctid) = stride · ctid + base` for EVERY position the launch runs, proven by
/// evaluating the chain at each one.
///
/// A proof, not a two-point guess: `remui(ctid, 4)` is affine on `{0, 1}` and wraps at
/// 4, and only the full domain separates the two. The domain is the launch's own
/// position count, so the proof covers exactly the trips that can run.
fn affine_over(index: &DefIndex, landmark: Ssa, v: Ssa, n: i64) -> Option<Uniform> {
    let base = eval_at(index, landmark, v, 0)?;
    let stride = eval_at(index, landmark, v, 1)?.checked_sub(base)?;
    for ctid in 2..n {
        let want = base.checked_add(stride.checked_mul(ctid)?)?;
        if eval_at(index, landmark, v, ctid)? != want {
            return None;
        }
    }
    Some(Uniform { stride, base })
}

/// Every SSA in `body` whose value the affine-form propagation can state as
/// `stride · ctid + base` over the landmark -- this crate's half of the shared
/// contract. See the module doc for the soundness and completeness claims.
///
/// `index` supplies the constants; `body` supplies the program order. ONE PASS in that
/// order: SSA names are single-assignment and a definition precedes its uses, so a form
/// computed at its own op sees every operand's form already -- the same single-pass
/// shape `index_constants` states for constants (that reader loops to a fixed point;
/// SSA makes the loop one trip, and this recognizer does not pay for the trip it
/// provably does not need).
pub fn uniform_corners(body: &[Op], index: &DefIndex, landmark: Ssa) -> HashMap<Ssa, Uniform> {
    let mut out: HashMap<Ssa, Uniform> = HashMap::new();
    out.insert(landmark, Uniform { stride: 1, base: 0 });
    for op in body {
        let Some(r) = op.result() else { continue };
        let a = op.operands.first().copied();
        let b = op.operands.get(1).copied();
        let form: Option<Uniform> = match op.kind {
            OpKind::ArithIndexCast => a.and_then(|v| out.get(&v).copied()),
            OpKind::ArithAddi | OpKind::ArithMuli => {
                let (Some(a), Some(b)) = (a, b) else { continue };
                let fa = out.get(&a).copied();
                let fb = out.get(&b).copied();
                let ka = const_int_index(index, a);
                let kb = const_int_index(index, b);
                match op.kind {
                    OpKind::ArithAddi => match (fa, fb, ka, kb) {
                        (Some(x), Some(y), _, _) => {
                            compose(x.stride.checked_add(y.stride), x.base.checked_add(y.base))
                        }
                        (Some(x), None, _, Some(c)) => {
                            compose(Some(x.stride), x.base.checked_add(c))
                        }
                        (None, Some(y), Some(c), _) => {
                            compose(Some(y.stride), y.base.checked_add(c))
                        }
                        _ => None,
                    },
                    // A product of two position terms is quadratic, not uniform: only
                    // mul by a CONSTANT propagates. This is the one op family where
                    // the recognizer is deliberately narrower than the arithmetic.
                    OpKind::ArithMuli => match (fa, fb, ka, kb) {
                        (Some(x), None, _, Some(c)) => {
                            compose(x.stride.checked_mul(c), x.base.checked_mul(c))
                        }
                        (None, Some(y), Some(c), _) => {
                            compose(y.stride.checked_mul(c), y.base.checked_mul(c))
                        }
                        _ => None,
                    },
                    _ => unreachable!("the outer match admits only these two kinds"),
                }
            }
            _ => None,
        };
        if let Some(f) = form {
            out.insert(r, f);
        }
    }
    out
}

/// `Some` for a form inside the i32 domain, `None` for one that left it (or whose
/// arithmetic overflowed -- `None` in, `None` out). A form the guard rejects is not
/// claimed: the door's twin states the same law.
fn compose(stride: Option<i64>, base: Option<i64>) -> Option<Uniform> {
    in_i32(stride?)?;
    in_i32(base?)?;
    Some(Uniform {
        stride: stride?,
        base: base?,
    })
}

/// One corner to rewrite: tile `tile`'s corner slot `slot` becomes `form`, minted in
/// `ty` and printed under `hint`.
struct Rewrite {
    tile: usize,
    slot: usize,
    corner: Ssa,
    form: Uniform,
    ty: IrType,
    hint: String,
}

/// True iff any op below `op`'s own region boundary states a window -- a tile, a load,
/// or a store. Nesting is walked to the bottom, because the door's readers do not stop
/// one level down.
fn nested_windows(op: &Op) -> bool {
    fn any_window(ops: &[Op]) -> bool {
        ops.iter().any(|o| {
            matches!(
                o.kind,
                OpKind::KtdpConstructAccessTile
                    | OpKind::KtdpConstructIndirectAccessTile
                    | OpKind::KtdpLoad
                    | OpKind::KtdpStore
            ) || o.regions.iter().any(|r| any_window(&r.ops))
        })
    }
    op.regions.iter().any(|r| any_window(&r.ops))
}

/// Rewrite program-id-derived 2-D tile corners into the per-core uniform spelling.
///
/// Runs late in `to_ktir`'s pipeline, after the work loop is folded/unrolled and the
/// constant-trip loops unrolled (so the corners are top-level chains over the
/// landmark), and before `unroll_grid_positions` -- whose fire gate exempts a corner
/// the recognizer claims, which is what lets an all-uniform grid keep its launch
/// instead of paying the clone protocol.
pub fn run(module: &mut Module, grid: &[i64]) -> Result<()> {
    let n = grid.iter().product::<i64>();
    // One position: the corner IS the position, and `fold_unit_grid_tile_id` has
    // already made it a constant.
    if n <= 1 {
        return Ok(());
    }
    let kernel = module.kernel()?;
    let body: Vec<Op> = kernel.regions[0].ops.clone();
    let Some(landmark) = body
        .iter()
        .find(|o| o.kind == OpKind::KtdpGetComputeTileId)
        .and_then(|o| o.result())
    else {
        return Ok(());
    };
    // THE DOOR'S OWN READING PRECONDITION, enforced here as a sit-out condition: every
    // window at the function's TOP LEVEL. A kernel whose tiles still live inside a
    // nested region -- the CAUSAL attention's KV sweep, whose trip count is a function
    // of the grid position and which `unroll_constant_trip_loops` therefore leaves
    // standing -- needs the clone protocol regardless of what its corners state,
    // because the door's every window reader walks the top level only. Rewriting that
    // kernel's corners would perturb the pinned clone output without opening anything:
    // the causal Q/O corner IS affine (head stride 128 = gx · BLOCK_M at grid (2,4)),
    // and it was the fire gate's computed-corner proxy for "this body needs cloning".
    // So the pass sits out, the gate sees the corner still computed, and the module is
    // exactly what it was.
    if body.iter().any(nested_windows) {
        return Ok(());
    }
    let index = module.def_index();
    let shipped = uniform_corners(&body, &index, landmark);

    // THE PLAN, collected before anything mutates: one entry per (tile, corner slot)
    // whose corner is computed, not already uniform, and affine over the domain.
    let mut plan: Vec<Rewrite> = Vec::new();
    for (i, op) in body.iter().enumerate() {
        // Only the 2-D tiles: the door resolves 1-D windows without reading corners,
        // and a 3-D+ tile is not a shape this pipeline's views state.
        if op.kind != OpKind::KtdpConstructAccessTile || op.operands.len() != 3 {
            continue;
        }
        for slot in [1usize, 2usize] {
            let c = op.operands[slot];
            // A constant corner is what the door reads directly.
            if const_int_index(&index, c).is_some() {
                continue;
            }
            // Already uniform -- the shipped form, or a previous run of this pass.
            if shipped.contains_key(&c) {
                continue;
            }
            let Some(form) = affine_over(&index, landmark, c, n) else {
                continue;
            };
            // A zero stride is a constant in disguise (the folds own it); a negative
            // one addresses tiles downward from the base, which no launch in this
            // pipeline does -- both stay computed rather than minted.
            if form.stride <= 0 {
                continue;
            }
            let Some(ty) = index.type_of(c) else {
                continue;
            };
            // Mint in the corner's own type, and only in the types the vocabulary
            // states: index and 32-bit integer. Anything else is left standing.
            if !matches!(ty, IrType::Index | IrType::Scalar(DType::I32 | DType::SI32)) {
                continue;
            }
            plan.push(Rewrite {
                tile: i,
                slot,
                corner: c,
                form,
                ty,
                hint: module.hint(c),
            });
        }
    }
    if plan.is_empty() {
        return Ok(());
    }

    // APPLY: rebuild the body, minting each distinct corner's chain immediately before
    // the first tile that consumes it -- after every operand's definition by
    // construction, because the tile consumed the corner's own chain.
    let mut minted: HashMap<Ssa, Ssa> = HashMap::new();
    let mut new_body: Vec<Op> = Vec::with_capacity(body.len() + plan.len() * 4);
    for (i, mut op) in body.into_iter().enumerate() {
        for r in plan.iter().filter(|r| r.tile == i) {
            let v = match minted.get(&r.corner) {
                Some(v) => *v,
                None => {
                    let v = mint_chain(module, landmark, r.form, &r.ty, &r.hint, &mut new_body);
                    minted.insert(r.corner, v);
                    v
                }
            };
            op.operands[r.slot] = v;
        }
        new_body.push(op);
    }
    let Some(f) = module.ops.iter_mut().find(|o| o.kind == OpKind::FuncFunc) else {
        return Ok(());
    };
    f.regions[0].ops = new_body;
    Ok(())
}

/// Mint `form.stride · landmark + form.base` in `ty`'s own type, pushing onto `out`,
/// and return the value that names it.
///
/// THE CANONICAL SPELLING, `mul_scale_small`'s shipped corner: the landmark cast to
/// i32, the stride as an i32 constant, one `muli`, and one `addi` for the base when it
/// is not zero. An index-typed corner takes that core and casts the result back -- one
/// cast in, one cast out. The spelling is pinned by the two recognizers that read it
/// (this crate's [`uniform_corners`] and the door's twin), so it is minted in exactly
/// this shape and no other.
fn mint_chain(
    module: &mut Module,
    landmark: Ssa,
    form: Uniform,
    ty: &IrType,
    hint: &str,
    out: &mut Vec<Op>,
) -> Ssa {
    let core = match ty {
        IrType::Index => IrType::Scalar(DType::I32),
        t => t.clone(),
    };
    let c = module.fresh_named(hint);
    out.push(
        Op::new(OpKind::ArithIndexCast)
            .with_result(c, core.clone())
            .with_operands([landmark]),
    );
    let s = module.fresh_named(hint);
    out.push(
        Op::new(OpKind::ArithConstant)
            .with_result(s, core.clone())
            .with_attr(AttrKey::Value, Attr::Int(form.stride)),
    );
    let m = module.fresh_named(hint);
    out.push(
        Op::new(OpKind::ArithMuli)
            .with_result(m, core.clone())
            .with_operands([c, s]),
    );
    let mut v = m;
    if form.base != 0 {
        let b = module.fresh_named(hint);
        out.push(
            Op::new(OpKind::ArithConstant)
                .with_result(b, core.clone())
                .with_attr(AttrKey::Value, Attr::Int(form.base)),
        );
        let a = module.fresh_named(hint);
        out.push(
            Op::new(OpKind::ArithAddi)
                .with_result(a, core)
                .with_operands([v, b]),
        );
        v = a;
    } else {
        let _ = core;
    }
    if matches!(ty, IrType::Index) {
        let i = module.fresh_named(hint);
        out.push(
            Op::new(OpKind::ArithIndexCast)
                .with_result(i, ty.clone())
                .with_operands([v]),
        );
        v = i;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse;

    /// The [4,4] Q/O corner arithmetic, as the delinearisation and the frontend build
    /// it: `start_m = ctid % 4`, `off_hz = ctid / 4` (index-typed bridges, cast to
    /// i32), corner `= off_hz · 256 + start_m · 64` -- which lands on `ctid · 64`
    /// because the head stride is exactly `gx · BLOCK_M`.
    const AFFINE: &str = "\
module {
  func.func @k(%q: !tt.ptr<f16>) attributes {grid = [16 : index]} {
    %tid = ktdp.get_compute_tile_id : index
    %c4 = arith.constant 4 : index
    %xm = arith.remui %tid, %c4 : index
    %x = arith.index_cast %xm : index to i32
    %yd = arith.divui %tid, %c4 : index
    %y = arith.index_cast %yd : index to i32
    %c64 = arith.constant 64 : i32
    %c256 = arith.constant 256 : i32
    %m = arith.muli %x, %c64 : i32
    %h = arith.muli %y, %c256 : i32
    %corner = arith.addi %h, %m : i32
    %c0 = arith.constant 0 : i32
    %v = ktdp.construct_memory_view %q, sizes: [1024, 64], strides: [64, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, d1 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1024x64xf16>
    %t = ktdp.construct_access_tile %v[%corner, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>, access_tile_set = affine_set<(d0, d1) : (d0 >= 0, d1 >= 0)>} : memref<1024x64xf16> -> !ktdp.access_tile<64x64xindex>
    return
  }
}
";

    fn tile_of(m: &Module) -> (Ssa, Ssa) {
        let f = m.kernel().unwrap();
        let t = f.regions[0]
            .ops
            .iter()
            .find(|o| o.kind == OpKind::KtdpConstructAccessTile)
            .unwrap();
        (t.operands[1], t.operands[2])
    }

    /// THE REWRITE: the [4,4] Q/O corner becomes `muli(index_cast(landmark), 64)`, the
    /// canonical spelling, and the recognizer claims it as `{stride: 64, base: 0}`.
    #[test]
    fn a_div_rem_corner_that_is_affine_over_the_launch_rewrites_to_the_uniform_spelling() {
        let mut m = parse::parse(AFFINE).unwrap();
        run(&mut m, &[4, 4]).unwrap();
        let (row, _col) = tile_of(&m);
        let index = m.def_index();
        let landmark = m.kernel().unwrap().regions[0]
            .ops
            .iter()
            .find(|o| o.kind == OpKind::KtdpGetComputeTileId)
            .and_then(|o| o.result())
            .unwrap();
        let forms = uniform_corners(&m.kernel().unwrap().regions[0].ops, &index, landmark);
        assert_eq!(
            forms.get(&row),
            Some(&Uniform {
                stride: 64,
                base: 0
            }),
            "the rewritten corner must be claimed as ctid · 64"
        );
        // AND IT IS THE MINTED CHAIN, not the old one: the tile's row corner is now a
        // muli over a cast of the landmark with a constant stride.
        let head = index.def_of(row).unwrap();
        assert_eq!(head.kind, OpKind::ArithMuli);
        let (a, b) = (head.operands[0], head.operands[1]);
        assert_eq!(const_int_index(&index, b), Some(64));
        let cast = index.def_of(a).unwrap();
        assert_eq!(cast.kind, OpKind::ArithIndexCast);
        assert_eq!(cast.operands[0], landmark);
        // The old corner chain is still present -- left standing for its other
        // consumers (here: none), not erased.
        assert!(
            m.kernel().unwrap().regions[0]
                .ops
                .iter()
                .any(|o| { o.kind == OpKind::ArithRemui && o.result().is_some() }),
            "the pass rewrites the use, not the corner's own chain"
        );
    }

    /// IDEMPOTENT: a second run finds the corner already uniform and mints nothing.
    #[test]
    fn a_second_run_mints_nothing() {
        let mut m = parse::parse(AFFINE).unwrap();
        run(&mut m, &[4, 4]).unwrap();
        let after_first = m.census();
        run(&mut m, &[4, 4]).unwrap();
        assert_eq!(
            m.census(),
            after_first,
            "the corner is in the shipped form now, so the pass must be a no-op"
        );
    }

    /// THE HEAD-GROUPED FAMILY STAYS COMPUTED: `(off_hz / 2) · 256` wraps twice per
    /// KV window (8 cores share one), so no line covers the domain and the fire gate
    /// still gets its computed corner.
    #[test]
    fn a_head_grouped_corner_stays_computed() {
        let src = AFFINE.replace(
            "    %corner = arith.addi %h, %m : i32",
            "    %c2 = arith.constant 2 : i32\n    %g = arith.divsi %y, %c2 : i32\n    %gh = arith.muli %g, %c256 : i32\n    %corner = arith.addi %gh, %m : i32",
        );
        let mut m = parse::parse(&src).unwrap();
        run(&mut m, &[4, 4]).unwrap();
        let (row, _) = tile_of(&m);
        let index = m.def_index();
        let landmark = m.kernel().unwrap().regions[0]
            .ops
            .iter()
            .find(|o| o.kind == OpKind::KtdpGetComputeTileId)
            .and_then(|o| o.result())
            .unwrap();
        let head = index.def_of(row).unwrap();
        assert_eq!(
            head.kind,
            OpKind::ArithAddi,
            "the head-grouped corner must stay the kernel's own chain"
        );
        assert!(
            affine_over(&index, landmark, row, 16).is_none(),
            "(ctid/8) · 256 + (ctid % 4) · 64 is not a line over [0, 16)"
        );
    }

    /// A WRAPPING CHAIN IS NOT UNIFORM, at any reader: `remui(ctid, 4) · 64` is affine
    /// on {0, 1} and wraps at 4 -- the two-point guess would claim it, the full-domain
    /// proof does not, and the propagation never sees a remui at all.
    #[test]
    fn a_wrapping_chain_is_not_uniform_at_any_reader() {
        let src = AFFINE.replace(
            "    %corner = arith.addi %h, %m : i32",
            "    %corner = arith.muli %x, %c64 : i32",
        );
        let mut m = parse::parse(&src).unwrap();
        run(&mut m, &[4, 4]).unwrap();
        let (row, _) = tile_of(&m);
        let index = m.def_index();
        let landmark = m.kernel().unwrap().regions[0]
            .ops
            .iter()
            .find(|o| o.kind == OpKind::KtdpGetComputeTileId)
            .and_then(|o| o.result())
            .unwrap();
        assert_eq!(
            index.def_of(row).unwrap().kind,
            OpKind::ArithMuli,
            "the wrapping corner must stay the kernel's own chain"
        );
        assert!(affine_over(&index, landmark, row, 16).is_none());
        let forms = uniform_corners(&m.kernel().unwrap().regions[0].ops, &index, landmark);
        assert!(!forms.contains_key(&row));
    }

    /// THE RECOGNIZER NAMES THE SHIPPED FORMS: the landmark itself, the folded
    /// configuration's `muli(cast(landmark), S)`, and the multi-trip unroll's
    /// `muli(cast(addi(landmark, num_cores)), S)` -- the three shapes the door's twin
    /// must also claim, or the bridge between them is broken.
    #[test]
    fn the_recognizer_names_the_shipped_forms() {
        let src = AFFINE.replace(
            "    %corner = arith.addi %h, %m : i32",
            "    %c32 = arith.constant 32 : index\n    %o = arith.addi %tid, %c32 : index\n    %oc = arith.index_cast %o : index to i32\n    %corner = arith.muli %oc, %c64 : i32",
        );
        let mut m = parse::parse(&src).unwrap();
        run(&mut m, &[4, 4]).unwrap();
        let f = m.kernel().unwrap();
        let body = &f.regions[0].ops;
        let index = m.def_index();
        let landmark = body
            .iter()
            .find(|o| o.kind == OpKind::KtdpGetComputeTileId)
            .and_then(|o| o.result())
            .unwrap();
        let forms = uniform_corners(body, &index, landmark);
        // The landmark itself.
        assert_eq!(forms.get(&landmark), Some(&Uniform { stride: 1, base: 0 }));
        // The multi-trip form: 64 · (ctid + 32) = 64 · ctid + 2048.
        let corner = body
            .iter()
            .find(|o| o.kind == OpKind::KtdpConstructAccessTile)
            .unwrap()
            .operands[1];
        assert_eq!(
            forms.get(&corner),
            Some(&Uniform {
                stride: 64,
                base: 2048
            }),
            "the multi-trip pid form is uniform with the trip offset folded into the base"
        );
        // Already uniform before the pass ran, so nothing was minted: the corner is
        // still the kernel's own muli.
        assert_eq!(index.def_of(corner).unwrap().kind, OpKind::ArithMuli);
        assert_eq!(
            index.def_of(corner).unwrap().operands[0],
            body.iter()
                .find(|o| o.kind == OpKind::ArithIndexCast
                    && matches!(index.def_of(o.operands[0]), Some(h) if h.kind == OpKind::ArithAddi))
                .and_then(|o| o.result())
                .expect("the cast of the addi")
        );
    }

    /// A QUADRATIC CHAIN IS DROPPED BY THE PROPAGATION: `muli(remui(ctid, 4), x)` with
    /// `x` itself position-derived has two position terms -- the one op family where
    /// the recognizer is deliberately narrower than the arithmetic.
    #[test]
    fn a_product_of_two_position_terms_is_not_uniform() {
        let src = AFFINE.replace(
            "    %corner = arith.addi %h, %m : i32",
            "    %corner = arith.muli %m, %h : i32",
        );
        let m = parse::parse(&src).unwrap();
        let f = m.kernel().unwrap();
        let body = &f.regions[0].ops;
        let index = m.def_index();
        let landmark = body
            .iter()
            .find(|o| o.kind == OpKind::KtdpGetComputeTileId)
            .and_then(|o| o.result())
            .unwrap();
        let forms = uniform_corners(body, &index, landmark);
        let corner = body
            .iter()
            .find(|o| o.kind == OpKind::KtdpConstructAccessTile)
            .unwrap()
            .operands[1];
        assert!(!forms.contains_key(&corner));
    }

    /// A WINDOW INSIDE A LOOP SITS THE PASS OUT, however affine its corner is. That is
    /// the causal-attention shape: the KV sweep's windows live in a standing `scf.for`
    /// (its trip count is a function of the grid position), the door's readers walk the
    /// top level only, and the kernel needs the clone protocol regardless -- so the
    /// corner must stay COMPUTED and keep tripping the fire gate, or the gate's silence
    /// would ship a module with a live landmark and a loop the door cannot read.
    /// MEASURED on the causal golden at grid (2,4): the Q/O corner there IS affine
    /// (head stride 128 = gx · BLOCK_M), and rewriting it without this sit-out left the
    /// causal module uncloned with a standing sweep -- the regression this test pins.
    #[test]
    fn a_window_inside_a_loop_sits_the_pass_out_however_affine_its_corner() {
        // The AFFINE corner, but consumed by a tile INSIDE an scf.for.
        let src = AFFINE
            .replace(
                "    %v = ktdp.construct_memory_view",
                "    %cz = arith.constant 0 : index\n    %cone = arith.constant 1 : index\n    scf.for %i = %cz to %c4 step %cone {\n    %v = ktdp.construct_memory_view",
            )
            .replace("    return\n", "    }\n    return\n");
        let mut m = parse::parse(&src).unwrap();
        let before = m.census();
        run(&mut m, &[4, 4]).unwrap();
        assert_eq!(
            m.census(),
            before,
            "the pass must mint nothing while a window sits inside a loop"
        );
        let f = m.kernel().unwrap();
        let body = &f.regions[0].ops;
        // And the tile's corner is still the kernel's own addi chain -- nothing was
        // minted into the loop either.
        let in_loop = body
            .iter()
            .find(|o| o.kind == OpKind::ScfFor)
            .expect("the loop")
            .regions[0]
            .ops
            .iter()
            .find(|o| o.kind == OpKind::KtdpConstructAccessTile)
            .unwrap();
        let index = m.def_index();
        assert_eq!(
            index.def_of(in_loop.operands[1]).unwrap().kind,
            OpKind::ArithAddi,
            "an affine corner inside a loop must stay the kernel's own chain"
        );
    }
}
