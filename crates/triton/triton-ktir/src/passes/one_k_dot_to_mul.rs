// SPDX-License-Identifier: Apache-2.0

//! A K=1 `tt.dot` is a multiply, and this pass says so before anything counts dots.
//!
//! `d = tt.dot a[m,1], b[1,n]` contracts ONE pair per output element:
//! `d[i,j] = a[i,0]·b[0,j]`. The sum has a single term, so there is no contraction
//! axis left to schedule — the op is an outer product, not a matmul — and this
//! target's matmul door states that itself: `assemble_matmul` emits no
//! `coordinateMasking_`, so K is read in whole 64-lane sticks. MEASURED on
//! attn.py's decode arm (granite-2b plan-only bake), the K=1 form is refused
//! outright — "MatmulTile t1150: K=1 not a multiple of the 64-fp16 stick" — and
//! `dot_to_linalg`'s paged-fixture tests record the same floor from the gathered
//! arm's side ("64 is the smallest K both doors admit").
//!
//! ⭐ THE SPELLING IS THE PIPELINE'S OWN, not an invention. Each degenerate operand
//! is widened to the product shape with `tt.broadcast` — the op `to_ktir` lowers to
//! the yield-only generic and `to_ktir_emit` names `linalg.broadcast`, i.e. exactly
//! the plumbing the whole-function door reads as an operand mode (`In::col` /
//! `In::mb`) — and ONE `arith.mulf` at the common shape computes the product. The
//! mulf KEEPS the dot's result SSA, so no consumer moves. No new op reaches KTIR:
//! every op this pass mints is one the softmax's own `m[:, None]` already exercises.
//!
//! ⛔ FAIL CLOSED ON EVERYTHING THE PRODUCT DOES NOT STATE. A dot is left untouched
//! — for the pass downstream that refuses it by name — unless ALL of these hold:
//! * both operands and the result are f16 (the device computes DL16-f16; a
//!   non-f16 dot is `dot_to_linalg`'s own refusal),
//! * the accumulator is a compile-time zero (a stated bias would be dropped by
//!   the multiply),
//! * the shapes are exactly `A [m,1]`, `B [1,n]`, result `[m,n]`.
//!
//! ⭐ AND A KERNEL WHOSE ONLY DOT IS A K=1 ONE becomes dotless here, so
//! `dot_to_linalg`'s single-dot whole-kernel contract never runs for it (zero dots
//! is that pass's early `Ok`) and the kernel lowers through the whole-function
//! door instead of being refused as a non-canonical matmul. That is not a bypass:
//! every op the rewrite mints still meets that door's own guards.

use crate::Result;
use crate::ir::*;
use crate::passes::dot_to_linalg::is_zero_const_index;
use crate::passes::walk::{self, OpPath};

pub fn run(module: &mut Module) -> Result<()> {
    // COLLECTED FIRST, walked DESCENDING — the `convert_all` law: the rewrite
    // replaces the dot in place and inserts its broadcasts before it in the same
    // block, so every later same-block index shifts, and descending means only
    // already-visited paths shift.
    let mut paths: Vec<OpPath> = walk::paths(module)
        .into_iter()
        .filter(|p| {
            walk::at(module, p)
                .map(|o| o.kind == OpKind::TtDot)
                .unwrap_or(false)
        })
        .collect();
    if paths.is_empty() {
        return Ok(());
    }
    paths.sort();
    paths.reverse();
    // ONE definition snapshot for the batch: the rewrite reads only PRE-EXISTING
    // definitions (the dot's own operands and result), which the snapshot has; the
    // values it mints are single-use, and no later iteration resolves them.
    let index = module.def_index();
    for path in paths {
        let dot = walk::at(module, &path).expect("collected path").clone();
        let Some(d) = dot.results.first().copied() else {
            continue;
        };
        let (Some(a), Some(b), Some(c)) = (
            dot.operands.first().copied(),
            dot.operands.get(1).copied(),
            dot.operands.get(2).copied(),
        ) else {
            continue;
        };
        let (Some(a_ty), Some(b_ty)) = (index.type_of(a), index.type_of(b)) else {
            continue;
        };
        let Some(d_ty) = dot.result_type().cloned() else {
            continue;
        };
        // Rank-2 extents only; anything else is a shape this rewrite has no
        // statement about and `dot_to_linalg` names.
        let dims2 = |t: &IrType| -> Option<(i64, i64)> {
            match t {
                IrType::Tensor { dims, .. } if dims.len() == 2 => Some((dims[0], dims[1])),
                _ => None,
            }
        };
        let (Some((am, ak)), Some((bk, bn)), Some((dm, dn))) =
            (dims2(&a_ty), dims2(&b_ty), dims2(&d_ty))
        else {
            continue;
        };
        // THE ONE-K LAW, on both operands AND the result: `A [m,1]`, `B [1,n]`,
        // result `[m,n]`. A disagreement is a malformed dot, not a product.
        if ak != 1 || bk != 1 || dm != am || dn != bn {
            continue;
        }
        // f16 only — the same gate `dot_to_linalg` states, so a dot this rewrite
        // leaves behind is refused there by name rather than mis-multiplied here.
        if a_ty.elem() != Some(DType::F16)
            || b_ty.elem() != Some(DType::F16)
            || d_ty.elem() != Some(DType::F16)
        {
            continue;
        }
        // ⛔ THE ACCUMULATOR MUST BE A STATED ZERO — `tl.dot`'s no-acc spelling
        // splats one (`semantic.rs`'s own law), and a real bias would be dropped
        // by the multiply. Anything else keeps the dot for the refusal that names
        // it.
        if !is_zero_const_index(&index, c) {
            continue;
        }

        // THE PRODUCT'S OPERANDS. A `[m,1]` needs widening iff the product has
        // more than one column (at `n == 1` it already IS the output shape, and a
        // no-op `tt.broadcast` is refused by `to_ktir` as one); a `[1,n]` iff the
        // product has more than one row; at `m == n == 1` neither, and the dot was
        // a scalar multiply all along.
        let hint = module.hint(d);
        let mut widened: Vec<Op> = Vec::new();
        let a_val = if bn != 1 {
            let v = module.fresh_named(&hint);
            widened.push(
                Op::new(OpKind::TtBroadcast)
                    .with_result(
                        v,
                        IrType::Tensor {
                            dims: vec![am, bn],
                            elem: DType::F16,
                        },
                    )
                    .with_operands([a]),
            );
            v
        } else {
            a
        };
        let b_val = if am != 1 {
            let v = module.fresh_named(&hint);
            widened.push(
                Op::new(OpKind::TtBroadcast)
                    .with_result(
                        v,
                        IrType::Tensor {
                            dims: vec![am, bn],
                            elem: DType::F16,
                        },
                    )
                    .with_operands([b]),
            );
            v
        } else {
            b
        };
        // The multiply, at the dot's own position and result SSA: every consumer
        // of the dot reads the product with no rewiring.
        let mul = Op::new(OpKind::ArithMulf)
            .with_result(d, d_ty)
            .with_operands([a_val, b_val]);
        let idx = path.index();
        let block = walk::block_mut(module, &path).expect("collected path");
        block[idx] = mul;
        for op in widened {
            block.insert(idx, op);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse;

    /// The decode arm's exact shape: `A [1,1] · B [1,64] -> [1,64]` with the
    /// no-accumulator zero splat (`semantic.rs`'s spelling: `arith.constant
    /// dense<0.0>`, not a `tt.splat`).
    const ONE_K: &str = "\
module {
  tt.func public @one_k(%pa: !tt.ptr<f16>, %pb: !tt.ptr<f16>) attributes {noinline = false} {
    %ca = arith.constant dense<2.000000e+00> : tensor<1x1xf16>
    %cb = arith.constant dense<3.000000e+00> : tensor<1x64xf16>
    %z = arith.constant dense<0.000000e+00> : tensor<1x64xf16>
    %d = tt.dot %ca, %cb, %z : tensor<1x1xf16> * tensor<1x64xf16> -> tensor<1x64xf16>
    tt.return
  }
}
";

    #[test]
    fn a_one_k_dot_becomes_one_broadcast_and_one_mulf() {
        let mut m = parse::parse(ONE_K).unwrap();
        run(&mut m).expect("a one-K dot is a product, not a refusal");
        let c = m.census();
        let get = |n: &str| c.iter().find(|(k, _)| k == n).map(|(_, v)| *v).unwrap_or(0);
        assert_eq!(get("tt.dot"), 0, "the dot is gone");
        assert_eq!(get("arith.mulf"), 1, "the product is one multiply");
        assert_eq!(
            get("tt.broadcast"),
            1,
            "the [1,1] operand is widened; the [1,64] one already has the product's shape"
        );
        // The mulf keeps the dot's result SSA: the single op defining a value that
        // the census still counts is the multiply.
        for op in m.ops_deep() {
            assert_ne!(op.kind, OpKind::TtDot, "no dot survives");
        }
    }

    /// The continuation shape `A [4,1] · B [1,8]`: BOTH operands are degenerate, so
    /// both widen — the Col vector and the Mb row of one multiply.
    #[test]
    fn both_degenerate_operands_widen() {
        let src = ONE_K
            .replacen(
                "%ca = arith.constant dense<2.000000e+00> : tensor<1x1xf16>",
                "%ca = arith.constant dense<2.000000e+00> : tensor<4x1xf16>",
                1,
            )
            .replacen(
                "%cb = arith.constant dense<3.000000e+00> : tensor<1x64xf16>",
                "%cb = arith.constant dense<3.000000e+00> : tensor<1x8xf16>",
                1,
            )
            .replacen(
                "%z = arith.constant dense<0.000000e+00> : tensor<1x64xf16>",
                "%z = arith.constant dense<0.000000e+00> : tensor<4x8xf16>",
                1,
            )
            .replacen(
                "%d = tt.dot %ca, %cb, %z : tensor<1x1xf16> * tensor<1x64xf16> -> tensor<1x64xf16>",
                "%d = tt.dot %ca, %cb, %z : tensor<4x1xf16> * tensor<1x8xf16> -> tensor<4x8xf16>",
                1,
            );
        let mut m = parse::parse(&src).unwrap();
        run(&mut m).expect("both-degenerate one-K dot lowers");
        let c = m.census();
        let get = |n: &str| c.iter().find(|(k, _)| k == n).map(|(_, v)| *v).unwrap_or(0);
        assert_eq!(get("tt.dot"), 0, "the dot is gone");
        assert_eq!(get("arith.mulf"), 1, "the product is one multiply");
        assert_eq!(get("tt.broadcast"), 2, "both operands widen");
    }

    /// CONTROL — a real contraction (`K=64`) is untouched: the whole point of the
    /// pass is that a K=1 dot is not a matmul, and everything that still is one
    /// must reach `dot_to_linalg` exactly as before.
    #[test]
    fn a_real_contraction_is_untouched() {
        let src = ONE_K
            .replacen(
                "%ca = arith.constant dense<2.000000e+00> : tensor<1x1xf16>",
                "%ca = arith.constant dense<2.000000e+00> : tensor<1x64xf16>",
                1,
            )
            .replacen(
                "%cb = arith.constant dense<3.000000e+00> : tensor<1x64xf16>",
                "%cb = arith.constant dense<3.000000e+00> : tensor<64x64xf16>",
                1,
            )
            .replacen(
                "%d = tt.dot %ca, %cb, %z : tensor<1x1xf16> * tensor<1x64xf16> -> tensor<1x64xf16>",
                "%d = tt.dot %ca, %cb, %z : tensor<1x64xf16> * tensor<64x64xf16> -> tensor<1x64xf16>",
                1,
            );
        let mut m = parse::parse(&src).unwrap();
        run(&mut m).expect("a K=64 dot is none of this pass's business");
        let c = m.census();
        let get = |n: &str| c.iter().find(|(k, _)| k == n).map(|(_, v)| *v).unwrap_or(0);
        assert_eq!(get("tt.dot"), 1, "the contraction survives");
        assert_eq!(get("arith.mulf"), 0, "no multiply was invented");
    }

    /// CONTROL — a stated accumulator (a bias) must not be dropped by a multiply:
    /// the dot stays for the pass that names it.
    #[test]
    fn a_stated_accumulator_keeps_the_dot() {
        let src = ONE_K.replacen(
            "%z = arith.constant dense<0.000000e+00> : tensor<1x64xf16>",
            "%z = arith.constant dense<1.000000e+00> : tensor<1x64xf16>",
            1,
        );
        let mut m = parse::parse(&src).unwrap();
        run(&mut m).expect("a biased dot is left in place, not refused here");
        let c = m.census();
        let get = |n: &str| c.iter().find(|(k, _)| k == n).map(|(_, v)| *v).unwrap_or(0);
        assert_eq!(get("tt.dot"), 1, "the biased dot survives");
        assert_eq!(get("arith.mulf"), 0, "no multiply was invented");
    }

    /// CONTROL — f32 operands are not this target's product; the dot stays for
    /// `dot_to_linalg`'s f16 refusal.
    #[test]
    fn an_f32_dot_keeps_the_dot() {
        let src = ONE_K.replace("xf16>", "xf32>");
        let mut m = parse::parse(&src).unwrap();
        run(&mut m).expect("an f32 dot is left in place, not refused here");
        let c = m.census();
        let get = |n: &str| c.iter().find(|(k, _)| k == n).map(|(_, v)| *v).unwrap_or(0);
        assert_eq!(get("tt.dot"), 1, "the f32 dot survives");
        assert_eq!(get("arith.mulf"), 0, "no multiply was invented");
    }
}
