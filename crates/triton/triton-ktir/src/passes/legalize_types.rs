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
//! WHY THE ISLAND GATE, and not a blanket module-wide retype: a blanket retype (a)
//! makes verifier-invalid IR wherever an f32 result is tied to an un-retyped
//! companion, and (b) SILENTLY MIS-NUMERICS a genuine-f32 kernel -- `ktdp.load`-ing
//! a `memref<f32>` as `tensor<f16>` verifies and reinterprets the f32 bytes as f16.
//!
//! RED-STOP, and it is real rather than comment-only: after the walk, any op result
//! still typed compute-domain f32 is GENUINE f32 that this pass did not, and must
//! not silently, legalize. Silent precision degradation is forbidden; legalizing
//! genuine f32 is an owner decision.

use crate::ir::*;
use crate::passes::walk::{self, OpPath};
use crate::{Refusal, Result};

const PASS: &str = "LegalizeTypes";

pub fn run(module: &mut Module) -> Result<()> {
    step_1_remove_extf(module);
    step_2_collapse_island(module);
    step_2b_island_constants(module);
    step_3_remove_truncf(module);
    red_stop_on_genuine_f32(module)
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
fn step_1_remove_extf(module: &mut Module) {
    loop {
        // ONE `find` with the WHOLE predicate. Finding on `kind` and then filtering
        // the Option stops at the first extf that is not f16->f32 and leaves every
        // later one in place -- a partial collapse, which is the silent-wrong-answer
        // shape this pass exists to avoid.
        let Some(path) = walk::paths(module).into_iter().find(|p| {
            let op = walk::at(module, p).expect("path");
            if op.kind != OpKind::ArithExtf {
                return false;
            }
            let in_elem =
                op.operands.first().and_then(|v| module.type_of(*v)).and_then(|t| t.elem());
            let out_elem = op.result_type().and_then(|t| t.elem());
            in_elem == Some(DType::F16) && out_elem == Some(DType::F32)
        }) else {
            break;
        };
        let op = walk::at(module, &path).expect("path").clone();
        let (from, to) = (op.results[0], op.operands[0]);
        walk::replace_all_uses(module, from, to);
        walk::erase(module, &[path]);
    }
}

/// --- Step 2: the gated f32 -> f16 retype, to a forward fixed point.
fn step_2_collapse_island(module: &mut Module) {
    loop {
        let mut changed = false;
        // Collect the retypes first, then apply, so the walk is not mutated
        // underneath itself.
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
            let operand_is_f16 = op
                .operands
                .iter()
                .any(|v| module.type_of(*v).and_then(|t| t.elem()) == Some(DType::F16));
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
            if op.kind == OpKind::ArithConstant {
                if let Some(a) = op.attr(&AttrKey::Value).cloned() {
                    if let Some(f) = a.as_float() {
                        if f.width == 32 {
                            let re = f.to_f16();
                            op.set_attr(
                                AttrKey::Value,
                                match a {
                                    Attr::SplatFloat(_) => Attr::SplatFloat(re),
                                    _ => Attr::Float(re),
                                },
                            );
                        }
                    }
                }
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
fn step_2b_island_constants(module: &mut Module) {
    let mut victims: Vec<OpPath> = Vec::new();
    for path in walk::paths(module) {
        let op = walk::at(module, &path).expect("path");
        if op.kind != OpKind::ArithConstant {
            continue;
        }
        if !op.result_types.iter().any(|t| t.is_compute_f32()) {
            continue;
        }
        let Some(res) = op.result() else { continue };
        let feeds_island = module.ops_deep().into_iter().any(|user| {
            user.operands.contains(&res)
                && !is_descriptor_or_ktdp_op(user)
                && user
                    .operands
                    .iter()
                    .any(|v| module.type_of(*v).and_then(|t| t.elem()) == Some(DType::F16))
        });
        if feeds_island {
            victims.push(path);
        }
    }
    for path in victims {
        let op = walk::at_mut(module, &path).expect("path");
        if let Some(a) = op.attr(&AttrKey::Value).cloned() {
            if let Some(f) = a.as_float() {
                if f.width == 32 {
                    let re = f.to_f16();
                    op.set_attr(
                        AttrKey::Value,
                        match a {
                            Attr::SplatFloat(_) => Attr::SplatFloat(re),
                            _ => Attr::Float(re),
                        },
                    );
                }
            }
        }
        for t in op.result_types.iter_mut() {
            if t.is_compute_f32() {
                *t = t.with_elem(DType::F16);
            }
        }
    }
    // A retyped constant can widen the island further, so re-run the fixed point.
    step_2_collapse_island(module);
}

/// --- Step 3: remove the now-redundant `arith.truncf` (f16 -> f16).
fn step_3_remove_truncf(module: &mut Module) {
    loop {
        let Some(path) = walk::paths(module).into_iter().find(|p| {
            let op = walk::at(module, p).expect("path");
            if op.kind != OpKind::ArithTruncf {
                return false;
            }
            let in_elem = op.operands.first().and_then(|v| module.type_of(*v)).and_then(|t| t.elem());
            let out_elem = op.result_type().and_then(|t| t.elem());
            in_elem == Some(DType::F16) && out_elem == Some(DType::F16)
        }) else {
            break;
        };
        let op = walk::at(module, &path).expect("path").clone();
        let (from, to) = (op.results[0], op.operands[0]);
        walk::replace_all_uses(module, from, to);
        walk::erase(module, &[path]);
    }
}

/// --- RED-stop: refuse to leave genuine f32 in the compute domain.
fn red_stop_on_genuine_f32(module: &Module) -> Result<()> {
    for op in module.ops_deep() {
        if is_descriptor_or_ktdp_op(op) {
            continue;
        }
        for t in &op.result_types {
            if t.is_out_of_scope_f32() {
                continue; // legitimately out-of-scope ptr/tensordesc/memref f32
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
        assert_eq!(get("arith.truncf"), 0, "step 3 removes every narrowing cast");

        // Every remaining compute type is f16, and the reduce's combiner too.
        for op in m.ops_deep() {
            for t in &op.result_types {
                assert!(!t.is_compute_f32(), "{} kept an f32 result", op.kind.spelling());
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
                    && o.result_type().map(|t| t.dims() == Some(&[64][..])).unwrap_or(false)
                    && o.attr(&AttrKey::Value).and_then(|a| a.as_float()).map(|f| !f.is_zero())
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
}
