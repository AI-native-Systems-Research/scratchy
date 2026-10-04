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

//! Ported from `DecomposeDenseConstants.cpp` (SP-E2-05).
//!
//! ```text
//!   Before:  %c = arith.constant dense<1.0> : tensor<4x8xf16>
//!   After:   %s = arith.constant 1.0 : f16
//!            %c = tensor.splat %s : tensor<4x8xf16>
//! ```
//!
//! The downstream ktir-cpu interpreter and the emitter path cannot parse `dense<>`
//! tensor-constant syntax.
//!
//! ORDERING CONTRACT (pinned by the pipeline driver, not enforced here): AFTER the
//! canonicalizer, which would otherwise fold `tensor.splat` of a constant straight
//! back into a `dense<>` constant, and AFTER `LegalizeTypes`, so the splat element
//! type is already f16.
//!
//! SCOPE: splat float constants. A NON-SPLAT dense constant is left as-is.
//!
//! RED-STOP BY OMISSION, and it is deliberate: this pass PRESERVES whatever element
//! type the splat carries and never retypes. So a non-f16 splat -- which would mean
//! a `LegalizeTypes` bug -- surfaces downstream rather than being silently "fixed"
//! here.

use crate::ir::*;
use crate::passes::walk::{self, OpPath};
use crate::Result;

pub fn run(module: &mut Module) -> Result<()> {
    // Collect first, rewrite after, so the op list is never mutated mid-walk.
    let splats: Vec<OpPath> = walk::paths(module)
        .into_iter()
        .filter(|p| {
            let op = walk::at(module, p).expect("path");
            op.kind == OpKind::ArithConstant
                && matches!(op.attr(&AttrKey::Value), Some(Attr::SplatFloat(_)))
                && op.result_type().map(|t| t.dims().is_some()).unwrap_or(false)
        })
        .collect();

    // Descending, so an insertion never shifts a path still to be visited.
    for path in splats.into_iter().rev() {
        let op = walk::at(module, &path).expect("path").clone();
        let tensor_ty = op.result_type().cloned().expect("filtered on a shaped result");
        let elem = tensor_ty.elem().expect("a shaped type has an element type");
        let value = match op.attr(&AttrKey::Value) {
            Some(Attr::SplatFloat(f)) => *f,
            _ => unreachable!("filtered"),
        };
        let result = op.results[0];
        let hint = module.hint(result);

        // The single splat element becomes a SCALAR constant. Preserve the element
        // type exactly -- no retype.
        let scalar = module.fresh_named(&hint);
        let scalar_op = Op::new(OpKind::ArithConstant)
            .with_result(scalar, IrType::Scalar(elem))
            .with_attr(AttrKey::Value, Attr::Float(value));

        // tensor.splat fans the scalar out to the original aggregate shape, and
        // KEEPS the original result name so every use is already wired.
        let splat_op = Op::new(OpKind::TensorSplat)
            .with_result(result, tensor_ty)
            .with_operands([scalar]);

        let idx = path.index();
        let block = walk::block_mut(module, &path).expect("path");
        block[idx] = splat_op;
        block.insert(idx, scalar_op);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse;

    #[test]
    fn a_splat_becomes_a_scalar_plus_a_splat_and_keeps_its_uses() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %a = arith.constant dense<1.000000e+00> : tensor<64xf16>
    %b = arith.addf %a, %a : tensor<64xf16>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).unwrap();
        let ops = &m.kernel().unwrap().regions[0].ops;
        let kinds: Vec<&str> = ops.iter().map(|o| o.kind.spelling()).collect();
        assert_eq!(kinds, vec!["arith.constant", "tensor.splat", "arith.addf", "tt.return"]);
        // The scalar constant is f16 -- the element type is PRESERVED, not retyped.
        assert_eq!(ops[0].result_type(), Some(&IrType::Scalar(DType::F16)));
        assert_eq!(
            ops[0].attr(&AttrKey::Value).and_then(|a| a.as_float()).map(|f| f.bits),
            Some(0x3c00)
        );
        // The addf still reads the SPLAT's value, so no use had to be rewired.
        assert_eq!(ops[2].operands[0], ops[1].results[0]);
    }

    #[test]
    fn a_non_splat_dense_constant_is_left_alone() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %a = arith.constant dense<[1, 2]> : tensor<2xi32>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).unwrap();
        let kinds: Vec<&str> =
            m.kernel().unwrap().regions[0].ops.iter().map(|o| o.kind.spelling()).collect();
        assert_eq!(kinds, vec!["arith.constant", "tt.return"], "OUT of scope, untouched");
    }

    #[test]
    fn a_scalar_constant_is_not_a_splat() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %a = arith.constant 0 : i32
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).unwrap();
        assert_eq!(m.kernel().unwrap().regions[0].ops.len(), 2);
    }
}
