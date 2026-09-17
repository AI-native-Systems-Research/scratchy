// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Shared `#[cfg(test)]` IR-construction helpers — the `ktir-optimizer` twin
//! of `ktir-emulator`'s `src/test_support.rs`. See that module's doc for why
//! this exists: %-name ergonomics over the arena-owned, typed-SSA `Operation`
//! / `IRFunction`, without resurrecting the deleted textual MLIR parser.

use std::collections::HashMap;

use ktir_core::affine::{AffineExpr, AffineMap, AffineSet, Constraint, ConstraintKind};
use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;

pub(crate) struct Ops {
    a: &'static Arena,
    next: u32,
    names: HashMap<&'static str, Ssa>,
}

impl Ops {
    pub(crate) fn new() -> Self {
        Ops {
            a: Arena::global(),
            next: 0,
            names: HashMap::new(),
        }
    }

    pub(crate) fn arena(&self) -> &'static Arena {
        self.a
    }

    /// The [`Ssa`] `name` refers to — minted on first use, stable after.
    pub(crate) fn ssa(&mut self, name: &'static str) -> Ssa {
        let next = &mut self.next;
        *self.names.entry(name).or_insert_with(|| {
            let s = Ssa(*next);
            *next += 1;
            s
        })
    }

    fn ssas(&mut self, names: &[&'static str]) -> Vec<Ssa> {
        names.iter().map(|n| self.ssa(n)).collect()
    }

    /// `%result = kind operands...`. `result: None` for a no-result op.
    pub(crate) fn op(
        &mut self,
        result: Option<&'static str>,
        kind: OpKind,
        operands: &[&'static str],
    ) -> Operation<'static> {
        let r = result.map(|n| self.ssa(n));
        let ops = self.ssas(operands);
        Operation::new(self.a, r, kind, &ops)
    }

    pub(crate) fn attr(
        &self,
        op: Operation<'static>,
        key: AttrKey,
        val: Attr<'static>,
    ) -> Operation<'static> {
        op.with_attr(self.a, key, val)
    }

    /// Set an op's result type directly (bypassing inference) — for tests that
    /// assert on a pass's `result_type` rewrite (e.g. a reshape prepending a dim).
    pub(crate) fn ty(&self, op: Operation<'static>, ty: IrType<'static>) -> Operation<'static> {
        Operation {
            result_type: Some(ty),
            ..op
        }
    }

    pub(crate) fn int_list(&self, v: Vec<i64>) -> Attr<'static> {
        Attr::IntList(self.a.ints(v))
    }

    /// Attach ONE region (a straight-line op list) to `op`.
    pub(crate) fn with_region(
        &self,
        op: Operation<'static>,
        body: Vec<Operation<'static>>,
    ) -> Operation<'static> {
        self.regions(op, vec![body])
    }

    pub(crate) fn regions(
        &self,
        op: Operation<'static>,
        bodies: Vec<Vec<Operation<'static>>>,
    ) -> Operation<'static> {
        let bodies: Vec<&'static [Operation<'static>]> =
            bodies.into_iter().map(|b| self.a.ops(b)).collect();
        Operation {
            regions: self.a.regions(bodies),
            ..op
        }
    }

    /// A rank-`n` identity affine map: `(d0..dn) -> (d0..dn)`.
    pub(crate) fn identity_map(&self, n: usize) -> AffineMap<'static> {
        self.perm_map(n, &(0..n).collect::<Vec<_>>())
    }

    pub(crate) fn sub(
        &self,
        a: AffineExpr<'static>,
        b: AffineExpr<'static>,
    ) -> AffineExpr<'static> {
        AffineExpr::Sub(self.a.expr(a), self.a.expr(b))
    }

    /// An inclusive box `[lo, hi]` as an `AffineSet` over `lo.len()` dims.
    pub(crate) fn box_set(&self, lo: &[i64], hi: &[i64]) -> AffineSet<'static> {
        let mut constraints = Vec::new();
        for i in 0..lo.len() {
            constraints.push(Constraint {
                expr: self.sub(AffineExpr::Dim(i), AffineExpr::Const(lo[i])),
                kind: ConstraintKind::GreaterEq,
            });
            constraints.push(Constraint {
                expr: self.sub(AffineExpr::Const(hi[i]), AffineExpr::Dim(i)),
                kind: ConstraintKind::GreaterEq,
            });
        }
        AffineSet {
            num_dims: lo.len(),
            num_syms: 0,
            constraints: self.a.constraints(constraints),
        }
    }

    pub(crate) fn perm_map(&self, num_dims: usize, perm: &[usize]) -> AffineMap<'static> {
        let exprs: Vec<AffineExpr<'static>> = perm.iter().map(|&d| AffineExpr::Dim(d)).collect();
        AffineMap {
            num_dims,
            num_syms: 0,
            exprs: self.a.exprs(exprs),
        }
    }

    /// Build an `IRFunction`: `name`, `(%arg_name, IrType)` pairs (minting each
    /// argument's Ssa from the same %-name table), a straight-line op list, and
    /// the grid shape.
    pub(crate) fn func(
        &mut self,
        name: &str,
        args: &[(&'static str, IrType<'static>)],
        ops: Vec<Operation<'static>>,
        grid: (usize, usize, usize),
    ) -> IRFunction<'static> {
        let arguments: Vec<(Ssa, IrType<'static>)> =
            args.iter().map(|&(n, t)| (self.ssa(n), t)).collect();
        IRFunction {
            name: self.a.str(name.to_string()),
            arguments: self.a.args(arguments),
            operations: self.a.ops(ops),
            grid,
            return_type: None,
        }
    }
}
