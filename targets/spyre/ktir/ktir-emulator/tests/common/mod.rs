// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Shared IR-construction helpers for integration tests — the `tests/`-tree
//! twin of `src/test_support.rs` (integration tests link the compiled crate
//! as an external dependency, so they can only see `pub` items; the library's
//! own `pub(crate) mod test_support` isn't reachable from here). See that
//! module's doc for why this exists: %-name ergonomics over the arena-owned,
//! typed-SSA `Operation`, without resurrecting the deleted textual parser.
#![allow(dead_code)]

use std::collections::HashMap;

use ktir_emulator::affine::{AffineExpr, AffineMap, AffineSet, Constraint, ConstraintKind};
use ktir_emulator::arena::Arena;
use ktir_emulator::attrkey::AttrKey;
use ktir_emulator::ir::{Attr, Operation, Ssa};
use ktir_emulator::irtype::IrType;
use ktir_emulator::opkind::OpKind;

pub struct Ops {
    a: &'static Arena,
    next: u32,
    names: HashMap<&'static str, Ssa>,
}

impl Ops {
    pub fn new() -> Self {
        Ops {
            a: Arena::global(),
            next: 0,
            names: HashMap::new(),
        }
    }

    pub fn arena(&self) -> &'static Arena {
        self.a
    }

    /// The [`Ssa`] `name` refers to — minted on first use, stable after.
    pub fn ssa(&mut self, name: &'static str) -> Ssa {
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
    pub fn op(
        &mut self,
        result: Option<&'static str>,
        kind: OpKind,
        operands: &[&'static str],
    ) -> Operation<'static> {
        let r = result.map(|n| self.ssa(n));
        let ops = self.ssas(operands);
        Operation::new(self.a, r, kind, &ops)
    }

    pub fn attr(
        &self,
        op: Operation<'static>,
        key: AttrKey,
        val: Attr<'static>,
    ) -> Operation<'static> {
        op.with_attr(self.a, key, val)
    }

    /// Set an op's result type directly (bypassing inference) — for tests that
    /// assert on a specific `result_type` (e.g. a cast's declared output dtype).
    pub fn ty(&self, op: Operation<'static>, ty: IrType<'static>) -> Operation<'static> {
        Operation {
            result_type: Some(ty),
            ..op
        }
    }

    pub fn int_list(&self, v: Vec<i64>) -> Attr<'static> {
        Attr::IntList(self.a.ints(v))
    }

    pub fn float_list(&self, v: Vec<f64>) -> Attr<'static> {
        Attr::FloatList(self.a.floats(v))
    }

    pub fn map_list(&self, v: Vec<AffineMap<'static>>) -> Attr<'static> {
        Attr::AffineMapList(self.a.maps(v))
    }

    pub fn str_list(&self, items: &[&str]) -> Attr<'static> {
        let owned: Vec<&'static str> = items.iter().map(|s| self.a.str((*s).to_string())).collect();
        Attr::StrList(self.a.names(owned))
    }

    /// A `%name`-referencing attribute — resolves the same %-name table
    /// [`Ops::op`] mints from.
    pub fn ssas_attr(&mut self, names: &[&'static str]) -> Attr<'static> {
        let ssas = self.ssas(names);
        Attr::Ssas(self.a.ssa(ssas))
    }

    /// Attach ONE region (a straight-line op list) to `op`.
    pub fn with_region(
        &self,
        op: Operation<'static>,
        body: Vec<Operation<'static>>,
    ) -> Operation<'static> {
        self.regions(op, vec![body])
    }

    pub fn regions(
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
    pub fn identity_map(&self, n: usize) -> AffineMap<'static> {
        self.perm_map(n, &(0..n).collect::<Vec<_>>())
    }

    /// A rank-`num_dims` affine map selecting/permuting dims by index.
    pub fn perm_map(&self, num_dims: usize, perm: &[usize]) -> AffineMap<'static> {
        let exprs: Vec<AffineExpr<'static>> = perm.iter().map(|&d| AffineExpr::Dim(d)).collect();
        AffineMap {
            num_dims,
            num_syms: 0,
            exprs: self.a.exprs(exprs),
        }
    }

    pub fn sub(&self, a: AffineExpr<'static>, b: AffineExpr<'static>) -> AffineExpr<'static> {
        AffineExpr::Sub(self.a.expr(a), self.a.expr(b))
    }

    pub fn mul(&self, a: AffineExpr<'static>, b: AffineExpr<'static>) -> AffineExpr<'static> {
        AffineExpr::Mul(self.a.expr(a), self.a.expr(b))
    }

    pub fn add_expr(&self, a: AffineExpr<'static>, b: AffineExpr<'static>) -> AffineExpr<'static> {
        AffineExpr::Add(self.a.expr(a), self.a.expr(b))
    }

    /// An inclusive box `[lo, hi]` as an `AffineSet`.
    pub fn box_set(&self, lo: &[i64], hi: &[i64]) -> AffineSet<'static> {
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
}

impl Default for Ops {
    fn default() -> Self {
        Self::new()
    }
}
