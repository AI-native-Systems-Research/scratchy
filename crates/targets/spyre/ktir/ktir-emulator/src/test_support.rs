// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Shared `#[cfg(test)]` IR-construction helpers.
//!
//! The dialect unit tests predate the arena-owned, typed-SSA `Operation` (they
//! built ops from string names — `Operation::new(Some("%r"), "linalg.fill",
//! &["%s"])` — against a textual MLIR parser that no longer exists). This
//! module gives tests back the %-name ergonomics WITHOUT resurrecting that
//! parser: [`Ops::op`] mints a fresh [`Ssa`] the first time a `%name` is used
//! and reuses it on every later use, so a test still reads as a small MLIR-ish
//! program while actually building typed, arena-backed IR.
//!
//! `'static` throughout: `execute_ops`/`execute_region` require
//! `Operation<'static>`, so every builder here goes through
//! [`Arena::global`] — the only arena that supplies it.

use std::collections::HashMap;

use crate::affine::{AffineExpr, AffineMap, AffineSet, Constraint, ConstraintKind};
use crate::arena::Arena;
use crate::attrkey::AttrKey;
use crate::ir::{Attr, Operation, Ssa};
use crate::opkind::OpKind;

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

    /// `%result = kind operands...`. `result: None` for a no-result op
    /// (`linalg.yield`, a store, ...).
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

    pub(crate) fn int_list(&self, v: Vec<i64>) -> Attr<'static> {
        Attr::IntList(self.a.ints(v))
    }

    pub(crate) fn map_list(&self, v: Vec<AffineMap<'static>>) -> Attr<'static> {
        Attr::AffineMapList(self.a.maps(v))
    }

    pub(crate) fn str_list(&self, items: &[&str]) -> Attr<'static> {
        let owned: Vec<&'static str> = items.iter().map(|s| self.a.str((*s).to_string())).collect();
        Attr::StrList(self.a.names(owned))
    }

    /// A `%name`-referencing attribute (e.g. `linalg.reduce`'s `outs_var`) —
    /// resolves the same %-name table [`Ops::op`] mints from.
    pub(crate) fn ssas_attr(&mut self, names: &[&'static str]) -> Attr<'static> {
        let ssas = self.ssas(names);
        Attr::Ssas(self.a.ssa(ssas))
    }

    /// Attach ONE region (a straight-line op list) to `op` — the inline-block
    /// combiner shape most dialect tests use (`(%in, %out) { ...; yield }`).
    /// For a multi-region op, build with [`Ops::regions`] instead.
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

    /// A rank-`num_dims` affine map selecting/permuting dims by index — e.g.
    /// `perm_map(2, &[1, 0])` spells `(d0,d1) -> (d1,d0)`, and a sub-rank
    /// `perm` (e.g. `&[2, 1]` at `num_dims=3`) spells a projection.
    pub(crate) fn sub(
        &self,
        a: AffineExpr<'static>,
        b: AffineExpr<'static>,
    ) -> AffineExpr<'static> {
        AffineExpr::Sub(self.a.expr(a), self.a.expr(b))
    }

    pub(crate) fn mul(
        &self,
        a: AffineExpr<'static>,
        b: AffineExpr<'static>,
    ) -> AffineExpr<'static> {
        AffineExpr::Mul(self.a.expr(a), self.a.expr(b))
    }

    pub(crate) fn add_expr(
        &self,
        a: AffineExpr<'static>,
        b: AffineExpr<'static>,
    ) -> AffineExpr<'static> {
        AffineExpr::Add(self.a.expr(a), self.a.expr(b))
    }

    /// An inclusive box `[lo, hi]` as an `AffineSet` over `lo.len()` dims: for
    /// each axis `i`, `d_i - lo_i >= 0` and `hi_i - d_i >= 0`.
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
}
