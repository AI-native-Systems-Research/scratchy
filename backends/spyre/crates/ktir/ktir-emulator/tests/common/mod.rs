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

use std::collections::HashMap;

use ktir_emulator::arena::Arena;
use ktir_emulator::attrkey::AttrKey;
use ktir_emulator::ir::{Attr, Operation, Ssa};
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
}

impl Default for Ops {
    fn default() -> Self {
        Self::new()
    }
}
