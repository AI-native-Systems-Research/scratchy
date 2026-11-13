// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Storage for IR built before it is baked.

use crate::affine::{AffineExpr, AffineMap, Constraint};
use crate::attrkey::AttrKey;
use crate::ir::{Attr, IRFunction, Operation, Ssa};
use crate::irtype::IrType;

/// One function argument as a store holds it: the SSA name and its type.
type Arg<'a> = (Ssa, IrType<'a>);

/// One attribute as a store holds it: the key and its value.
type KeyedAttr<'a> = (AttrKey, Attr<'a>);

/// One heap allocation the arena keeps for its whole life and hands out `&'a` borrows INTO.
///
/// Spelled as its own type rather than a bare `Box` because the box is LOAD-BEARING, not
/// incidental: a `Box`'s heap contents keep their address when the backing `Vec` reallocates, and
/// that fixed address is the entire reason a value is boxed here instead of stored inline in the
/// `Vec`. Inline storage would move on the next `push` and dangle every borrow already handed out.
struct Pinned<T: ?Sized>(Box<T>);

impl<T: ?Sized> Pinned<T> {
    /// The address these contents keep for as long as the arena holds them.
    fn addr(&self) -> *const T {
        &*self.0
    }
}

/// Storage for IR built at COMPILE TIME — the lowering and the optimizer both run inside
/// `#[forward]`, never at runtime — handing out the `&'a` borrows [`Operation`] is made of.
///
/// A BAKED program is `'static`: const literals the macro emits into the binary, which is what
/// `inventory::submit!` requires. A program still being built borrows from one of these. One type
/// serves both, so nothing is converted between a "build" form and a "baked" form.
///
/// One store per element type rather than `Box<dyn Any>`, because the IR is SELF-REFERENTIAL —
/// an `Operation<'a>` held by the arena borrows from that same arena — and `dyn Any` would demand
/// `'static` of exactly the types that cannot be.
#[derive(Default)]
pub struct Arena {
    strs: std::sync::Mutex<Vec<Pinned<str>>>,
    strn: std::sync::Mutex<Vec<Pinned<[&'static str]>>>,
    ssas: std::sync::Mutex<Vec<Pinned<[Ssa]>>>,
    exprs: std::sync::Mutex<Vec<Pinned<AffineExpr<'static>>>>,
    exprv: std::sync::Mutex<Vec<Pinned<[AffineExpr<'static>]>>>,
    cons: std::sync::Mutex<Vec<Pinned<[Constraint<'static>]>>>,
    ints: std::sync::Mutex<Vec<Pinned<[i64]>>>,
    flts: std::sync::Mutex<Vec<Pinned<[f64]>>>,
    maps: std::sync::Mutex<Vec<Pinned<[AffineMap<'static>]>>>,
    args: std::sync::Mutex<Vec<Pinned<[Arg<'static>]>>>,
    attrs: std::sync::Mutex<Vec<Pinned<[KeyedAttr<'static>]>>>,
    ops: std::sync::Mutex<Vec<Pinned<[Operation<'static>]>>>,
    regs: std::sync::Mutex<Vec<Pinned<[&'static [Operation<'static>]]>>>,
    funs: std::sync::Mutex<Vec<Pinned<[IRFunction<'static>]>>>,
}

/// The stores are only ever pushed to, and nothing between the `lock` and the `push` can panic, so a
/// poisoned arena is unreachable rather than merely unlikely.
const POISON: &str = "the compile-time arena cannot be poisoned: nothing between `lock` and `push` \
                      can panic";

/// Store `v` in `slot` and hand back a borrow tied to the arena.
///
/// Two arms, because only SOME element types carry the arena's lifetime:
///
/// * `keep!(self, slot, v)` — the element type has no lifetime parameter at all (`Ssa`, `i64`,
///   `f64`), so the store's type is already the one the caller has. Nothing to erase, and so no
///   `unsafe` cast at all.
/// * `keep!(self, slot, v, From<'a> => To<'static>)` — the element type carries `'a`, which is
///   erased to the `'static` the store is spelled with. `$from` and `$to` differ in NOTHING but
///   that lifetime, and both are named at the call site so the erasure is spelled out rather than
///   inferred.
///
/// SAFETY: the box is owned by the arena for its whole life and never handed out mutably, and
/// [`Pinned`] is why its address holds. The `'static` on the stores is the erased form of the
/// arena's own lifetime — every borrow returned here is re-tied to `&self`, and the pointer it is
/// rebuilt from is taken BEFORE the erasure, so it carries the caller's `'a` rather than `'static`.
macro_rules! keep {
    ($self:ident, $slot:ident, $v:expr) => {{
        let pinned = Pinned($v.into_boxed_slice());
        let ptr = pinned.addr();
        $self.$slot.lock().expect(POISON).push(pinned);
        // SAFETY: as above.
        unsafe { &*ptr }
    }};
    ($self:ident, $slot:ident, $v:expr, $from:ty => $to:ty) => {{
        let boxed: Box<[$from]> = $v.into_boxed_slice();
        let ptr: *const [$from] = &*boxed;
        // SAFETY: as above.
        let pinned = Pinned(unsafe { std::mem::transmute::<Box<[$from]>, Box<[$to]>>(boxed) });
        $self.$slot.lock().expect(POISON).push(pinned);
        // SAFETY: as above.
        unsafe { &*ptr }
    }};
}

impl Arena {
    pub fn new() -> Arena {
        Arena::default()
    }

    /// ⭐ THE ONE ARENA A BUILD HAS, and the reason a built program is spelled `IRFunction<'static>`.
    ///
    /// `#[forward]` expands once per process and every program it builds must outlive the collector
    /// the macro drains, so the storage those programs borrow from is the process itself. Handing out
    /// `&'static Arena` is what makes a program a `static` — the exact shape `inventory::submit!`
    /// stores — with no second "baked" form to convert into.
    pub fn global() -> &'static Arena {
        static A: std::sync::OnceLock<Arena> = std::sync::OnceLock::new();
        A.get_or_init(Arena::new)
    }

    /// Keep `s` alive for the arena's life and hand back a borrow of it.
    pub fn str(&self, s: String) -> &str {
        let pinned = Pinned(s.into_boxed_str());
        let ptr: *const str = pinned.addr();
        self.strs.lock().expect(POISON).push(pinned);
        // SAFETY: as `keep!`.
        unsafe { &*ptr }
    }

    pub fn names<'a>(&'a self, v: Vec<&'a str>) -> &'a [&'a str] {
        keep!(self, strn, v, &'a str => &'static str)
    }

    pub fn ssa(&self, v: Vec<Ssa>) -> &[Ssa] {
        keep!(self, ssas, v)
    }

    pub fn ints(&self, v: Vec<i64>) -> &[i64] {
        keep!(self, ints, v)
    }

    pub fn floats(&self, v: Vec<f64>) -> &[f64] {
        keep!(self, flts, v)
    }

    pub fn maps<'a>(&'a self, v: Vec<AffineMap<'a>>) -> &'a [AffineMap<'a>] {
        keep!(self, maps, v, AffineMap<'a> => AffineMap<'static>)
    }

    pub fn args<'a>(&'a self, v: Vec<Arg<'a>>) -> &'a [Arg<'a>] {
        keep!(self, args, v, Arg<'a> => Arg<'static>)
    }

    pub fn attrs<'a>(&'a self, v: Vec<KeyedAttr<'a>>) -> &'a [KeyedAttr<'a>] {
        keep!(self, attrs, v, KeyedAttr<'a> => KeyedAttr<'static>)
    }

    pub fn ops<'a>(&'a self, v: Vec<Operation<'a>>) -> &'a [Operation<'a>] {
        keep!(self, ops, v, Operation<'a> => Operation<'static>)
    }

    pub fn regions<'a>(&'a self, v: Vec<&'a [Operation<'a>]>) -> &'a [&'a [Operation<'a>]] {
        keep!(self, regs, v, &'a [Operation<'a>] => &'static [Operation<'static>])
    }

    /// Keep a launch group's whole programs alive — the form a bundle carries them in.
    pub fn funcs<'a>(&'a self, v: Vec<IRFunction<'a>>) -> &'a [IRFunction<'a>] {
        keep!(self, funs, v, IRFunction<'a> => IRFunction<'static>)
    }
}

impl Arena {
    /// Keep one affine expression alive and hand back a borrow — what the symbolic-bound builders
    /// need now that a node's children are borrows rather than refcounts.
    pub fn expr<'a>(&'a self, e: AffineExpr<'a>) -> &'a AffineExpr<'a> {
        let boxed: Box<AffineExpr<'a>> = Box::new(e);
        let ptr: *const AffineExpr<'a> = &*boxed;
        // SAFETY: as `keep!` — the only difference between the two types is the lifetime.
        let pinned = Pinned(unsafe {
            std::mem::transmute::<Box<AffineExpr<'a>>, Box<AffineExpr<'static>>>(boxed)
        });
        self.exprs.lock().expect(POISON).push(pinned);
        // SAFETY: as `keep!`.
        unsafe { &*ptr }
    }

    pub fn exprs<'a>(&'a self, v: Vec<AffineExpr<'a>>) -> &'a [AffineExpr<'a>] {
        keep!(self, exprv, v, AffineExpr<'a> => AffineExpr<'static>)
    }

    pub fn constraints<'a>(&'a self, v: Vec<Constraint<'a>>) -> &'a [Constraint<'a>] {
        keep!(self, cons, v, Constraint<'a> => Constraint<'static>)
    }
}
