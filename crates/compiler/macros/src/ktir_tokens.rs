// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ THE PROGRAM, AS CONST TOKENS.
//!
//! A launch group carries the `IRFunction`s it runs, and `#[forward]` bakes them into the binary
//! through `inventory::submit!`. That is what these emit: every operation, attribute, operand and
//! nested region rendered as a Rust literal of `ktir-core`'s own types.
//!
//! ⛔ NOTHING IS SERIALIZED AND NOTHING IS PARSED. The value the lowering constructed and the value
//! the device executes are the same value; these tokens are how it survives the compile, not an
//! interchange format. There is no text form of a program at any point.
//!
//! ⭐ THE ENUM SPELLINGS ARE THE COMPILER'S TO CHECK. A variant is emitted by its own `Debug` name
//! and lands in the generated source as a path, so a variant that does not exist is a build error
//! at the use site rather than a string nobody validates.

use quote::quote;

/// ⭐ THE ALIASES THE RENDERED PROGRAMS USE. Every operation names its kind, its operands' `Ssa`,
/// its attribute keys and its result type, so a fully-qualified path per reference is the dominant
/// cost of the emission: `::scratchy_target_spyre::ktir::ir::Ssa` is nine tokens and forty
/// characters wrapping one integer, and there are over a million of them.
///
/// MEASURED on llama-3.2-1b: 266,503 rendered operations. Aliasing collapses each path to a single
/// token, which is a ~5x cut in both the generated source's size and the token count rustc parses
/// and name-resolves.
pub fn preamble() -> proc_macro2::TokenStream {
    quote! {
        #[allow(unused_imports)]
        use ::scratchy_target_spyre::ktir::{
            affine::{AffineExpr as __KExpr, AffineMap as __KMap},
            attrkey::AttrKey as __KAttrKey,
            dtypes::DType as __KDTy,
            ir::{Attr as __KAttr, IRFunction as __KFn, Operation as __KOp, Ssa as __KSsa},
            irtype::IrType as __KTy,
            opkind::OpKind as __KOpKind,
        };
        #[allow(unused_imports)]
        use ::scratchy_target_spyre::bundle_code::{
            LaunchProgram as __KProg, PlaceId as __KPlace, SynthRole as __KRole,
        };
        #[allow(unused_imports)]
        use ::std::borrow::Cow::Borrowed as __KCow;
    }
}

fn variant(name: String) -> proc_macro2::TokenStream {
    let id = proc_macro2::Ident::new(&name, proc_macro2::Span::call_site());
    quote! { #id }
}

fn ssa(s: ktir_core::ir::Ssa) -> proc_macro2::TokenStream {
    let n = proc_macro2::Literal::u32_unsuffixed(s.0);
    quote! { __KSsa(#n) }
}

fn dtype(d: ktir_core::dtypes::DType) -> proc_macro2::TokenStream {
    let v = variant(format!("{d:?}"));
    quote! { __KDTy::#v }
}

fn opkind(k: ktir_core::opkind::OpKind) -> proc_macro2::TokenStream {
    let v = variant(format!("{k:?}"));
    quote! { __KOpKind::#v }
}

fn attrkey(k: ktir_core::attrkey::AttrKey) -> proc_macro2::TokenStream {
    let v = variant(format!("{k:?}"));
    quote! { __KAttrKey::#v }
}

fn ints(v: &[i64]) -> proc_macro2::TokenStream {
    let it = v.iter().map(|i| proc_macro2::Literal::i64_unsuffixed(*i));
    quote! { &[#(#it),*] }
}

fn ir_type(t: ktir_core::irtype::IrType<'_>) -> proc_macro2::TokenStream {
    use ktir_core::irtype::IrType;
    match t {
        IrType::Tensor { dims, elem } => {
            let (d, e) = (ints(dims), dtype(elem));
            quote! { __KTy::Tensor { dims: #d, elem: #e } }
        }
        IrType::MemRef { dims, elem } => {
            let (d, e) = (ints(dims), dtype(elem));
            quote! { __KTy::MemRef { dims: #d, elem: #e } }
        }
        IrType::AccessTile { dims } => {
            let d = ints(dims);
            quote! { __KTy::AccessTile { dims: #d } }
        }
        IrType::Index => quote! { __KTy::Index },
        IrType::Scalar(e) => {
            let e = dtype(e);
            quote! { __KTy::Scalar(#e) }
        }
    }
}

fn affine_expr(e: &ktir_core::affine::AffineExpr<'_>) -> proc_macro2::TokenStream {
    use ktir_core::affine::AffineExpr as E;
    let path = quote! { __KExpr };
    match e {
        E::Dim(i) => {
            let i = proc_macro2::Literal::usize_unsuffixed(*i);
            quote! { #path::Dim(#i) }
        }
        E::Sym(i) => {
            let i = proc_macro2::Literal::usize_unsuffixed(*i);
            quote! { #path::Sym(#i) }
        }
        E::Const(c) => {
            let c = proc_macro2::Literal::i64_unsuffixed(*c);
            quote! { #path::Const(#c) }
        }
        E::Ref(s) => quote! { #path::Ref(#s) },
        E::Neg(a) => {
            let a = affine_expr(a);
            quote! { #path::Neg(&#a) }
        }
        E::Add(a, b) => {
            let (a, b) = (affine_expr(a), affine_expr(b));
            quote! { #path::Add(&#a, &#b) }
        }
        E::Sub(a, b) => {
            let (a, b) = (affine_expr(a), affine_expr(b));
            quote! { #path::Sub(&#a, &#b) }
        }
        E::Mul(a, b) => {
            let (a, b) = (affine_expr(a), affine_expr(b));
            quote! { #path::Mul(&#a, &#b) }
        }
        E::FloorDiv(a, b) => {
            let (a, b) = (affine_expr(a), affine_expr(b));
            quote! { #path::FloorDiv(&#a, &#b) }
        }
        E::Mod(a, b) => {
            let (a, b) = (affine_expr(a), affine_expr(b));
            quote! { #path::Mod(&#a, &#b) }
        }
        E::Max(a, b) => {
            let (a, b) = (affine_expr(a), affine_expr(b));
            quote! { #path::Max(&#a, &#b) }
        }
        E::Min(a, b) => {
            let (a, b) = (affine_expr(a), affine_expr(b));
            quote! { #path::Min(&#a, &#b) }
        }
    }
}

fn affine_map(m: &ktir_core::affine::AffineMap<'_>) -> proc_macro2::TokenStream {
    let nd = proc_macro2::Literal::usize_unsuffixed(m.num_dims);
    let ns = proc_macro2::Literal::usize_unsuffixed(m.num_syms);
    let ex = m.exprs.iter().map(affine_expr);
    quote! {
        __KMap {
            num_dims: #nd,
            num_syms: #ns,
            exprs: &[#(#ex),*],
        }
    }
}

fn attr(a: &ktir_core::ir::Attr<'_>) -> proc_macro2::TokenStream {
    use ktir_core::ir::Attr as A;
    let path = quote! { __KAttr };
    match a {
        A::Int(i) => {
            let i = proc_macro2::Literal::i64_unsuffixed(*i);
            quote! { #path::Int(#i) }
        }
        A::Float(f) => {
            let f = proc_macro2::Literal::f64_unsuffixed(*f);
            quote! { #path::Float(#f) }
        }
        A::IntList(v) => {
            let v = ints(v);
            quote! { #path::IntList(#v) }
        }
        A::Str(s) => quote! { #path::Str(#s) },
        A::StrList(v) => {
            let it = v.iter();
            quote! { #path::StrList(&[#(#it),*]) }
        }
        A::Bool(b) => quote! { #path::Bool(#b) },
        A::FloatList(v) => {
            let it = v.iter().map(|f| proc_macro2::Literal::f64_unsuffixed(*f));
            quote! { #path::FloatList(&[#(#it),*]) }
        }
        A::Dtype(d) => {
            let d = dtype(*d);
            quote! { #path::Dtype(#d) }
        }
        A::Op(k) => {
            let k = opkind(*k);
            quote! { #path::Op(#k) }
        }
        A::Ssas(v) => {
            let it = v.iter().map(|s| ssa(*s));
            quote! { #path::Ssas(&[#(#it),*]) }
        }
        A::AffineMap(m) => {
            let m = affine_map(m);
            quote! { #path::AffineMap(#m) }
        }
        A::AffineMapList(v) => {
            let it = v.iter().map(affine_map);
            quote! { #path::AffineMapList(&[#(#it),*]) }
        }
        // An affine SET is a constraint system, and nothing this lowering emits carries one: a
        // tile's coordinate set is the full one, which is what its absence means.
        A::AffineSet(_) => {
            panic!("an affine set in a baked program: this lowering emits none")
        }
    }
}

fn operation(op: &ktir_core::ir::Operation<'static>) -> proc_macro2::TokenStream {
    let result = match op.result {
        Some(r) => {
            let r = ssa(r);
            quote! { Some(#r) }
        }
        None => quote! { None },
    };
    let kind = opkind(op.op_type);
    let operands = op.operands.iter().map(|s| ssa(*s));
    let attrs = op.attributes.iter().map(|(k, v)| {
        let (k, v) = (attrkey(*k), attr(v));
        quote! { (#k, #v) }
    });
    let rty = match op.result_type {
        Some(t) => {
            let t = ir_type(t);
            quote! { Some(#t) }
        }
        None => quote! { None },
    };
    let regions = op.regions.iter().map(|r| {
        let ops = r.iter().map(operation);
        quote! { &[#(#ops),*] as &[_] }
    });
    quote! {
        __KOp {
            result: #result,
            op_type: #kind,
            operands: &[#(#operands),*],
            attributes: &[#(#attrs),*],
            result_type: #rty,
            regions: &[#(#regions),*],
        }
    }
}

/// One program of a launch group: its function, and which placed tensor each parameter carries.
/// ⭐⭐⭐ ONE `const` PER DISTINCT PROGRAM, REFERENCED BY EVERY LAUNCH THAT RUNS IT.
///
/// A program is BUNDLE-INVARIANT: what a launch varies is the ADDRESSES it binds
/// (`LaunchProgram::args`), never the operations. So the same `IRFunction` is reached by every rung
/// of the decode ladder, every prefill rung, and every layer that shares a class — and rendering
/// its operations inline at each of those sites emits the identical item once per site.
///
/// MEASURED on llama-3.2-1b: 33 distinct programs reached from 1,668 launches — 80 copies each —
/// which rendered inline is a 337 MB generated source that rustc chews through single-threaded for
/// ~25 minutes. Interning is not a size optimization on top of a working emission; at that scale
/// the inline form is not a viable emission.
#[derive(Default)]
pub struct ProgramInterner {
    /// Structural hash of the `IRFunction` → its index in `items`. `Operation` hashes by value —
    /// kind, operands, attributes, result type and nested regions — so two entries collapse only
    /// when the programs really are the same program.
    seen: std::collections::HashMap<u64, usize>,
    items: Vec<proc_macro2::TokenStream>,
}

impl ProgramInterner {
    /// The `const` items to place at the top of the emission, in definition order.
    pub fn items(&self) -> &[proc_macro2::TokenStream] {
        &self.items
    }

    fn ident(&mut self, f: &ktir_core::ir::IRFunction<'static>) -> proc_macro2::Ident {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        f.hash(&mut h);
        let key = h.finish();
        let n = self.items.len();
        let idx = *self.seen.entry(key).or_insert(n);
        let id = proc_macro2::Ident::new(
            &format!("__KTIR_PROG_{idx}"),
            proc_macro2::Span::call_site(),
        );
        if idx == n {
            let body = function_tokens(f);
            self.items.push(quote! {
                const #id: __KFn<'static> = #body;
            });
        }
        id
    }
}

/// The function itself — every operation, rendered once.
fn function_tokens(f: &ktir_core::ir::IRFunction<'static>) -> proc_macro2::TokenStream {
    let name = f.name;
    let args = f.arguments.iter().map(|(s, t)| {
        let (s, t) = (ssa(*s), ir_type(*t));
        quote! { (#s, #t) }
    });
    let ops = f.operations.iter().map(operation);
    let (gx, gy, gz) = f.grid;
    let [gx, gy, gz] = [gx, gy, gz].map(proc_macro2::Literal::usize_unsuffixed);
    quote! {
        __KFn {
            name: #name,
            arguments: &[#(#args),*],
            operations: &[#(#ops),*],
            grid: (#gx, #gy, #gz),
            return_type: None,
        }
    }
}

pub fn program_tokens(
    p: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::LaunchProgram<'static>,
    interner: &mut ProgramInterner,
) -> proc_macro2::TokenStream {
    let func = interner.ident(&p.func);
    let binds = p.args.iter().map(|(s, id)| {
        let (s, id) = (ssa(*s), place_id_tokens(*id));
        quote! { (#s, #id) }
    });
    quote! {
        __KProg {
            func: #func,
            args: __KCow(&[#(#binds),*]),
        }
    }
}

/// The identity a parameter's address is bound through.
fn place_id_tokens(
    id: scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::PlaceId,
) -> proc_macro2::TokenStream {
    use scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::PlaceId;
    match id {
        PlaceId::Act(t) => {
            let t = proc_macro2::Literal::u32_unsuffixed(t);
            quote! { __KPlace::Act(#t) }
        }
        PlaceId::Synth { of, role } => {
            let of = proc_macro2::Literal::u32_unsuffixed(of);
            let r = synth_role_tokens(role);
            quote! { __KPlace::Synth { of: #of, role: #r } }
        }
    }
}

fn synth_role_tokens(
    role: scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::SynthRole,
) -> proc_macro2::TokenStream {
    use scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::SynthRole as R;
    let path = quote! { __KRole };
    // The four INDEXED roles carry a block number; every other is a unit variant, emitted by name.
    match role {
        R::Blk(i) => {
            let i = proc_macro2::Literal::u32_unsuffixed(i);
            quote! { #path::Blk(#i) }
        }
        R::Acc(i) => {
            let i = proc_macro2::Literal::u32_unsuffixed(i);
            quote! { #path::Acc(#i) }
        }
        R::LBlk(i) => {
            let i = proc_macro2::Literal::u32_unsuffixed(i);
            quote! { #path::LBlk(#i) }
        }
        R::LAcc(i) => {
            let i = proc_macro2::Literal::u32_unsuffixed(i);
            quote! { #path::LAcc(#i) }
        }
        other => {
            let v = variant(format!("{other:?}"));
            quote! { #path::#v }
        }
    }
}
