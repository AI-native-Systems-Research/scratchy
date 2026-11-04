// SPDX-License-Identifier: Apache-2.0
//! SHAPES AS RUST TYPES: the op signatures as generic Rust functions, and every carrier the
//! typed-Rust emit pass (`typed_emit.rs`, run by the build script) turned into a Rust function over them.
//!
//! A dim is a type: a config bound is a marker named for it (`HeadDim` for `head_dim`), a product
//! is [`Mul`] in the one canonical order [`canonical_mul`] writes. An activation is [`Act<T, D>`]:
//! `T` rows (the token axis) of `D` features. A weight is [`W1`]/[`W2`] at the shape its arch's
//! `weights.json` declares. So `gemm(x: Act<T, K>, w: W2<K, N>) -> Act<T, N>` is the law
//! `x`'s last dim equals `w`'s first, and rustc checks it where a carrier calls it: a shape error
//! in a carrier is a type error at the generated line, which names the carrier's file and line.
//!
//! The emitted functions also RUN: [`Trace`] records, per [`LocalId`], the shape each binding's
//! type states. That record is the carrier's shapes, symbolic in its config's bounds. Until phase
//! C of the shapes-as-Rust-types plan (issue #1086) makes it the source, the bake checks it against
//! `shape::infer`'s ([`check_parity`]).

use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;

use crate::classified::LocalId;
use crate::dim::{Dim, canonical_mul};

/// A dim as a type: the symbolic [`Dim`] it stands for.
pub trait DimType {
    /// The dim.
    fn dim() -> Dim;
}

/// A config bound as a type: the bound's name.
pub trait Bound {
    /// The config's name for it.
    const NAME: &'static str;
}

/// The marker types of config bounds, each `Name => "config_name"`.
macro_rules! bounds {
    ($($marker:ident => $name:literal),* $(,)?) => {$(
        #[doc = concat!("The `", $name, "` config bound, as a type.")]
        pub enum $marker {}
        impl Bound for $marker {
            const NAME: &'static str = $name;
        }
        impl DimType for $marker {
            fn dim() -> Dim {
                Dim::Bound($name.to_string())
            }
        }
    )*};
}

/// The product `A · B`. Products are written in [`canonical_mul`]'s order, so equal products are
/// equal types.
pub struct Mul<A, B>(PhantomData<(A, B)>);

impl<A: DimType, B: DimType> DimType for Mul<A, B> {
    fn dim() -> Dim {
        canonical_mul(vec![A::dim(), B::dim()])
    }
}

/// Phantom values are copies whatever their parameters: a shape handle holds no data.
macro_rules! phantom_copy {
    ($($t:ident<$($p:ident),*>),* $(,)?) => {$(
        impl<$($p),*> Clone for $t<$($p),*> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<$($p),*> Copy for $t<$($p),*> {}
    )*};
}

/// An activation: `T` rows of `D` features each.
pub struct Act<T, D>(PhantomData<(T, D)>);
/// One index per row: `[T]` (token ids, positions).
pub struct Tokens<T>(PhantomData<T>);
/// A rank-1 weight `[D]`.
pub struct W1<D>(PhantomData<D>);
/// A rank-2 weight `[K, N]`.
pub struct W2<K, N>(PhantomData<(K, N)>);

phantom_copy!(Act<T, D>, Tokens<T>, W1<D>, W2<K, N>);

/// A weight at the shape its type declares.
pub trait Declared {
    /// The weight.
    fn declared() -> Self;
}

impl<D> Declared for W1<D> {
    fn declared() -> Self {
        Self(PhantomData)
    }
}

impl<K, N> Declared for W2<K, N> {
    fn declared() -> Self {
        Self(PhantomData)
    }
}

/// An `if`'s merge: both arms produce the same type, which is the law that their bindings agree.
/// The trace runs both arms, so each arm's locals are recorded; which arm a layer takes changes no
/// shape.
pub fn merge<A>(then_arm: A, _else_arm: A) -> A {
    then_arm
}

/// The rotary tables: opaque to shapes.
#[derive(Clone, Copy)]
pub struct Rotary;
/// One layer's KV cache: opaque to shapes.
#[derive(Clone, Copy)]
pub struct KvCache;
/// The paged block table: opaque to shapes.
#[derive(Clone, Copy)]
pub struct BlockTable;

/// A value whose type states a shape.
pub trait Traced {
    /// The shape its type states.
    fn shape() -> Vec<Dim>;
}

impl<T: DimType, D: DimType> Traced for Act<T, D> {
    fn shape() -> Vec<Dim> {
        vec![T::dim(), D::dim()]
    }
}

impl<T: DimType> Traced for Tokens<T> {
    fn shape() -> Vec<Dim> {
        vec![T::dim()]
    }
}

/// A tuple of traced values, bound to as many locals.
pub trait TracedTuple<const N: usize> {
    /// Each element's shape, in order.
    fn shapes() -> [Vec<Dim>; N];
}

impl<A: Traced, B: Traced, C: Traced> TracedTuple<3> for (A, B, C) {
    fn shapes() -> [Vec<Dim>; 3] {
        [A::shape(), B::shape(), C::shape()]
    }
}

/// The shapes a run of an emitted carrier bound, per local.
#[derive(Debug, Default)]
pub struct Trace {
    shapes: HashMap<LocalId, Vec<Dim>>,
}

impl Trace {
    /// Record `value`'s shape as local `id`'s, and pass it on.
    pub fn bind<A: Traced>(&mut self, id: u32, value: A) -> A {
        self.shapes.insert(LocalId(id), A::shape());
        value
    }

    /// [`Self::bind`], for a tuple bound to the locals `ids`.
    pub fn bind_tuple<const N: usize, A: TracedTuple<N>>(&mut self, ids: [u32; N], value: A) -> A {
        for (id, shape) in ids.into_iter().zip(A::shapes()) {
            self.shapes.insert(LocalId(id), shape);
        }
        value
    }

    /// A loop's body, run once: its locals' shapes are their types', the same every iteration.
    pub fn loop_once(&self) -> std::ops::Range<u32> {
        0..1
    }

    /// The shapes recorded, per local.
    pub fn shapes(&self) -> &HashMap<LocalId, Vec<Dim>> {
        &self.shapes
    }
}

/// A decoder program's inputs at `T` rows, each the field its DSL name names (`ExternKind::name`).
pub struct Inputs<T> {
    pub input_ids: Tokens<T>,
    pub positions: Tokens<T>,
    pub rotary: Rotary,
    /// `kv_cache[layer]`: every layer's is opaque alike.
    pub kv_cache: KvCache,
    pub block_table: BlockTable,
}

impl<T> Inputs<T> {
    /// The inputs, at their shapes.
    pub fn declared() -> Self {
        Self {
            input_ids: Tokens(PhantomData),
            positions: Tokens(PhantomData),
            rotary: Rotary,
            kv_cache: KvCache,
            block_table: BlockTable,
        }
    }
}

/// The op signatures, each named as its `OpKind` is in a carrier.
pub mod ops {
    use super::{Act, BlockTable, HeadDim, KvCache, Mul, NumAttentionHeads, NumKeyValueHeads};
    use super::{PhantomData, Rotary, Tokens, W1, W2};

    /// `embed(ids: [T], table: [V, D]) -> [T, D]`.
    pub fn embed<T, V, D>(_ids: Tokens<T>, _table: W2<V, D>) -> Act<T, D> {
        Act(PhantomData)
    }

    /// `rmsnorm(x: [T, D], w: [D]) -> [T, D]`.
    pub fn rmsnorm<T, D>(x: Act<T, D>, _w: W1<D>) -> Act<T, D> {
        x
    }

    /// `gemm(x: [T, K], w: [K, N]) -> [T, N]`.
    pub fn gemm<T, K, N>(_x: Act<T, K>, _w: W2<K, N>) -> Act<T, N> {
        Act(PhantomData)
    }

    /// `rope_append(q, k, v, positions, rotary, kv_cache)`: q, k and v, each its own shape.
    pub fn rope_append<T, Q, K, V>(
        q: Act<T, Q>,
        k: Act<T, K>,
        v: Act<T, V>,
        _positions: Tokens<T>,
        _rotary: Rotary,
        _kv_cache: KvCache,
    ) -> (Act<T, Q>, Act<T, K>, Act<T, V>) {
        (q, k, v)
    }

    /// `attention(q: [T, num_attention_heads · head_dim], k, v: [T, num_key_value_heads ·
    /// head_dim], kv_cache, block_table) -> [T, num_attention_heads · head_dim]`.
    pub fn attention<T>(
        q: Act<T, Mul<HeadDim, NumAttentionHeads>>,
        _k: Act<T, Mul<HeadDim, NumKeyValueHeads>>,
        _v: Act<T, Mul<HeadDim, NumKeyValueHeads>>,
        _kv_cache: KvCache,
        _block_table: BlockTable,
    ) -> Act<T, Mul<HeadDim, NumAttentionHeads>> {
        q
    }

    /// `add(a: [T, D], b: [T, D]) -> [T, D]`.
    pub fn add<T, D>(a: Act<T, D>, _b: Act<T, D>) -> Act<T, D> {
        a
    }

    /// `a * b`, elementwise: `[T, D] · [T, D] -> [T, D]`.
    pub fn mul<T, D>(a: Act<T, D>, _b: Act<T, D>) -> Act<T, D> {
        a
    }

    /// `silu(x: [T, D]) -> [T, D]`.
    pub fn silu<T, D>(x: Act<T, D>) -> Act<T, D> {
        x
    }

    /// `capture_hidden(x: [T, D]) -> [T, D]`: the rows a drafter reads.
    pub fn capture_hidden<T, D>(x: Act<T, D>) -> Act<T, D> {
        x
    }
}

// The emit pass's output: the marker of every bound the signatures above and the carriers' weights
// name (`typed_emit::SIGNATURE_BOUNDS` and each `weights.json`), one module per carrier, and
// `typed_program`, which names each carrier's traced function by its arch.
include!(concat!(env!("OUT_DIR"), "/typed_carriers.rs"));

/// A traced shape and `shape::infer`'s for the same local, under one model's bounds.
#[derive(Debug)]
pub struct ParityMismatch {
    pub local: LocalId,
    pub name: String,
    pub line: Option<u32>,
    pub traced: Option<Vec<Dim>>,
    pub inferred: Option<Vec<Dim>>,
}

/// TRANSITIONAL — deleted with `shape.rs`'s solver in phase C, when the trace becomes the source.
///
/// The carrier `arch`'s traced shapes against `inferred` (what `shape::infer` closed for one model),
/// local by local, each evaluated under `bounds` with `num_tokens` set. A local `inferred` shapes
/// that `classified` never bound is one reshape recovery added, which the trace (over the
/// classifier's program) cannot name: those are returned apart, as `recovered`.
///
/// `None` when `arch` has no typed program.
pub fn check_parity(
    arch: &str,
    classified: &crate::classified::Program,
    inferred: &HashMap<LocalId, Vec<Dim>>,
    bounds: &BTreeMap<String, u64>,
) -> Option<Parity> {
    let forward = typed_program(arch)?;
    let mut trace = Trace::default();
    forward(&mut trace);
    // A row count no bound equals, so a dim that confused rows with features shows.
    const NUM_TOKENS: u64 = 7919;
    let mut bounds = bounds.clone();
    bounds.insert(NumTokens::NAME.to_string(), NUM_TOKENS);
    let eval = |shape: &Vec<Dim>| -> Option<Vec<u64>> {
        shape
            .iter()
            .map(|d| crate::shape::eval_closed_dim(d, &bounds))
            .collect()
    };
    let mut mismatches = Vec::new();
    let mut compared = 0;
    for id in classified.locals.ids() {
        let traced = trace.shapes().get(&id);
        let inferred = inferred.get(&id);
        let agree = match (traced, inferred) {
            (Some(t), Some(i)) => {
                compared += 1;
                eval(t).is_some() && eval(t) == eval(i)
            }
            (None, None) => true,
            _ => false,
        };
        if !agree {
            mismatches.push(ParityMismatch {
                local: id,
                name: classified.locals.name(id).to_string(),
                line: classified.locals.line(id),
                traced: traced.cloned(),
                inferred: inferred.cloned(),
            });
        }
    }
    let bound: std::collections::HashSet<LocalId> = classified.locals.ids().collect();
    let mut recovered: Vec<LocalId> = (inferred.keys())
        .filter(|id| !bound.contains(id))
        .copied()
        .collect();
    recovered.sort_by_key(|id| id.0);
    Some(Parity {
        compared,
        mismatches,
        recovered,
    })
}

/// [`check_parity`]'s verdict.
#[derive(Debug)]
pub struct Parity {
    /// How many locals both sides shaped, and so were compared.
    pub compared: usize,
    /// Every local whose shapes differ, or that one side shaped and the other did not.
    pub mismatches: Vec<ParityMismatch>,
    /// The locals reshape recovery added: none of the classifier's, so outside the trace.
    pub recovered: Vec<LocalId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each product a signature spells is in [`canonical_mul`]'s order: otherwise it would be a
    /// different type from the same product a `weights.json` declares.
    #[test]
    fn the_signatures_spell_products_canonically() {
        for (written, factors) in [
            (
                <Mul<HeadDim, NumAttentionHeads>>::dim(),
                vec![HeadDim::dim(), NumAttentionHeads::dim()],
            ),
            (
                <Mul<HeadDim, NumKeyValueHeads>>::dim(),
                vec![HeadDim::dim(), NumKeyValueHeads::dim()],
            ),
        ] {
            assert_eq!(written, Dim::Mul(factors));
        }
    }
}
