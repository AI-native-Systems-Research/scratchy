// SPDX-License-Identifier: Apache-2.0
//! A MODEL'S DIMS AND WEIGHTS AS TYPES.
//!
//! The bake emits, per model, one marker type per config bound its weights name, carrying the
//! model's number ([`Dim`]); a product or sum of bounds is [`Mul`] / [`Add`] over them, spelled as
//! the carrier spells it. A loaded weight is a handle at its declared shape: [`W1`]`<D, L>` and
//! [`W2`]`<K, N, L>`, `L` the device layer it holds. The one way to make one is [`Mint`]: the
//! loader's layer, its shape checked against the type's numbers.

use std::marker::PhantomData;

/// A dim as a type: its number for the model whose module declares it.
pub trait Dim {
    const N: u32;
}

/// The product `A · B`.
pub struct Mul<A, B>(PhantomData<(A, B)>);

impl<A: Dim, B: Dim> Dim for Mul<A, B> {
    const N: u32 = A::N * B::N;
}

/// The sum `A + B`: a packed projection's parts side by side.
pub struct Add<A, B>(PhantomData<(A, B)>);

impl<A: Dim, B: Dim> Dim for Add<A, B> {
    const N: u32 = A::N + B::N;
}

/// The quotient `A / B`.
pub struct Div<A, B>(PhantomData<(A, B)>);

impl<A: Dim, B: Dim> Dim for Div<A, B> {
    const N: u32 = A::N / B::N;
}

/// A literal dim.
pub enum Lit<const V: u32> {}

impl<const V: u32> Dim for Lit<V> {
    const N: u32 = V;
}

/// A device layer's extents, in the order its weight's declared shape states them.
pub trait Extents<const R: usize> {
    fn extents(&self) -> [usize; R];
}

/// A rank-1 weight `[D]`, held as `L`.
#[repr(transparent)]
pub struct W1<D, L> {
    layer: L,
    shape: PhantomData<fn() -> D>,
}

/// A rank-2 weight `[K, N]`, held as `L`.
#[repr(transparent)]
pub struct W2<K, N, L> {
    layer: L,
    shape: PhantomData<fn() -> (K, N)>,
}

/// Two weights packed side by side along `N` (a gated MLP's gate and up projections, one
/// buffer): `[K, A]` and `[K, B]` pack to `[K, A + B]`. Only weights of one `K` pack — two
/// that differ have no `Out`.
///
/// ```
/// # use scratchy_tensors::typed::{Add, Dim, PackedN, W2};
/// # enum Hidden {}
/// # impl Dim for Hidden { const N: u32 = 2048; }
/// # enum Inter {}
/// # impl Dim for Inter { const N: u32 = 8192; }
/// let _: Option<<(W2<Hidden, Inter, ()>, W2<Hidden, Inter, ()>) as PackedN>::Out> =
///     None::<W2<Hidden, Add<Inter, Inter>, ()>>;
/// ```
///
/// ```compile_fail,E0277
/// # use scratchy_tensors::typed::{Add, Dim, PackedN, W2};
/// # enum Hidden {}
/// # impl Dim for Hidden { const N: u32 = 2048; }
/// # enum Inter {}
/// # impl Dim for Inter { const N: u32 = 8192; }
/// let _: Option<<(W2<Hidden, Inter, ()>, W2<Inter, Inter, ()>) as PackedN>::Out> =
///     None::<W2<Hidden, Add<Inter, Inter>, ()>>;
/// ```
pub trait PackedN {
    type Out;
}

impl<K, A, B, L> PackedN for (W2<K, A, L>, W2<K, B, L>) {
    type Out = W2<K, Add<A, B>, L>;
}

/// A rank-2 weight's dims, as types.
pub trait Shape2 {
    type K: Dim;
    type N: Dim;
}

impl<K: Dim, N: Dim, L> Shape2 for W2<K, N, L> {
    type K = K;
    type N = N;
}

/// A rank-1 weight's dim, as a type.
pub trait Shape1 {
    type D: Dim;
}

impl<D: Dim, L> Shape1 for W1<D, L> {
    type D = D;
}

impl<D, L> std::ops::Deref for W1<D, L> {
    type Target = L;
    fn deref(&self) -> &L {
        &self.layer
    }
}

impl<K, N, L> std::ops::Deref for W2<K, N, L> {
    type Target = L;
    fn deref(&self) -> &L {
        &self.layer
    }
}

/// A loaded layer whose extents are not its type's.
#[derive(Debug)]
pub struct ShapeMismatch {
    pub tensor: String,
    pub loaded: Vec<usize>,
    pub declared: Vec<usize>,
}

impl std::fmt::Display for ShapeMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "weight `{}` loaded at {:?}, its type declares {:?}",
            self.tensor, self.loaded, self.declared
        )
    }
}

impl std::error::Error for ShapeMismatch {}

/// ⛔ THE ONE DOOR from a loaded layer to a typed handle: `layer`'s extents checked against the
/// type's numbers, the mismatch naming `tensor`. A layered field mints each layer's.
pub trait Mint<L>: Sized {
    fn mint(layer: L, tensor: &str) -> anyhow::Result<Self>;
}

/// The one check every handle type's mint runs: each of `layers`' extents against `declared`,
/// the first mismatch naming `tensor` (and, for a layered field, its layer). Generic over the
/// layer type alone, so the binary holds one copy per device layer type, not one per handle.
#[inline(never)]
fn check_layers<const R: usize, L: Extents<R>>(
    layers: &[L],
    declared: [u32; R],
    tensor: &str,
    layered: bool,
) -> anyhow::Result<()> {
    let declared = declared.map(|d| d as usize);
    let Some((at, loaded)) =
        (layers.iter().map(Extents::extents).enumerate()).find(|(_, loaded)| *loaded != declared)
    else {
        return Ok(());
    };
    Err(anyhow::Error::new(ShapeMismatch {
        tensor: match layered {
            true => format!("{tensor}[{at}]"),
            false => tensor.to_string(),
        },
        loaded: loaded.to_vec(),
        declared: declared.to_vec(),
    }))
}

/// A handle over layer type `L`: its check (its numbers handed to [`check_layers`]) and its
/// wrap. Both mints read it.
pub trait Shaped<L>: Sized {
    fn check(layers: &[L], tensor: &str, layered: bool) -> anyhow::Result<()>;
    fn wrap(layer: L) -> Self;
}

impl<D: Dim, L: Extents<1>> Shaped<L> for W1<D, L> {
    fn check(layers: &[L], tensor: &str, layered: bool) -> anyhow::Result<()> {
        check_layers(layers, [D::N], tensor, layered)
    }

    fn wrap(layer: L) -> Self {
        Self {
            layer,
            shape: PhantomData,
        }
    }
}

impl<K: Dim, N: Dim, L: Extents<2>> Shaped<L> for W2<K, N, L> {
    fn check(layers: &[L], tensor: &str, layered: bool) -> anyhow::Result<()> {
        check_layers(layers, [K::N, N::N], tensor, layered)
    }

    fn wrap(layer: L) -> Self {
        Self {
            layer,
            shape: PhantomData,
        }
    }
}

impl<D: Dim, L: Extents<1>> Mint<L> for W1<D, L> {
    fn mint(layer: L, tensor: &str) -> anyhow::Result<Self> {
        Self::check(std::slice::from_ref(&layer), tensor, false)?;
        Ok(Self::wrap(layer))
    }
}

impl<K: Dim, N: Dim, L: Extents<2>> Mint<L> for W2<K, N, L> {
    fn mint(layer: L, tensor: &str) -> anyhow::Result<Self> {
        Self::check(std::slice::from_ref(&layer), tensor, false)?;
        Ok(Self::wrap(layer))
    }
}

/// A layered field: every layer checked, then every layer wrapped.
impl<T: Shaped<L>, L> Mint<Vec<L>> for Vec<T> {
    fn mint(layers: Vec<L>, tensor: &str) -> anyhow::Result<Self> {
        T::check(&layers, tensor, true)?;
        Ok(layers.into_iter().map(T::wrap).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    enum Hidden {}
    impl Dim for Hidden {
        const N: u32 = 2048;
    }
    enum Vocab {}
    impl Dim for Vocab {
        const N: u32 = 128256;
    }

    /// A layer of the extents it states.
    struct Layer([usize; 2]);
    impl Extents<2> for Layer {
        fn extents(&self) -> [usize; 2] {
            self.0
        }
    }

    #[test]
    fn the_mint_takes_a_layer_of_its_types_extents() {
        let w = W2::<Hidden, Vocab, Layer>::mint(Layer([2048, 128256]), "lm_head");
        assert_eq!(w.map(|w| w.extents()).ok(), Some([2048, 128256]));
    }

    #[test]
    fn the_mint_refuses_a_layer_of_other_extents_naming_its_tensor() {
        let e = W2::<Hidden, Vocab, Layer>::mint(Layer([128256, 2048]), "lm_head");
        assert_eq!(
            e.err().map(|e| e.to_string()).as_deref(),
            Some("weight `lm_head` loaded at [128256, 2048], its type declares [2048, 128256]")
        );
    }

    #[test]
    fn a_layered_field_names_the_layer_it_refuses() {
        let layers = vec![Layer([2048, 128256]), Layer([2048, 4])];
        let e = <Vec<W2<Hidden, Vocab, Layer>>>::mint(layers, "lm_head");
        assert_eq!(
            e.err()
                .and_then(|e| e.downcast::<ShapeMismatch>().ok())
                .map(|e| e.tensor)
                .as_deref(),
            Some("lm_head[1]")
        );
    }
}
