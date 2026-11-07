// SPDX-License-Identifier: Apache-2.0
//! A MODEL'S DIMS AND WEIGHTS AS TYPES.
//!
//! The bake emits, per model, one marker type per config bound its weights name, carrying the
//! model's number ([`Dim`]); a product or sum of bounds is [`Mul`] / [`Add`] over them, spelled as
//! the carrier spells it. A loaded weight is a handle at its declared shape: [`W1`]`<D, L>` and
//! [`W2`]`<K, N, L>`, `L` the device layer it holds. The one way to make one is [`Mint`]: the
//! loader's layer, its shape checked against the type's numbers.

use std::marker::PhantomData;

/// The type-level unsigned integers a dim's number is: `typenum`, re-exported so the code the bake
/// emits names it through this crate.
pub use typenum;

use typenum::{UInt, UTerm, Unsigned};

/// A dim as a type: its number, as a type (`Num`, a [`typenum`] unsigned), for the model whose
/// module declares it. A law between dims is a bound on their numbers, which rustc checks.
pub trait Dim {
    type Num: Unsigned;
}

/// A dim's number, as a value: derived from its [`Dim::Num`], the one impl every dim has.
pub trait DimN {
    const N: u32;
}

impl<D: Dim> DimN for D {
    const N: u32 = <D::Num as Unsigned>::U32;
}

/// A number is its own dim: a literal in a shape is spelled as the number itself.
impl Dim for UTerm {
    type Num = Self;
}

impl<U: Unsigned, B: typenum::Bit> Dim for UInt<U, B> {
    type Num = Self;
}

/// `n` as the path of its [`typenum`] type, through this crate's re-export: what the bake emits for
/// a number it states as a type.
pub fn unsigned_path(n: u64) -> String {
    let t = "::scratchy_tensors::typed::typenum";
    let width = u64::BITS - n.leading_zeros();
    (0..width).rev().fold(format!("{t}::UTerm"), |acc, bit| {
        let b = if (n >> bit) & 1 == 1 { "B1" } else { "B0" };
        format!("{t}::UInt<{acc}, {t}::{b}>")
    })
}

/// `n` as the path of its [`typenum`] type, by the name `typenum` gives it where it gives one (every
/// number to 1024, and every power of two): a bucket's rows, spelled wherever its arena is.
///
/// ```
/// # use scratchy_tensors::typed::{DimN, unsigned_alias_path};
/// # use scratchy_tensors::typed::typenum::{U4096, U768};
/// assert_eq!(unsigned_alias_path(4096), "::scratchy_tensors::typed::typenum::U4096");
/// assert_eq!(unsigned_alias_path(1536).matches("UInt").count(), 11);
/// assert_eq!((<U4096 as DimN>::N, <U768 as DimN>::N), (4096, 768));
/// ```
pub fn unsigned_alias_path(n: u64) -> String {
    match n <= 1024 || n.is_power_of_two() {
        true => format!("::scratchy_tensors::typed::typenum::U{n}"),
        false => unsigned_path(n),
    }
}

/// The product `A · B`.
pub struct Mul<A, B>(PhantomData<(A, B)>);

impl<A: Dim, B: Dim> Dim for Mul<A, B>
where
    A::Num: core::ops::Mul<B::Num, Output: Unsigned>,
{
    type Num = typenum::Prod<A::Num, B::Num>;
}

/// The sum `A + B`: a packed projection's parts side by side.
pub struct Add<A, B>(PhantomData<(A, B)>);

impl<A: Dim, B: Dim> Dim for Add<A, B>
where
    A::Num: core::ops::Add<B::Num, Output: Unsigned>,
{
    type Num = typenum::Sum<A::Num, B::Num>;
}

/// A number that is zero: the remainder of a division that is exact.
#[diagnostic::on_unimplemented(
    message = "a remainder of `{Self}` is not zero: the division is not exact",
    label = "the divisor does not divide the dividend"
)]
pub trait Zero {}

impl Zero for UTerm {}

/// `Self` divides `A` exactly: the quotient, which only a zero remainder has.
///
/// ```
/// # use scratchy_tensors::typed::{Div, DimN};
/// # use scratchy_tensors::typed::typenum::{U4, U32};
/// assert_eq!(<Div<U32, U4> as DimN>::N, 8);
/// ```
///
/// ```compile_fail,E0277
/// # use scratchy_tensors::typed::{Div, DimN};
/// # use scratchy_tensors::typed::typenum::{U5, U32};
/// let _ = <Div<U32, U5> as DimN>::N;
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not divide `{A}` exactly",
    label = "a quotient needs a divisor that divides its dividend"
)]
pub trait Divides<A: Dim>: Dim {
    type Quot: Unsigned;
}

impl<A: Dim, B: Dim> Divides<A> for B
where
    A::Num: core::ops::Rem<B::Num, Output: Zero> + core::ops::Div<B::Num, Output: Unsigned>,
{
    type Quot = typenum::Quot<A::Num, B::Num>;
}

/// A number that is true: the outcome of a comparison that holds.
#[diagnostic::on_unimplemented(
    message = "the comparison does not hold",
    label = "a dim that must fit in another is wider"
)]
pub trait Holds {}

impl Holds for typenum::B1 {}

/// `Self` fits in `A`: its number is at most `A`'s.
///
/// ```
/// # use scratchy_tensors::typed::FitsIn;
/// # use scratchy_tensors::typed::typenum::{U64, U128};
/// fn fits<R: FitsIn<H>, H>() {}
/// fits::<U64, U128>();
/// ```
///
/// ```compile_fail,E0277
/// # use scratchy_tensors::typed::FitsIn;
/// # use scratchy_tensors::typed::typenum::{U64, U128};
/// fn fits<R: FitsIn<H>, H>() {}
/// fits::<U128, U64>();
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not fit in `{A}`",
    label = "a dim that must fit in another is wider"
)]
pub trait FitsIn<A>: Dim {}

impl<A: Dim, R: Dim> FitsIn<A> for R where R::Num: typenum::IsLessOrEqual<A::Num, Output: Holds> {}

/// A number that is true: the outcome of an equality that holds.
#[diagnostic::on_unimplemented(
    message = "the two dims are not equal",
    label = "a dim that must equal another differs from it"
)]
pub trait Same {}

impl Same for typenum::B1 {}

/// `Self` equals `A`: their numbers are one number, however each is spelled.
///
/// ```
/// # use scratchy_tensors::typed::{Add, Equals, Mul};
/// # use scratchy_tensors::typed::typenum::{U2, U3, U4, U10};
/// fn equal<R: Equals<A>, A>() {}
/// equal::<Add<Mul<U2, U3>, U4>, U10>();
/// ```
///
/// ```compile_fail,E0277
/// # use scratchy_tensors::typed::{Add, Equals, Mul};
/// # use scratchy_tensors::typed::typenum::{U2, U3, U4, U11};
/// fn equal<R: Equals<A>, A>() {}
/// equal::<Add<Mul<U2, U3>, U4>, U11>();
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not equal `{A}`",
    label = "a dim that must equal another differs from it"
)]
pub trait Equals<A>: Dim {}

impl<A: Dim, R: Dim> Equals<A> for R where R::Num: typenum::IsEqual<A::Num, Output: Same> {}

/// The quotient `A / B`, exact: a `B` that does not divide `A` has none (E0277).
pub struct Div<A, B>(PhantomData<(A, B)>);

impl<A: Dim, B: Divides<A>> Dim for Div<A, B> {
    type Num = B::Quot;
}

/// The floor quotient `A // B`, a remainder dropped: the one quotient that may have one, as an
/// arch spells a round-up (`(A + 7) // 8 * 8`).
///
/// ```
/// # use scratchy_tensors::typed::{DimN, FloorDiv};
/// # use scratchy_tensors::typed::typenum::{U5, U32};
/// assert_eq!(<FloorDiv<U32, U5> as DimN>::N, 6);
/// ```
pub struct FloorDiv<A, B>(PhantomData<(A, B)>);

impl<A: Dim, B: Dim> Dim for FloorDiv<A, B>
where
    A::Num: core::ops::Div<B::Num, Output: Unsigned>,
{
    type Num = typenum::Quot<A::Num, B::Num>;
}

/// A device layer's extents, in the order its weight's declared shape states them.
pub trait Extents<const R: usize> {
    fn extents(&self) -> [usize; R];
}

/// A raw tensor declared a vector is read as one, whatever its stored rank: its extent is its
/// element count (a checkpoint may store `[8192]` as `[2048, 4]`; the kernel reads it flat).
impl Extents<1> for crate::GpuTensor {
    fn extents(&self) -> [usize; 1] {
        [self.numel()]
    }
}

/// A raw tensor declared a matrix is one: its shape, and none (all zero, which no declared shape
/// is) for a tensor of another rank.
impl Extents<2> for crate::GpuTensor {
    fn extents(&self) -> [usize; 2] {
        match *self.shape() {
            [a, b] => [a as usize, b as usize],
            _ => [0; 2],
        }
    }
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
/// # impl Dim for Hidden { type Num = scratchy_tensors::typed::typenum::U2048; }
/// # enum Inter {}
/// # impl Dim for Inter { type Num = scratchy_tensors::typed::typenum::U8192; }
/// let _: Option<<(W2<Hidden, Inter, ()>, W2<Hidden, Inter, ()>) as PackedN>::Out> =
///     None::<W2<Hidden, Add<Inter, Inter>, ()>>;
/// ```
///
/// ```compile_fail,E0277
/// # use scratchy_tensors::typed::{Add, Dim, PackedN, W2};
/// # enum Hidden {}
/// # impl Dim for Hidden { type Num = scratchy_tensors::typed::typenum::U2048; }
/// # enum Inter {}
/// # impl Dim for Inter { type Num = scratchy_tensors::typed::typenum::U8192; }
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

/// An MLX-affine weight's code width, as a type: `B` bits a weight, its number its [`Dim`]'s.
pub trait Bits: Dim {}

/// An MLX-affine weight's quantization group, as a type: `G` weights a scale and bias, its number
/// its [`Dim`]'s.
pub trait Group: Dim {}

macro_rules! markers {
    ($trait:ident: $($name:ident = $n:ident),+ $(,)?) => {$(
        #[doc = concat!("`", stringify!($n), "`, as a [`", stringify!($trait), "`].")]
        pub enum $name {}
        impl Dim for $name {
            type Num = typenum::$n;
        }
        impl $trait for $name {}
    )+};
}

// Every width and group MLX's affine format writes; which of them a target's kernels take is that
// target's set of these.
markers!(Bits: B2 = U2, B3 = U3, B4 = U4, B5 = U5, B6 = U6, B8 = U8);
markers!(Group: G32 = U32, G64 = U64, G128 = U128);

/// `G` divides `K`: a matrix quantized in groups of `G` along its inner dim `K` holds whole groups.
#[diagnostic::on_unimplemented(
    message = "a group of `{Self}` does not divide the inner dim `{K}`",
    label = "an MLX-affine weight's groups tile its inner dim"
)]
pub trait Tiles<K: Dim>: Group {}

impl<K: Dim, G: Group + Divides<K>> Tiles<K> for G {}

/// A rank-2 MLX-affine weight `[K, N]`, its codes `B` bits wide in groups of `G`, held as `L`.
/// Not a [`Shape2`]: a quantized weight is read by the kernels of its codes alone, never as a
/// dense one. Its groups tile its inner dim ([`Tiles`]): there is no `Q2` whose `G` does not
/// divide its `K`.
///
/// ```
/// # use scratchy_tensors::typed::{B4, DimN, G64, Q2, QShape2};
/// # use scratchy_tensors::typed::typenum::{U1024, U2048};
/// type W = Q2<U2048, U1024, B4, G64, ()>;
/// assert_eq!(<<W as QShape2>::K as DimN>::N / <<W as QShape2>::Group as DimN>::N, 32);
/// ```
///
/// ```compile_fail,E0277
/// # use scratchy_tensors::typed::{B4, DimN, G64, Q2, QShape2};
/// # use scratchy_tensors::typed::typenum::{U1000, U1024};
/// type W = Q2<U1000, U1024, B4, G64, ()>;
/// let _ = <<W as QShape2>::K as DimN>::N;
/// ```
#[repr(transparent)]
pub struct Q2<K: Dim, N, B, G: Tiles<K>, L> {
    layer: L,
    shape: PhantomData<fn() -> (K, N)>,
    codes: PhantomData<fn() -> (B, G)>,
}

/// A rank-2 MLX-affine weight's dims and codes, as types.
pub trait QShape2 {
    type K: Dim;
    type N: Dim;
    type Bits: Bits;
    type Group: Group;
}

impl<K: Dim, N: Dim, B: Bits, G: Tiles<K>, L> QShape2 for Q2<K, N, B, G, L> {
    type K = K;
    type N = N;
    type Bits = B;
    type Group = G;
}

impl<K: Dim, N, B, G: Tiles<K>, L> std::ops::Deref for Q2<K, N, B, G, L> {
    type Target = L;
    fn deref(&self) -> &L {
        &self.layer
    }
}

/// A device layer's MLX-affine codes, `(bits, group)`; `None` for a layer of other storage.
pub trait AffineCodes {
    fn affine_codes(&self) -> Option<(u32, u32)>;
}

/// A weight bundle's role at `[D]`, as a type.
pub struct S1<D>(PhantomData<fn() -> D>);
/// A weight bundle's role at `[K, N]` (a matrix `[in, out]`, a conv `[channels, taps]`).
pub struct S2<K, N>(PhantomData<fn() -> (K, N)>);
/// A weight bundle's role at `[E, K, N]` (an expert bank `[experts, in, out]`).
pub struct S3<E, K, N>(PhantomData<fn() -> E>, PhantomData<fn() -> (K, N)>);

/// A bundle role's declared extents.
pub trait RoleShape {
    fn declared() -> Vec<u32>;
}

impl<D: Dim> RoleShape for S1<D> {
    fn declared() -> Vec<u32> {
        vec![D::N]
    }
}

impl<K: Dim, N: Dim> RoleShape for S2<K, N> {
    fn declared() -> Vec<u32> {
        vec![K::N, N::N]
    }
}

impl<E: Dim, K: Dim, N: Dim> RoleShape for S3<E, K, N> {
    fn declared() -> Vec<u32> {
        vec![E::N, K::N, N::N]
    }
}

/// A bundle's roles' declared extents, in its kind's role order: a tuple of [`RoleShape`]s.
pub trait RoleShapes {
    fn declared() -> Vec<Vec<u32>>;
}

macro_rules! role_shapes {
    ($($t:ident),+) => {
        impl<$($t: RoleShape),+> RoleShapes for ($($t,)+) {
            fn declared() -> Vec<Vec<u32>> {
                vec![$($t::declared()),+]
            }
        }
    };
}
role_shapes!(A);
role_shapes!(A, B);
role_shapes!(A, B, C);
role_shapes!(A, B, C, D);
role_shapes!(A, B, C, D, E);
role_shapes!(A, B, C, D, E, F);
role_shapes!(A, B, C, D, E, F, G);
role_shapes!(A, B, C, D, E, F, G, H);

/// A weight bundle's kind, as a type: its roles are [`BundleKind::roles`].
///
/// [`BundleKind::roles`]: crate::bundle_roles::BundleKind::roles
pub trait BundleKindOf {
    const KIND: crate::bundle_roles::BundleKind;
}

macro_rules! bundle_kinds {
    ($($marker:ident = $kind:ident),+ $(,)?) => {$(
        #[doc = concat!("[`BundleKind::", stringify!($kind), "`](crate::bundle_roles::BundleKind::", stringify!($kind), "), as a [`BundleKindOf`].")]
        pub enum $marker {}
        impl BundleKindOf for $marker {
            const KIND: crate::bundle_roles::BundleKind = crate::bundle_roles::BundleKind::$kind;
        }
    )+};
}
bundle_kinds!(
    KRoutedExperts = RoutedExperts,
    KCorrectedExperts = CorrectedExperts,
    KGatedSharedExperts = GatedSharedExperts,
    KUngatedExperts = UngatedExperts,
    KBiasedExperts = BiasedExperts,
    KScaledRouter = ScaledRouter,
    KExpertBank = ExpertBank,
    KGatedDeltaNet = GatedDeltaNet,
    KMamba2 = Mamba2,
);

/// A device bundle's tensor at `role`, its extents in the role's declared orientation (a matrix
/// `[in, out]`, an expert bank `[experts, in, out]`, a conv `[channels, taps]`); `None` for a role
/// it does not hold.
pub trait RoleExtents {
    fn role_extents(&self, role: crate::bundle_roles::BundleRole) -> Option<Vec<usize>>;
}

/// A weight bundle of kind `K`, its roles at `R` ([`RoleShapes`]), held as `L`.
#[repr(transparent)]
pub struct Wb<K, R, L> {
    layer: L,
    roles: PhantomData<fn() -> (K, R)>,
}

impl<K, R, L> std::ops::Deref for Wb<K, R, L> {
    type Target = L;
    fn deref(&self) -> &L {
        &self.layer
    }
}

/// The DSL weights a `Weights` field holds hold `W`, a DSL weight's marker: a field of one weight
/// holds that weight, a packed pair each of its two. The bake emits each impl, concretely, beside
/// the model's markers (`__weights`): what a field holds is a fact of the field, not of the tape.
#[diagnostic::on_unimplemented(
    message = "the field `{Self}` holds does not hold the weight `{W}`",
    label = "a tape step's weight slot names another weight than the op reads"
)]
pub trait HoldsWeight<W> {}

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

/// A loaded layer whose MLX-affine codes are not its type's: `loaded` `None` for a layer that
/// holds no affine codes.
#[derive(Debug)]
pub struct CodesMismatch {
    pub tensor: String,
    pub loaded: Option<(u32, u32)>,
    pub declared: (u32, u32),
}

impl std::fmt::Display for CodesMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (bits, group) = self.declared;
        match self.loaded {
            Some((b, g)) => write!(
                f,
                "weight `{}` loaded at {b} bits in groups of {g}, its type declares {bits} in \
                 groups of {group}",
                self.tensor
            ),
            None => write!(
                f,
                "weight `{}` loaded without affine codes, its type declares {bits} bits in groups \
                 of {group}",
                self.tensor
            ),
        }
    }
}

impl std::error::Error for CodesMismatch {}

/// A loaded bundle whose tensor at `role` is not at its type's extents: `loaded` `None` for a
/// bundle that holds no tensor there.
#[derive(Debug)]
pub struct RoleMismatch {
    pub tensor: String,
    pub role: crate::bundle_roles::BundleRole,
    pub loaded: Option<Vec<usize>>,
    pub declared: Vec<usize>,
}

impl std::fmt::Display for RoleMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let role = self.role.name();
        match &self.loaded {
            Some(loaded) => write!(
                f,
                "bundle `{}`'s {role} loaded at {loaded:?}, its type declares {:?}",
                self.tensor, self.declared
            ),
            None => write!(
                f,
                "bundle `{}` loaded without its {role}, its type declares {:?}",
                self.tensor, self.declared
            ),
        }
    }
}

impl std::error::Error for RoleMismatch {}

/// The check a bundle handle's mint runs: each of `layers`' roles of `kind` against `declared`
/// (in the kind's role order), the first mismatch naming `tensor` (and its layer) and the role.
#[inline(never)]
fn check_roles<L: RoleExtents>(
    layers: &[L],
    kind: crate::bundle_roles::BundleKind,
    declared: Vec<Vec<u32>>,
    tensor: &str,
    layered: bool,
) -> anyhow::Result<()> {
    let roles = kind.roles();
    if roles.len() != declared.len() {
        anyhow::bail!(
            "bundle `{tensor}`: its type declares {} roles, its kind has {}",
            declared.len(),
            roles.len()
        );
    }
    for (at, layer) in layers.iter().enumerate() {
        for (&role, declared) in roles.iter().zip(&declared) {
            let declared: Vec<usize> = declared.iter().map(|&d| d as usize).collect();
            let loaded = layer.role_extents(role);
            if loaded.as_ref() != Some(&declared) {
                return Err(anyhow::Error::new(RoleMismatch {
                    tensor: match layered {
                        true => format!("{tensor}[{at}]"),
                        false => tensor.to_string(),
                    },
                    role,
                    loaded,
                    declared,
                }));
            }
        }
    }
    Ok(())
}

/// ⛔ THE ONE DOOR from a loaded layer to a typed handle: `layer`'s extents checked against the
/// type's numbers, the mismatch naming `tensor`. A layered field mints each layer's.
///
/// ```
/// # use scratchy_tensors::typed::{Dim, Extents, Mint, W2};
/// # enum Hidden {}
/// # impl Dim for Hidden { type Num = scratchy_tensors::typed::typenum::U2048; }
/// # struct Layer;
/// # impl Extents<2> for Layer { fn extents(&self) -> [usize; 2] { [2048, 2048] } }
/// let w = W2::<Hidden, Hidden, Layer>::mint(Layer, "o_proj");
/// assert!(w.is_ok());
/// ```
///
/// There is no other: a handle's wrap takes a [`Checked`] layer, which only this module makes.
///
/// ```compile_fail,E0423
/// # use scratchy_tensors::typed::{Checked, Dim, Extents, Shaped, W2};
/// # enum Hidden {}
/// # impl Dim for Hidden { type Num = scratchy_tensors::typed::typenum::U2048; }
/// # struct Layer;
/// # impl Extents<2> for Layer { fn extents(&self) -> [usize; 2] { [4, 4] } }
/// let w = W2::<Hidden, Hidden, Layer>::wrap(Checked(Layer));
/// ```
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

/// The codes check a quantized handle's mint runs beside [`check_layers`]: each of `layers`'
/// `(bits, group)` against `declared`, the first mismatch naming `tensor` (and its layer).
#[inline(never)]
fn check_codes<L: AffineCodes>(
    layers: &[L],
    declared: (u32, u32),
    tensor: &str,
    layered: bool,
) -> anyhow::Result<()> {
    let Some((at, loaded)) = (layers.iter().map(AffineCodes::affine_codes).enumerate())
        .find(|(_, loaded)| *loaded != Some(declared))
    else {
        return Ok(());
    };
    Err(anyhow::Error::new(CodesMismatch {
        tensor: match layered {
            true => format!("{tensor}[{at}]"),
            false => tensor.to_string(),
        },
        loaded,
        declared,
    }))
}

/// A layer its handle type's mint checked: the one thing a handle's wrap takes. Its field is
/// private, and nothing in this module builds one but the mints below, each right after
/// its `check` passed on that same layer (a layered field's mint: on those same layers), so a
/// handle is reached through a mint alone.
pub struct Checked<L>(L);

/// A handle over layer type `L`: its check (its numbers handed to [`check_layers`]) and its
/// wrap of a layer a mint checked. Both mints read it.
pub trait Shaped<L>: Sized {
    fn check(layers: &[L], tensor: &str, layered: bool) -> anyhow::Result<()>;
    fn wrap(layer: Checked<L>) -> Self;
}

impl<D: Dim, L: Extents<1>> Shaped<L> for W1<D, L> {
    fn check(layers: &[L], tensor: &str, layered: bool) -> anyhow::Result<()> {
        check_layers(layers, [D::N], tensor, layered)
    }

    fn wrap(Checked(layer): Checked<L>) -> Self {
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

    fn wrap(Checked(layer): Checked<L>) -> Self {
        Self {
            layer,
            shape: PhantomData,
        }
    }
}

impl<K: Dim, N: Dim, B: Bits, G: Tiles<K>, L: Extents<2> + AffineCodes> Shaped<L>
    for Q2<K, N, B, G, L>
{
    fn check(layers: &[L], tensor: &str, layered: bool) -> anyhow::Result<()> {
        check_layers(layers, [K::N, N::N], tensor, layered)?;
        check_codes(layers, (B::N, G::N), tensor, layered)
    }

    fn wrap(Checked(layer): Checked<L>) -> Self {
        Self {
            layer,
            shape: PhantomData,
            codes: PhantomData,
        }
    }
}

impl<K: Dim, N: Dim, B: Bits, G: Tiles<K>, L: Extents<2> + AffineCodes> Mint<L>
    for Q2<K, N, B, G, L>
{
    fn mint(layer: L, tensor: &str) -> anyhow::Result<Self> {
        Self::check(std::slice::from_ref(&layer), tensor, false)?;
        Ok(Self::wrap(Checked(layer)))
    }
}

impl<K: BundleKindOf, R: RoleShapes, L: RoleExtents> Shaped<L> for Wb<K, R, L> {
    fn check(layers: &[L], tensor: &str, layered: bool) -> anyhow::Result<()> {
        check_roles(layers, K::KIND, R::declared(), tensor, layered)
    }

    fn wrap(Checked(layer): Checked<L>) -> Self {
        Self {
            layer,
            roles: PhantomData,
        }
    }
}

impl<K: BundleKindOf, R: RoleShapes, L: RoleExtents> Mint<L> for Wb<K, R, L> {
    fn mint(layer: L, tensor: &str) -> anyhow::Result<Self> {
        Self::check(std::slice::from_ref(&layer), tensor, false)?;
        Ok(Self::wrap(Checked(layer)))
    }
}

impl<D: Dim, L: Extents<1>> Mint<L> for W1<D, L> {
    fn mint(layer: L, tensor: &str) -> anyhow::Result<Self> {
        Self::check(std::slice::from_ref(&layer), tensor, false)?;
        Ok(Self::wrap(Checked(layer)))
    }
}

impl<K: Dim, N: Dim, L: Extents<2>> Mint<L> for W2<K, N, L> {
    fn mint(layer: L, tensor: &str) -> anyhow::Result<Self> {
        Self::check(std::slice::from_ref(&layer), tensor, false)?;
        Ok(Self::wrap(Checked(layer)))
    }
}

/// A layered field: every layer checked, then every layer wrapped.
impl<T: Shaped<L>, L> Mint<Vec<L>> for Vec<T> {
    fn mint(layers: Vec<L>, tensor: &str) -> anyhow::Result<Self> {
        T::check(&layers, tensor, true)?;
        Ok(layers.into_iter().map(|l| T::wrap(Checked(l))).collect())
    }
}

/// An activation: `T` rows of `D` elements each, as an arena buffer holds it.
pub struct Act<T, D>(PhantomData<fn() -> (T, D)>);

/// What an arena buffer holding a value of this type must fit: its element count.
pub trait Extent {
    const ELEMS: u64;
}

impl<T: Dim, D: Dim> Extent for Act<T, D> {
    const ELEMS: u64 = <T as DimN>::N as u64 * <D as DimN>::N as u64;
}

/// An extent of a model's arena, as a marker type of its own: the extent it is at a bucket's rows
/// `M` (its `sk_bucket` `S`). A marker, not the extent's type, is what a colour hosts: two
/// extents whose types agree at some bucket's rows (`[M, D]` and `[1, D]`, where `M` is 1) are
/// still two markers, so their hosts never overlap.
pub trait ExtentAt<M, S> {
    type At: Extent;
}

/// An arena colour that holds extent `E` (an [`ExtentAt`] marker) at some point of its tape: what
/// the colouring assigned it. A buffer read or written as `E` must be a colour that hosts `E`.
pub trait Hosts<E> {}

#[cfg(test)]
mod tests {
    use super::*;

    enum Hidden {}
    impl Dim for Hidden {
        type Num = typenum::U2048;
    }
    enum Vocab {}
    impl Dim for Vocab {
        type Num = typenum::Prod<typenum::U1002, typenum::U128>;
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
