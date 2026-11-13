// SPDX-License-Identifier: Apache-2.0
//! A loaded model, as spyre runs it: [`SpyreWeights`] extends the target-neutral
//! [`ScratchyWeights`] with the baked bundles, the launch wiring and the generated weight
//! bindings, and [`SpyreArchRegistration`] is the row every spyre-compiled arch registers its
//! loaders under. A worker loads through [`try_load`] and holds a `Box<dyn SpyreWeights>`.

use scratchy_forward_compiler::{
    ArchLoad, ArchRegistration, HfFingerprint, ScratchyWeights, TargetRegistration,
};
use scratchy_tensors::{DeviceAllocator, GpuTensor, LoadStream};

/// One bound weight source: which tensor id it fills, the staged tensor, and
/// whether it is a GEMM operand.
///
/// ⛔ `is_gemm` IS CARRIED, NOT INFERRED. It decides the staged shape's
/// orientation — a GEMM weight sits on disk as `[out, in]` and the bundle wants
/// `[n, k]`; a norm is a flat gain staged `[1, hidden]`. It cannot be recovered from the
/// tensor's RANK: a per-channel fp8 `weight_scale` is 2-D on disk (`[N, 1]`) and
/// is NOT a gemm operand, so a rank test would mis-orient every quantized
/// model. The emitter knows which external is the gemm's operand-1, so it says
/// so, once, at bake.
#[derive(Clone, Copy, Debug)]
pub struct BoundWeight {
    pub id: u32,
    pub tensor: GpuTensor,
    pub is_gemm: bool,
}

impl BoundWeight {
    /// The tensor's own `[rows, cols]` — its ON-DISK shape, flattened for the
    /// two ranks that are not a plain matrix.
    ///
    /// ⛔ NOT THE BUNDLE'S `[k, n]` CONVENTION, AND NOT A STAGING DECISION. A
    /// GEMM weight sits on disk as `[out, in]` = `[n, k]` and that is what this
    /// reports. The staging needs the LOGICAL `[k, n]` and reads it from
    /// `Wiring::tensor_shapes`. Deriving it here instead returned the same two
    /// numbers in the other order under the same names, which mis-padded
    /// granite's lm_head; see the note at that call site.
    pub fn staged_shape(&self) -> [usize; 2] {
        match self.tensor.ndim() {
            // A flat gain (norm weight) is one row.
            1 => [1, self.tensor.dim(0)],
            // ⛔ A PER-CHANNEL SCALE IS A COLUMN ON DISK AND A ROW HERE. fp8
            // `weight_scale` ships as `[N, 1]`; staging it verbatim would
            // present N rows of one element where the bundle expects one row of
            // N. Only non-GEMM operands take this arm, so a genuine `[n, 1]`
            // GEMM weight is untouched.
            2 if !self.is_gemm && self.tensor.dim(1) == 1 => [1, self.tensor.dim(0)],
            _ => [self.tensor.dim(0), self.tensor.dim(1)],
        }
    }
}

/// A loaded model's spyre surface. Every spyre-compiled arch's `Weights` implements it beside
/// [`ScratchyWeights`]; each method answers from the ONE fingerprint-matched variant, so the
/// bundle, the wiring and the weights can never come from two different variants.
pub trait SpyreWeights: ScratchyWeights {
    /// The embedded KTIR bundle for the host run path: a [`crate::manifest::KtirBundle`] typed
    /// `Any`, the variant module's baked `KTIR_BUNDLE`.
    fn ktir_bundle(&self) -> Option<&'static (dyn core::any::Any + Send + Sync)>;

    /// This model's launch sources as `(tensor id, staged tensor)` — the
    /// GENERATED binding of a source id to a `Weights` FIELD.
    ///
    /// ⛔ THIS EXISTS SO NO ONE HAS TO NAME A WEIGHT BY STRING. A rule that rebuilds every
    /// weight's on-disk key at model load (`format!("{disk}.weight")`) and re-reads it out of
    /// the weight store cannot express a bare `nn.Parameter` (gemma-4's `layer_scalar`, which
    /// has no `.weight`) or a non-default decoder root, and it re-answers a question the
    /// compiler had already settled. Only generated code can name a generated field, so the
    /// emitter writes this per model.
    ///
    /// ⛔ THE TENSORS BORROW THE WEIGHT STORE. They are views over buffers the
    /// `GpuWeights` allocator owns, so the caller must keep it alive until the
    /// bytes have been staged.
    fn superdsc_weights(&self) -> Option<::anyhow::Result<Vec<BoundWeight>>>;

    /// The embedded **sendnn** bundle (the on-silicon path), peer of [`Self::ktir_bundle`].
    ///
    /// ⛔ ON THE LOADED MODEL SO ONE MATCHED VARIANT ANSWERS EVERYTHING. A second, independent
    /// registry walk after loading weights is a second variant match, and an arch registers many
    /// base models — so the bundle could come from one variant while the weights and geometry
    /// came from another, with nothing checking they agreed.
    fn sengraph_bundle(&self) -> Option<&'static (dyn core::any::Any + Send + Sync)>;

    /// The per-model launch WIRING — source roles, tensor shapes, per-layer
    /// AttnDecode ids and baked geometry, one set per program, as a static (a typed value, never
    /// a manifest the worker parses back into the shape it started as).
    fn superdsc_wiring(&self) -> Option<&'static (dyn core::any::Any + Send + Sync)>;

    /// The per-model SHARDED WORLDS — one forward per tensor-parallel world the bake sharded
    /// this model for, beside the one-rank wiring [`Self::superdsc_wiring`] carries.
    fn superdsc_tp(&self) -> Option<&'static (dyn core::any::Any + Send + Sync)>;

    /// The embedding table, for the host-side token gather.
    ///
    /// ⛔ IT CANNOT BE RE-READ BY NAME AFTER LOADING. The generated
    /// `Weights::load` TAKES the table out of the weight store, so a worker
    /// that later asks `gw` for `"model.embed_tokens.weight"` finds nothing —
    /// and the hardcoded spelling of that key is itself the gemma-4 bug (its
    /// decoder root is `model.language_model`). The loaded model hands over
    /// its own field instead.
    fn superdsc_embed(&self) -> Option<::anyhow::Result<GpuTensor>>;
}

/// The row every spyre-compiled arch registers its loaders under: the shared
/// [`ArchRegistration`], over [`SpyreWeights`].
pub struct SpyreArchRegistration(pub ArchRegistration<dyn SpyreWeights>);

inventory::collect!(SpyreArchRegistration);

impl TargetRegistration for SpyreArchRegistration {
    type Weights = dyn SpyreWeights;
    fn arch(&self) -> &ArchRegistration<dyn SpyreWeights> {
        &self.0
    }
}

/// Load the compiled arch that claims `arch_hint` from `gw`, over the spyre registry — the shared
/// [`scratchy_forward_compiler::try_load`] walk.
pub fn try_load<A: DeviceAllocator>(
    gw: &mut scratchy_layers::weights::GpuWeights<A>,
    stream: LoadStream,
    arch_hint: &str,
    tp_world_size: u8,
    tp_rank: u8,
    max_model_len: usize,
    hf: HfFingerprint<'_>,
) -> anyhow::Result<ArchLoad<dyn SpyreWeights>> {
    scratchy_forward_compiler::try_load::<SpyreArchRegistration, A>(
        gw,
        stream,
        arch_hint,
        tp_world_size,
        tp_rank,
        max_model_len,
        hf,
    )
}
