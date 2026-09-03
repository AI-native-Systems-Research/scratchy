// SPDX-License-Identifier: Apache-2.0
//! THE COMPILER -> TARGET HANDOFF TYPES.
//!
//! ⭐ THESE LIVE HERE SO A TARGET'S COMPILER CAN NAME THEM. They were in
//! `scratchy-forward-compiler-macro`, which DEPENDS ON the target crates — so nothing on the
//! target side could refer to them, and any lowering that needed them was forced to live in the
//! shared compiler crate no matter whose logic it was. That dependency direction is the reason
//! metal's lowering sat in `compiler/macros` instead of under `targets/metal`.
//!
//! None of it is target-specific: a `TileId` names a FUF tile, a `SlotMap` is a coloring keyed
//! on one, a `SourceBinding` says what a graph source is bound to, and a `LoweredDecode` is the
//! op list plus the provenance a target needs to bind weights to it.
//!
//! ⛔ NO `syn` HERE. `WeightSlot` stayed behind precisely because its `base` is a `syn::Ident`,
//! and this crate is compiled into the metal RUNTIME. A target lowering hands back
//! `(WeightKind, String)` and the macro crate mints the `Ident`.

use crate::lower::LoweringInput;
use std::collections::BTreeMap;

/// A weight tensor's identity in the model's weight table. The same id spans every layer;
/// [`UnrollIndex`] picks which one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightId(pub u32);

/// The unrolled (former loop-variable) position of a weight reference. `Some(UnrollIndex(n))`
/// names the `n`th per-layer instance of a layered weight; `None` a global tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UnrollIndex(pub u64);

impl std::fmt::Display for UnrollIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Dense tile index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId(pub u32);

/// Kind of weight a tape position consumes. The variant of
/// the carrying [`Instruction`] determines the *count* and *kind
/// list*; what the proc-macro records per-instance is the per-arch
/// base name(s) so the per-arch [`WeightAccessors`] impl can emit
/// the right match arm.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightKind {
    RmsNorm,
    Embedding,
    Linear,
    LayerNorm,
    Marlin,
    Bnb4,
    Fp8,
    DeepSeekMoe,
    DeepSeekMoeFp8,
    DeepSeekMoeGgml,
    FusedMoe,
    SharedFusedMoe,
    /// Gemma-4 router bundle (`GemmaMoe` op) — resolves to
    /// `WeightAccessors::gemma_router_at` → `&GemmaRouterLayer`.
    GemmaRouter,
    /// Gemma-4 SwitchGLU experts bundle (`GemmaMoe` op) — resolves to
    /// `WeightAccessors::gemma_switch_glu_at` → `&SwitchGluExpertsLayer`.
    GemmaSwitchGlu,
    /// Gated-DeltaNet per-layer weight bundle — resolves to
    /// `WeightAccessors::gated_delta_net_at`.
    GatedDeltaNet,
    CosSin,
    /// Metal-only: MLX-affine int4 quantized embedding. Resolves to
    /// `WeightAccessors::affine_quant_embedding_at`.
    AffineQuantEmbedding,
}

/// Compile-time mapping from `(TileId, output_slot)` → flat slot
/// index in the runtime tile table. Built once per (variant ×
/// workload-point) FUF by codegen and consumed by `fan_out` so it
/// can render `i32` slot ids into `Instruction` fields.
///
/// Indices are dense in `0..total()`. The runtime tile table is
/// allocated as `Vec<Option<TileEntry>>` of size `total()` and
/// indexed directly by these values.
///
/// Lives in the macro crate (not in `scratchy-forward-compiler`) because
/// it's a *compile-time* artifact: by the time a forward fn runs,
/// every i32 in the const `Instruction` array is already a flat
/// slot index. The runtime never sees a `(TileId, slot)` pair.
#[derive(Clone, Debug, Default)]
pub struct SlotMap {
    map: BTreeMap<(TileId, u8), u32>,
    total: u32,
}
impl SlotMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `(tile, output_slot)` and assign the next dense
    /// index. Idempotent on repeats — same `(tile, output_slot)`
    /// keeps its first-assigned index.
    pub fn insert(&mut self, tile: TileId, output_slot: u8) -> u32 {
        let key = (tile, output_slot);
        if let Some(&existing) = self.map.get(&key) {
            return existing;
        }
        let idx = self.total;
        self.map.insert(key, idx);
        self.total += 1;
        idx
    }

    /// Insert `(tile, output_slot)` at a specific color. The colored
    /// slot map (linear-scan register allocation) walks tiles in
    /// def order and picks a color from the free pool, so two
    /// non-overlapping tiles can share an index. Use this instead of
    /// `insert` when the caller has already decided the color.
    /// `total()` ends up = 1 + max color seen.
    pub fn insert_at(&mut self, tile: TileId, output_slot: u8, color: u32) {
        self.map.insert((tile, output_slot), color);
        if color + 1 > self.total {
            self.total = color + 1;
        }
    }

    /// Resolve `(tile, output_slot)` → flat slot index. Panics
    /// when the pair was never inserted — codegen invariant.
    #[inline]
    /// Every `(tile, slot) -> color` entry, ordered — the coloring as
    /// DATA, for cross-checking one colorer against another.
    pub fn entries(&self) -> impl Iterator<Item = (&(TileId, u8), &u32)> {
        self.map.iter()
    }

    pub fn of(&self, tile: TileId, output_slot: u8) -> u32 {
        match self.map.get(&(tile, output_slot)) {
            Some(&v) => v,
            None => panic!(
                "SlotMap::of: tile {tile:?} slot {output_slot} not registered \
                 — codegen forgot to insert this tile output before fan_out"
            ),
        }
    }

    /// Total number of slots allocated. Size of the runtime tile table.
    #[inline]
    pub fn total(&self) -> u32 {
        self.total
    }

    /// Iterate `((tile, output_slot), color)` for every registered
    /// pair. Multiple pairs may share a `color` when the linear-scan
    /// register allocator coalesced them into the same arena slot.
    pub fn iter(&self) -> impl Iterator<Item = ((TileId, u8), u32)> + '_ {
        self.map.iter().map(|(k, v)| (*k, *v))
    }
}

/// What real tensor each wavefront `Source` is bound to at run time —
/// parallel to [`LoweringInput::sources`]. The bridge anonymizes sources
/// to bare shapes; this manifest is how the Tier-B host executor (and
/// later the GPU tape player) maps `SourceId(i)` back to a concrete
/// buffer. `Weight` carries the raw [`crate::classified::WeightId`] +
/// unrolled index (resolve to a tensor name via the same `Program`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceBinding {
    /// A model weight: the `index` is the unrolled (former loop-var)
    /// integer, so `(id, index)` uniquely names a per-layer tensor.
    Weight {
        id: u32,
        index: Option<UnrollIndex>,
    },
    /// The per-CHANNEL fp8 `weight_scale` sibling of a [`SourceBinding::Weight`]: SAME `(id, index)` as
    /// the fp8 weight, but the worker loads it from `{prefix}.weight_scale` (vs `.weight`). It is the
    /// 3rd input of an arity-3 fp8 W8A8 [`LoweredOp::Gemm`] (`[act, weight_fp8, weight_scale]`). Shape
    /// `[1, n]` (n = output channels), byte-identical to the on-disk `[n, 1]` per-output-channel scale.
    WeightScale {
        id: u32,
        index: Option<UnrollIndex>,
    },
    /// The embedded hidden-state row the runtime gathers from
    /// `embed_tokens[input_id]` before the kernel — embed is a cheap
    /// host lookup, not a megakernel op.
    EmbeddedHidden,
    /// The new token's rotary `cos` / `sin` row `[1, head_dim]` (shared
    /// across layers at one decode position). `local` = the
    /// sliding-class dual-theta rotary (`rotary_local`) on arches that
    /// declare one (Gemma4).
    Cos {
        local: bool,
    },
    Sin {
        local: bool,
    },
    /// The read-only prefix KV cache for a layer, `[prefix_len, kvdim]`.
    PrefixK {
        layer: u64,
    },
    PrefixV {
        layer: u64,
    },
}

#[derive(Debug, Clone)]
pub struct LoweredDecode {
    pub input: LoweringInput,
    /// Parallel to `input.sources`: what each source is bound to.
    pub bindings: Vec<SourceBinding>,
    /// Parallel to `input.ops`: the FUF `(tile, output slot)` each op's
    /// output realizes, when it realizes one (fusion intermediates
    /// don't). Provenance lets per-target emitters reuse the SAME
    /// slot-coloring authority (`colored_slot_map` keys on tiles)
    /// instead of re-deriving a coloring.
    pub op_tiles: Vec<Option<(u32, u8)>>,
    /// For an rmsnorm op whose gain routed through a folded
    /// `(w + scalar)` Add: the ADD TILE's id. The instruction-selection
    /// impls CLAIM that tile, so its colored slot participates in their
    /// hazard signatures — emitters must mirror it.
    pub norm_gain_add_tiles: std::collections::HashMap<usize, u32>,
}

impl WeightKind {
    /// The Rust type an accessor of this kind resolves to.
    ///
    /// ⭐ ONE SOURCE OF TRUTH. The macro crate parses tokens from this to emit the per-arch
    /// `WeightAccessors` impl; metal's compiler compares accessor GROUP SIGNATURES against it.
    /// Two hand-kept copies of the same mapping is how a kind comes to mean one type on one
    /// side and a different one on the other, with nothing to catch it.
    ///
    /// Exhaustive by construction: adding a variant without a row here is a build error.
    pub const fn rust_type_name(&self) -> &'static str {
        match self {
            Self::RmsNorm => "&crate::__gpu::layers::RmsNorm",
            Self::Embedding => "&crate::__gpu::layers::Embedding",
            Self::Linear => "&crate::__gpu::layers::LinearLayer",
            Self::LayerNorm => "&crate::__gpu::layers::LayerNorm",
            Self::Marlin => "&crate::__gpu::layers::MarlinLinear",
            Self::Bnb4 => "&crate::__gpu::layers::Bnb4bitLinear",
            Self::Fp8 => "&crate::__gpu::layers::Fp8AnyLinear",
            Self::DeepSeekMoe => "&crate::__gpu::layers_moe::DeepSeekV2MoELayer",
            Self::DeepSeekMoeFp8 => "&crate::__gpu::layers_moe::DeepSeekV2Fp8BlockMoELayer",
            Self::DeepSeekMoeGgml => "&crate::__gpu::layers_moe::DeepSeekV2GgmlMoELayer",
            Self::GatedDeltaNet => "&crate::__gpu::layers::GatedDeltaNetLayer",
            Self::FusedMoe => "&crate::__gpu::layers_moe::FusedMoELayer",
            Self::SharedFusedMoe => "&crate::__gpu::layers_moe::SharedFusedMoELayer",
            Self::GemmaRouter => "&crate::__gpu::layers_moe::GemmaRouterLayer",
            Self::GemmaSwitchGlu => "&crate::__gpu::layers_moe::SwitchGluExpertsLayer",
            // NOTE: no leading `&` — this one is passed by value.
            Self::CosSin => "crate::__gpu::tensor::GpuTensor",
            Self::AffineQuantEmbedding => "&crate::__gpu::layers::AffineQuantEmbedding",
        }
    }
}

/// One weight slot consumed by an op instance: which `WeightAccessors` method the per-arch
/// match arm goes under (`kind`), and the user-source accessor method name it calls (`base`).
///
/// ⛔ `base` IS A `String`, NOT A `syn::Ident`. This crate is linked into the metal RUNTIME, and
/// an `Ident` would drag `syn` in with it. The macro crate mints the identifier at emission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightSlot {
    pub kind: WeightKind,
    pub base: String,
}
