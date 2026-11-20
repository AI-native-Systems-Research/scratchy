// SPDX-License-Identifier: Apache-2.0
//! THE COMPILER -> TARGET HANDOFF TYPES.
//!
//! ⭐ THESE LIVE HERE SO A TARGET'S COMPILER CAN NAME THEM. They were in
//! `scratchy-forward-compiler-macro`, which DEPENDS ON the target crates — so nothing on the
//! target side could refer to them, and any lowering that needed them was forced to live in the
//! shared compiler crate no matter whose logic it was. That dependency direction is the reason
//! a target's lowering once sat in `compiler/macros` instead of under its target.
//!
//! None of it is target-specific: a `TileId` names a FUF tile, a `SlotMap` is a coloring keyed
//! on one, a `SourceBinding` says what a graph source is bound to, and a `LoweredDecode` is the
//! op list plus the provenance a target needs to bind weights to it.
//!
//! ⛔ NO `syn` HERE. `WeightSlot` stayed behind precisely because its `base` is a `syn::Ident`,
//! and this crate is compiled into a target's RUNTIME. A target lowering hands back
//! `(WeightKind, String)` and the macro crate mints the `Ident`.

use crate::lower::LoweringInput;
use std::collections::BTreeMap;

pub use scratchy_tensors::compiler_ids::{UnrollIndex, WeightId};

/// Dense tile index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId(pub u32);

/// Which tensor inside a multi-tensor MoE weight bundle a
/// [`SourceBinding::BundleTensor`] names.
///
/// ⭐ THE SHARED SUBSET ONLY. A variant belongs here when MORE THAN ONE target binds that
/// tensor off the bundle — a staging target binds each as its own weight and another reads it
/// through its own `WeightTensor` at lowering, so both must agree on the vocabulary. A
/// tensor exactly one target reads (a shared-expert slab, a GDN table) stays that target's
/// own enum and never appears here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BundleTensor {
    /// Gemma-4 router: the dense `[num_experts, hidden]` projection (`router.proj`,
    /// dequantized at load when the checkpoint ships it quantized).
    RouterGate,
    /// Gemma-4 router: `[num_experts]` per-expert score scale
    /// (`router.per_expert_scale`), gathered after the softmax.
    RouterPerExpertScale,
    /// Gemma-4 router: `[hidden]` RMSNorm gain on the router input (`router.scale`).
    RouterScale,
    /// Experts: the gate projection's codes. Rank-3 `[E, out, in]` (or its packed
    /// form — the LOADER decides the representation, the target reads what the
    /// bundle's kind says it holds).
    ExpertGateW,
    /// Experts: the gate projection's per-channel fp8 scale (`[E, out, 1]` stacked).
    ExpertGateS,
    ExpertUpW,
    ExpertUpS,
    ExpertDownW,
    ExpertDownS,
}

impl BundleTensor {
    /// The tensor's registry name — for dumps and the manifest, so no consumer keeps a
    /// parallel string table. Lowercase snake, matching the on-disk leaf it loads.
    pub const fn name(self) -> &'static str {
        match self {
            Self::RouterGate => "router_gate",
            Self::RouterPerExpertScale => "router_per_expert_scale",
            Self::RouterScale => "router_scale",
            Self::ExpertGateW => "expert_gate_w",
            Self::ExpertGateS => "expert_gate_s",
            Self::ExpertUpW => "expert_up_w",
            Self::ExpertUpS => "expert_up_s",
            Self::ExpertDownW => "expert_down_w",
            Self::ExpertDownS => "expert_down_s",
        }
    }

    /// Whether this tensor is a GEMM's weight operand — the fact that decides its staged
    /// shape's orientation (a gemm weight is `[n, k]` on disk; a gain or scale is not).
    pub const fn is_gemm_operand(self) -> bool {
        matches!(
            self,
            Self::RouterGate | Self::ExpertGateW | Self::ExpertUpW | Self::ExpertDownW
        )
    }
}

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
    /// Gemma-4 router bundle (`RouterBundle::Gemma`) — resolves to
    /// `WeightAccessors::gemma_router_at` → `&GemmaRouterLayer`.
    GemmaRouter,
    /// Gemma-4 SwitchGLU experts bundle (`ExpertBundle::SwitchGlu`) — resolves to
    /// `WeightAccessors::gemma_switch_glu_at` → `&SwitchGluExpertsLayer`.
    GemmaSwitchGlu,
    /// gpt-oss's whole MoE bundle (`RouterBundle::GptOss` /
    /// `ExpertBundle::GptOss`): the biased router, the SwiGLU-OAI
    /// expert stacks and their per-expert linear biases under one
    /// `mlp[layer]` base — resolves to
    /// `WeightAccessors::gpt_oss_moe_at` → `&GptOssMoELayer`.
    GptOssMoe,
    /// Gated-DeltaNet per-layer weight bundle — resolves to
    /// `WeightAccessors::gated_delta_net_at`.
    GatedDeltaNet,
    CosSin,
    /// MLX-affine int4 quantized embedding. Resolves to
    /// `WeightAccessors::affine_quant_embedding_at`.
    AffineQuantEmbedding,
    /// Per-layer attention-sink logits (`[num_attention_heads]`,
    /// dense, never quantized — gpt-oss). An extra softmax column
    /// added unscaled after qk·sm_scale; the sink column is dropped
    /// before the ·V accumulation.
    AttnSinks,
    /// A bare tensor read verbatim (`<prefix>`, no `.weight`, dense, never quantized): a DFlash2
    /// drafter's convolution base kernels and selector codebooks, read whole by Splash's kernels.
    Tensor,
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
    /// 3rd input of an arity-3 fp8 W8A8 [`crate::subtile_ir::SubOp::MatmulTile`]
    /// (`[act, weight_fp8, weight_scale]`). Shape `[1, n]` (n = output channels), byte-identical
    /// to the on-disk `[n, 1]` per-output-channel scale.
    WeightScale {
        id: u32,
        index: Option<UnrollIndex>,
    },
    /// ONE TENSOR inside a MULTI-TENSOR weight bundle (a MoE block's router or expert bank).
    ///
    /// ⭐ WHY A DISTINCT VARIANT, NOT A `Weight` WITH A SUFFIXED PATH. A bundle is ONE
    /// `Weights`-struct field (one accessor, one load, one on-disk prefix) that HOLDS several
    /// tensors — gemma-4's router alone is a projection, a per-expert scale, and a norm gain.
    /// The source a tape operand binds must name BOTH the bundle (`id`, `index`, exactly as
    /// [`SourceBinding::Weight`] names a single-tensor weight) and WHICH tensor inside it
    /// (`tensor`). Suffixing the path instead would mint a phantom accessor the load walk never
    /// declared, exactly the two-producers-disagree shape this file exists to prevent.
    ///
    /// The `tensor` vocabulary is the SHARED one — the subset of a bundle's tensors that more
    /// than one target binds ([`BundleTensor`]). A target-only tensor stays the target's own
    /// fact (a target's own tensor vocabulary is its superset, with shared-expert and GDN arms); nothing
    /// here may name a tensor only one target reads.
    BundleTensor {
        id: u32,
        index: Option<UnrollIndex>,
        tensor: BundleTensor,
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

// ── The resident KV state's declaration ──────────────────────────────

/// One plane of a layer's resident KV state: the minted source index of the plane's
/// tensor, and its ELEMENT COUNT PER GROWTH UNIT — one page of the state's token slots,
/// each slot of the plane's own row width. Fields are PRIVATE: a plane reaches a
/// declaration only through [`KvStateDecl::new`] / [`KvStateDecl::paged_kv`] /
/// [`KvStateDecl::shard_cols`], the constructors that state the whole-pages law.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvStatePlane {
    source: usize,
    per_unit_elems: u64,
}

impl KvStatePlane {
    /// The plane's own mint — its source index and per-unit element count.
    pub fn new(source: usize, per_unit_elems: u64) -> Self {
        Self {
            source,
            per_unit_elems,
        }
    }

    /// The plane's minted source index.
    pub fn source(&self) -> usize {
        self.source
    }

    /// The plane's element count per growth unit.
    pub fn per_unit_elems(&self) -> u64 {
        self.per_unit_elems
    }
}

/// A state whose row count is not a whole number of growth units is unrepresentable — its
/// unit count would not exist. Refused AT THE MINT (a compile-time error in the bake),
/// never re-checked downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartialKvStatePage {
    pub rows: u32,
    pub page_slots: u32,
}

/// ⭐ ONE LAYER'S RESIDENT KV STATE — DECLARED WHERE ITS TENSORS ARE MINTED (the bridge's
/// `prefix_for`, at the moment it pushes the layer's cache sources), carried in the
/// handoff, and consumed by the planner and the splice door. Nobody else re-derives any of
/// it: no op is read to learn a state's shape, and no `K + V` total is restated anywhere
/// downstream. The state is its planes (in the splice's operand order) plus ONE GROWTH
/// UNIT — a page of `page_slots` token slots holding EVERY plane — and every figure a
/// consumer needs is a VIEW of this one declaration:
///  * UNIT COUNT is `rows / page_slots` — whole by construction (the minting constructor
///    refuses a partial page);
///  * UNIT BYTES, at the plane dtype's bytes-per-element, is the SUM of the planes'
///    per-unit elems times that width — the sum over planes, never a restated pair total;
///  * A PLANE'S OFFSET IN THE UNIT is the prefix sum of the planes before it;
///  * A PLANE'S EXTENT from its own base is the whole bank (`units × unit bytes`) minus
///    its offset.
///
/// The elem forms below are the same views before the dtype conversion; the byte forms
/// take the width as their ONE argument, so rows and bytes can never be two spellings of
/// the law.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvStateDecl {
    planes: Vec<KvStatePlane>,
    page_slots: std::num::NonZeroU32,
    rows: u32,
}

impl KvStateDecl {
    /// The minting constructor — whole pages only (see [`PartialKvStatePage`]).
    pub fn new(
        planes: Vec<KvStatePlane>,
        page_slots: std::num::NonZeroU32,
        rows: u32,
    ) -> Result<Self, PartialKvStatePage> {
        if rows % page_slots.get() != 0 {
            return Err(PartialKvStatePage {
                rows,
                page_slots: page_slots.get(),
            });
        }
        Ok(Self {
            planes,
            page_slots,
            rows,
        })
    }

    /// ⭐ THE PAGED K/V STATE'S OWN MINT — the ONE statement of what one layer's attention cache
    /// planes hold per growth unit: a page of `page_slots` token slots, each slot `kv_width`
    /// elements (`kv_heads · head_dim`) on EACH of the K and V planes. The bridge's `prefix_for`
    /// mints its declarations here, and every fixture that states a paged KV state consumes this
    /// same constructor — a per-unit element count for a paged K or V plane spelled anywhere else
    /// is the law stated twice.
    pub fn paged_kv(
        k_source: usize,
        v_source: usize,
        kv_width: u32,
        page_slots: std::num::NonZeroU32,
        rows: u32,
    ) -> Result<Self, PartialKvStatePage> {
        let per_unit = u64::from(page_slots.get()) * u64::from(kv_width);
        Self::new(
            vec![
                KvStatePlane {
                    source: k_source,
                    per_unit_elems: per_unit,
                },
                KvStatePlane {
                    source: v_source,
                    per_unit_elems: per_unit,
                },
            ],
            page_slots,
            rows,
        )
    }

    /// UNIT COUNT — whole by construction.
    pub fn units(&self) -> u64 {
        u64::from(self.rows) / u64::from(self.page_slots.get())
    }

    /// The declaration's planes, in the splice's operand order.
    pub fn planes(&self) -> &[KvStatePlane] {
        &self.planes
    }

    /// The growth unit's token slots — one page.
    pub fn page_slots(&self) -> u32 {
        self.page_slots.get()
    }

    /// The state's declared row count.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// UNIT ELEMS — the sum over planes.
    pub fn unit_elems(&self) -> u64 {
        self.planes.iter().map(|p| p.per_unit_elems).sum()
    }

    /// A PLANE'S OFFSET IN THE UNIT, in elems — the prefix sum of the planes before it.
    pub fn plane_offset_elems(&self, plane: usize) -> u64 {
        self.planes[..plane].iter().map(|p| p.per_unit_elems).sum()
    }

    /// A PLANE'S EXTENT from its own base, in elems — the whole bank minus its offset.
    pub fn plane_span_elems(&self, plane: usize) -> u64 {
        self.plane_span_elems_at(self.units(), plane)
    }

    /// ⭐ THE SPAN LAW AT AN ARBITRARY UNIT COUNT of this same declaration —
    /// `units · unit − offset`, the ONE statement ([`Self::plane_span_elems`]'s
    /// own law), stated once and read at any unit count a placement or a kernel
    /// descriptor declares FOR THIS DECLARATION: the declaration's own units at
    /// its mint, and the addressing bound (`units_max`, the most units one
    /// device region may hold) where the compiled programs state the pool's
    /// span without baking the pages a particular run funds.
    pub fn plane_span_elems_at(&self, units: u64, plane: usize) -> u64 {
        units * self.unit_elems() - self.plane_offset_elems(plane)
    }

    /// UNIT BYTES at one elems→bytes width — the sum over planes, the ONE statement.
    pub fn unit_bytes(&self, width: u64) -> u64 {
        self.unit_elems() * width
    }

    /// A PLANE'S OFFSET IN THE UNIT, in bytes.
    pub fn plane_offset_bytes(&self, plane: usize, width: u64) -> u64 {
        self.plane_offset_elems(plane) * width
    }

    /// A PLANE'S EXTENT from its own base, in bytes.
    pub fn plane_span_bytes(&self, plane: usize, width: u64) -> u64 {
        self.plane_span_bytes_at(self.units(), plane, width)
    }

    /// The span law's byte form at an arbitrary unit count — see
    /// [`Self::plane_span_elems_at`].
    pub fn plane_span_bytes_at(&self, units: u64, plane: usize, width: u64) -> u64 {
        self.plane_span_elems_at(units, plane) * width
    }

    /// The declaration as ONE TP RANK holds it: `cols[plane]` is that plane's source's
    /// `(local_cols, full_cols)` under the world's column slice, and a plane's per-unit
    /// elems scale by exactly that ratio — a growth unit is a page of the plane's rows,
    /// so its elems follow the columns' own slice law. Rows and the page are every
    /// rank's alike, so the whole-page law holds un-rechecked. A per-unit elem count
    /// that is not a WHOLE multiple of `full_cols` is no slice this handoff states — a
    /// build panic naming both numbers, never a runtime fact.
    pub fn shard_cols(&self, cols: &[(u64, u64)]) -> Self {
        let planes = self
            .planes
            .iter()
            .zip(cols)
            .map(|(p, &(local, full))| {
                // ⭐ THE SLICE LAW, STATED EXACTLY: a growth unit is a page of the plane's
                // rows, so its elems must be a WHOLE multiple of the source's full column
                // count — a remainder truncates slots silently, which is not a slice this
                // handoff states. The panic names both numbers.
                let per_full = p
                    .per_unit_elems
                    .checked_div(full)
                    .filter(|&q| q * full == p.per_unit_elems)
                    .unwrap_or_else(|| {
                        panic!(
                            "a resident KV state plane's per-unit elems ({}) are not a whole \
                             multiple of its source's full column count ({}) — no TP slice of \
                             it exists",
                            p.per_unit_elems, full
                        )
                    });
                KvStatePlane {
                    source: p.source,
                    per_unit_elems: per_full * local,
                }
            })
            .collect();
        Self {
            planes,
            page_slots: self.page_slots,
            rows: self.rows,
        }
    }
}

#[cfg(test)]
mod kv_state_decl_tests {
    use super::*;

    fn one_state(per_unit: u64) -> KvStateDecl {
        KvStateDecl::new(
            vec![KvStatePlane::new(0, per_unit), KvStatePlane::new(1, per_unit)],
            std::num::NonZeroU32::new(128).expect("128 != 0"),
            128,
        )
        .expect("one page is a whole number of pages")
    }

    /// ⛔ NEAR-MISS: a plane whose per-unit elems are NOT a whole multiple of its source's
    /// full column count is no slice — the mint panics naming BOTH numbers, never truncates
    /// (the old `checked_div` silently dropped the remainder while its panic text claimed
    /// this law).
    #[test]
    #[should_panic(
        expected = "per-unit elems (300) are not a whole multiple of its source's full column count (128)"
    )]
    fn a_partial_column_slice_refuses_naming_both_numbers() {
        let _ = one_state(300).shard_cols(&[(64, 128), (64, 128)]);
    }

    /// The whole multiple slices exactly: `local/full` of the per-unit elems, page and rows
    /// untouched (every rank's alike).
    #[test]
    fn a_whole_column_slice_scales_the_per_unit_elems() {
        let sharded = one_state(256).shard_cols(&[(64, 128), (64, 128)]);
        assert_eq!(sharded.planes()[0].per_unit_elems(), 128);
        assert_eq!(sharded.page_slots(), 128);
        assert_eq!(sharded.rows(), 128);
    }
}

/// What an arena slot holds at a step: a FUF tile's output (`(tile, output)`, the key
/// [`SlotMap`] colours), or a graph source (the embedded hidden, which a target's colour 0 is).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ArenaValue {
    Tile { tile: u32, out: u8 },
    Source(u32),
}

/// The extent each arena value is, by its marker's name: a tile output's from its shape, the
/// embedded hidden's (the one graph source a step names in the arena) likewise. What a target
/// names a binding of the value by.
#[derive(Clone, Debug, Default)]
pub struct ArenaExtents {
    tiles: std::collections::HashMap<(u32, u8), String>,
    embedded: Option<String>,
}

impl ArenaExtents {
    /// Note `value`'s extent's marker.
    pub fn insert(&mut self, value: ArenaValue, marker: String) {
        match value {
            ArenaValue::Tile { tile, out } => {
                self.tiles.insert((tile, out), marker);
            }
            ArenaValue::Source(_) => self.embedded = Some(marker),
        }
    }

    /// `value`'s extent's marker, when it has been noted.
    pub fn of(&self, value: ArenaValue) -> Option<&str> {
        match value {
            ArenaValue::Tile { tile, out } => self.tiles.get(&(tile, out)),
            ArenaValue::Source(_) => self.embedded.as_ref(),
        }
        .map(String::as_str)
    }
}

/// ⭐ THE SOURCE A TAPE GATHERS ITS EMBEDDED ROW INTO — the one `SourceBinding::EmbeddedHidden`
/// of its bindings, found once by [`LoweredDecode::new`]. A decoder's tape has exactly one; a
/// tower's (pixels in) has none; two is refused at construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbedRow(usize);

impl EmbedRow {
    /// The source index.
    pub fn index(self) -> usize {
        self.0
    }
}

/// What a tape knows of its embed row: `Option<EmbedRow>` for any lowered tape, `EmbedRow` for a
/// DECODER's tape ([`LoweredDecode::into_decode`]) — the type the decode bake takes, so its read of
/// the row cannot miss.
pub trait EmbedState: Clone + std::fmt::Debug {}
impl EmbedState for Option<EmbedRow> {}
impl EmbedState for EmbedRow {}

/// A tape whose bindings name two embed rows — the bridge mints one per `Embed` op and the host
/// gathers one row, so a second is never written. Refused at [`LoweredDecode::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoEmbedRows {
    pub first: usize,
    pub second: usize,
}

impl std::fmt::Display for TwoEmbedRows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sources t{} and t{} are both EmbeddedHidden — the host gathers one row, so the \
             second is never written",
            self.first, self.second
        )
    }
}

/// A tape that gathers no embed row, asked to be a decoder's ([`LoweredDecode::into_decode`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoEmbedRow;

#[derive(Debug, Clone)]
pub struct LoweredDecode<E: EmbedState = Option<EmbedRow>> {
    pub input: LoweringInput,
    /// Parallel to `input.sources`: what each source is bound to. Private, with the roles it
    /// states ([`Self::embed`]) computed from it once at [`Self::new`]: nothing can change one
    /// without the other.
    bindings: Vec<SourceBinding>,
    /// The embed row the bindings name — [`EmbedState`].
    embed: E,
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
    /// Parallel to `input.ops`: the construct an op was expanded from, when it was (every op of
    /// one MoE block, or of one KV codec site, shares one id), and the runtime guard it runs
    /// under. A target fences a construct's steps as one.
    pub op_expansion: Vec<Option<Expansion>>,
    /// `(writer, reader)`: op `reader` reads storage op `writer` wrote that none of its operands
    /// names — a KV encode packs the cache half its writer filled, a coded attention reads the
    /// K/V its stage steps staged. Every other read is an operand.
    pub unnamed_reads: Vec<(usize, usize)>,
    /// ⭐ THE RESIDENT KV STATES THIS TAPE DECLARES — minted with their tensors (the bridge's
    /// `prefix_for`), one per layer, and re-derived by nobody: the planner's placement and the
    /// splice door consume these and nothing else. An op-rewritten tape carries them unchanged
    /// (an op rewrite keeps every source at its index, so the planes' source indices stay
    /// valid).
    kv_states: Vec<KvStateDecl>,
}

impl LoweredDecode {
    /// A lowered tape: its roles (the embed row) read off `bindings` here, once. Two embed rows
    /// are refused ([`TwoEmbedRows`]).
    pub fn new(
        input: LoweringInput,
        bindings: Vec<SourceBinding>,
        op_tiles: Vec<Option<(u32, u8)>>,
        norm_gain_add_tiles: std::collections::HashMap<usize, u32>,
        op_expansion: Vec<Option<Expansion>>,
        unnamed_reads: Vec<(usize, usize)>,
        kv_states: Vec<KvStateDecl>,
    ) -> Result<Self, TwoEmbedRows> {
        let mut rows = (bindings.iter().enumerate())
            .filter(|(_, b)| matches!(b, SourceBinding::EmbeddedHidden))
            .map(|(i, _)| i);
        let embed = match (rows.next(), rows.next()) {
            (first, None) => first.map(EmbedRow),
            (Some(first), Some(second)) => return Err(TwoEmbedRows { first, second }),
            (None, Some(_)) => unreachable!("an iterator yields its first item first"),
        };
        Ok(Self {
            input,
            bindings,
            embed,
            op_tiles,
            norm_gain_add_tiles,
            op_expansion,
            unnamed_reads,
            kv_states,
        })
    }

    /// The embed row, when the tape gathers one (a tower's does not).
    pub fn embed_row(&self) -> Option<EmbedRow> {
        self.embed
    }

    /// This tape as a DECODER's — its embed row held as a fact ([`EmbedRow`], no `Option`), the
    /// type the decode bake takes. [`NoEmbedRow`] when the tape gathers none.
    pub fn into_decode(self) -> Result<LoweredDecode<EmbedRow>, NoEmbedRow> {
        match self.embed {
            Some(row) => Ok(LoweredDecode {
                input: self.input,
                bindings: self.bindings,
                embed: row,
                op_tiles: self.op_tiles,
                norm_gain_add_tiles: self.norm_gain_add_tiles,
                op_expansion: self.op_expansion,
                unnamed_reads: self.unnamed_reads,
                kv_states: self.kv_states,
            }),
            None => Err(NoEmbedRow),
        }
    }
}

impl LoweredDecode<EmbedRow> {
    /// The embed row — a decoder's tape always gathers one.
    pub fn embed_row(&self) -> EmbedRow {
        self.embed
    }
}

impl<E: EmbedState> LoweredDecode<E> {
    /// What each source is bound to (parallel to `input.sources`).
    pub fn bindings(&self) -> &[SourceBinding] {
        &self.bindings
    }

    /// The resident KV states this tape declares (see [`Self`]'s own `kv_states` note).
    pub fn kv_states(&self) -> &[KvStateDecl] {
        &self.kv_states
    }

    /// This tape's bindings and embed row over a rewritten forward — the form every op-rewriting
    /// pass (a codec expansion, a shard, a reorder) returns. Such a pass keeps every source at its
    /// index (a shard changes only its per-rank shape), so the roles, which are per index, carry
    /// over as they are, never re-derived.
    pub fn with_ops(
        &self,
        input: LoweringInput,
        op_tiles: Vec<Option<(u32, u8)>>,
        norm_gain_add_tiles: std::collections::HashMap<usize, u32>,
        op_expansion: Vec<Option<Expansion>>,
        unnamed_reads: Vec<(usize, usize)>,
    ) -> Self {
        assert_eq!(
            input.sources.len(),
            self.input.sources.len(),
            "an op rewrite keeps its tape's sources at their indices — a pass that adds or drops \
             one mints a new tape (`LoweredDecode::new`)"
        );
        Self {
            input,
            bindings: self.bindings.clone(),
            embed: self.embed.clone(),
            op_tiles,
            norm_gain_add_tiles,
            op_expansion,
            unnamed_reads,
            kv_states: self.kv_states.clone(),
        }
    }

    /// This tape over its states re-declared for one TP rank ([`KvStateDecl::shard_cols`] — the
    /// planes' source indices are the shard's own, so only the per-unit elems move). The form
    /// [`crate::tensor_parallel::shard_lowered`] returns.
    pub fn with_kv_states(self, kv_states: Vec<KvStateDecl>) -> Self {
        Self {
            kv_states,
            ..self
        }
    }
}

/// One construct's expansion into ops (see [`LoweredDecode::op_expansion`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpansionId(pub u32);

/// An op's place in an expansion: the construct, and the guard it runs under (`None`: always).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Expansion {
    pub id: ExpansionId,
    pub guard: Option<crate::kv_codec::CodecGuard>,
}

impl WeightKind {
    /// The Rust type an accessor of this kind resolves to.
    ///
    /// ⭐ ONE SOURCE OF TRUTH. The macro crate parses tokens from this to emit the per-arch
    /// `WeightAccessors` impl; a target's compiler compares accessor GROUP SIGNATURES against it.
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
            Self::GptOssMoe => "&crate::__gpu::layers_moe::GptOssMoELayer",
            // NOTE: no leading `&` — this one is passed by value.
            Self::CosSin => "crate::__gpu::tensor::GpuTensor",
            Self::AffineQuantEmbedding => "&crate::__gpu::layers::AffineQuantEmbedding",
            // NOTE: no leading `&` — a per-layer plain dense
            // `[num_heads]` GpuTensor, taken verbatim at load.
            Self::AttnSinks => "crate::__gpu::tensor::GpuTensor",
            // NOTE: no leading `&` — a plain dense GpuTensor, taken verbatim at load.
            Self::Tensor => "crate::__gpu::tensor::GpuTensor",
        }
    }
}

/// One weight slot consumed by an op instance: which `WeightAccessors` method the per-arch
/// match arm goes under (`kind`), and the user-source accessor method name it calls (`base`).
///
/// ⛔ `base` IS A `String`, NOT A `syn::Ident`. This crate is linked into a target's RUNTIME, and
/// an `Ident` would drag `syn` in with it. The macro crate mints the identifier at emission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightSlot {
    pub kind: WeightKind,
    pub base: String,
}

/// The `(tape_index, op_idx, slot)` accessor slot of weight `i` on an op's site: its ordinal
/// among the site's weights of the same kind. The per-arch `WeightAccessors` are emitted by
/// this rule, so a lowering that binds a site's weight computes the slot by it too.
pub fn accessor_slot(site: &[WeightKind], i: usize) -> u32 {
    site[..i].iter().filter(|k| **k == site[i]).count() as u32
}
