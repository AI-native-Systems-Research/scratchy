// SPDX-License-Identifier: Apache-2.0
//! Typed per-kernel binding-set structs.
//!
//! Each `BindingSet` struct captures the per-call-variable bindings
//! (`ArenaSlotIdx`, `LayerId`, weight thunks) for one kernel family.
//! The runtime-fixed bindings (`CuSeqlensQ`, `SeqUsedK`, `BlockTable`,
//! `NumTokensU32`) are baked into the `From<Self> for Vec<Binding<W>>`
//! conversion — they live at the same per-kernel binding index every
//! time so the lowering arm doesn't restate them.
//!
//! This catches the buffer-slot analogue of bug class #1 (the
//! `ATTN_PAGED_DEBUG_MODE` slot-99 omission): emitting a Vec<Binding>
//! by hand makes it possible to skip a runtime binding silently;
//! constructing the struct + converting can't.
//!
//! Phase 3 lands the high-binding-count attention / RoPE / QKV
//! kernels. The simpler 2-3 binding kernels (RmsNorm, SiluMul,
//! ScalarMul, Add) stay on hand-rolled `vec![Binding::…]` for now —
//! the per-kernel struct boilerplate isn't pulling its weight at that
//! size, and the visual inspection is trivial.

use crate::tape::ids::{ArenaSlotIdx, LayerId};
use crate::tape::lowered::{
    Binding, RuntimeBindingKind, WeightBundleKind, WeightLocator, WeightTensor,
};

/// Spans rope-on-read: append the one extra attention binding — the
/// class-resolved cos_sin table at `cossin_idx`. The per-block "stored
/// unrotated → rotate on read" flag rides in `block_table` bit 31 (set
/// by the worker), so there is NO separate flag buffer. cos_sin is
/// resolved by layer class via `WeightBundleKind::RopeOnReadCosSin`, so
/// its `WeightLocator` is unused (a nominal zero locator). Only called
/// when the BindingSet's `rope_on_read` is `Some`.
fn push_rope_on_read_bindings(
    v: &mut Vec<Binding>,
    layer: LayerId,
    is_global: bool,
    cossin_idx: u8,
) {
    v.push(Binding::Weight {
        kind: WeightBundleKind::RopeOnReadCosSin { is_global },
        which: WeightTensor::Weight,
        layer,
        // Unused for RopeOnReadCosSin (resolved by class, not operand).
        locator: WeightLocator {
            bucket: 0,
            op_idx: 0,
            slot: 0,
        },
        binding_index: cossin_idx,
    });
}

// ── AttentionPrefillSdpaPaged (both sdpa_vector and steel variants) ─

/// Bindings for `KernelId::AttentionPrefillSdpaPaged`. Seven slots:
/// 0 = output (arena), 1 = Q (arena), 2 = CuSeqlensQ (runtime),
/// 3 = SeqUsedK (runtime), 4 = BlockTable (runtime),
/// 5 = KvCacheK\[layer\] (runtime), 6 = KvCacheV\[layer\] (runtime).
pub struct AttentionPrefillPagedBindingSet {
    pub output: ArenaSlotIdx,
    pub q: ArenaSlotIdx,
    pub kv_layer: LayerId,
    /// Spans rope-on-read: `Some(is_global)` binds the class-resolved
    /// cos_sin (slot 7) for the IN-KERNEL rope path (sdpa-paged,
    /// gqa_shared, simdgroup steel); `None` → the 7-binding ABI
    /// (byte-identical). The per-block "stored unrotated" flag rides in
    /// `block_table` bit 31, so there is no separate flag buffer.
    pub rope_on_read: Option<bool>,
    /// Spans rope-on-read on NAX (rope-once-to-scratch): when `true`,
    /// slot 7 binds the shared `Binding::RopedKScratch` (the pre-roped K
    /// written by `KernelId::RopeOnceNax`) INSTEAD of cos_sin — the NAX
    /// attention reads pre-roped K and does no per-tile rotation. Mutually
    /// exclusive with the cos_sin slot-7 binding (only one writes slot 7).
    pub nax_roped_k_scratch: bool,
}

impl From<AttentionPrefillPagedBindingSet> for Vec<Binding> {
    fn from(s: AttentionPrefillPagedBindingSet) -> Vec<Binding> {
        let mut v = vec![
            Binding::ArenaSlot {
                slot: s.output.get(),
                binding_index: 0,
            },
            Binding::ArenaSlot {
                slot: s.q.get(),
                binding_index: 1,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::CuSeqlensQ,
                binding_index: 2,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::SeqUsedK,
                binding_index: 3,
            },
            Binding::Runtime {
                // `layer`'s KV-cache group block table (full or sliding).
                kind: RuntimeBindingKind::BlockTable { layer: s.kv_layer },
                binding_index: 4,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheK { layer: s.kv_layer },
                binding_index: 5,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheV { layer: s.kv_layer },
                binding_index: 6,
            },
        ];
        if s.nax_roped_k_scratch {
            // NAX spans: pre-roped K from the scratch at slot 7.
            v.push(Binding::RopedKScratch { binding_index: 7 });
        } else if let Some(is_global) = s.rope_on_read {
            push_rope_on_read_bindings(&mut v, s.kv_layer, is_global, 7);
        }
        // Block-diagonal span attention: the per-block span-label buffer at slot
        // 8, bound whenever spans/rope-on-read is active (the kernel reads it
        // only under ATTN_ROR). All-zero on a non-spans request ⇒ mask inert.
        if s.nax_roped_k_scratch || s.rope_on_read.is_some() {
            v.push(Binding::Runtime {
                kind: RuntimeBindingKind::SpanIds,
                binding_index: 8,
            });
        }
        v
    }
}

/// Bindings for `KernelId::RopeOnceNax` (spans rope-on-read, NAX prefill):
/// ropes the cache's K into the shared `Binding::RopedKScratch` ONCE.
///   0 = RopedKScratch (out)   1 = BlockTable[layer] (bit 31 = unrotated)
///   2 = KvCacheK[layer]       3 = SeqUsedK
///   4 = class-resolved cos_sin (`RopeOnReadCosSin`)
pub struct RopeOnceNaxBindingSet {
    pub kv_layer: LayerId,
    pub is_global: bool,
}

impl From<RopeOnceNaxBindingSet> for Vec<Binding> {
    fn from(s: RopeOnceNaxBindingSet) -> Vec<Binding> {
        let mut v = vec![
            Binding::RopedKScratch { binding_index: 0 },
            Binding::Runtime {
                kind: RuntimeBindingKind::BlockTable { layer: s.kv_layer },
                binding_index: 1,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheK { layer: s.kv_layer },
                binding_index: 2,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::SeqUsedK,
                binding_index: 3,
            },
        ];
        push_rope_on_read_bindings(&mut v, s.kv_layer, s.is_global, 4);
        v
    }
}

// ── AttentionViaCache (decode) ─────────────────────────────────────

/// Bindings for `KernelId::AttentionViaCache`. Six slots:
/// 0 = output (arena), 1 = Q (arena), 2 = SeqUsedK (runtime),
/// 3 = BlockTable (runtime), 4 = KvCacheK\[layer\] (runtime),
/// 5 = KvCacheV\[layer\] (runtime).
///
/// Same `kv_layer` payload appears at slots 4 + 5 — the layer index
/// is single-field on the BindingSet so they can't drift.
pub struct AttentionViaCacheBindingSet {
    pub output: ArenaSlotIdx,
    pub q: ArenaSlotIdx,
    pub kv_layer: LayerId,
    /// Spans rope-on-read: `Some(is_global)` binds the per-physical-block
    /// unrotated-flag buffer (slot 6) + the class-resolved cos_sin (slot
    /// 7); `None` (every non-spans dispatch) → the 6-binding ABI, exactly
    /// as before. `is_global` selects the global vs sliding RoPE cache.
    pub rope_on_read: Option<bool>,
}

impl From<AttentionViaCacheBindingSet> for Vec<Binding> {
    fn from(s: AttentionViaCacheBindingSet) -> Vec<Binding> {
        let mut v = vec![
            Binding::ArenaSlot {
                slot: s.output.get(),
                binding_index: 0,
            },
            Binding::ArenaSlot {
                slot: s.q.get(),
                binding_index: 1,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::SeqUsedK,
                binding_index: 2,
            },
            Binding::Runtime {
                // `layer`'s KV-cache group block table.
                kind: RuntimeBindingKind::BlockTable { layer: s.kv_layer },
                binding_index: 3,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheK { layer: s.kv_layer },
                binding_index: 4,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheV { layer: s.kv_layer },
                binding_index: 5,
            },
        ];
        if let Some(is_global) = s.rope_on_read {
            push_rope_on_read_bindings(&mut v, s.kv_layer, is_global, 6);
        }
        v
    }
}

/// `KernelId::AttentionViaCacheTq`: the bindings its `AttentionViaCache`
/// twin's set gains — the layer's packed K/V codes + norms, the shared
/// signs/codebook, and `slot_mapping` (slots 7..=13).
pub struct TqAttentionBindingSet {
    pub kv_layer: LayerId,
}

impl From<TqAttentionBindingSet> for Vec<Binding> {
    fn from(s: TqAttentionBindingSet) -> Vec<Binding> {
        let layer = s.kv_layer;
        [
            RuntimeBindingKind::TqPackedK { layer },
            RuntimeBindingKind::TqPackedV { layer },
            RuntimeBindingKind::TqNormsK { layer },
            RuntimeBindingKind::TqNormsV { layer },
            RuntimeBindingKind::TqSigns,
            RuntimeBindingKind::TqCentroids,
            RuntimeBindingKind::SlotMapping { layer },
        ]
        .into_iter()
        .zip(7u8..)
        .map(|(kind, binding_index)| Binding::Runtime {
            kind,
            binding_index,
        })
        .collect()
    }
}

/// `KernelId::TqStageRotated`: the layer's K (or V) scratch, the step's
/// addressing, the layer's packed codes + norms, the codebook, and — for K
/// under rope-on-read — the class-resolved cos_sin (slot 9).
pub struct TqStageBindingSet {
    pub kv_layer: LayerId,
    pub is_v: bool,
    /// `Some(is_global)` for K under rope-on-read.
    pub rope_on_read: Option<bool>,
}

impl From<TqStageBindingSet> for Vec<Binding> {
    fn from(s: TqStageBindingSet) -> Vec<Binding> {
        let layer = s.kv_layer;
        let (cache, packed, norms) = if s.is_v {
            (
                RuntimeBindingKind::KvCacheV { layer },
                RuntimeBindingKind::TqPackedV { layer },
                RuntimeBindingKind::TqNormsV { layer },
            )
        } else {
            (
                RuntimeBindingKind::KvCacheK { layer },
                RuntimeBindingKind::TqPackedK { layer },
                RuntimeBindingKind::TqNormsK { layer },
            )
        };
        let mut v: Vec<Binding> = [
            cache,
            RuntimeBindingKind::BlockTable { layer },
            RuntimeBindingKind::SeqUsedK,
            RuntimeBindingKind::CuSeqlensQ,
            RuntimeBindingKind::SlotMapping { layer },
            packed,
            norms,
            RuntimeBindingKind::TqSigns,
            RuntimeBindingKind::TqCentroids,
        ]
        .into_iter()
        .zip(0u8..)
        .map(|(kind, binding_index)| Binding::Runtime {
            kind,
            binding_index,
        })
        .collect();
        if let Some(is_global) = s.rope_on_read {
            push_rope_on_read_bindings(&mut v, layer, is_global, 9);
        }
        v
    }
}

/// `KernelId::TqRotateRows`: the rows (in/out, arena) and the codebook signs.
pub struct TqRotateRowsBindingSet {
    pub rows: ArenaSlotIdx,
}

impl From<TqRotateRowsBindingSet> for Vec<Binding> {
    fn from(s: TqRotateRowsBindingSet) -> Vec<Binding> {
        vec![
            Binding::ArenaSlot {
                slot: s.rows.get(),
                binding_index: 0,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::TqSigns,
                binding_index: 1,
            },
        ]
    }
}

// ── RopeAppend ─────────────────────────────────────────────────────

/// Bindings for `KernelId::RopeAppend`. Eight slots:
/// 0 = Q-rotated out (arena), 1 = K out (arena), 2 = V out (arena),
/// 3 = CosSin\[layer\] (weight), 4 = Positions (runtime),
/// 5 = SlotMapping (runtime), 6 = KvCacheK\[layer\] (runtime),
/// 7 = KvCacheV\[layer\] (runtime).
///
/// The `layer` on the cos/sin table MUST match the layer on the
/// KvCache* slots — single-field `layer` on the BindingSet enforces
/// it.
pub struct RopeAppendBindingSet {
    pub q_out: ArenaSlotIdx,
    pub k_out: ArenaSlotIdx,
    pub v_out: ArenaSlotIdx,
    pub cos_sin_locator: WeightLocator,
    pub layer: LayerId,
    // Spans rope-on-read: the "store K unrotated" flag rides in
    // slot_mapping bit 31 (set by the worker), so this kernel needs no
    // extra binding — only the ROPE_ROR function constant (slot 9).
}

impl From<RopeAppendBindingSet> for Vec<Binding> {
    fn from(s: RopeAppendBindingSet) -> Vec<Binding> {
        vec![
            Binding::ArenaSlot {
                slot: s.q_out.get(),
                binding_index: 0,
            },
            Binding::ArenaSlot {
                slot: s.k_out.get(),
                binding_index: 1,
            },
            Binding::ArenaSlot {
                slot: s.v_out.get(),
                binding_index: 2,
            },
            Binding::Weight {
                kind: WeightBundleKind::CosSin,
                which: WeightTensor::Weight,
                layer: s.layer,
                locator: s.cos_sin_locator,
                binding_index: 3,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::Positions,
                binding_index: 4,
            },
            Binding::Runtime {
                // `layer`'s KV-cache group slot_mapping.
                kind: RuntimeBindingKind::SlotMapping { layer: s.layer },
                binding_index: 5,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheK { layer: s.layer },
                binding_index: 6,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheV { layer: s.layer },
                binding_index: 7,
            },
        ]
    }
}

// ── RopeAppendNormed (Gemma4 norm-prologue rope) ───────────────────

/// Bindings for `KernelId::RopeAppendNormed`. Slots 0..7 mirror
/// [`RopeAppendBindingSet`] except 1/2 bind the RAW (pre-norm) K/V
/// inputs (read-only; the kernel writes K/V to the cache only);
/// 8/9 = q/k norm gains (RmsNorm-kind weight sub-slots 0/1).
pub struct RopeAppendNormedBindingSet {
    pub q_out: ArenaSlotIdx,
    pub k_in: ArenaSlotIdx,
    pub v_in: ArenaSlotIdx,
    pub cos_sin_locator: WeightLocator,
    pub q_gains_locator: WeightLocator,
    pub k_gains_locator: WeightLocator,
    pub layer: LayerId,
    // Spans rope-on-read: flag rides in slot_mapping bit 31 (no binding;
    // only the ROPE_ROR fn-const at slot 9).
}

impl From<RopeAppendNormedBindingSet> for Vec<Binding> {
    fn from(s: RopeAppendNormedBindingSet) -> Vec<Binding> {
        vec![
            Binding::ArenaSlot {
                slot: s.q_out.get(),
                binding_index: 0,
            },
            Binding::ArenaSlot {
                slot: s.k_in.get(),
                binding_index: 1,
            },
            Binding::ArenaSlot {
                slot: s.v_in.get(),
                binding_index: 2,
            },
            Binding::Weight {
                kind: WeightBundleKind::CosSin,
                which: WeightTensor::Weight,
                layer: s.layer,
                locator: s.cos_sin_locator,
                binding_index: 3,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::Positions,
                binding_index: 4,
            },
            Binding::Runtime {
                // `layer`'s KV-cache group slot_mapping.
                kind: RuntimeBindingKind::SlotMapping { layer: s.layer },
                binding_index: 5,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheK { layer: s.layer },
                binding_index: 6,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheV { layer: s.layer },
                binding_index: 7,
            },
            Binding::Weight {
                kind: WeightBundleKind::RmsNorm,
                which: WeightTensor::Weight,
                layer: s.layer,
                locator: s.q_gains_locator,
                binding_index: 8,
            },
            Binding::Weight {
                kind: WeightBundleKind::RmsNorm,
                which: WeightTensor::Weight,
                layer: s.layer,
                locator: s.k_gains_locator,
                binding_index: 9,
            },
        ]
    }
}

// ── FusedQkvRopeCache (dense BF16/F16) ─────────────────────────────

/// Bindings for `KernelId::FusedQkvRopeCache`. Eight slots:
/// 0 = Q out (arena), 1 = input (arena),
/// 2 = packed \[Q|K|V\] weight (LinearLayer, weight),
/// 3 = CosSin\[layer\] (weight), 4 = Positions (runtime),
/// 5 = SlotMapping (runtime), 6 = KvCacheK\[layer\] (runtime),
/// 7 = KvCacheV\[layer\] (runtime).
pub struct FusedQkvRopeCacheBindingSet {
    pub q_out: ArenaSlotIdx,
    pub input: ArenaSlotIdx,
    pub qkv_locator: WeightLocator,
    pub cos_sin_locator: WeightLocator,
    pub layer: LayerId,
}

impl From<FusedQkvRopeCacheBindingSet> for Vec<Binding> {
    fn from(s: FusedQkvRopeCacheBindingSet) -> Vec<Binding> {
        vec![
            Binding::ArenaSlot {
                slot: s.q_out.get(),
                binding_index: 0,
            },
            Binding::ArenaSlot {
                slot: s.input.get(),
                binding_index: 1,
            },
            Binding::Weight {
                kind: WeightBundleKind::LinearLayer,
                which: WeightTensor::Weight,
                layer: s.layer,
                locator: s.qkv_locator,
                binding_index: 2,
            },
            Binding::Weight {
                kind: WeightBundleKind::CosSin,
                which: WeightTensor::Weight,
                layer: s.layer,
                locator: s.cos_sin_locator,
                binding_index: 3,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::Positions,
                binding_index: 4,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::SlotMapping { layer: s.layer },
                binding_index: 5,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheK { layer: s.layer },
                binding_index: 6,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheV { layer: s.layer },
                binding_index: 7,
            },
        ]
    }
}
