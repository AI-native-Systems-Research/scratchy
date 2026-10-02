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

use crate::tape::ids::{ArenaSlotIdx, LayerId, SourceIx};
use crate::tape::lowered::{Binding, RuntimeBindingKind, WeightTensor};

/// A weight source's `which` tensor at `layer`, bound at `binding_index`.
pub fn source(ix: SourceIx, which: WeightTensor, layer: LayerId, binding_index: u8) -> Binding {
    Binding::Source {
        ix,
        which,
        layer,
        binding_index,
    }
}

/// The cos/sin table a rope reads: the model's rotary source, or — MRoPE — the per-forward
/// band-split table the forward writes (read with identity positions).
#[derive(Clone, Copy)]
pub enum CosSinTable {
    Static(SourceIx),
    Mrope,
}

impl CosSinTable {
    pub fn binding(self, layer: LayerId, binding_index: u8) -> Binding {
        match self {
            Self::Static(ix) => source(ix, WeightTensor::Weight, layer, binding_index),
            Self::Mrope => Binding::Runtime {
                kind: RuntimeBindingKind::MropeCosSin,
                binding_index,
            },
        }
    }
}

/// Spans rope-on-read: append the one extra attention binding — the layer
/// class's rotary `table` at `cossin_idx`. The per-block "stored unrotated →
/// rotate on read" flag rides in `block_table` bit 31 (set by the worker), so
/// there is NO separate flag buffer. Only called when the BindingSet's
/// `rope_on_read` is `Some`.
fn push_rope_on_read_bindings(
    v: &mut Vec<Binding>,
    layer: LayerId,
    table: SourceIx,
    cossin_idx: u8,
) {
    v.push(source(table, WeightTensor::Weight, layer, cossin_idx));
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
    /// Spans rope-on-read: `Some(table)` binds the layer class's rotary
    /// table (slot 7) for the IN-KERNEL rope path (sdpa-paged,
    /// gqa_shared, simdgroup steel); `None` → the 7-binding ABI
    /// (byte-identical). The per-block "stored unrotated" flag rides in
    /// `block_table` bit 31, so there is no separate flag buffer.
    pub rope_on_read: Option<SourceIx>,
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
        } else if let Some(table) = s.rope_on_read {
            push_rope_on_read_bindings(&mut v, s.kv_layer, table, 7);
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
///   4 = the layer class's rotary table
pub struct RopeOnceNaxBindingSet {
    pub kv_layer: LayerId,
    pub table: SourceIx,
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
        push_rope_on_read_bindings(&mut v, s.kv_layer, s.table, 4);
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
    /// Spans rope-on-read: `Some(table)` binds the layer class's rotary
    /// table (slot 6); `None` (every non-spans dispatch) → the 6-binding
    /// ABI, exactly as before.
    pub rope_on_read: Option<SourceIx>,
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
        if let Some(table) = s.rope_on_read {
            push_rope_on_read_bindings(&mut v, s.kv_layer, table, 6);
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
    /// The layer class's rotary table, for K under rope-on-read.
    pub rope_on_read: Option<SourceIx>,
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
        if let Some(table) = s.rope_on_read {
            push_rope_on_read_bindings(&mut v, layer, table, 9);
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
/// 3 = the cos/sin table, 4 = Positions (runtime),
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
    pub cos_sin: CosSinTable,
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
            s.cos_sin.binding(s.layer, 3),
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
/// 8/9 = q/k norm gains (the site's RmsNorm sources 0/1).
pub struct RopeAppendNormedBindingSet {
    pub q_out: ArenaSlotIdx,
    pub k_in: ArenaSlotIdx,
    pub v_in: ArenaSlotIdx,
    pub cos_sin: CosSinTable,
    pub q_gains: SourceIx,
    pub k_gains: SourceIx,
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
            s.cos_sin.binding(s.layer, 3),
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
            source(s.q_gains, WeightTensor::Weight, s.layer, 8),
            source(s.k_gains, WeightTensor::Weight, s.layer, 9),
        ]
    }
}
