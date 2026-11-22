// SPDX-License-Identifier: Apache-2.0
//! The bake's memory plan: where every buffer of a bundle sits on the device — its segment, bank
//! and byte offset — and how big each segment is. Everything an address is made of.

pub use scratchy_spyre_bundle::segments::{KvBank, SegRole, WeightTail};

/// One buffer's placement.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct TensorPlacement {
    pub tid: u32,
    pub role: SegRole,
    pub segment: usize,
    /// Which bank of the segment: a weight segment past one device region is banked at a layer
    /// boundary; 0 everywhere else.
    pub bank: u32,
    /// Byte offset within the segment (within the bank, when banked), 128 B aligned.
    pub offset: u64,
    pub size: u64,
}

/// How a staged weight's host bytes are walked into its device layout: per device axis, outermost
/// first, its extent and the host-element stride it advances.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RetileDescriptor {
    pub axes: Vec<(u64, u64)>,
    pub stick_size: u32,
    pub word_length: u32,
}

/// A bundle's memory plan.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct BundleLayout {
    pub placements: std::collections::BTreeMap<u32, TensorPlacement>,
    pub segment_bytes: [u64; scratchy_spyre_bundle::NUM_SEGMENTS],
    /// The weight segment's banks, each one device region; empty when it fits one.
    #[serde(default)]
    pub weight_bank_bytes: Vec<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kv_bank_bytes: Vec<u64>,
    /// Each KV pool's per-block banks.
    #[serde(skip)]
    pub kv_block_banks: std::collections::BTreeMap<u32, Vec<KvBank>>,
    #[serde(default)]
    pub weight_tail: WeightTail,
    /// The staged weights re-laid-out for their readers, by tensor.
    #[serde(default)]
    pub kernel_weights: std::collections::BTreeMap<u32, RetileDescriptor>,
    /// Each staged weight's held layout, agreed by every program reading it.
    #[serde(skip)]
    pub held: std::collections::BTreeMap<u32, spyre_superdsc::Held>,
    /// Each held weight's memory-view shape and element format: what its walk reads.
    #[serde(skip)]
    pub held_views: std::collections::BTreeMap<u32, (Vec<u64>, spyre_superdsc::format::DataFormat)>,
    /// The constant sticks the rung's descriptors read, by the tid each is placed under.
    pub stick_consts: Vec<(u32, spyre_superdsc::graph::ConstStick)>,
    /// `(weight tensor, multiplier)`: compile-time multipliers folded into a weight at staging.
    #[serde(default)]
    pub weight_scale_folds: Vec<(u32, f32)>,
    #[serde(default)]
    pub kv_request_stride_bytes: u64,
    #[serde(default)]
    pub num_sources: u32,
    #[serde(default)]
    pub resident_source_tids: Vec<u32>,
}

/// The segments an over-size weight segment may spill its non-per-layer tail into. Segment 6 is
/// the one slot no per-step mechanism touches: it is never re-staged from the host, never read back,
/// never shifted by a launch, and no program but the suffix's reads it — so a tail there is the sole
/// dataspace of its segment in the one program that reads it. Only the TAIL spills: the rolled body
/// reaches layer `v` by advancing the weight segment's base, so a per-layer weight anywhere else
/// would read layer 0's copy at every layer.
pub const WEIGHT_SPILL_SEGS: [usize; 1] = [6];

/// `n` rounded up to the 128-byte stick.
pub fn align128(n: u64) -> u64 {
    n.next_multiple_of(128)
}

impl BundleLayout {
    /// The device address of `tid`'s first byte: its segment's base plus its offset.
    pub fn address(&self, tid: u32) -> Option<u64> {
        self.placements
            .get(&tid)
            .map(|p| segment_base(p.segment) + p.offset)
    }
}

/// The device address segment `s` starts at.
pub const fn segment_base(s: usize) -> u64 {
    s as u64 * scratchy_spyre_bundle::MAX_SEGMENT_BYTES
}
