/// HBM SEGMENT each tensor ROLE owns in the packed bundle (≤7; seg7 is the
/// flex-reserved program segment, never a data operand).
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize)]
pub enum SegRole {
    /// Per-step graph inputs (embeddings / cos / sin / positions) — DMA'd each step.
    Activation = 3,
    /// Model weights — packed, uploaded ONCE, resident across decode steps.
    Weight = 1,
    /// Paged K/V pools — resident, one bank a layer ([`KvBank`]).
    Kv = 2,
    /// Produced-and-consumed scratch — lifetime-colored (slots reused across
    /// non-overlapping live ranges, sized to max-concurrent-live, not the sum).
    Intermediate = 0,
    /// The single graph output (logits) — D2H'd each step.
    Logits = 4,
}

/// ⭐ WHERE THE WEIGHT SEGMENT'S NON-PER-LAYER TAIL LIVES, as the bake decided it: the segment budget's
/// verdict, carried in the baked layout as a typed fact rather than left for the runtime to infer
/// from where the placements happen to sit.
///
/// ⛔ A SPILLED TAIL IS NOT FREE. The spill slot is an intermediate colour, so it cannot be aliased
/// between bundles: every borrowing session gets its own copy of the tail (a 400 MiB tied embedding
/// spilled across 30 sessions reserves 12,000 MB of copies). So the verdict is a field of the
/// bundle, every consumer matches on it, and a model whose weights are meant to fit pins it in a
/// test. What the spill costs is memory and load, not decode time: the suffix that reads a spilled
/// embedding measured 3.37 ms a step against 3.35 ms in place.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum WeightTail {
    /// Every weight sits in the weight segment: it fits one device region.
    #[default]
    InPlace,
    /// The tail sits in `segment` (one of the target's `WEIGHT_SPILL_SEGS`) at `[0, bytes)`, which every
    /// borrowing session is handed its own copy of.
    Spilled { segment: u32, bytes: u64 },
}

impl SegRole {
    pub fn segment(self) -> usize {
        self as usize
    }
}

/// ⭐ ONE LAYER'S KV REGION — the bank of the KV segment that holds one (class, iteration) launch
/// block's pools. Every layer's pools are their OWN device allocation, at the same intra-layer
/// offsets for every layer of a class, so a layer's launch re-bases the KV slot by handing it this
/// bank's region and nothing else: no shift, no packing against a region size. Bank 0 is the
/// segment's own region (the first block, plus any pool no layer owns).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KvBank(u32);

impl KvBank {
    /// The KV segment's own region.
    pub const BASE: KvBank = KvBank(0);

    /// The bank a KV placement names.
    pub const fn of_placement(bank: u32) -> KvBank {
        KvBank(bank)
    }

    /// The index into the segment's extra-bank list, or `None`
    /// for [`Self::BASE`].
    pub const fn extra(self) -> Option<usize> {
        (self.0 as usize).checked_sub(1)
    }

    /// The bank's number, as a placement and the bake's token stream spell it.
    pub const fn get(self) -> u32 {
        self.0
    }
}
