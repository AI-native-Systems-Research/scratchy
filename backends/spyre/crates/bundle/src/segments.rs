/// HBM SEGMENT each tensor ROLE owns in the packed bundle (≤7; seg7 is the
/// flex-reserved program segment, never a data operand). Indexes [`crate::wire::SEGMENT_OFFSETS`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize)]
pub enum SegRole {
    // DIAGNOSTIC SWAP (2026-06-28): Intermediate↔Activation (was 0/3) to land the SFP
    // transcendental's operands (meps/inv = Intermediates) in REGION 0 like torch-spyre's
    // working rsqrt (input region1 / output region0), vs scratchy's region3. Tests whether
    // the operand region is the lever for the SFP Newton-Raphson refine (the SOLE remaining
    // byte-difference vs torch-spyre's byte-identical program). All segs are MemoryType::Tensor
    // so this SHOULD be inert — if it refines, the region matters despite same memory type.
    /// Per-step graph inputs (embeddings / cos / sin / positions) — DMA'd each step.
    Activation = 3,
    /// Model weights — packed, uploaded ONCE, resident across decode steps.
    Weight = 1,
    /// Paged K/V cache — resident (reserved; attention is host-routed today, see
    /// `lower_graph_to_superdsc` — populated once attention moves on-device).
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
/// ⛔ A SPILLED TAIL IS NOT FREE, AND IT WAS FOUND IN A SERVE LOG. The spill slot is an intermediate
/// colour, so it cannot be aliased between bundles: every borrowing session gets its own copy of the
/// tail (granite-3.1-8b fp16 padded its gate/up to 224 sticks, spilled its 400 MiB tied embedding
/// into seg6, and reserved 12,000 MB of copies across 30 sessions — 28,644 MB against 15,683 MB
/// unspilled). Nothing at `cargo build` said it had spilled: the bake's own line goes to a stderr
/// that cargo swallows on success. So the verdict is a field of the bundle, every consumer matches on
/// it, and a model whose weights are meant to fit pins it in a test.
///
/// ⚠️ WHAT THE SPILL COSTS IS MEMORY AND LOAD, NOT DECODE TIME — MEASURED. That padded build decoded
/// 18–26% slower, and a keyed `SCRATCHY_SDSC_GROUP_TIME` split of one 10k-context step put all of it
/// (49.1 of 49.6 ms) in the MLP group: 224 sticks on 32 cores is 7 sticks a core, a prime, so each of
/// gate and up ran 7 one-stick trips where 200 sticks on 25 cores runs 4 two-stick trips. The suffix
/// that reads the spilled embedding ran 3.37 ms a step against 3.35 ms in place.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum WeightTail {
    /// Every weight sits in the weight segment: it fits one device region.
    #[default]
    InPlace,
    /// The tail was moved to `segment` (one of [`WEIGHT_SPILL_SEGS`]) at `[0, bytes)`, which every
    /// borrowing session has to be handed its own copy of.
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

    /// The bank a KV placement names ([`TensorPlacement::bank`]).
    pub const fn of_placement(bank: u32) -> KvBank {
        KvBank(bank)
    }

    /// The index into the segment's extra-bank list ([`BundleLayout::kv_bank_bytes`]), or `None`
    /// for [`Self::BASE`].
    pub const fn extra(self) -> Option<usize> {
        (self.0 as usize).checked_sub(1)
    }

    /// The bank's number, as a placement and the bake's token stream spell it.
    pub const fn get(self) -> u32 {
        self.0
    }
}
