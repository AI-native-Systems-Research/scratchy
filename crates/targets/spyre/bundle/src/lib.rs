// SPDX-License-Identifier: Apache-2.0
//! A baked SuperDSC bundle: the artifact the Spyre emitter produces and the Spyre runtime consumes.
//!
//! ## One type family, two lifetimes
//!
//! Every struct here is generic over `'a` and holds [`Cow`], so ONE type serves both ends:
//!
//! * the **emitter** builds `BundleCode<'static>` with `Cow::Owned` from data it is about to drop,
//! * the **macro** emits it as a `const`-constructed `Cow::Borrowed` value inside
//!   `inventory::submit!`, with the device image as `include_bytes!`,
//! * the **runtime** reads that `&'static BundleCode` directly.
//!
//! ⛔ THE SINGLE TYPE IS THE POINT. A separate emit-side and runtime-side struct means a
//! field-for-field conversion between them, and that conversion is where a field goes missing without
//! anything failing. Here a field exists once, so the only way to have it at one end and not the other
//! is not to declare it.
//!
//! ## Identity is a fingerprint
//!
//! [`bundle`] looks a bundle up by its content fingerprint. A rolled body names its prefix, suffix,
//! fold-fused twin and every ladder rung as fingerprints ([`SiblingFp`]), and each of those is another
//! registered [`BundleCode`] — so resolving a sibling is a lookup, and one this binary does not carry
//! is `None` at the call site that asked for it.
//!
//! ## Why this is its own crate
//!
//! Three crates name these types and none of them may depend on another:
//!
//! * `scratchy-subtile`'s SuperDSC lowering BUILDS them — and subtile is the generic scheduling layer
//!   shared with the cuda/metal targets, so it must not own Spyre artifacts.
//! * the `#[forward]` macro BAKES them, emitting `inventory::submit!` tokens into each arch crate.
//! * `scratchy-target-spyre` READS them at load, and re-exports this crate as its `bundle_code` so the
//!   emitted tokens have one stable path to name.
//!
//! So the family lives below all three, with no features and no backend deps.

/// The DeepTools `processComputeOnHostCommand` port, run at BUILD time — see [`correction`].
pub mod correction;

/// Re-exported so the `inventory::submit!` the macro emits into each arch crate resolves through the
/// same path as the types it submits — the arch crate need not depend on `inventory` itself.
pub use inventory;

use std::borrow::Cow;
use std::num::NonZeroU32;

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  The two axes of the rung grid
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐⭐⭐⭐⭐ THE BATCH WIDTH A DECODE RUNG WAS BAKED AT — how many requests one launch holds.
///
/// ⛔ NOT A SWEEP EXTENT. [`SweptCols`] is the number of KV COLUMNS one fold pass sweeps. Both are
/// counts, both were bare `u32` in lists that have both been called `decode_rungs`, and the two
/// ladders even share plausible values — so "smallest rung >= live request count" compiled fine
/// against either. A body baked for a 64-column sweep, selected because four requests are live, is a
/// wrong ANSWER and not a fault: the fold covers `active_cap * pages` columns, and a row deeper than
/// that silently attends its prompt and its newest token alone.
///
/// One quantity, one type. There is no `From<u32>`: the only way in is [`Self::new`], at the site that
/// knows which axis it is holding.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct RungSeqs(u32);

impl RungSeqs {
    /// The width, from a site that has just decided this is a WIDTH and not a sweep.
    pub const fn new(seqs: u32) -> RungSeqs {
        RungSeqs(seqs)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// ⭐⭐⭐⭐⭐ THE COLUMNS ONE FOLD PASS SWEEPS in the body a rung actually runs — its `active_cap`.
///
/// The fold covers `swept * pages` columns, NOT `PAGE_SLOTS * pages`. The emitter bakes bodies at 64,
/// 128 and 256; at 128 with 6 pages the covered window is 768 slots, so a batch whose shared write
/// slot is 1441 leaves the DEEPEST row's tail described by no swept column — a wrong answer with no
/// fault.
///
/// A distinct type from [`RungSeqs`] because they are the two axes of the rung grid and both are
/// counts. Defined HERE, in the leaf, so the baked ladder ([`LadderRung::active_cap`]) can be typed:
/// while the launch path still crossed into C++ this had to be a bare `u32` on the way through, which
/// is the whole reason the two axes were ever interchangeable.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct SweptCols(u32);

impl SweptCols {
    /// The swept extent of the body this rung names — the emitter's `active_cap` for that bundle.
    pub const fn new(cols: u32) -> SweptCols {
        SweptCols(cols)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    /// Does a fold whose window is this wide cover `positions` of KV?
    ///
    /// The one comparison rung selection makes, named so `swept >= valid_len` cannot be written
    /// operand-swapped on bare integers — and so it cannot be asked of a [`RungSeqs`] at all.
    pub const fn covers(self, positions: u64) -> bool {
        self.0 as u64 >= positions
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  Placements — the memory plan
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The ≤7-packed-segment executor model: one device region per occupied segment.
pub const NUM_SEGMENTS: usize = 7;

/// ⛔⛔⛔ THE BYTES ONE SEGMENT MAY HOLD — because "one device region per occupied segment" (above)
/// is not just a mental model, it is an ALLOCATION, and a flex region is capped at 16 GiB
/// (`flex::MAX_REGION_SIZE`; the port is `flex_rs::allocator::MAX_REGION_BYTES`, and
/// `scratchy-target-spyre` const-asserts the two are equal so they cannot drift). A `FlexAllocator`
/// request is served from ONE region and never spans two, so this is a hard ceiling on a segment
/// regardless of how much of the card is free.
///
/// 🛑 MEASURED, and it is why this constant exists rather than being discovered on a card:
/// granite-3.1-8b at **fp16** packs a 17,365,082,112 B (16.17 GiB) weight segment — 40 layers ×
/// 423,641,088 B, plus the 51200-wide tied embedding — which is 177 MiB past this ceiling. The load
/// dies in `prepare` with an OOM whose own numbers look self-contradictory
/// (`requested_bytes=17365082112, free_space_bytes=103048964224`) because the free space is spread
/// across the five regions the request cannot reach. Every byte in that sum is a compile-time
/// constant of the bundle, so the refusal belongs at `cargo build`, naming the segment — see
/// `audit_layout_addresses`.
pub const MAX_SEGMENT_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// ⭐ THE OPERAND IDENTITY — declared in the lowering's own scratchy-free leaf and re-exported here,
/// so every `bundle::PlaceId` / `bundle::SynthRole` path in the tree resolves unchanged. See
/// [`ktir_superdsc::place`] for both types and for why a [`SynthRole`] belongs to the lowering that
/// invents the intermediate rather than to the bake that places it.
///
/// ⚠️ THIS CRATE'S HEADER SAYS IT MAY "DEPEND ON NOTHING EITHER END OWNS", AND THIS IS A DEPENDENCY.
/// `ktir-superdsc` is neither end: it is a leaf BELOW all three crates that header names, with no
/// scratchy dependency of its own, and all three still reach these types through this crate.
/// Recorded rather than reworded: the rule stands as written, and this note is the exception it does
/// not name.
pub use ktir_superdsc::place::{PlaceId, SynthRole};

/// One tensor's baked placement inside its segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    /// Which tensor this places.
    pub id: PlaceId,
    /// Segment id (`0..NUM_SEGMENTS`).
    pub segment: u32,
    /// ⭐ WHICH **BANK** OF THAT SEGMENT — the second coordinate a weight needs once one segment's
    /// worth of addresses is no longer one device region's worth of bytes.
    ///
    /// A segment is ONE region and a region is [`MAX_SEGMENT_BYTES`], so the weight segment used to
    /// be the ceiling on a model. It is not a ceiling on the *addresses*: the rolled body bakes only
    /// LAYER 0's offsets and reaches layer `v` by advancing the base it is handed
    /// (`off[SEG_WEIGHT] = v·weight_stride`), and `tensor_allocs` is positional PER LAUNCH with no
    /// segment identity in a `DevAddr`. So the weight segment can be a BANK of regions, each holding
    /// a whole number of layers and each addressed as segment 1 by the same descriptors.
    ///
    /// ⛔ A LAUNCH HAS ONE BASE PER SEGMENT, so a launch may only touch ONE bank — which is why this
    /// is a property of the PLACEMENT and the split is at a LAYER boundary. The emitter proves the
    /// one-bank-per-launch-group rule at build time (`bank_weight_segment`); nothing here can.
    ///
    /// 0 for every tensor of every bundle whose weights fit one region — which is every bundle that
    /// worked before banking existed, so their placements are byte-identical.
    pub bank: u32,
    /// Byte offset WITHIN the segment region (128 B aligned) — and within its BANK, when `bank` is
    /// set: each bank is packed from 0, so this is what the descriptor bakes either way.
    pub offset: u64,
    /// Byte size.
    pub size: u64,
    /// This is the graph result read back each step (`SegRole::Logits`).
    pub is_logits: bool,
}

/// A matmul KERNEL weight's device-tile walk — the PT array reads the device TILE layout
/// (`[out/64, in, 64]`), not the row-major flat bytes, so staging re-tiles through this.
///
/// Derived solely from the emitter's `DeviceTileLayout` witness, which is also what the per-core
/// address is derived from, so the staging walk and the baked address cannot disagree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelWeight<'a> {
    /// The tensor this describes.
    pub id: PlaceId,
    /// Device extents in dim_map order.
    pub device_size: Cow<'a, [u64]>,
    /// Host-element stride per device axis.
    pub stride_map: Cow<'a, [u64]>,
    /// Elements per stick (64 for fp16).
    pub stick_size: u32,
    /// Bytes per element (2 for fp16, 1 for fp8/int8).
    pub word_length: u32,
}

/// The whole-bundle memory plan.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BundleLayout<'a> {
    /// Bytes occupied per segment. For the weight segment this is BANK 0's bytes — see
    /// [`BundleLayout::weight_bank_bytes`].
    pub segment_bytes: [u64; NUM_SEGMENTS],
    /// ⭐ THE WEIGHT SEGMENT'S EXTRA BANKS, in bank order — bytes of banks `1..N`, so bank `b`'s
    /// extent is `weight_bank_bytes[b - 1]` and bank 0's is `segment_bytes[SEG_WEIGHT]`.
    ///
    /// EMPTY for every bundle whose weights fit one device region, which is what keeps every such
    /// bundle's layout and every runtime path byte-identical: an empty list means "one bank", the
    /// single-region case that existed before banking. See [`Placement::bank`] for why a bank is a
    /// coordinate rather than another segment (there is no free segment slot: 0/3 are
    /// intermediates+activations, 2 is KV, 4 logits, 5+6 intermediate COLORS).
    pub weight_bank_bytes: Cow<'a, [u64]>,
    /// Every placed tensor, in a deterministic order.
    pub places: Cow<'a, [Placement]>,
    /// The matmul kernel weights that must be staged tiled.
    pub kernel_weights: Cow<'a, [KernelWeight<'a>]>,
    /// Distinct ScalarMul scale VALUES, in a stable order. Index `i` ↔ the reserved const tid
    /// `scalarmul_scale_tid(i)`, which the worker binds as a `[1,1]` const so the on-device
    /// pointwise `mul` gets its real multiplier (unbound = 0 = wrong output).
    pub scalarmul_scales: Cow<'a, [f32]>,
    /// Bytes between two requests' KV inside one page+layer, or 0 for an unpaged bundle. The
    /// EMITTER's number, travelling to the runtime rather than being re-derived there.
    pub kv_request_stride_bytes: u64,
}

impl<'a> BundleLayout<'a> {
    /// The placement of `id`, if this bundle placed it.
    ///
    /// ⛔ BY IDENTITY, NOT BY A FORMATTED NAME. The reserved ids are `u32` consts
    /// (`IDENTITY_TID`, `ATTN_MASK_TID`, `ksplit_block_tid(0)`), so asking by id keeps the question in
    /// the type the answer is in — no `format!("t{}", CONST)` round trip. This was TWO lookups, one
    /// by name and one by tid, which is what having two spellings of one identity costs.
    pub fn place(&self, id: PlaceId) -> Option<&Placement> {
        self.places.iter().find(|p| p.id == id)
    }

    /// The placement of SubtileIR tensor `tid`, if this bundle placed it.
    pub fn place_of_tid(&self, tid: u32) -> Option<&Placement> {
        self.place(PlaceId::Act(tid))
    }

    /// The graph result's placement (`SegRole::Logits`).
    pub fn logits(&self) -> Option<&Placement> {
        self.places.iter().find(|p| p.is_logits)
    }

    /// How many BANKS the weight segment occupies — always ≥ 1, and exactly 1 for every bundle
    /// whose weights fit one device region. Derived from [`Self::weight_bank_bytes`] so the count
    /// and the extents cannot disagree.
    pub fn weight_banks(&self) -> usize {
        1 + self.weight_bank_bytes.len()
    }

    /// Bank `b`'s extent in bytes: bank 0 is the weight segment's own total, the rest come from
    /// [`Self::weight_bank_bytes`]. `None` for a bank this bundle does not have.
    pub fn weight_bank_extent(&self, b: usize, weight_seg: usize) -> Option<u64> {
        match b.checked_sub(1) {
            None => self.segment_bytes.get(weight_seg).copied(),
            Some(i) => self.weight_bank_bytes.get(i).copied(),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  The per-op launch index
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Which rows one fold pass of a bundle's page-fold group covers.
///
/// ⛔ A CLOSED ENUM AND NOT A `bool`, so "whole batch" is a value and not the absence of one. The
/// worker derives its intermediate-segment rebase stride from this, and a wrong stride is a request
/// answering from another request's history: fluent, wrong, and undetectable downstream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FoldRows {
    /// One pass covers the WHOLE batch — the intermediate segment is not rebased per row.
    #[default]
    WholeBatch,
    /// One pass covers ONE request — each row rebases the intermediate segment by its own stride.
    PerRequest,
}

/// The address shifts a launch applies to its program's segment bases.
///
/// A group that carries any of these is a SINGLETON at every group size, because the shift is applied
/// once to the whole group — `group_ranges` breaks a run at such a trip for exactly that reason.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvShifts {
    /// seg2 shift per write slot, for a group that writes one token into the KV cache. `0` for a group
    /// that needs no slot shift.
    pub slot_stride_bytes: u32,
    /// The incremental Kᵀ restickify's shift, or `None` for a group that is not one.
    ///
    /// ⭐⭐⭐⭐⭐ THE STRIDE AND THE PAGE IT INDEXES INTO ARE ONE VALUE, and that is the whole reason this
    /// is a [`SlabShift`] and not a `u32` beside [`Self::page_slots`]. A slab shift is a distance per
    /// 64-slot BLOCK; a block index exists only inside a page. As two independent fields, "a stride with
    /// no page" was a value you could write down — and the runtime that read it divided an absolute
    /// history by 64 with nothing bounding the quotient, so from slot 256 on it shifted a whole page past
    /// where the launch had been rebased. Below 257 tokens the two forms agree, which is why it shipped.
    /// [`SlabShift::new`] takes both or neither, so that shape no longer exists to be checked for.
    pub slab: Option<SlabShift>,
    /// The write-slot modulus AND the declaration that this bundle is PAGED — a runtime that does not
    /// know the key must refuse rather than apply an absolute-position shift and land past the page.
    /// `0` for an unpaged bundle.
    pub page_slots: u32,
    /// Which request's KV this group addresses (batched decode). `0` when unbatched.
    pub request: u32,
    /// This is the group the runtime re-launches once per page of resident KV.
    pub page_fold: bool,
    /// The fold's kernels carry the request axis, so one pass per PAGE suffices instead of one per
    /// (request, page).
    pub batched_requests: bool,
    /// What one fold pass covers — see [`FoldRows`]. Meaningful only when [`Self::page_fold`].
    pub fold_rows: FoldRows,
}

impl KvShifts {
    /// Does this group write one token into the KV cache? DERIVED from the stride, so a stride and its
    /// flag cannot disagree.
    pub const fn slot_write(self) -> bool {
        self.slot_stride_bytes > 0
    }

    /// Does this group restickify a Kᵀ slab? Derived, for the same reason.
    pub const fn slab_write(self) -> bool {
        self.slab.is_some()
    }
}

/// ⭐⭐⭐⭐⭐ THE INCREMENTAL Kᵀ RESTICKIFY'S SEGMENT SHIFT — a distance per 64-slot stick-block, TOGETHER
/// WITH THE PAGE THAT BOUNDS THE BLOCK INDEX, because neither is meaningful without the other.
///
/// The op is baked at block 0 and the runtime moves it onto the live block. "Which block" is
/// `position mod page / 64`, so the page is not context the caller may or may not supply — it is half
/// the arithmetic. Carrying it here means [`shift_at`](Self::shift_at) is the only way to spend the
/// stride, the wrap happens inside it, and there is no arrangement of these fields that skips it.
///
/// Both extents are [`NonZeroU32`], so a zero page (which would divide by zero) and a zero stride (a
/// slab group whose every step re-transposes block 0) are not values either.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlabShift {
    stride_bytes: NonZeroU32,
    page_slots: NonZeroU32,
}

impl SlabShift {
    /// Both, or neither — there is no other constructor.
    pub const fn new(stride_bytes: NonZeroU32, page_slots: NonZeroU32) -> SlabShift {
        SlabShift {
            stride_bytes,
            page_slots,
        }
    }

    /// Bytes to shift the KV segment so this op lands on the block holding `pos`.
    ///
    /// `pos` is a request's ABSOLUTE history and is wrapped into its page HERE — the one place it can
    /// be, because this is the one place that holds both numbers. `rem_euclid` so a negative position
    /// cannot produce a negative block.
    pub const fn shift_at(self, pos: i64) -> u64 {
        let page = self.page_slots.get() as i64;
        let block = pos.rem_euclid(page) / SLAB_SLOTS;
        (block as u64) * (self.stride_bytes.get() as u64)
    }

    /// The declared stride, for the group-uniformity comparison that is the only other reader.
    pub const fn stride_bytes(self) -> NonZeroU32 {
        self.stride_bytes
    }

    /// The page, for the ONE caller that has to take this value apart and put it back together: the
    /// `#[forward]` codegen, which re-emits it through [`new`](Self::new) so the baked `static` cannot
    /// hold a half of it either.
    pub const fn page_slots(self) -> NonZeroU32 {
        self.page_slots
    }
}

/// Slots per stick-block — the fp16 stick, and the block's slot count.
pub const SLAB_SLOTS: i64 = 64;

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  A launch: the shifts and the program they apply to, as ONE value
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// One program of a launch group: the function, and which PLACED tensor each of its parameters
/// points at.
///
/// ⛔ A PARAMETER IS AN `index` — A START ADDRESS. `ktdp.construct_memory_view`'s first operand is
/// that offset and the memref is its result, so a launch binds an ADDRESS per parameter: the
/// segment's base, plus that launch's shift, plus the placement's offset. Nothing here is a shape
/// or a name.
#[derive(Clone, Debug, PartialEq)]
pub struct LaunchProgram<'a> {
    /// The constructed program.
    pub func: ktir_core::ir::IRFunction<'a>,
    /// Parameter -> the tensor it carries, in parameter order. Carried from the construction that
    /// minted the parameter, never re-derived from a name.
    pub args: Cow<'a, [(ktir_core::ir::Ssa, PlaceId)]>,
}

/// ONE LAUNCH: a contiguous run of up to `group_size()` trips that dxp compiled into a single device
/// program, together with the address shifts the launch applies to it.
///
/// ⛔ THE SHIFTS AND THE PROGRAM ARE ONE STRUCT, AND LAUNCH ORDER IS SLICE ORDER. They were two
/// parallel slices joined by a `group: u32` stored in BOTH, rejoined by a linear search, with the load
/// path refusing when an index did not resolve. A bundle naming a program it does not carry was
/// constructible, and the refusal was the proof it was constructible. There is no index to dangle now.
///
/// `SCRATCHY_SUPERDSC_GROUP_SIZE=1` makes this one trip per program, which is the fault-isolation end of
/// that knob — not a different mode.
// NOT `Eq`: a group carries its PROGRAMS, and a program carries float constants.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LaunchGroup<'a> {
    /// What the launch shifts this program's segment bases by.
    pub kv: KvShifts,
    /// ⭐⭐⭐ THE PROGRAMS THIS LAUNCH RUNS, IN ORDER — the group's KTIR, constructed by the lowering
    /// and baked here as const data. A launch group IS its programs.
    ///
    /// ⛔ NOTHING PARSES THESE. They are `IRFunction` values from end to end: the lowering builds
    /// them, `#[forward]` submits them, and the device executes them. On the emulator that is
    /// direct; on a card the device image below is compiled FROM these, and is their artifact
    /// rather than a second source of truth.
    pub programs: Cow<'a, [LaunchProgram<'a>]>,
    /// dxp's device image. Empty for a group dxp compiled to a job plan alone.
    pub init_binary: Cow<'a, [u8]>,
    /// The device VA execution starts at. The launch passes `job_bin_ptr - PROG_OFFSET_BASE` as the
    /// bootstrap OFFSET (flex bounds the seg7 translation to the program allocation's size).
    pub job_bin_ptr: u64,
    /// The program-correction flits, already computed. Non-empty ⇒ H2D these into the head of the
    /// program allocation before each launch, so the device patches its own symbolic addresses; EMPTY ⇒
    /// this program needs no correction step (every concretely-addressed bundle).
    ///
    /// ⭐ ITS LENGTH IS THE CORRECTION REGION'S SIZE — one quantity, carried once. dxp states that size
    /// separately as `ComputeOnHost.size`; the bake checks the finished payload against it and refuses a
    /// mismatch, so only the payload needs to reach the binary.
    pub correction: Cow<'a, [u8]>,
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  The re-rolled layer loop
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A fingerprint naming ANOTHER registered [`BundleCode`].
///
/// ⛔ A NEWTYPE AND NOT A `String`, because a fingerprint here is an OBLIGATION: the load will resolve
/// it and fail if this binary does not carry it. [`sibling`] is the one resolver, so there is no second
/// list of "which fields hold a fingerprint" to keep in step with this one — [`RerollMeta::siblings`]
/// walks the fields themselves.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SiblingFp<'a>(pub Cow<'a, str>);

impl<'a> SiblingFp<'a> {
    /// No sibling of this kind.
    pub const fn none() -> SiblingFp<'a> {
        SiblingFp(Cow::Borrowed(""))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_none(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<String> for SiblingFp<'static> {
    fn from(s: String) -> SiblingFp<'static> {
        SiblingFp(Cow::Owned(s))
    }
}

/// One rung of the decode sk_bucket ladder: the same tape re-lowered at a narrower attention sweep.
///
/// ⛔ KEYED BY [`SweptCols`], NOT BY A BARE `u32`. Two other ladders in this codebase are keyed by the
/// BATCH WIDTH ([`RungSeqs`]); while all three were `(u32, fingerprint)` pairs, reading the wrong one to
/// answer "which body serves four live requests" type-checked and returned a plausible, wrong body —
/// one baked for a 64-column sweep, which attends the prompt and the newest token and nothing between.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LadderRung<'a> {
    /// The columns one fold pass of this rung's body sweeps.
    pub active_cap: SweptCols,
    /// This rung's body.
    pub body: SiblingFp<'a>,
    /// The same body with the per-page fold fused in — one fewer launch per layer, run whenever the
    /// context fits ONE page. [`SiblingFp::none`] when this rung has no fused twin, which simply
    /// means it always uses the split body: correct, one launch per layer dearer.
    pub body_fused: SiblingFp<'a>,
}

/// The re-rolled layer loop: the body is ONE layer, re-launched `iters` times with the weight and KV
/// segment bases advanced per layer, between a prefix (embed) and a suffix (lm_head).
///
/// ⛔ THE DESTRUCTURE THAT BAKES THIS IS EXHAUSTIVE (`codegen::reroll_tokens`), so adding a field here
/// is a build error until it is baked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RerollMeta<'a> {
    /// The prefix (embedding → first layer input).
    pub prefix: SiblingFp<'a>,
    /// The suffix (final norm → lm_head → logits).
    pub suffix: SiblingFp<'a>,
    /// This body with the per-page fold fused in, or [`SiblingFp::none`].
    pub body_fused: SiblingFp<'a>,
    /// The sk_bucket ladder, ASCENDING by `active_cap`; the top rung is this body itself.
    pub rungs: Cow<'a, [LadderRung<'a>]>,
    /// How many times the body runs — the layer count.
    pub iters: u32,
    /// Byte stride between one layer's weights and the next (seg1).
    pub weight_stride: u64,
    /// Byte stride between one layer's KV and the next (seg2). `page_stride = iters × kv_stride`,
    /// which is why an absent meta used to yield a zero-byte KV pool rather than an error.
    pub kv_stride: u64,
    /// ⭐ HOW MANY LAYERS ONE WEIGHT BANK HOLDS — the divisor that turns a layer index into
    /// `(bank, offset)`: layer `v` lives in bank `v / layers_per_bank` at
    /// `(v % layers_per_bank) · weight_stride`.
    ///
    /// `iters` (every layer in one bank) whenever the weights fit one device region, which makes the
    /// division a no-op and every existing bundle's launch sequence byte-identical. See
    /// [`Placement::bank`].
    pub layers_per_bank: u32,
    /// The weight bank the PREFIX program's weight operands live in, and the SUFFIX's.
    ///
    /// ⛔ A LAUNCH HAS ONE BASE PER SEGMENT, so each group's weights must be in ONE bank — proven at
    /// build time by `bank_weight_segment`, carried here because the runtime cannot re-derive which
    /// bank a program's baked operands came from. 0 for an unbanked bundle.
    pub prefix_weight_bank: u32,
    pub suffix_weight_bank: u32,
}

impl<'a> RerollMeta<'a> {
    /// Every sibling fingerprint this meta names, whatever kind it is — one walk over the fields, so a
    /// sibling cannot be named in the struct and missed by the resolver.
    pub fn siblings(&self) -> impl Iterator<Item = &str> {
        [&self.prefix, &self.suffix, &self.body_fused]
            .into_iter()
            .chain(self.rungs.iter().flat_map(|r| [&r.body, &r.body_fused]))
            .filter(|s| !s.is_none())
            .map(|s| s.as_str())
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  The bundle, and the registry
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE baked SuperDSC bundle: its memory plan, its launch index, its compiled device programs, and
/// — when it is a re-rolled body — the layer-loop parameters and the siblings that complete it.
#[derive(Clone, Debug, PartialEq)]
pub struct BundleCode<'a> {
    /// Content fingerprint — this bundle's identity, and how a sibling names it.
    pub fp: Cow<'a, str>,
    /// The memory plan every launch addresses through.
    pub layout: BundleLayout<'a>,
    /// This bundle's launches, IN LAUNCH ORDER. EMPTY on a cardless build (no `dxp_standalone` to
    /// run), which is why [`have_device_code`] exists.
    pub groups: Cow<'a, [LaunchGroup<'a>]>,
    /// `Some` ⇒ this bundle is a re-rolled BODY (one layer).
    pub reroll: Option<RerollMeta<'a>>,
}

impl<'a> BundleCode<'a> {
    /// ⭐ THE FOLD-ROW REGIME THIS BUNDLE DECLARES — what the launch derives its intermediate-segment
    /// rebase stride from.
    ///
    /// No fold groups at all ⇒ [`FoldRows::WholeBatch`]: nothing re-launches, so there is no pass to
    /// rebase. Fold groups that DISAGREE are refused — a session carries ONE intermediate stride, so a
    /// mixed bundle has no stride that is right for all of its passes.
    ///
    /// [`FoldRows`] is a closed enum every entry carries, so "whole batch" is a declaration rather
    /// than a silence.
    pub fn fold_rows(&self) -> Result<FoldRows, String> {
        let mut seen: Option<FoldRows> = None;
        for g in self.groups.iter().filter(|g| g.kv.page_fold) {
            match seen {
                None => seen = Some(g.kv.fold_rows),
                Some(s) if s == g.kv.fold_rows => {}
                Some(s) => {
                    return Err(format!(
                        "bundle {}: fold groups disagree about their row regime ({s:?} vs {:?}) — a \
                         session carries one intermediate stride, so a mixed bundle cannot be launched",
                        self.fp, g.kv.fold_rows
                    ));
                }
            }
        }
        Ok(seen.unwrap_or_default())
    }
}

inventory::collect!(BundleCode<'static>);

/// The bundle registered under `fp`, if this binary carries it.
pub fn bundle(fp: &str) -> Option<&'static BundleCode<'static>> {
    inventory::iter::<BundleCode<'static>>().find(|b| b.fp == fp)
}

/// The bundle a [`SiblingFp`] names — the whole of "resolve a sibling".
pub fn sibling(fp: &SiblingFp<'_>) -> Option<&'static BundleCode<'static>> {
    if fp.is_none() {
        return None;
    }
    bundle(fp.as_str())
}

/// Is any dxp-compiled device code in this binary? False for every cardless build, where the emit
/// had no `dxp_standalone` to run.
pub fn have_device_code() -> bool {
    inventory::iter::<BundleCode<'static>>().any(|b| !b.groups.is_empty())
}

/// Where the NUMERIC-BISECTION GOLDEN for `fp` lives — written by the emit under
/// `SCRATCHY_SUPERDSC_DBG`, read back by the worker under `SPYRE_SUPERDSC_SELFTEST`.
///
/// ⛔ ONE FUNCTION BECAUSE IT IS ONE AGREEMENT between a build-time writer and a run-time reader that
/// share no other state. Both ends call this; neither spells the path.
///
/// `SCRATCHY_SUPERDSC_DBG` holding a path uses it; any other value (`=1`) puts it under the temp dir.
/// `None` ⇒ the golden was not requested, and the self-test says so rather than reading a stale one.
pub fn dbg_golden_dir(fp: &str) -> Option<std::path::PathBuf> {
    let v = std::env::var_os("SCRATCHY_SUPERDSC_DBG")?;
    let base = std::path::Path::new(&v);
    let root = if base.is_absolute() {
        base.to_path_buf()
    } else {
        std::env::temp_dir().join("scratchy-superdsc-dbg")
    };
    Some(root.join(fp))
}

/// Every registered fingerprint, for diagnostics that need to say what IS here.
pub fn registered_fps() -> Vec<&'static str> {
    let mut v: Vec<&str> = inventory::iter::<BundleCode<'static>>()
        .map(|b| b.fp.as_ref())
        .collect();
    v.sort_unstable();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⭐ RUNG SELECTION ASKS THE SWEEP EXTENT ITS OWN QUESTION, and the batch width cannot answer it.
    ///
    /// The two axes of the rung grid are both counts and both used to be bare `u32` in lists that have
    /// both been called `decode_rungs`, so `rung.0 >= live` compiled against either — and selecting a
    /// body baked for a 64-column sweep because four requests are live is a wrong ANSWER with no fault.
    ///
    /// ⛔ THE COMPILE-TIME HALF OF THIS GUARD CANNOT BE ASSERTED HERE, WHICH IS THE POINT: there is no
    /// `SweptCols::covers` on [`RungSeqs`], no `From` between them and no public field on either, so
    /// asking a width whether it covers a KV length does not compile. The lines this test would need in
    /// order to check that are in `compile_fail` territory; what it CAN pin is that the surviving
    /// question is the one the fold's geometry actually poses.
    #[test]
    fn a_rung_covers_by_its_swept_extent() {
        let rung = SweptCols::new(128);
        assert!(rung.covers(128), "the extent covers exactly its own width");
        assert!(rung.covers(127));
        assert!(
            !rung.covers(129),
            "one position past the sweep is NOT covered"
        );
        // The ladder is ordered by this axis, and ordering is all `select_body` needs of it.
        let mut ladder = [SweptCols::new(256), SweptCols::new(64), SweptCols::new(128)];
        ladder.sort();
        assert_eq!(ladder.map(SweptCols::get), [64, 128, 256]);
        // A width is a different type with a different question — it has no `covers` at all.
        assert_eq!(RungSeqs::new(4).get(), 4);
    }

    /// ⭐ A LAUNCH IS ITS PROGRAM, so a bundle cannot name one it does not carry.
    ///
    /// The shifts and the program were two parallel slices joined by a `group: u32` held in BOTH, with
    /// the load path refusing when an index did not resolve — i.e. a bundle with a dangling program
    /// reference was constructible, and the refusal was the proof of it. Launch order is slice order
    /// now, so there is no index to dangle and no refusal to write.
    /// The KTIR the two launches below carry, in the CONST form `#[forward]` bakes: a program is
    /// data, so a test can hold one the same way the binary does.
    const F_A: ktir_core::ir::IRFunction<'static> = ktir_core::ir::IRFunction {
        name: "a",
        arguments: &[],
        operations: &[],
        grid: (1, 1, 1),
        return_type: None,
    };
    const F_B: ktir_core::ir::IRFunction<'static> = ktir_core::ir::IRFunction {
        name: "b",
        arguments: &[],
        operations: &[],
        grid: (2, 1, 1),
        return_type: None,
    };
    const PROGS_A: &[LaunchProgram<'static>] = &[LaunchProgram {
        func: F_A,
        args: Cow::Borrowed(&[]),
    }];
    const PROGS_B: &[LaunchProgram<'static>] = &[LaunchProgram {
        func: F_B,
        args: Cow::Borrowed(&[]),
    }];

    #[test]
    fn a_launch_carries_its_own_program() {
        let code = BundleCode {
            fp: Cow::Borrowed("fp"),
            layout: BundleLayout::default(),
            groups: Cow::Borrowed(&[
                LaunchGroup {
                    kv: KvShifts {
                        slot_stride_bytes: 128,
                        ..KvShifts::default()
                    },
                    programs: Cow::Borrowed(PROGS_A),
                    init_binary: Cow::Borrowed(&[1, 2, 3]),
                    job_bin_ptr: 64,
                    correction: Cow::Borrowed(&[]),
                },
                LaunchGroup {
                    kv: KvShifts::default(),
                    programs: Cow::Borrowed(PROGS_B),
                    init_binary: Cow::Borrowed(&[4]),
                    job_bin_ptr: 128,
                    correction: Cow::Borrowed(&[]),
                },
            ]),
            reroll: None,
        };
        // Every launch has a program, by construction — there is nothing to resolve. The program is
        // reached through the launch ITSELF, with no index and no registry lookup in between, which
        // is the whole claim: `groups[i].programs` cannot name a program some other slice holds.
        assert_eq!(code.groups.len(), 2);
        assert_eq!(code.groups[0].programs[0].func.name, "a");
        assert_eq!(code.groups[1].programs[0].func.name, "b");
        assert_eq!(code.groups[1].programs[0].func.grid, (2, 1, 1));
        assert_eq!(code.groups[0].job_bin_ptr, 64);
        assert_eq!(code.groups[1].init_binary.as_ref(), &[4]);
        // The write predicates are DERIVED from the strides, so a stride and its flag cannot disagree.
        assert!(code.groups[0].kv.slot_write());
        assert!(!code.groups[0].kv.slab_write());
        assert!(!code.groups[1].kv.slot_write());
    }

    /// ⛔ EVERY ONE OF THESE IS A BUNDLE THE LOAD OPENS, and a missing interior ladder rung is a fatal
    /// load error — so `siblings` must reach the fingerprints buried inside `rungs`, not just the scalar
    /// fields. Fail-first: dropping the `rungs` arm returns three of six.
    #[test]
    fn siblings_cover_every_scalar_and_every_ladder_rung() {
        let m = RerollMeta {
            prefix: SiblingFp::from("cdfd5b3768384100".to_string()),
            suffix: SiblingFp::from("de8ab8a9205d4b1e".to_string()),
            body_fused: SiblingFp::from("443aebd9890d269af".to_string()),
            rungs: Cow::Owned(vec![
                LadderRung {
                    active_cap: SweptCols::new(64),
                    body: SiblingFp::from("2f65540feaf0b24b".to_string()),
                    body_fused: SiblingFp::from("2f65540feaf0b24bf".to_string()),
                },
                LadderRung {
                    active_cap: SweptCols::new(256),
                    body: SiblingFp::from("443aebd9890d269a".to_string()),
                    body_fused: SiblingFp::none(),
                },
            ]),
            iters: 40,
            weight_stride: 60872704,
            kv_stride: 786432,
            // Every layer in one bank — the unbanked case, where `v / layers_per_bank` is always 0.
            layers_per_bank: 40,
            prefix_weight_bank: 0,
            suffix_weight_bank: 0,
        };
        let sibs: Vec<&str> = m.siblings().collect();
        for want in [
            "cdfd5b3768384100",
            "de8ab8a9205d4b1e",
            "443aebd9890d269af",
            "2f65540feaf0b24b",
            "2f65540feaf0b24bf",
            "443aebd9890d269a",
        ] {
            assert!(sibs.contains(&want), "sibling {want} lost: {sibs:?}");
        }
        // An ABSENT sibling contributes nothing — an empty fingerprint used to name the scratch
        // ROOT as if it were a bundle.
        assert!(!sibs.iter().any(|s| s.is_empty()));
        assert_eq!(sibs.len(), 6);
    }

    /// ⭐ A LAYOUT ANSWERS BY IDENTITY. It used to answer by NAME *and* by TID — two keys for one
    /// thing, which is what let a caller ask the wrong one. The sharp case is below: a synthetic
    /// DERIVED FROM 728 must not answer "did the bake place tensor 728", even though it carries
    /// that tid, because it is a different tensor.
    #[test]
    fn a_layout_answers_by_identity_and_a_synthetic_is_not_its_source() {
        let l = BundleLayout {
            segment_bytes: [10, 20, 30, 40, 50, 60, 70],
            places: Cow::Owned(vec![
                Placement {
                    id: PlaceId::Act(788),
                    segment: 4,
                    bank: 0,
                    offset: 0,
                    size: 98304,
                    is_logits: true,
                },
                Placement {
                    id: PlaceId::Act(728).synth(SynthRole::Sq16),
                    segment: 0,
                    bank: 0,
                    offset: 4096,
                    size: 2048,
                    is_logits: false,
                },
            ]),
            ..BundleLayout::default()
        };
        assert_eq!(l.logits().unwrap().id, PlaceId::Act(788));
        assert_eq!(l.place_of_tid(788).unwrap().size, 98304);
        assert_eq!(
            l.place(PlaceId::Act(728).synth(SynthRole::Sq16))
                .unwrap()
                .offset,
            4096
        );
        // ⛔ THE ONE THAT MATTERS. `t728_sq16` shares tid 728, so a tid-only key would return it
        // here and the host would bind an activation onto a device-internal scratch buffer.
        assert!(
            l.place_of_tid(728).is_none(),
            "a synthetic must not answer for the tensor it derives from"
        );
        assert_eq!(l.places.iter().filter(|p| !p.id.is_bindable()).count(), 1);
    }
}

#[cfg(test)]
mod place_id_tests {
    use super::*;

    /// ⭐ THE SPELLING IS DERIVED, NOT AGREED. Both ends used to build `format!("t{tid}")`
    /// independently and join on the result; this pins that there is now ONE renderer, so the
    /// emitted program's operand and the host's bind cannot drift.
    #[test]
    fn an_act_spells_as_the_tid_and_a_synth_as_its_role() {
        assert_eq!(PlaceId::Act(7).to_string(), "t7");
        assert_eq!(PlaceId::Act(7).synth(SynthRole::Rot).to_string(), "t7_rot");
        assert_eq!(
            PlaceId::Act(7).synth(SynthRole::Blk(3)).to_string(),
            "t7_blk3"
        );
    }

    /// A synthetic shares its `of` tid with the tensor it derives from, so a tid ALONE cannot
    /// identify a placement — which is why `bindable()` exists and why the host, which keys by
    /// tid, may only ever be handed `Act`s.
    #[test]
    fn a_synthetic_is_not_bindable_even_though_it_shares_a_tid() {
        let a = PlaceId::Act(7);
        let s = a.synth(SynthRole::Rot);
        assert_eq!(a.tid(), s.tid(), "same tid");
        assert_ne!(a, s, "different identities");
        assert_eq!(a.bindable(), Some(7));
        assert_eq!(
            s.bindable(),
            None,
            "device-internal: the host must not bind it"
        );
    }

    /// ⛔ EVERY ROLE SPELLS DISTINCTLY. Two roles sharing a rendering would give two tensors one
    /// operand name, and the allocator — which is keyed by the rendering, because a SuperDSC op
    /// names its operands — would hand them the SAME address.
    #[test]
    fn no_two_roles_share_a_spelling() {
        use SynthRole::*;
        let roles = [
            Rot,
            Xc,
            Rs,
            Silu,
            Qs,
            KRep,
            VRep,
            NewKRep,
            NewVRep,
            NewKScaled,
            FqAbsX,
            FqAfp8,
            FqAmax,
            FqAmaxFl,
            FqAscale,
            FqChi,
            FqCl,
            FqInvS,
            FqSc,
            FqDqA,
            FqMm,
            FqRaw,
            Sq16,
            Mean,
            Meps,
            Rinv,
            Xn,
            NewKt,
            Sc,
            BMax,
            NewM,
            Corr,
            CorrSubT,
            ExpB,
            ESubT,
            BSum,
            OTmp,
            LTmp,
            RunM,
            RunL,
            RunO,
            Blk(0),
            Acc(0),
            LBlk(0),
            LAcc(0),
        ];
        let mut seen = std::collections::BTreeSet::new();
        for r in roles {
            assert!(seen.insert(r.to_string()), "duplicate spelling: {r}");
        }
        assert_eq!(seen.len(), roles.len());
    }

    /// An `Act`'s spelling can never collide with a synthetic's — the host binds by `Act`, and a
    /// collision would let a device-internal buffer answer to a bound tensor's name.
    #[test]
    fn an_act_never_spells_like_a_synthetic() {
        for tid in [0u32, 7, 4242, u32::MAX] {
            let a = PlaceId::Act(tid).to_string();
            assert!(!a.contains('_'), "an act's spelling has no role part: {a}");
        }
    }
}
