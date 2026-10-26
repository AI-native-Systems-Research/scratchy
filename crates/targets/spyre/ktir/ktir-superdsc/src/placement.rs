// SPDX-License-Identifier: Apache-2.0
//! THE WHOLE-BUNDLE MEMORY PLAN — every tensor's `(segment, bank, offset, size, role)`, the
//! synthetic-intermediate allocator, and the arrangement authority.
//!
//! ⛔⛔⛔ THIS IS NOT A TRAIT, AND IT MUST NOT BECOME ONE. The extraction plan filed everything
//! reached as `layout: Option<&BundleLayout>` as scratchy's plan, to be abstracted behind a
//! `trait Placement`, and two attempts began by designing that trait. It is not scratchy's plan: it
//! is a DEVICE memory plan, and — established by grep over the whole moving set, not by reading —
//! the only scratchy-shaped names in the whole region were `bundle::PlaceId` and
//! `sdsc_abstract::StickLayout`, both of which already live in this crate. The trait would have cost
//! 71 signatures changed to `Option<&dyn Placement>`, which does not coerce through `Option`, and
//! every build-time refusal below re-derived behind a `&dyn`.
//!
//! ⭐ THE SEAM IS THE **CONSTRUCTION**, NOT THE STRUCT. What is scratchy's is how a plan gets built:
//! `compute_bundle_layout` walks SubtileIR liveness to colour intermediates, `audit_layout_addresses`
//! audits the baked `bundle::Placement` list, `bake_layout` converts to the bundle's own type. All
//! three stayed. A third-party KTIR producer builds a [`BundleLayout`] from whatever it has and hands
//! in the same struct — which is strictly easier for it than implementing a trait, and keeps the
//! refusals here (the footprint guard, [`BundleLayout::declare_arrangement`], the segment ceiling)
//! as the ONE implementation rather than something each consumer re-derives.

use crate::addr::{DevOff, Win};
use crate::place::PlaceId;
use crate::sdsc_abstract::{KindTag, StickLayout, Stk};
use crate::superdsc_error::SuperDscError;
use crate::superdsc_opspec::Df;

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

// ⛔⛔⛔ THERE IS NO FREE SEGMENT FOR A SECOND WEIGHT SEGMENT, so an over-size weight segment
// (see `bundle::MAX_SEGMENT_BYTES`) has to SHARE one. All 7 of `SEGMENT_OFFSETS`'s non-aliasing
// slots are allocated: 0 intermediates, 1 weights, 2 KV, 3 activations/mask, 4 logits, and
// **5 + 6 are intermediate COLORS** (`inter_segs` in `compute_bundle_layout` — dxp's ModuleStitcher
// wires producer→consumer BY SEGMENT, so simultaneously-live intermediates must not share one;
// cramming them together was the "multi-op all-seg3 orphans" bug). Trying `WeightOverflow = 5`
// TOOK one of those colors, and a per-layer intermediate colored into seg5 then tripped `reroll`'s
// per-layer-stride guard. A color is not available to take: MEASURED on granite-3.1-8b-fp16, all
// three are occupied (seg0 13,495,424 B / seg5 2,850,816 B / seg6 1,826,816 B), so dropping to two
// pushes live intermediates into the offset-packed `None` arm that the stitcher cannot tell apart.
//
// ⭐ BUT A COLOR SEGMENT IS 16 GiB AND HOLDS 1.8 MB OF IT. Sharing needs no color, so the ceiling
// moves without touching the coloring at all — see [`WEIGHT_SPILL_SEGS`].

/// ⭐ THE SLOTS AN OVER-SIZE WEIGHT SEGMENT MAY SPILL ITS **NON-PER-LAYER** TAIL INTO, and the
/// reason this table has exactly one row.
///
/// A spill slot has to survive four independent per-step mechanisms, and each one eliminates a
/// candidate outright. All four were read off the code, not assumed:
///
/// | slot | per-step H2D | per-step D2H | fold-shifted | also in the SUFFIX program |
/// |------|--------------|--------------|--------------|----------------------------|
/// | 0 intermediates | no | no | **yes** (`off[SEG_INTERMEDIATE] += delta.intermediate`) | the residual + the lm_head's activation input |
/// | 3 activations   | covering runs | no | **yes** (`off[SEG_MASK] += delta.mask`) | — |
/// | 4 logits        | no | **THE WHOLE SEGMENT** | no | the logits |
/// | 5 color 2       | no | no | no | **the lm_head's OUTPUT** |
/// | **6 color 3**   | **no** | **no** | **no** | **nothing** |
///
/// 🛑 seg4 is the trap, and it looks like the obvious answer: it holds ONE tensor (the 102,400 B
/// logits row) in a 16 GiB slot. But `predict`'s logits readback is `d2h_bytes =
/// addr.total_size()` — the whole segment, every token, as its own comment measures ("~3 MB of
/// untouched memory (~1.7 ms)"). A 419 MB tail there costs ~240 ms PER TOKEN. Nothing about the
/// placement would look wrong; the model would simply be an order of magnitude slow.
///
/// seg6 survives all four: intermediates are never in `Executor::bound`, so `refill_activations`
/// never marks it dirty; only the logits segment is read back; no `LaunchDelta` names it; and
/// `zero_seg` is only ever called for seg2. What makes it *correct* rather than merely cheap is the
/// last column — MEASURED on granite-3.1-8b-fp16, the suffix's four tensors are t1127 and t1128 in
/// seg0, t1129 in seg5 and t1130 in seg4, so a tail spilled here is the SOLE seg6 dataspace in the
/// one program that reads it, and the stitcher has nothing to confuse it with. Sharing with seg5
/// would put the lm_head's weight input and its own output in one segment, which is the
/// producer→consumer ambiguity this whole coloring pass exists to prevent.
///
/// ⛔ ONLY THE NON-PER-LAYER TAIL. The rolled body reaches layer `v` by advancing ONE segment's base
/// (`off[SEG_WEIGHT] = v · weight_stride`), so a per-layer weight outside [`SegRole::Weight`] gets no
/// stride and every layer reads layer 0's copy — silent garbage. `reroll` refuses that by ROLE (a
/// per-layer *intermediate* colored here is fine and must stay fine — that distinction is what the
/// `WeightOverflow = 5` attempt got wrong). So this raises the ceiling on the tail, NOT on the
/// per-layer block: the per-layer block still has to fit one segment, which caps dense fp16 near 8B.
/// Splitting THAT needs the body launch to pick a segment per layer — expressible, for the reasons
/// `spill_weight_tail` records, but not built here.
pub const WEIGHT_SPILL_SEGS: [usize; 1] = [6];

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

// ⛔⛔⛔ THERE IS NO `MAX_PAGES_PER_REQUEST` AND NO `MAX_FOLD_PASSES`, AND THE GUARD THAT TIED THEM
// TO THE ROW CONST IS GONE WITH ALL THREE.
//
// Both constants were hand-picked page ceilings on the prefix mask — 32 was chosen to keep the
// reservation byte-identical to a pre-paged mask that no longer exists, and 64 was the pass budget
// of an uncollapsed fold the gathered form superseded — and together they bound a row's context
// before the pool's memory ever got a vote. The mask's block count is now DERIVED at bake from the
// device (`compute_bundle_layout`: `MAX_SEGMENT_BYTES / page_bytes`, the most pages one region of
// KV can hold), one expression feeding the `rows_are_requests` and broadcast arms alike, so there
// are no two constants left to relate and no guard left to fire. The runtime reads the ceiling
// from the artifact (`BakedMaskBlocks`) — see `sdsc_abstract`'s fold-pass ceiling block for the
// enforcement chain.

impl SegRole {
    pub fn segment(self) -> usize {
        self as usize
    }
}

/// ⭐⭐⭐⭐⭐ THE WRITE-COVER LAW — WHAT A BIND MAY LEAVE UNWRITTEN, AS A TYPE.
///
/// A placement's `size` is what the bake RESERVES; a bind's byte count is what
/// a producer WRITES. Those were two unrelated numbers until now: every bind door
/// refused only the over-bind (`bytes > size`), and a short bind silently left
/// the placement's tail holding whatever the shadow had — which the consumers
/// then read. [`crate::sdsc_abstract`]'s staging docs have named the hazard for
/// the whole campaign ("an under-bind of an ADDITIVE mask leaves a tail the ops
/// read as ZERO, i.e. as 'no mask'"); this type is that prose, enforced.
///
/// Every placement STATES its cover at construction — there is no default and no
/// omission: a construction site that cannot name its cover has not finished
/// stating its placement. Two classes exist because two bind laws exist:
///
/// * **`Full`** — every bind fills the placement exactly (`bytes == size`). The
///   weights, the embeddings, the rotary tables, the causal mask, the constants,
///   the spliced step inputs (whose fill door already demands `==`), and the
///   BROADCAST prefix mask of a prompt chunk or solo decode, whose staging law is
///   "THE EXTENT IS THE BAKE'S RESERVE … every swept column gets an explicit
///   value" — the one-row form is bound whole.
/// * **`FoldBlocks`** — the fold's per-pass tensors. A decode-BATCH prefix mask
///   places `[nqh·mq rows, max_pages blocks]` and the host stages only the LIVE
///   blocks ("3 blocks instead of 24"); the KV block index rides the mask's own
///   per-pass shift, so its pitch IS the mask's block stride — one number, both
///   tensors. A bind is a WHOLE NUMBER of blocks, never a partial one: a partial
///   block is a truncated row of a `[pass][row][page]` grid, read as mask bytes
///   nobody staged (which read as ZERO and therefore VALID) or as block numbers
///   that point somewhere real.
///
/// ⛔ THE DOORS ASK, THE BAKE PROVES. `admits` is the one extent law every bind
/// door consults ([`crate::Executor`]'s `place_raw` / `bind_input` family); the
/// address audit proves the class's arithmetic against the placement's own size
/// (`block_bytes` divides `size`, is nonzero, and does not exceed it) at bake,
/// where those are constants. A new bind law is a new VARIANT, minted where its
/// geometry is computed, refused by every door until then — never a relaxation
/// of an existing one.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum BindCover {
    /// Every bind fills the placement exactly: `bytes == size`.
    Full,
    /// Every bind is a whole number of `block_bytes` blocks, at most the
    /// placement's: the fold's per-pass mask and index tables.
    FoldBlocks {
        /// One `[pass][row][page]` block — the mask's own pass stride.
        block_bytes: u64,
    },
}

impl BindCover {
    /// ⭐ THE ONE EXTENT LAW — does a bind of `bytes` into a `size`-byte
    /// placement obey this cover? Every bind door asks this and nothing else;
    /// the over-bind and the silent-short-bind are the same refusal here.
    pub fn admits(&self, bytes: u64, size: u64) -> bool {
        match self {
            BindCover::Full => bytes == size,
            BindCover::FoldBlocks { block_bytes } => {
                bytes != 0
                    && *block_bytes != 0
                    && bytes.is_multiple_of(*block_bytes)
                    && bytes <= size
            }
        }
    }
}

/// One tensor's global placement in the packed bundle.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct TensorPlacement {
    /// SubtileIR tensor id (the stable global name is `t{id}`).
    pub tid: u32,
    pub role: SegRole,
    pub segment: usize,
    /// Which BANK of that segment — see `bundle::Placement::bank`. 0 for everything except a
    /// weight in a bundle whose weights need more than one device region.
    pub bank: u32,
    /// Byte offset WITHIN the segment region (128 B aligned), and within its BANK when banked.
    pub offset: u64,
    /// Byte size (f16 = 2 B/elem, rows*cols*2).
    pub size: u64,
    /// ⭐ WHAT A BIND MAY LEAVE UNWRITTEN — the write-cover law, stated at
    /// construction. See [`BindCover`].
    pub cover: BindCover,
}

/// The device-tile re-tile spec the executor (C++ shim) consumes to stage a matmul
/// KERNEL weight in the PT array's device layout. DERIVED SOLELY from the
/// [`DeviceTileLayout`] witness (`host_retile_descriptor`-style), so the shim's
/// re-tile and the emitter's per-core address read the SAME device layout — they
/// cannot diverge. `device(coord)` reads `host[Σ coord·stride_map]`, written in
/// contiguous `device_size` order.
///
/// [`DeviceTileLayout`]: crate::superdsc_opspec::DeviceTileLayout
#[derive(Clone, Debug, serde::Serialize)]
pub struct RetileDescriptor {
    /// Device extents in dim_map order, e.g. `[out/64, in, 64]` for a `[in,out]` KERNEL.
    pub device_size: Vec<u64>,
    /// Host-element stride per device axis, e.g. `[64, out, 1]`.
    pub stride_map: Vec<u64>,
    /// Elements per stick (64 for fp16).
    pub stick_size: u32,
    /// Bytes per element (2 for fp16).
    pub word_length: u32,
}

/// One row of the step-input discovery table ([`BundleLayout::step_inputs`]): the SLOT the one
/// numbering law ([`step_input_tid`](crate::ktir_node::step_input_tid)) gave the tid, and the
/// ELEMENT FORMAT the program's own view states for that buffer.
///
/// ⛔ BOTH FACTS ARE THE PROGRAM'S OWN STATEMENTS, recorded by the placement pass at the same
/// statements it sizes the placement from — the slot is what the producer numbered the binding
/// through, and the element format is the dtype of the buffer's own `construct_memory_view`. The
/// slot is what the runtime ADDRESSES the input by; the element format is what the bake maps to
/// the staging law. Neither is given a meaning here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepInputRow {
    pub slot: u32,
    pub df: Df,
}

// `Df`'s own doc law is that the typed core does NOT derive `Serialize` (the wire structs do), so
// the row serializes its format as the dataspace's own `dataFormat_` string — the same spelling
// the emitted descriptors carry, which is what a runtime diagnostic reading `bundle_layout.json`
// can actually compare against.
impl serde::Serialize for StepInputRow {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("StepInputRow", 2)?;
        st.serialize_field("slot", &self.slot)?;
        st.serialize_field("df", self.df.dataformat())?;
        st.end()
    }
}

/// ⭐⭐⭐ THE KV BLOCK INDEX, HELD UNPLACED — the ONE activation a session's bodies may
/// legitimately disagree about, which is why its offset cannot be taken where the masks take
/// theirs.
///
/// Only a gathering body reserves the index (`gathers_kv`: the batched rungs and a chunked
/// one-request ceiling gather through the page table; an unchunked sweep rung sweeps the
/// resident KV and needs no table), while the host binds every step at the CEILING bundle's
/// placements — so the index's PRESENCE must move nothing the host stages.
/// [`BundleLayout`] therefore carries the reservation as a VALUE and takes no offset for it in
/// the placement pass; the one placement door is
/// [`BundleLayout::place_kv_block_index_last`], which a walk must call AFTER the last
/// host-staged placement (`place_step_inputs`, `complete_program_scales`). A reservation
/// still pending at the bake is a `cargo build` panic, never a silently unplaced index: the
/// placement's PRESENCE is `BakeFacts::gathers_kv`, and a gather that finds no table reads
/// entry 0 — which is a real address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvBlockIndexReservation {
    /// The MASK'S OWN PASS-GRID BYTES (`pmbytes`) — one derivation for both tensors: the index
    /// rides the mask's per-pass shift, so its per-pass pitch is this over the mask's block
    /// count and its reservation is the whole grid. There is no second spelling of this number.
    pub bytes: u64,
    /// ⭐ THE SAME GRID'S BLOCK STRIDE — `pmbytes / pm_blocks`, the one number the mask's own
    /// [`BindCover::FoldBlocks`] mint uses. The index's binds are whole blocks of exactly this
    /// many bytes (the host stages the live passes' entries at the head, one block per live
    /// page), so this is the cover the placement door mints — a bind that is not a whole
    /// number of these blocks is a truncated row of the pass grid, read as block numbers.
    pub block_bytes: u64,
}

/// The whole-bundle memory plan: every tensor → (segment, offset, size, role),
/// plus each segment's total packed bytes. Exported as `bundle_layout.json` for
/// the resident executor (which allocates one region per occupied segment).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct BundleLayout {
    /// Placement per tensor id, ordered by id for determinism.
    pub placements: std::collections::BTreeMap<u32, TensorPlacement>,
    /// IDENTITY BY OPERAND SPELLING — populated wherever a name is MINTED.
    ///
    /// ⛔ THIS EXISTS TO DELETE A PARSE. `resolve_seg_base` used to recover a tensor's id with
    /// `name.strip_prefix('t').and_then(|s| s.parse::<u32>())` — the emitter rendering `t{tid}`
    /// at one site and reading the number back out of the rendering at another, which is the
    /// compile-time-fact-through-a-string round trip in its purest form. An operand carries a
    /// STRING because the layout handle it comes from (`scratchy_subtile`'s `Stk`) is shared,
    /// target-neutral code that must not learn a Spyre type; so spyre keeps its own table from
    /// that spelling to its own identity, filled where the spelling is created.
    #[serde(skip)]
    pub ids: std::cell::RefCell<std::collections::BTreeMap<String, PlaceId>>,
    /// ⭐ THE INTERMEDIATE-TID CURSOR — the ONE source of fresh intermediate tids, bundle-unique.
    ///
    /// ⛔ MEASURED on the card (2026-10-20, granite-2b fp16 at `load_model`): the mint used to
    /// be seeded per NODE from that node's own bindings' max, so node B's mint landed on node
    /// A's placement — `1310 placements collapse to 1214 distinct ids` (96 duplicates = the
    /// number of intermediates one rope mints per node), the runtime refused to load, and the
    /// load-time aliasing is worse than the refusal: a `BTreeMap` keeps ONE entry per id, so
    /// the other tensor has NO placement at all and the device reads whatever shares its
    /// address. The cursor is seeded LAZILY, on first mint, above the highest non-reserved
    /// placement id in the WHOLE bundle — `#[forward]` expands every node before any lowering
    /// runs, so the seed sees every placement — and advances monotonically past reserved tids
    /// (the rope P / identity / scalarmul const tables live above `u32::MAX/2` and are skipped,
    /// never minted onto). `take_inter_tid(floor)` also honors a caller-passed floor, so a
    /// node lowered into TWO layouts (declare pass then real pass) can demand the same
    /// tid again; [`KtirNode::minted`] replays the node's own record for exactly that.
    #[serde(skip)]
    pub inter_cursor: std::cell::Cell<Option<u32>>,
    /// Bytes occupied per segment (index = segment id 0..6). For the weight segment this is BANK
    /// 0's bytes — see [`BundleLayout::weight_bank_bytes`].
    pub segment_bytes: [u64; 7],
    /// ⭐ THE WEIGHT SEGMENT'S EXTRA BANKS: bytes of banks `1..N`, empty when the weights fit ONE
    /// device region (every model that worked before banking existed). See
    /// `bundle::Placement::bank` and `bank_weight_segment`.
    #[serde(default)]
    pub weight_bank_bytes: Vec<u64>,
    /// ⭐ THE SEGMENT BUDGET'S VERDICT on the non-per-layer weight tail — see [`WeightTail`].
    #[serde(default)]
    pub weight_tail: WeightTail,
    /// Matmul KERNEL weights → their device-tile [`RetileDescriptor`] (built ONLY via
    /// `DeviceTileLayout`, the same witness the per-core address uses). The PT array
    /// reads the device TILE layout `[out/64, in, 64]`, NOT the row-major flat bytes;
    /// the shim re-tiles every weight in this map during H2D staging via the descriptor.
    /// Tensors absent here (activations, embed gather table, 1-D rmsnorm gains) stay
    /// flat — `[1,N]`/`[N]` are tiling-invariant. Includes per-layer copies.
    #[serde(default)]
    pub kernel_weights: std::collections::BTreeMap<u32, RetileDescriptor>,
    /// Distinct granite ScalarMul scale VALUES (embedding/residual/attn/logits multipliers), in a stable
    /// order. Index `i` ↔ reserved const TID `scalarmul_scale_tid(i)` (a worker-bound `[1,1]` const,
    /// exactly the ATTN_SCALE mechanism). The emitter (the door's scalarmul lowering) looks up its scale's index
    /// here → the const TID it multiplies by; the WORKER reads this list (serde) and binds each
    /// `t{tid} = [scale]`. So the emitter and worker agree on scale↔TID by construction (no reward-hack
    /// fold, no host-route — the op is a real on-device pointwise `mul`).
    #[serde(default)]
    pub scalarmul_scales: Vec<f32>,
    /// ⭐ THE SCALE REGISTRY'S FORMAT COLUMN — `scalarmul_scales[i]`'s own `df`, parallel by
    /// construction. The PAYLOAD half (the census) is all `Fp16`, the staging-twin law: the
    /// value the card applies is `f16(v)`, written by the existing `[1,1]` const fill. The
    /// PROGRAM-STATED half carries the element type each constant's own `tensor.splat` states
    /// (`Fp32` for the f32 m-merge's constants), appended by the completion pass in the same
    /// bits-sorted order the values are. The runtime reads this column to pick each slot's
    /// staging channel — f16 fill vs raw f32 word — so it is baked into the artifact beside the
    /// values, never re-derived on the host.
    #[serde(default)]
    pub scalarmul_scales_df: Vec<Df>,
    /// ⭐ THE WEIGHT FOLDS — `(weight tensor id, multiplier)`, one per fold the producer half
    /// recognized, of EITHER kind: a `ScalarMul` multiplier scaling a MATMUL's whole, unshared
    /// output (`scalar_mul_weight_folds`), or an attention's `√scale` scaling its `W_q`/`W_k`
    /// (`attn_scale_weight_folds` — the node stays on the tape at scale 1.0). The weight's
    /// staged bytes are multiplied by `multiplier` at load — dense `[A, W]` folds into `W`, fp8
    /// `[A, qW, ws]` into the `ws` dequant row — and neither the `[1,1]` const bind nor the
    /// pointwise `mul` is emitted. EMPTY for every model whose multipliers scale something
    /// other than a private matmul product (granite's tied-embedding `embedding_multiplier` is
    /// exactly that), which is what keeps those bundles byte-identical to before this field
    /// existed.
    #[serde(default)]
    pub weight_scale_folds: Vec<(u32, f32)>,
    /// ⭐ THE HEAD-MAJOR o_proj PERMUTED COPIES — `(companion weight tid, permuted copy tid)`, one
    /// pair per o_proj weight the head-major restructure companions. Where the bundle's
    /// single-row attention finalize writes `out` SLAB-MAJOR, the o matmul's contraction axis
    /// arrives permuted and its B operand must be the SECOND, k-shuffled o_proj copy staged under
    /// `oproj_headmajor_tid` — the copy whose retile walk (in [`Self::kernel_weights`]) enumerates
    /// exactly that order. The worker materializes each `to` tid from its `from` tid's bytes at
    /// load. MINTED BUNDLE-INVARIANTLY — keyed on the geometry (multi-slab head dim + fp8 W8A8 o
    /// matmul), never on the bundle's row count, because every bundle of a model shares ONE
    /// staged weight segment; only the EMISSION gates on rows. EMPTY for every model the gate
    /// excludes (granite-3.1-2b's one-slab head dim, every dense o_proj), which keeps those
    /// bundles byte-identical to before this field existed.
    #[serde(default)]
    pub weight_copies: Vec<(u32, u32)>,
    /// ⭐ THE ROPE-P CLASS SET — this tape's DISTINCT rope head dims, sorted descending. Index `i` ↔
    /// [`rope_p_class_tid`](crate::reserved_tids::rope_p_class_tid)`(i)` (class 0 = the widest =
    /// `ROPE_P_TID`). One `[hd,hd]` P table per class: a hybrid (gemma-4) carries two rope head dims
    /// and a composite single P is IMPOSSIBLE (the big class's contraction reads the small class's
    /// nonzero block as spurious ±1 terms). The DOOR resolves a rope program's class index here —
    /// the same registry lookup `scalarmul_scales` established — so the emitter, the placement pass
    /// and the worker's load-time bind all agree on class↔tid by construction.
    #[serde(default)]
    pub rope_class_hds: Vec<u32>,
    /// ⭐ THE IDENTITY CLASS SET — this tape's DISTINCT `AttnDecode` head dims, sorted descending.
    /// Index `i` ↔ [`identity_class_tid`](crate::reserved_tids::identity_class_tid)`(i)` (class 0 =
    /// the widest = `IDENTITY_TID`). Same defect, same fix: the GQA krep/vrep matmuls read the
    /// identity at THEIR class's head dim, and one max-sized placement holding a base-hd table zeroes
    /// the wide class's upper lanes (dump-proven on gemma-4 tiny-allglobal).
    #[serde(default)]
    pub attn_class_hds: Vec<u32>,
    /// ⭐⭐⭐ THE HOST-STAGED PER-STEP INPUTS — tid → its [`StepInputRow`].
    ///
    /// This is the WORKER'S DISCOVERY TABLE, and it is DATA, exactly like [`Self::weight_copies`]
    /// and [`Self::scalarmul_scales`]: a per-step input is placed like any other Activation (its
    /// placement is in [`Self::placements`], sized from the program's own view), and this field is
    /// how the runtime tells a staged buffer apart from an intermediate WITHOUT recomputing the
    /// producer's numbering — the runtime reads the tid, the slot and the element format off the
    /// artifact and never re-derives `graph.tensors.len() + k`. The element format is the
    /// program's own view statement, recorded at the same statement the placement was sized from,
    /// and it is what the bake maps to the staging law (which runtime channel narrows the host's
    /// bytes); this crate serializes all of it and branches on none of it.
    #[serde(default)]
    pub step_inputs: std::collections::BTreeMap<u32, StepInputRow>,
    /// SYNTHETIC intermediates created during lowering (silu's `{out}_silu`,
    /// rmsnorm's `{out}_sq/mean/meps/inv/tmp/eps`) are NOT SubtileIR tensors, so
    /// they have no `t{id}` placement. They are lazily assigned a STABLE offset in
    /// the Intermediate segment (seg3) ABOVE the colored intermediates, so a
    /// producer and consumer of the same synthetic name resolve to the SAME address
    /// and they NEVER collide with weights (seg1)/activations (seg0). Interior-
    /// mutable so `resolve_seg_base` can allocate on first sight through `&layout`. Exported to
    /// `bundle_layout.json` (segment is always [`SegRole::Intermediate`], offset is `synth.map[name]`)
    /// so a runtime diagnostic can look up a synthetic intermediate's address by NAME, the same way
    /// it looks up a `t{id}`'s.
    pub synth: std::cell::RefCell<SynthAlloc>,
    /// ⭐ PINNED-LX INTERMEDIATES — the residency twin of [`Self::synth`]. A tensor declared
    /// here (via [`Self::synth_lx_op_local`]) is addressed in the per-core LX scratchpad at a pinned
    /// byte address instead of the Intermediate HBM segment: its producer replicates across
    /// all cores ([`crate::superdsc_opspec::WorkPlan::replicate`]) and its consumers read
    /// their own core's copy. Empty for every bundle that declares none (all models before
    /// the exx2 rmsnorm chain), keeping those byte-identical. Interior-mutable so the
    /// `&layout`-threaded emit can consult it by name.
    #[serde(skip)]
    pub lx: std::cell::RefCell<LxAlloc>,
    /// THE arrangement authority: `tensor name → its ONE device [`StickLayout`]`. The FIRST op to
    /// address a tensor declares its layout; every later op must address it EQUIVALENTLY
    /// ([`StickLayout::addr_eq`]) or `declare_arrangement` returns a build `Err` naming the tensor. A
    /// tensor has ONE physical layout — so a producer that writes it stick-major and a consumer that
    /// reads it flat (the M>1 prefill scramble) is a COMPILE-TIME failure, not silently-wrong bytes.
    /// Interior-mutable so the `&layout`-threaded `emit_sdsc` can declare on first sight.
    #[serde(skip)]
    pub arrangements: std::cell::RefCell<std::collections::BTreeMap<String, StickLayout>>,
    /// BYTES BETWEEN TWO REQUESTS' KV inside one page+layer — [`PagedKvPool::request_stride`] in
    /// bytes, or 0 when this bundle has no paged KV.
    ///
    /// The EMITTER's number, travelling to the runtime rather than being re-derived there. Every
    /// baked KV address already has the request dimension in it (`block_index` puts `ROWS` requests
    /// inside each kv head), so the launch shifting the KV segment by `r * this` is the other half of
    /// one law. A runtime that computed its own would be free to disagree, and disagreeing means
    /// reading another request's keys under a mask that thinks they are this one's — fluent output,
    /// wrong tokens, nothing to catch it.
    ///
    /// [`PagedKvPool::request_stride`]: crate::sdsc_abstract::PagedKvPool::request_stride
    #[serde(default)]
    pub kv_request_stride_bytes: u64,
    /// ⭐ THE GATHER REACH — source tid → the bound on the index VALUES a GATHERED read of that
    /// tensor may hold, as the typed pair ([`crate::sdsc_abstract::GatherReach`]): the entry
    /// ceiling and the entry unit those values address at.
    ///
    /// Declared by the producer for a POOLED tensor only. A pooled tensor's placement covers
    /// ONE slice of a pool (the paged KV cache's per-layer plane: one page, one layer, one
    /// plane), while a gathered read's index names stick-blocks across the WHOLE pool the
    /// runtime re-bases per launch — RELATIVE TO THE PLANE'S OWN BASE, which is why one reach
    /// serves every plane. So the placement footprint, the bound every plain window runs
    /// against, is the WRONG bound for the indexed read, and this pair is the right one: the
    /// gather door (`crate::emit::assemble_pointwise_broadcast_gather`) checks the descriptor's
    /// own entry stride against it — the unit the host's `GatherEntry` composition counts in —
    /// and the host-side composition mints through the SAME ceiling
    /// ([`crate::sdsc_abstract::gather_entry_ceiling`]), so a value above the pool's bound is
    /// unconstructible at both halves and the two cannot disagree. Nothing else widens — a
    /// plain sweep or a producer's write of the same tid is still bounded by the placement
    /// alone, so the reach is a fact about the INDEXED read, never about the tensor. Empty for
    /// every bundle with no pooled tensor, keeping those byte-identical.
    #[serde(default)]
    pub gather_reach: std::collections::BTreeMap<u32, crate::sdsc_abstract::GatherReach>,
    /// ⭐ THE KV BLOCK INDEX'S PENDING RESERVATION — carried UNPLACED out of the placement pass
    /// ([`KvBlockIndexReservation`] states why), spent by the one door
    /// [`BundleLayout::place_kv_block_index_last`] after the last host-staged activation has its
    /// offset. Compile-time only: the baked artifact carries the PLACEMENT the door mints, not
    /// the pending value.
    #[serde(skip)]
    pub kv_block_index: Option<KvBlockIndexReservation>,
    /// ⭐ THE PAGED-FOLD REGION'S PAGE CEILING — the ONE derivation, recorded where the prefix
    /// mask's pass grid is sized: the device's own cap over this bundle's page bytes (a pool is
    /// a single flex region) BOUNDED BY the pool this graph actually laid (its cache tensor's
    /// row extent — a reach fact: no context can out-reach the pool its entries gather). Read
    /// by the tape walk to state the region the fold family's tables and mask describe. `None`
    /// for a bundle with no attention — and for every bundle baked before the field existed.
    #[serde(default)]
    pub paged_fold_region_pages: Option<u32>,
}

/// Bump allocator for synthetic-intermediate offsets within the Intermediate
/// segment (task #55). `next` starts at the colored-intermediate high-water; each
/// distinct synthetic name gets one 128B-aligned offset, reused on later sight.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct SynthAlloc {
    pub next: u64,
    pub map: std::collections::BTreeMap<String, u64>,
    /// Declared FULL footprint (bytes) per synth name — set by [`BundleLayout::synth`].
    /// Present ⇒ the tensor OWNS its shape: `resolve_seg_base` reserves this whole size up
    /// front (so a per-head write at offset h·stride can't land on the next tensor) and GUARDS
    /// that no access exceeds it — an out-of-footprint access is a shape mismatch and a
    /// `cargo build` Err, not silent on-card aliasing.
    pub sizes: std::collections::BTreeMap<String, u64>,
}

/// LX scratchpad size per core: 2 MiB (`sysdef.cpp:587`, `lxSize = 2L*1024*1024`).
pub const LX_SIZE: u64 = 2 * 1024 * 1024;

/// ⭐ THE CEILING EVERY PINNED LX ADDRESS + FOOTPRINT MUST FIT UNDER — the frontend's end of the LX
/// split, [`crate::superdsc_opspec::FRONTEND_LX_END`] (1,625,344 B), not the 2 MiB scratchpad. dxp
/// allocates its own landings and staging from that address up (dbo's `reserveFrontendLx`), and it
/// does not know about our pins. A pin past it would share bytes with them on card. The pin allocator's
/// overflow is a `cargo build` panic, never an on-card overlap.
pub const LX_PIN_END: u64 = crate::superdsc_opspec::FRONTEND_LX_END;

const _: () = assert!(
    LX_PIN_BASE < LX_PIN_END && LX_PIN_END <= LX_SIZE,
    "the pinning window [LX_PIN_BASE, LX_PIN_END) lies inside the scratchpad"
);

/// Where PINNED LX-resident intermediates start: stick 1024. The runtime's own L3 landing
/// allocator demonstrably never touches the low region — every landing in a real 128-dsc
/// granite bundle sits in sticks [12698, 14778] (1.62–1.84 MiB), unchanged when a pin is
/// placed inside the low zone (measured: a stick-1000 pin compiled with a byte-identical
/// landing set and zero collisions in the compiled register image) — and the vendor's own
/// standalone fixtures pin `lxStartAddress_` at exactly this scale (`128*1000`, `128*2000`,
/// `1000000` in dsc_standalone.cpp:195/:259). Low LX is the pinning zone by the device's
/// own conventions; the landings live high.
pub const LX_PIN_BASE: u64 = 128 * 1024;

/// The pinned-LX twin of [`SynthAlloc`]: byte-granular addresses in the per-core LX
/// scratchpad for intermediates declared via [`BundleLayout::synth_lx_op_local`]. LX residency
/// exists for ONE measured reason: a SUB-STICK read offset on a collapsed stick dim (the
/// rmsnorm `rinv` lane read at element 8 of the exx2 pair) survives the whole compile chain
/// only on the LX path — the HBM path floors it away (the L3 EBR is stick-granular, and the
/// reader's 2bsplat reads the landing stick base with imm 0), which silently reads element 0
/// instead. An LX-pinned allocate is byte-granular end-to-end (`lbrInit_` in bytes → lxlu
/// register → LDSTU address), so writer-at-A / reader-at-A+16 is exactly expressible.
/// Bake-verified against dxp_standalone: the +16 lands in the compiled reader registers and
/// every HBM leg for the tensor vanishes.
#[derive(Clone, Debug)]
pub struct LxAlloc {
    /// Next free pin byte for CORE-LOCAL pins (starts at [`LX_PIN_BASE`]; bumps by each
    /// tensor's 128-aligned footprint so pinned tensors never overlap each other).
    pub next: u64,
    /// Next free byte for OP-LOCAL shared slots, bumping DOWN from [`LX_PIN_END`]. The two
    /// cursors are independent by construction — only [`lx_window_fits`] asserts they never
    /// meet — so a chunk pin landing later in walk order can never eat a slot reserved earlier.
    pub next_down: u64,
    /// Sum of the op-local slot footprints reserved so far — the [`lx_window_fits`] term the
    /// allocator guard and the `lx_form` witness both read (one number, two consults).
    pub op_local_bytes: u64,
    /// The probability-part pins the chunk form mints BESIDE its gathers — the other
    /// core-local term of the `lx_form` witness's budget, recorded at the chunk's admission
    /// ([`BundleLayout::note_chunk_prob_pin_bytes`]) so the baked artifact carries it to the
    /// host's own `chunk_gather_index_table` read. RECORDED, not bumped: the mint itself bumps
    /// `next`; this is the number the ADMISSION budgeted, noted so the two arms cannot drift.
    pub prob_pin_bytes: u64,
    /// The shared slot's start address per footprint class (bytes) — one slot per class,
    /// every op-local pin of that class resolving to the same address ([`LxAlloc::next_down`]).
    pub op_local_slots: std::collections::BTreeMap<u64, u64>,
    /// The names declared OP-LOCAL ([`BundleLayout::synth_lx_op_local`]): short-lived scratch
    /// whose def and use both live inside one assembly call, so same-class names share a slot
    /// under the per-group alternation walk's proof.
    pub op_local: std::collections::BTreeSet<String>,
    /// Pinned byte address per synth name.
    pub map: std::collections::BTreeMap<String, u64>,
    /// Declared footprint (bytes) per name — the resolve-time window guard's bound. For a
    /// CORE-LOCAL tensor this is ONE core's slice, not the tensor.
    pub sizes: std::collections::BTreeMap<String, u64>,
    /// The names declared CORE-LOCAL ([`BundleLayout::synth_lx_core_local`]): each core holds only its
    /// own work slice at the pin, so they are readable only under the work division that wrote them.
    pub core_local: std::collections::BTreeSet<String>,
    /// The work division the first op to touch each core-local tensor used, as a comparable
    /// signature ([`BundleLayout::record_lx_division`]). Every later op must match it.
    pub divisions: std::collections::BTreeMap<String, String>,
}
impl Default for LxAlloc {
    fn default() -> Self {
        LxAlloc {
            next: LX_PIN_BASE,
            next_down: LX_PIN_END,
            op_local_bytes: 0,
            prob_pin_bytes: 0,
            op_local_slots: std::collections::BTreeMap::new(),
            op_local: std::collections::BTreeSet::new(),
            map: std::collections::BTreeMap::new(),
            sizes: std::collections::BTreeMap::new(),
            core_local: std::collections::BTreeSet::new(),
            divisions: std::collections::BTreeMap::new(),
        }
    }
}

/// ⭐⭐ THE ONE PIN-WINDOW BUDGET — the single place the window's arithmetic lives. Both
/// regions of the pin window are summed here and nowhere else: core-local chunk pins bump
/// UP from [`LX_PIN_BASE`], op-local shared slots bump DOWN from [`LX_PIN_END`], and the
/// window holds exactly when the two never meet. Every guard consults THIS function — the
/// allocator's per-declaration asserts (below) with the live cursor state, and the
/// `ChunkScratch::lx_form` admission witness with the chunk's would-be bytes — so the
/// witness can never admit a form the allocator would refuse: one law, two readers,
/// zero drift (#645's ruling 4).
pub const fn lx_window_fits(core_local_bytes: u64, op_local_slot_bytes: u64) -> bool {
    LX_PIN_BASE + core_local_bytes + op_local_slot_bytes <= LX_PIN_END
}

/// Footprint (bytes) of a synth tensor with shape `dims` (outermost→innermost; the inner dim
/// is the stick axis, padded up to the dtype's stick-elem multiple). The allocator reserves THIS —
/// the tensor's true size — not a single per-op view (which under-reserved multi-head tensors → HBM
/// overlap). The dtype is the tensor's typed [`Df`]: fp32 split-K merge partials are 4-byte /
/// 32-elem-stick, fp8/int8 are 1-byte / 128-elem-stick (½ fp16), everything else fp16 (2-byte /
/// 64-elem-stick). INERT for fp16 (the working 1318-emit build is unchanged).
pub fn synth_footprint_bytes(dims: &[u32], df: Df) -> u64 {
    let stick_elems = df.elems_per_stick();
    let word_length = df.word_length();
    let inner = dims.last().copied().unwrap_or(stick_elems);
    let inner_stick = inner.div_ceil(stick_elems) * stick_elems;
    let outer: u64 = dims
        .iter()
        .rev()
        .skip(1)
        .map(|&d| d as u64)
        .product::<u64>()
        .max(1);
    outer * inner_stick as u64 * word_length as u64
}

/// Round `n` up to a 128-byte boundary (Spyre requires tensor start addresses to be
/// multiples of 128 B — `runtime_operation.hpp:365`). This is the ACTUAL primitive the segment `pack`
/// closure advances every tensor base by; `pub` so the SegLayout island's Kani proof
/// (`align128_emitter_equals_model`) can assert the on-card packing rests on the proven round-up.
pub fn align128(n: u64) -> u64 {
    (n + 127) & !127
}

/// Mint a synthetic's operand SPELLING and record its identity with the layout in one step.
///
/// Every `{tensor}_{role}` name goes through here, so `resolve_seg_base` can recover the identity
/// by LOOKUP. Minting and registering are one call because a name minted without being registered
/// is exactly the case the parse used to paper over.
pub fn syn(layout: Option<&BundleLayout>, id: PlaceId) -> String {
    let n = id.to_string();
    if let Some(l) = layout {
        l.ids.borrow_mut().insert(n.clone(), id);
    }
    n
}

/// Door 1 of the window mints — the DECLARED-SHAPE bound, on the operand facts (name + declared
/// [`StickLayout`]) rather than a handle. Shared by [`BundleLayout::check_window`] (which adds the
/// placement door) and the pointwise family's operand mint, whose callers hold handles but no
/// `BundleLayout` (every unit test; the gate probes). `off_elems` and the extents are ELEMENTS of
/// the operand's own `df`.
pub(crate) fn window_fits_declared(
    name: &str,
    layout: StickLayout,
    off_elems: u64,
    rows: u32,
    cols: u32,
    what: &str,
) -> Result<(), String> {
    let shape_elems = layout.rows as u64 * layout.cols as u64;
    let end_elems = off_elems + rows as u64 * cols as u64;
    if end_elems > shape_elems {
        return Err(format!(
            "{what}: the operand window [{off_elems}, {end_elems}) (elements) of '{name}' \
             runs past the end its handle declares — {rows}×{cols} = {shape_elems} elements \
             at {:?}. An op that sweeps past its own tensor is a defect in the computation \
             that produced this offset; it cannot compile.",
            layout.df,
        ));
    }
    Ok(())
}

impl BundleLayout {
    /// The identity behind an emitted operand's spelling, or `None` if nothing minted it.
    pub fn id_of(&self, name: &str) -> Option<PlaceId> {
        self.ids.borrow().get(name).copied()
    }

    /// ⭐⭐⭐ THE PLACEMENT-AUTHORITY MINT — a [`Win`] over a tensor THIS layout places.
    ///
    /// Closes the class the bake-time guard could only catch late: an operand's offset and its
    /// name arriving from different laws and sweeping past the tensor they name (the windowed
    /// matmul that read 1/8 past its own tensor; the pad law that swept 512 columns out of a
    /// 64-column window). Identity, offset and extents are minted as ONE value here, checked
    /// against BOTH bounds at expansion time — a refusal is a `cargo build` failure:
    ///
    /// 1. THE DECLARED SHAPE — the handle's own [`StickLayout`] (`rows × cols` at its `df`): a
    ///    window whose `off + rows·cols` runs past the tensor's own element count is an `Err`.
    /// 2. THE PLACEMENT FOOTPRINT — resolved by NAME exactly the way `resolve_seg_base` resolves
    ///    it at bake (LX pin → `Act` placement → declared synth), against the same `align128`
    ///    granule, so the mint and the bake-time guard can never disagree about where a tensor
    ///    ends. A window past the footprint is an `Err` naming the segment it would alias.
    ///
    /// And a placement SMALLER than the tensor's declared shape is itself an `Err`: the two
    /// authorities disagree about the tensor, and every window over it is suspect — caught even
    /// when the window itself fits both.
    ///
    /// `what` names the computation the window is for (op + operand role); every refusal carries
    /// it with the offset, the extents and the bound it exceeded, so a failed build names the
    /// defect, not just the number.
    pub fn win<'a, K: KindTag>(
        &self,
        stk: &'a Stk<K>,
        off: DevOff,
        rows: u32,
        cols: u32,
        what: &str,
    ) -> Result<Win<'a, K>, String> {
        self.check_window(
            stk.name(),
            stk.layout(),
            off.into_raw_elems() as u64,
            rows,
            cols,
            what,
        )?;
        Ok(Win::minted(stk, off, rows, cols))
    }

    /// ⭐ The placement-authority CHECK behind [`Self::win`], on the operand FACTS — the handle's
    /// name and declared [`StickLayout`] — instead of the handle itself. For the kind-erased
    /// operand lists the pointwise family lowers ([`crate::emit::EwOperand`]): one slice holds
    /// many layout kinds, so none of them can mint a kind-tagged [`Win`], but every one of them
    /// runs the SAME two doors against the same numbers. A facts-checked operand and a
    /// handle-minted window can never disagree about where a tensor ends.
    pub(crate) fn check_window(
        &self,
        name: &str,
        layout: StickLayout,
        off_elems: u64,
        rows: u32,
        cols: u32,
        what: &str,
    ) -> Result<(), String> {
        let word = layout.df.word_length() as u64;
        let shape_elems = layout.rows as u64 * layout.cols as u64;
        let shape_bytes = shape_elems * word;
        let end_elems = off_elems + rows as u64 * cols as u64;
        let end_bytes = end_elems * word;
        // 1. THE DECLARED SHAPE — the window fits the tensor the HANDLE declares.
        window_fits_declared(name, layout, off_elems, rows, cols, what)?;
        // 2. THE PLACEMENT FOOTPRINT — `resolve_seg_base`'s three arms, same numbers, so the mint
        //    and the bake-time guard read ONE window model.
        if let Some((_pin, fp)) = self.lx_pin(name) {
            // A CORE-LOCAL tensor's pin holds ONE core's slice, so the whole-operand window is
            // not its bound (`core_local_reads_agree` checks the per-core slice against it) and
            // its declared shape may be the full tensor's — the same early return the bake-time
            // guard takes. But it has no addressable offset: every core's slice starts AT the pin.
            if self.lx_core_local(name) {
                if off_elems != 0 {
                    return Err(format!(
                        "{what}: core-local LX tensor '{name}' windowed at element {off_elems} — \
                         each core's slice sits at the pin, so a view into it has no offset to take"
                    ));
                }
                return Ok(());
            }
            if end_bytes > fp {
                return Err(format!(
                    "{what}: the operand window [{off_elems}, {end_elems}) = {end_bytes}B of \
                     '{name}' exceeds its pinned-LX footprint {fp}B — the sweep would run past \
                     the pin into the next pinned intermediate. Fix the declared shape \
                     (`BundleLayout::synth_lx_op_local`) or the op's view/extents."
                ));
            }
            if fp < shape_bytes {
                return Err(format!(
                    "{what}: '{name}' is pinned in LX at {fp}B but its handle declares \
                     {rows}×{cols} = {shape_bytes}B — a pin smaller than the tensor's own \
                     declared shape. The layout and the handle disagree about this tensor."
                ));
            }
            return Ok(());
        }
        // A real caller-bound buffer → its global placement. The id comes from the layout's own
        // mint table, NOT from picking the digits back out of the spelling.
        if let Some(PlaceId::Act(tid)) = self.id_of(name)
            && let Some(p) = self.placements.get(&tid)
        {
            let fp = align128(p.size).max(128);
            if end_bytes > fp {
                return Err(format!(
                    "{what}: the operand window [{off_elems}, {end_elems}) = {end_bytes}B of \
                     '{name}' (t{tid}) exceeds its placement footprint {fp}B (seg{}) — the sweep \
                     addresses PAST its own tensor and would alias whatever is placed next. The \
                     offset arithmetic and the placement must come from the SAME layout \
                     authority.",
                    p.segment
                ));
            }
            if fp < shape_bytes {
                return Err(format!(
                    "{what}: '{name}' (t{tid}) is placed at {fp}B but its handle declares \
                     {rows}×{cols} = {shape_bytes}B at {:?} — a placement smaller than the \
                     tensor's own declared shape. The layout and the handle disagree about this \
                     tensor; every window over it is suspect.",
                    layout.df,
                ));
            }
            return Ok(());
        }
        // A SYNTHETIC intermediate — declared or refused; the mint never allocates.
        let s = self.synth.borrow();
        match s.sizes.get(name).copied() {
            None => Err(format!(
                "{what}: '{name}' is windowed but was never declared to the layout. Every \
                 intermediate is known at bake — declare it with `BundleLayout::synth(id, dims)` \
                 at the site that mints the name."
            )),
            Some(fp) => {
                if end_bytes > fp {
                    return Err(format!(
                        "{what}: the operand window [{off_elems}, {end_elems}) = {end_bytes}B of \
                         '{name}' exceeds its declared synth footprint {fp}B — the sweep would \
                         run past the intermediate into the next one. Fix the declared shape \
                         (`BundleLayout::synth`) or the op's view/extents."
                    ));
                }
                if fp < shape_bytes {
                    return Err(format!(
                        "{what}: '{name}' is declared at {fp}B but its handle declares \
                         {rows}×{cols} = {shape_bytes}B — a declaration smaller than the \
                         tensor's own declared shape. The layout and the handle disagree about \
                         this tensor."
                    ));
                }
                Ok(())
            }
        }
    }

    /// ⭐ LAW (b)'s AUTHORITY — a gathered source's window at a sweep, under the TYPED-REACH
    /// law: a sweep that fits the placement keeps its window (no reach needed, byte-identical);
    /// a sweep that EXCEEDS the placement must carry a DECLARED reach — the typed statement of
    /// its index values' bound (entry ceiling + unit, `BundleLayout::gather_reach`) — and the
    /// family is then read from the placement (a placement holding the whole reach is a source
    /// extent, `min(swept, ceiling)`; a slice is a pool, one entry). The placement footprint
    /// (real buffer, LX pin, or synth declaration — the same three arms [`Self::check_window`]'s
    /// footprint doors read, so the two can never disagree about where a tensor ends) supplies
    /// the bytes through [`gathered_window_rows`], the one pure statement of the law the proof
    /// series pins. Refuses, fail-closed: an over-sweep with no reach, a source whose placement
    /// holds less than one whole row of the stated width, and a name the layout never declared.
    pub(crate) fn gathered_source_rows(
        &self,
        name: &str,
        swept: u32,
        row_elems: u32,
        df: crate::superdsc_opspec::Df,
        reach: Option<crate::sdsc_abstract::GatherReach>,
    ) -> Result<u32, String> {
        let row_bytes = row_elems as u64 * df.word_length() as u64;
        let fp = if let Some((_pin, fp)) = self.lx_pin(name) {
            fp
        } else if let Some(crate::place::PlaceId::Act(tid)) = self.id_of(name)
            && let Some(p) = self.placements.get(&tid)
        {
            align128(p.size).max(128)
        } else {
            *self.synth.borrow().sizes.get(name).ok_or_else(|| {
                format!(
                    "'{name}' is read through an index but was never declared to the layout \
                         — every intermediate is known at bake; declare it (`BundleLayout::synth`) \
                         or bind it before gathering through it"
                )
            })?
        };
        let swept_bytes = u64::from(swept).saturating_mul(row_bytes);
        gathered_window_rows(swept, fp, row_bytes, reach.map(|r| r.max_entry().get() as u64))
            .ok_or_else(|| {
                if fp < row_bytes {
                    format!(
                        "'{name}' (placed at {fp}B) holds less than one {row_elems}-element row of \
                         the width this gather strides — the index has no row it could legally name"
                    )
                } else {
                    format!(
                        "'{name}' is swept for {swept} rows × {row_elems} elements ({swept_bytes}B) \
                         but placed at {fp}B — the swept window exceeds its placement footprint, and \
                         a gathered read may address past its placement only under a DECLARED REACH \
                         (the typed bound on its index values: an entry ceiling and unit, \
                         `BundleLayout::gather_reach`). Declare the reach for '{name}' or place the \
                         whole source"
                    )
                }
            })
    }

    /// ⭐ THE `[hd, hd]` IDENTITY TABLE an attention class of head dim `hd` reads — its tid
    /// from this bundle's [`Self::attn_class_hds`] registry, the one lookup every identity-kernel copy (krep/vrep,
    /// the KV cache write, the query group re-layout) resolves through. `None` when `hd` is not one of
    /// the tape's attention classes — a registry desync each caller refuses at build.
    pub fn identity_tid_of_class(&self, hd: u32) -> Option<u32> {
        self.attn_class_hds
            .iter()
            .position(|&h| h == hd)
            .map(crate::reserved_tids::identity_class_tid)
    }

    /// Mint the NEXT intermediate tid — bundle-unique by construction. First call seeds the
    /// cursor above the highest non-reserved placement id in the whole bundle (reserved tids
    /// are the const tables and are skipped, never landed on); every later call advances
    /// monotonically. `floor` is the caller's node-derived bound: the returned tid is at or
    /// above it, so a node whose bindings sit above the current cursor still mints clear of
    /// its own tensors too. See [`Self::inter_cursor`] for why the seed is bundle-wide.
    pub fn take_inter_tid(&self, floor: u32) -> u32 {
        let mut cur = match self.inter_cursor.get() {
            Some(c) => c,
            None => {
                let above_places = self
                    .placements
                    .keys()
                    .copied()
                    .filter(|t| !crate::reserved_tids::is_reserved(*t))
                    .max()
                    .map_or(0, |m| m + 1);
                above_places.max(floor)
            }
        };
        if cur < floor {
            cur = floor;
        }
        while crate::reserved_tids::is_reserved(cur) {
            cur += 1;
        }
        let tid = cur;
        self.inter_cursor.set(Some(cur + 1));
        tid
    }

    /// Declare a SYNTHETIC intermediate at its FULL shape (`dims`, outermost→innermost),
    /// reserving its true footprint in the Intermediate-segment allocator UP FRONT. The tensor
    /// then OWNS its shape: a later per-head write at offset h·stride cannot land on the next
    /// synthetic (no under-reservation), and `resolve_seg_base` turns any access beyond the
    /// footprint into a `cargo build` Err (the typed-shape contract). Idempotent per name.
    ///
    /// This is the principled replacement for the old bump allocator that sized each synthetic
    /// from its FIRST op's per-op view (ONE head) — which aliased sp/spprod/mxp on-card and
    /// produced the garbage attention. The shape is now part of the tensor, not guessed per-op.
    pub fn synth(&self, id: PlaceId, dims: &[u32]) {
        self.synth_df(id, dims, Df::Fp16);
    }

    /// [`synth`](Self::synth) for a NON-fp16 intermediate — the packed fp8/int8 (1-byte / 128-stick)
    /// or fp32-merge (4-byte / 32-stick) tensors. The footprint is reserved at the given [`Df`]'s true
    /// width, so a fp8 activation reserves HALF the fp16 bytes (the residency win) — not a name-suffix
    /// guess. `synth(name, dims)` is exactly `synth_df(name, dims, Df::Fp16)`.
    pub fn synth_df(&self, id: PlaceId, dims: &[u32], df: Df) {
        self.synth_bytes(id, synth_footprint_bytes(dims, df));
    }

    /// Declare a SYNTHETIC intermediate LX-RESIDENT and OP-LOCAL at its FULL shape (`dims`,
    /// outermost→innermost): short-lived scratch whose DEF and USE both live inside one
    /// assembly call (the rmsnorm `rinv`: rsqrt produces it, apply consumes it, nothing else
    /// touches it). Like [`Self::synth`] this REPLACES the seg3 HBM slot — an LX-resident
    /// tensor has no HBM copy at all — and like [`Self::synth_lx_core_local`] it pins in the
    /// per-core scratchpad. Idempotent per name.
    ///
    /// ⭐ SAME-CLASS PINS SHARE ONE SLOT. The class is the 128-aligned footprint, and every
    /// name of one class resolves to the SAME address — one 64-KiB slot serves every norm in
    /// the group, so a bundle's pin demand is the number of CLASSES, not of norms (#645: a
    /// rung's ~23 stacked `[rows,64]` rinvs refused the window; as ~3 classes they fit with
    /// room for the chunk forms). This is sound because each name's live range is confined to
    /// its own assembly call: the per-group alternation walk
    /// (`lower_subtile_tape_to_superdsc::walk_lx_slot_alternation`) proves at build, per launch
    /// group over its programs in order, that no same-class def lands inside another's live
    /// range — so sharing is checked, never assumed, and the fallback on a refusal is to split
    /// the conflicting class, never to un-share globally.
    ///
    /// Slots bump DOWN from [`LX_PIN_END`] so the two pin families are independent by
    /// construction: a chunk core-local pin declared later in walk order (bumping up from
    /// [`LX_PIN_BASE`]) can never eat a slot reserved earlier — the only place the two regions
    /// are compared is [`lx_window_fits`].
    ///
    /// ⛔ PANICS when the two regions would meet — computed by [`lx_window_fits`], the same
    /// one-function budget the `lx_form` admission witness reads — so this runs at
    /// `#[forward]` expansion AS the cargo-build guard: a bundle whose pins do not fit the
    /// frontend's LX window is unbuildable, not silently overlapping dxp's own LX allocations.
    pub fn synth_lx_op_local(&self, id: PlaceId, dims: &[u32]) {
        let name = id.to_string();
        self.ids.borrow_mut().insert(name.clone(), id);
        let mut a = self.lx.borrow_mut();
        if a.map.contains_key(&name) {
            return;
        }
        let fp = align128(synth_footprint_bytes(dims, Df::Fp16).max(1));
        let (slot, added) = match a.op_local_slots.get(&fp) {
            Some(&s) => (s, 0),
            None => (a.next_down - fp, fp),
        };
        assert!(
            lx_window_fits(a.next - LX_PIN_BASE, a.op_local_bytes + added),
            "op-local LX slots meet the core-local pins inside the frontend's LX window: the \
             {fp}-byte class of '{name}' slots at {slot}, with core-local pins through {} and \
             {} B of op-local slots against the {LX_PIN_END} boundary. The op-local classes are \
             per-FOOTPRINT, so this is one class too many at this geometry — either the norm \
             rows shrank a class away, or a chunk form this bundle also declares leaves no room \
             for it. The `lx_form` witness reads the same budget (`lx_window_fits`), so it \
             refusing a chunk form here is the designed alternative, not this panic.",
            a.next,
            a.op_local_bytes + added
        );
        if added > 0 {
            a.next_down = slot;
            a.op_local_bytes += added;
            a.op_local_slots.insert(fp, slot);
        }
        a.map.insert(name.clone(), slot);
        a.sizes.insert(name.clone(), fp);
        a.op_local.insert(name);
    }

    /// Declare a synthetic intermediate LX-resident CORE-LOCALLY: every core holds only ITS OWN work
    /// slice, `per_core_bytes` of it, at one pinned address shared by all cores — the reference's form
    /// for every softmax intermediate (`allocate … component_ lx`, one `startAddressCoreCorelet_` value
    /// on all 32 cores). Like [`Self::synth_lx_op_local`] it REPLACES the seg3 HBM slot: the tensor never leaves
    /// LX. Idempotent per name.
    ///
    /// ⛔ ONLY AN OP THAT DIVIDES THE WORK EXACTLY AS THE PRODUCER DID CAN READ IT — core `c`'s slice is
    /// whatever core `c`'s producer pass wrote. The emitter records the first op's division and refuses
    /// every op that differs, at build ([`Self::record_lx_division`]).
    ///
    /// ⛔ PANICS past [`LX_PIN_END`], the same `cargo build` guard as [`Self::synth_lx_op_local`].
    pub fn synth_lx_core_local(&self, id: PlaceId, per_core_bytes: u64) {
        let name = id.to_string();
        self.ids.borrow_mut().insert(name.clone(), id);
        let mut a = self.lx.borrow_mut();
        if a.map.contains_key(&name) {
            return;
        }
        let o = a.next;
        let fp = align128(per_core_bytes.max(1));
        assert!(
            lx_window_fits(a.next + fp - LX_PIN_BASE, a.op_local_bytes),
            "a core-local LX intermediate overflows the frontend's LX window: '{name}' pins at {o} \
             with a {fp}-byte per-core slice, meeting the {} B of op-local slots already reserved \
             against the {LX_PIN_END} boundary. Cut the op into smaller parts so one core's slice \
             fits, or keep the tensor in HBM (`BundleLayout::synth`). The `lx_form` witness reads \
             the same budget (`lx_window_fits`), so this chunk form being refused there instead is \
             the designed alternative, not this panic.",
            a.op_local_bytes
        );
        a.next = o + fp;
        a.map.insert(name.clone(), o);
        a.sizes.insert(name.clone(), fp);
        a.core_local.insert(name);
    }

    /// The op-local slot bytes reserved so far — one slot per footprint class
    /// ([`Self::synth_lx_op_local`]). THE `lx_form` witness's op-local term: by the time the
    /// consumer pass runs, the declare pass has already declared every op-local pin of the
    /// rung, so this number is final and the witness and the allocator's guard read the same
    /// quantity through [`lx_window_fits`].
    pub fn lx_op_local_slot_bytes(&self) -> u64 {
        self.lx.borrow().op_local_bytes
    }

    /// Record the probability-part pins the chunk form mints beside its gathers
    /// ([`crate::ir::bridge::tiled_op_sdsc_op::attn::chunk_prob_pins`]) — the second core-local
    /// term of the `lx_form` witness's budget, summed with the gather pins at the chunk's
    /// admission. Noted so the baked artifact carries the term to the host's own
    /// `chunk_gather_index_table` read — the bake/host agreement law: the host must refuse
    /// exactly what the bake refused. REPLACEMENT, not a sum: one bundle mints one attention
    /// op's probability parts, so re-emitting the same chunk (the declare pass, then the real
    /// one) notes the same number.
    pub fn note_chunk_prob_pin_bytes(&self, bytes: u64) {
        self.lx.borrow_mut().prob_pin_bytes = bytes;
    }

    /// The recorded probability-part term ([`Self::note_chunk_prob_pin_bytes`]) — `0` for a
    /// bundle whose chunk form mints none.
    pub fn lx_prob_pin_bytes(&self) -> u64 {
        self.lx.borrow().prob_pin_bytes
    }

    /// The core-local pin bytes declared so far (the up-cursor's distance from
    /// [`LX_PIN_BASE`]) — the other [`lx_window_fits`] term.
    pub fn lx_core_local_bytes(&self) -> u64 {
        self.lx.borrow().next - LX_PIN_BASE
    }

    /// Whether `name` was declared OP-LOCAL ([`Self::synth_lx_op_local`]) — a same-class
    /// shared-slot pin, as opposed to a core-local one.
    pub fn lx_op_local(&self, name: &str) -> bool {
        self.lx.borrow().op_local.contains(name)
    }

    /// Whether `name` is LX-resident CORE-LOCALLY ([`Self::synth_lx_core_local`]).
    pub fn lx_core_local(&self, name: &str) -> bool {
        self.lx.borrow().core_local.contains(name)
    }

    /// ⭐ THE OP-LOCAL PIN TABLE — one row per shared-slot pin, `(name, class bytes, slot address)`,
    /// name-sorted. THE alternation walk's and the dump ledger's whole view of the op-local family:
    /// the class is the row's `bytes` (the footprint IS the class key, [`LxAlloc::op_local_slots`]),
    /// and every name of one class carries the same slot address. Empty for a bundle that declares
    /// no op-local pins — the walk's own no-op case.
    pub fn lx_op_local_pins(&self) -> Vec<(String, u64, u64)> {
        let a = self.lx.borrow();
        a.op_local
            .iter()
            .map(|name| {
                let fp = a.sizes[name];
                let slot = a.map[name];
                (name.clone(), fp, slot)
            })
            .collect()
    }

    /// ⛔ THE CORE-LOCAL READ LAW, at build: the FIRST op to touch core-local `name` records its work
    /// division (`signature`); every later op must present the same one, or this is an `Err` naming both.
    /// A different division hands core `c` a slice core `c`'s producer never wrote — fluent garbage on
    /// card with no fault.
    pub fn record_lx_division(
        &self,
        name: &str,
        op: &str,
        signature: String,
    ) -> Result<(), String> {
        let mut a = self.lx.borrow_mut();
        match a.divisions.get(name) {
            None => {
                a.divisions.insert(name.to_string(), signature);
                Ok(())
            }
            Some(first) if *first == signature => Ok(()),
            Some(first) => Err(format!(
                "op '{op}' divides core-local LX tensor '{name}' as {signature}, but its first op \
                 divided it as {first} — each core holds only the slice its own producer pass wrote, so \
                 every op on a core-local tensor must divide the work identically"
            )),
        }
    }

    /// The pinned-LX address + declared footprint of `name`, or `None` when the tensor is not
    /// LX-resident. THE name-keyed residency consult — `resolve_seg_base` turns it into the
    /// view's address and the pointwise builders' LX pass turns it into `Allocation::Lx`.
    pub fn lx_pin(&self, name: &str) -> Option<(u64, u64)> {
        let a = self.lx.borrow();
        a.map
            .get(name)
            .and_then(|&pin| a.sizes.get(name).map(|&fp| (pin, fp)))
    }

    /// ⛔⛔⛔ DECLARE A SYNTH AT THE FOOTPRINT OF THE TENSOR IT SHADOWS, NOT OF ONE CHUNK OF IT.
    ///
    /// A decomposition's intermediate usually has EXACTLY the shape of the node's output —
    /// `silu(gate) -> tmp` then `multiply(tmp, up) -> out`. But a wide op is CHUNKED: the tape splits
    /// granite's 12800-wide MLP intermediate into column blocks, and `lower_*_node` sees only its own
    /// block. Declaring `[rows, cols]` from the node therefore reserves the FIRST chunk's width, and
    /// since declaration is first-one-wins ([`Self::synth_bytes`]) every later chunk writes past it.
    ///
    /// On granite-3.1-8b that is chunk 2 — offset 8192, width 4608 of 12800 — landing 9216 B past a
    /// 16384 B reservation, straight into the next intermediate. The footprint guard turns it into a
    /// build error rather than on-card garbage, and this is the cure: take the width from the OUTPUT
    /// TENSOR's placement, which `compute_bundle_layout` sized from `ir.tensors` before any chunking.
    ///
    /// Falls back to `dims` when the tensor has no placement (a synth OF a synth, and every unit test
    /// that lowers without a layout) — the old behaviour, correct whenever the op is not chunked.
    pub fn synth_like(&self, id: PlaceId, shadows: u32, dims: &[u32], df: Df) {
        match self.placements.get(&shadows) {
            Some(p) => self.synth_bytes(id, p.size),
            None => self.synth_bytes(id, synth_footprint_bytes(dims, df)),
        }
    }

    /// The byte-exact core of [`Self::synth_df`]: bump-allocate `bytes` for `id`, once.
    fn synth_bytes(&self, id: PlaceId, bytes: u64) {
        let name = id.to_string();
        let a = self.synth.borrow_mut();
        // The IDENTITY is recorded even when the OFFSET already exists. `resolve_seg_base` bump-
        // allocates an intermediate the first time an op references it, which can happen BEFORE the
        // declaring `synth()` call; the early return below then skipped the id and the placement had
        // no identity to be built from. Declaration order decides the offset, never the identity.
        drop(a);
        self.ids.borrow_mut().insert(name.clone(), id);
        let mut a = self.synth.borrow_mut();
        if a.map.contains_key(&name) {
            return;
        }
        let o = align128(a.next);
        let fp = align128(bytes.max(1));
        a.next = o + fp;
        a.map.insert(name.clone(), o);
        a.sizes.insert(name, fp);
    }

    /// THE arrangement authority. Record that tensor `name` is addressed through `want`; a later op that
    /// addresses it NON-equivalently ([`StickLayout::addr_eq`]) is a build `Err` naming the tensor.
    /// `addr_eq` is true for the same layout, and for a byte-identical RESHAPE (same total elements, both
    /// contiguous — e.g. `[1,2048]` ↔ `[32,64]`), which IS safe: identical bytes. It is FALSE for the M>1
    /// same-shape Dense-vs-stick scramble AND for a different-TOTAL disagreement (`[4,64]` vs `[32,64]`),
    /// because a consumer that reads more elements than the producer wrote reads garbage. That is NOT
    /// silently allowed — it names the tensor at build time so the size mismatch is confronted, not baked.
    pub fn declare_arrangement(
        &self,
        name: &str,
        want: StickLayout,
    ) -> Result<StickLayout, SuperDscError> {
        let mut m = self.arrangements.borrow_mut();
        match m.get(name) {
            Some(existing) if !existing.addr_eq(&want) => Err(SuperDscError(format!(
                "tensor '{name}': device arrangement {want:?} conflicts with the earlier {existing:?} — a \
                 tensor has ONE physical layout, but a producer and consumer address it differently (a \
                 same-shape Dense-vs-stick scramble, OR a different-TOTAL size mismatch where the consumer \
                 reads bytes the producer never wrote). Address it ONE way, or insert an explicit \
                 restickify/replication op. This is the arrangement authority (declare_arrangement)."
            ))),
            Some(existing) => Ok(*existing),
            None => {
                m.insert(name.to_string(), want);
                Ok(want)
            }
        }
    }

    /// ⭐⭐⭐ THE KV BLOCK INDEX'S ONE PLACEMENT DOOR — the LAST host-staged activation.
    ///
    /// ⛔ WHY A DOOR, AND WHY "LAST" IS LOAD-BEARING. One session runs bodies that gather and
    /// bodies that do not (the one-request ladder's chunked ceiling beside its unchunked sweep
    /// rungs), and the host binds each step at the CEILING's placements — so
    /// `ladder_stages_one_layout` exempts exactly `KV_BLOCK_INDEX_TID` from its comparison, an
    /// exemption that is sound ONLY if the index's presence moves nothing the guard compares.
    /// Placed where the reservation used to sit — beside the masks, inside the placement pass —
    /// it pushed every LATER host-staged tensor (the step inputs, then the program-stated
    /// scales) down by its reservation (279,552 B on granite-3.1-2b), and the guard refused the
    /// first rung whose gathering differed: the ceiling staged `t1412` at 603,392 while the
    /// rung's own layout read it at 323,840. Called AFTER `place_step_inputs` and
    /// `complete_program_scales`, a gathering body's index lands past every compared offset and
    /// a non-gathering body simply has none — the exemption holds by construction, and every
    /// gathering body of the session places its index at the SAME tail (their compared
    /// placements agree, so their tails agree), which is the offset the host stages the table
    /// at for all of them.
    pub fn place_kv_block_index_last(&mut self) {
        let Some(res) = self.kv_block_index.take() else {
            return;
        };
        let seg = SegRole::Activation.segment();
        let off = self.segment_bytes[seg];
        let tid = crate::reserved_tids::KV_BLOCK_INDEX_TID;
        self.placements.insert(
            tid,
            TensorPlacement {
                tid,
                bank: 0,
                role: SegRole::Activation,
                segment: seg,
                offset: off,
                size: res.bytes,
                // ⭐ THE WRITE-COVER, FROM THE RESERVATION'S OWN PITCH — the mask's block
                // stride, the one number both tensors derive from. A bind is a whole number
                // of these blocks: the host stages the live passes' entries, one block per
                // live page, and a partial block is a truncated pass row.
                cover: BindCover::FoldBlocks {
                    block_bytes: res.block_bytes,
                },
            },
        );
        self.segment_bytes[seg] = align128(off + res.bytes);
    }
}

/// ⭐⭐⭐ LAW (b)'s ONE PURE STATEMENT — a gathered source's window rows at a sweep, a placement
/// footprint, and the reach the table declared (its entry ceiling; `None` = no reach). For a
/// gathered source the addressable set is the index's IMAGE over the source's rows — `addr =
/// idx·skip_addr + base`, one row per entry — so the bytes an op can touch are rows the INDEX
/// names, never the swept rectangle. The law, in two halves:
///
/// * **A SWEEP THAT FITS KEEPS ITS WINDOW** — `swept·row_bytes ≤ fp` ⇒ `window == swept`, no
///   reach needed. Every existing gather form (the embedding's card-proven emission, every pool
///   leg under its plane) lands here: byte-identical, no reach required.
/// * **AN OVER-SWEEP REQUIRES A DECLARED REACH** — the placement alone cannot bound the values:
///   for a source placed WHOLE the image ⊆ the placement, but for a SLICE of a pool the values
///   address past the placement into the pool the layout cannot see, and `min(swept, fp)` would
///   bound nothing they obey. So the over-sweep must carry the TYPED statement of its index
///   values' bound — a [`GatherReach`](crate::sdsc_abstract::GatherReach), whose unit the door
///   has already checked against the read's stride — and the family is then read from the
///   placement: `fp ≥ ceiling·row_bytes` (the placement holds the WHOLE reach) is a SOURCE
///   EXTENT, `window = min(swept, ceiling)`; a placement holding less is a POOL SLICE the
///   runtime re-bases, `window = 1` entry's direct coverage. NO reach ⇒ `None`: fail-closed,
///   the refusal `gathered_source_rows` words.
///
/// * **SAFETY** — `window·row_bytes ≤ fp` always: the fitting half is the sweep itself; the
///   source-extent min is at most the ceiling, and `ceiling·row_bytes ≤ fp` is that family's
///   own condition; the pool's one entry is at most `fp` because `fp ≥ row_bytes` is a
///   precondition. The guard's purpose (no aliasing past the placement) is preserved by
///   construction.
/// * **MONOTONE** — `window ≤ swept` always: the reach-bounded window never exceeds the swept
///   window. The law only ever shrinks a handle, the same shape as the residency fix.
/// * **BYTE-IDENTITY** — where the sweep fits, `window == swept` exactly.
/// * **TABLE-VALUE CORRECTNESS IS NOT THIS LAW'S** — whether the staged values name in-range
///   rows is the staging family's own law at birth (the reach's ceiling is enforced where the
///   values are BORN), not a fact a window over the placement could check.
/// * **THE FAMILY FORK IS SOUNDNESS-BASED, NOT PROVENANCE-BASED** — `fp ≥
///   ceiling·row_bytes ⇒ source extent` is safe WHATEVER family minted the reach: a placement
///   that holds the whole reach cannot be exceeded by the source-extent window (`min(swept,
///   ceiling)` rows of in-range values, `ceiling·row_bytes ≤ fp` by the fork's own
///   condition), and the one-entry pool window fires only where the placement does NOT hold
///   the reach — which is exactly where the pool's own machinery (the runtime re-base, the
///   birth-door ceiling) governs what the values address. The fork reads WHICH BOUND THE
///   PLACEMENT CAN CERTIFY, never which producer minted the reach; nobody should read it as a
///   family recognizer.
///
/// The proof series pins the properties above (see `proofs` below).
pub(crate) fn gathered_window_rows(
    swept: u32,
    fp: u64,
    row_bytes: u64,
    reach_ceiling: Option<u64>,
) -> Option<u32> {
    if row_bytes == 0 || fp < row_bytes {
        // No whole row of this width to name — the index has no row it could legally address.
        return None;
    }
    if u64::from(swept)
        .checked_mul(row_bytes)
        .is_some_and(|b| b <= fp)
    {
        // The sweep fits: byte-identical, no reach needed.
        return Some(swept);
    }
    // The over-sweep: fail-closed without a declared reach.
    let ceiling = reach_ceiling?;
    // The reach's own extent at the read's stride — `ceiling · entry_elems · word`, the unit
    // check having equated the read's row with the reach's entry. Overflow is a refusal: a
    // reach whose extent does not fit u64 bounds nothing.
    let coverage = ceiling.checked_mul(row_bytes)?;
    if fp >= coverage {
        // SOURCE EXTENT — the placement holds the whole reach; the values name its rows.
        Some(swept.min(u32::try_from(ceiling).unwrap_or(swept)))
    } else {
        // POOL SLICE — the runtime re-bases; the direct window is one entry's coverage.
        Some(1)
    }
}

#[cfg(kani)]
mod proofs {
    use super::{Df, gathered_window_rows};

    /// LAW (b), pinned at the compiler's own vocabulary. The grind diagnosis first: this
    /// harness's first draft gave `row_bytes`, `swept`, `ceiling` and `fp` to z3 as FREE u64s,
    /// and every product in the law and the harness (`swept × row_bytes`, `ceiling ×
    /// row_bytes`, `window × row_bytes`) had TWO symbolic sides — nonlinear bitvector
    /// arithmetic no solver converges on (8 CPU-hours, no verdict; the standing 15-minute
    /// no-verdict rule exists because of it). The fix is TYPING, not narrowing: `row_bytes`
    /// is one of `Df::word_length`'s values ({1, 2, 4} — fp8/int8 are 1-byte, fp16/bf16 are
    /// 2-byte, fp32/uint32 are 4-byte; there is no fifth word length), so every product has
    /// one CONCRETE side — a multiply by 1, 2 or 4, a shift. `swept`, `ceiling` and `fp` are
    /// u32, the walk's own row-count and placement-footprint types; a gathered source's
    /// placement is a pool activation (tile-sized — the real gathered tables are
    /// hundreds of KiB), far under the 4 GiB the type bounds, and the 16 GiB weight regions
    /// are not gathered sources. At this vocabulary the u64 headroom the general signature
    /// keeps for the overflow refusals is unreachable (ceiling ≤ u32::MAX, row_bytes ≤ 4),
    /// and the law is proved over exactly the values the placement code can hand it.
    #[kani::proof]
    fn gathered_source_window_never_exceeds_its_placement() {
        let df: u8 = kani::any();
        kani::assume(df < 3);
        let row_bytes: u64 = match df {
            0 => u64::from(Df::Fp8.word_length()),
            1 => u64::from(Df::Fp16.word_length()),
            _ => u64::from(Df::Fp32.word_length()),
        };
        let swept: u32 = kani::any();
        kani::assume(swept >= 1);
        let fp = u64::from(kani::any::<u32>());
        let has_reach: bool = kani::any();
        let ceiling = u64::from(kani::any::<u32>());
        let reach = has_reach.then_some(ceiling);
        let fits = u64::from(swept)
            .checked_mul(row_bytes)
            .is_some_and(|b| b <= fp);
        let Some(w) = gathered_window_rows(swept, fp, row_bytes, reach) else {
            // The only refusals: no whole row of this width, an over-sweep with no reach, or a
            // reach whose own extent overflows — each names a fact, none is a silent admit.
            if fp >= row_bytes && fits {
                assert!(false, "a fitting sweep never refuses");
            }
            if fp >= row_bytes && !fits && reach.is_some() {
                assert!(
                    ceiling.checked_mul(row_bytes).is_none(),
                    "an over-sweep with a reach refuses only when the reach's extent overflows"
                );
            }
            return;
        };
        // SAFETY — the window, at the row width, stays inside the footprint.
        assert!(u64::from(w) * row_bytes <= fp);
        // BYTE-IDENTITY — a sweep that fits its placement keeps its window exactly.
        if fits {
            assert!(w == swept);
        }
        // MONOTONE — the reach-bounded window never exceeds the swept window.
        assert!(w <= swept);
        // FAIL-CLOSED — an over-sweep admits only under a declared reach.
        if !fits {
            assert!(reach.is_some(), "an over-sweep with no reach must refuse");
        }
    }
}
