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

use crate::place::PlaceId;
use crate::sdsc_abstract::StickLayout;
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

/// Pages one request's block table can address — the validity rows the mask reserves.
///
/// 32 is chosen so the reservation is `32 * 256 * 2` = 16384 B, byte-identical to the pre-paged
/// mask (`nqh * mq * cap * 2` at mq=1), which keeps every activation placed after it at the offset
/// it had before paging. A larger reserve pushes seven activation tensors along by its excess, and
/// this path is documented as placement-sensitive. 32 pages is 8192 positions per request; the
/// binding limit in practice is the runtime pool.
pub const MAX_PAGES_PER_REQUEST: u64 = 32;

/// Fold passes a batched-decode bundle reserves prefix-mask room for.
///
/// A pass is one `(request, page)`, and EVERY page comes from the one shared pool, so the passes in
/// flight can never exceed the pool's size however the batch is shaped: 32 requests of one page and
/// one request of 32 pages are the same 32 passes. Sizing the mask as
/// `requests * MAX_PAGES_PER_REQUEST` instead counts a batch where every request is simultaneously
/// at full context, which the pool cannot hold — at 32 requests that is 537 MB of mask against 17 MB
/// of actual worst case, and it is re-uploaded every step.
pub const MAX_FOLD_PASSES: u64 = 64;

impl SegRole {
    pub fn segment(self) -> usize {
        self as usize
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
    /// Bytes occupied per segment (index = segment id 0..6). For the weight segment this is BANK
    /// 0's bytes — see [`BundleLayout::weight_bank_bytes`].
    pub segment_bytes: [u64; 7],
    /// ⭐ THE WEIGHT SEGMENT'S EXTRA BANKS: bytes of banks `1..N`, empty when the weights fit ONE
    /// device region (every model that worked before banking existed). See
    /// `bundle::Placement::bank` and `bank_weight_segment`.
    #[serde(default)]
    pub weight_bank_bytes: Vec<u64>,
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
    /// exactly the ATTN_SCALE mechanism). The emitter (`lower_scalarmul_node`) looks up its scale's index
    /// here → the const TID it multiplies by; the WORKER reads this list (serde) and binds each
    /// `t{tid} = [scale]`. So the emitter and worker agree on scale↔TID by construction (no reward-hack
    /// fold, no host-route — the op is a real on-device pointwise `mul`).
    #[serde(default)]
    pub scalarmul_scales: Vec<f32>,
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

impl BundleLayout {
    /// The identity behind an emitted operand's spelling, or `None` if nothing minted it.
    pub fn id_of(&self, name: &str) -> Option<PlaceId> {
        self.ids.borrow().get(name).copied()
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
}
