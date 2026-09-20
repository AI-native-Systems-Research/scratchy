// SPDX-License-Identifier: Apache-2.0
//! Execute an emitted **SuperDSC** bundle on the real IBM Spyre AIU — the `--target sendnn` silicon
//! run path, as the worker sees it.
//!
//! This is the worker-facing wrapper: f32 host buffers in, f32 logits out, names as `t{id}` (the
//! SubtileIR tensor id). The work is [`crate::superdsc_exec`]'s, which drives the card through the
//! one C-ABI seam in [`crate::sdk_abi`].
//!
//! ⛔ THE SENGRAPH PATH IS GONE. The offline_decoder `GraphLoader`/`Predict` families (`SpyreSession`
//! / `GroupedSession` and their C++ shim) were the pre-SuperDSC bring-up path and had no live
//! caller: the worker routes every sendnn bundle to `SuperDsc`. Under `sendnn` the only bundle that
//! runs is the dxp-compiled SuperDSC bundle.

use crate::superdsc_exec::{Executor, KvDim, NewTokens, PoolPages, SegIdx, SeqPos, Vocab};
use anyhow::{Result, anyhow, bail};
use scratchy_subtile::sdsc_abstract::{
    FoldPages, FoldRowRegime, FoldWalker, MaskBlocks, PrefixMaskShape,
};
use scratchy_tensors::DType as SDType;
use tracing::debug;

use crate::manifest::bytes_to_f32;

/// f32 → little-endian f16 bytes (the device tensor dtype at the bind boundary).
fn f32_to_f16_le(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    out
}

/// The three model extents every constructor here forwards unchanged — one value because they are
/// always read from the same parse and passed together, and separately they pushed `new_inner` past the
/// argument limit. Typed wrappers (`Vocab`/`KvDim`/`PoolPages`) are minted from these downstream.
#[derive(Clone, Copy)]
struct SessionDims {
    vocab: usize,
    kv_dim: usize,
    num_blocks: usize,
}

/// LE f16 bytes → f32.
fn f16_le_to_f32(bytes: &[u8]) -> Vec<f32> {
    // `as_chunks` over `chunks_exact`: the chunk width is a CONSTANT, so this yields `&[u8; 2]` and the
    // element indexing below cannot be out of bounds by construction (clippy::chunks_exact_to_as_chunks).
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| half::f16::from_le_bytes(*c).to_f32())
        .collect()
}

/// ⛔ GUARD (env precondition): the deeptools host functions the executor calls open their hardware
/// arch config at `${DEEPTOOLS_PATH}/perfdsc/HardwareArchMapping/…`. When `DEEPTOOLS_PATH` is unset
/// the SDK composes a broken relative `..` path and aborts deep inside with
/// `DtException: Unable to open arch config` (observed on dd2 when the var was not exported).
/// Convert that cryptic abort into an early, actionable error.
///
/// This is an *environment* precondition, so it cannot be a build-time guard — the runtime env is
/// unknown at `cargo build`. Validate only (no env mutation): the SDK env block is supplied
/// externally, exactly like `LD_LIBRARY_PATH` / `FLEX_COMPUTE`.
fn ensure_deeptools_arch_config() -> Result<()> {
    let root = std::env::var_os("DEEPTOOLS_PATH").filter(|v| !v.is_empty());
    let Some(root) = root else {
        bail!(
            "DEEPTOOLS_PATH is not set — the deeptools host functions need it to locate the arch \
             config (${{DEEPTOOLS_PATH}}/perfdsc/HardwareArchMapping/…). Set it to the SDK 'share' \
             dir, e.g. DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share."
        );
    };
    let arch_dir = std::path::Path::new(&root)
        .join("perfdsc")
        .join("HardwareArchMapping");
    if !arch_dir.is_dir() {
        bail!(
            "DEEPTOOLS_PATH={} but the arch config dir {} does not exist — point DEEPTOOLS_PATH at \
             the SDK 'share' dir (e.g. /opt/ibm/spyre/deeptools/share).",
            std::path::Path::new(&root).display(),
            arch_dir.display()
        );
    }
    Ok(())
}

/// The prefix mask's per-fold-pass BYTE stride, minted only from the mask's own declared shape —
/// never a loose integer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MaskRepStride(u64);

impl MaskRepStride {
    /// One block of `shape`, in f16 bytes — the only constructor.
    pub const fn of<const STICK: u32, const COLS: u32>(
        shape: PrefixMaskShape<STICK, COLS>,
    ) -> MaskRepStride {
        MaskRepStride(shape.rep_stride_bytes())
    }
}

/// The per-request row-block stride in the INTERMEDIATE segment, minted only from the bundle's own
/// declared [`FoldRowRegime`] — a whole-batch bundle's 0 arrives by derivation, not by choice.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IntRepStride(u64);

impl IntRepStride {
    /// The regime's stride across one request's `nqh` head rows — the only constructor.
    pub const fn of(regime: FoldRowRegime, nqh: u32) -> IntRepStride {
        IntRepStride(regime.int_rep_stride_bytes(nqh))
    }
}

/// A resident SUPERDSC session: the dxp-compiled bundle loaded from its AoT cache directory and run
/// through the flex runtime. Weights pack into resident device segments and the KV grows in a
/// resident pool; `run_step` appends `n_new` NEW positions at `[seq_pos..seq_pos+n_new)` and returns
/// the last new position's logits.
pub struct SuperDscSession {
    exec: Executor,
    /// Logits column count.
    pub vocab: usize,
    /// Resident capacity in slots (= the longest context THIS request can hold).
    pub cap: usize,
    /// PAGED: pages in the SHARED pool, and positions per page (0 ⇒ not paged).
    pub pool_pages: usize,
    pub page_slots: usize,
    /// One resident K/V row width (= kv_heads·head_dim).
    pub kv_dim: usize,
    /// ⛔ THE BROADCAST PREFIX MASK'S EXTENT, from the bundle's own placement — see
    /// [`Executor::pmask_slots`]. NOT `cap`: `cap` is the resident POOL's slot count (what the KV probes
    /// read back at), and using it for the mask is what over-bound the mask by 4x on every prompt.
    /// `None` ⇒ the bundle placed no broadcast mask, and staging one must be REFUSED, not guessed.
    pub pmask_slots: Option<usize>,
    /// Whether this bundle's ops read KV through the PAGED pool — see [`Executor::is_paged`]. Load-time
    /// callers pair it with [`Self::pmask_slots`]: unpaged, the attention sweeps the whole `cap`, so a
    /// `cap` past the mask's placement is unservable and must be refused rather than staged either way.
    pub paged: bool,
}

impl SuperDscSession {
    /// Load the bundle, bind the weights, and prepare (allocate + upload the resident program,
    /// weights and KV pool).
    #[allow(clippy::type_complexity)]
    pub fn new(
        code: &'static crate::bundle_code::BundleCode<'static>,
        vocab: usize,
        kv_dim: usize,
        num_blocks: usize,
        num_sources: u32,
        resident_sources: &[u32],
        weights: Vec<(usize, Vec<u8>, SDType, Vec<usize>)>,
    ) -> Result<Self> {
        Self::new_inner(
            code,
            SessionDims {
                vocab,
                kv_dim,
                num_blocks,
            },
            num_sources,
            resident_sources,
            weights,
            &[],
        )
    }

    /// [`new`](Self::new) for a session whose WEIGHTS come from another session by alias — every rung
    /// of the prefill width ladder. Binds nothing and stages nothing: each `borrowed` segment is
    /// declared dead-on-arrival before prepare, so prepare gives it a placeholder and skips its H2D.
    ///
    /// The caller MUST [`alias_seg_from`](Self::alias_seg_from) every listed segment immediately
    /// after this returns — a borrowed segment left un-aliased holds ONLY the placeholder, so a
    /// failed alias must be fatal for this session, never a warning that leaves it on a stub.
    pub fn new_borrowing(
        code: &'static crate::bundle_code::BundleCode<'static>,
        vocab: usize,
        kv_dim: usize,
        num_blocks: usize,
        num_sources: u32,
        resident_sources: &[u32],
        borrowed: &[i64],
    ) -> Result<Self> {
        Self::new_inner(
            code,
            SessionDims {
                vocab,
                kv_dim,
                num_blocks,
            },
            num_sources,
            resident_sources,
            Vec::new(),
            borrowed,
        )
    }

    /// Allocate every segment at FULL size but populate none — no weights bound, so staging no-ops.
    ///
    /// This exists for the batched-prefill session, whose 2.6 GB seg1 reservation is what PLACES the
    /// decode session's weight region: borrowing (a 128 B placeholder) instead moves decode's weights
    /// ~2.6 GB earlier in HBM and costs 1.8 tok/s, measured. The placement effect comes from the
    /// ALLOCATION, not from the bytes — and those bytes are dead, since the session is aliased onto
    /// decode's copy immediately afterwards. So reserving without populating keeps the placement and
    /// drops the bind + stage (6.97s of startup for granite-3.1-8b, measured).
    pub fn new_reserving(
        code: &'static crate::bundle_code::BundleCode<'static>,
        vocab: usize,
        kv_dim: usize,
        num_blocks: usize,
        // ⛔ REQUIRED, from the generated wiring. Ids `0..num_sources` are host-filled every
        // forward; the executor refuses to launch with one unwritten. Passing it as an ARGUMENT
        // rather than a setter is the lock: a new session site cannot forget it.
        num_sources: u32,
        resident_sources: &[u32],
    ) -> Result<Self> {
        Self::new_inner(
            code,
            SessionDims {
                vocab,
                kv_dim,
                num_blocks,
            },
            num_sources,
            resident_sources,
            Vec::new(),
            &[],
        )
    }

    fn new_inner(
        code: &'static crate::bundle_code::BundleCode<'static>,
        dims: SessionDims,
        num_sources: u32,
        // The prefix-KV source ids — placed and inside `0..num_sources`, but device-resident, so
        // no per-step bind names them.
        resident_sources: &[u32],
        weights: Vec<(usize, Vec<u8>, SDType, Vec<usize>)>,
        borrowed: &[i64],
    ) -> Result<Self> {
        let SessionDims {
            vocab,
            kv_dim,
            num_blocks,
        } = dims;
        ensure_deeptools_arch_config()?;
        let mut exec = Executor::load(
            code,
            Vocab(vocab),
            KvDim(kv_dim),
            PoolPages(num_blocks as i64),
        )?;
        exec.declare_sources(num_sources, resident_sources.iter().copied());
        // BORROWED segments must be declared BEFORE prepare — afterwards the full region is already
        // allocated and transferred, so the flag would be a silent no-op (the executor refuses).
        for &seg in borrowed {
            let s = SegIdx::checked(seg)
                .ok_or_else(|| anyhow!("mark_borrowed: segment {seg} out of range"))?;
            exec.mark_borrowed(s)?;
        }

        // Bind every weight to its input `t{id}` as device-dtype bytes. Prepare packs them, so they
        // must be bound before prepare. Ownership MOVES in — no copy anywhere on this path.
        let t_bind = std::time::Instant::now();
        let n_bound = weights.len();
        for (id, data, dt, _shape) in weights {
            let bytes = match dt {
                SDType::F16 => data,
                SDType::Fp8E4m3 => data, // fp8 binds VERBATIM (1-byte); do NOT dequant to f16
                other => f32_to_f16_le(&bytes_to_f32(&data, other)),
            };
            exec.bind_weight(crate::bundle_code::PlaceId::Act(id as u32), bytes);
        }
        debug!(
            "[timing] bind weights loop took {:.2}s",
            t_bind.elapsed().as_secs_f64()
        );

        // Convert + stage the bound weights into host staging NOW, on this thread, while the
        // background device prewarm is still bringing the card up — this takes the ~1s parallel
        // IEEE→SEN convert off the prepare critical path. Bytes reaching the device are identical;
        // pure scheduling.
        //
        // NOTHING BOUND is the same situation as BORROWED: with no weights this call only zero-fills
        // a host shadow of a segment that is about to be replaced by an alias.
        if !borrowed.is_empty() || n_bound == 0 {
            debug!(
                "[timing] weight bind + stage SKIPPED ({})",
                if borrowed.is_empty() {
                    "nothing bound — segments reserved, populated by alias".to_string()
                } else {
                    format!("segments {borrowed:?} borrowed by alias")
                }
            );
        } else {
            let t_stage = std::time::Instant::now();
            exec.stage_weights()?;
            debug!(
                "[timing] weight stage took {:.2}s",
                t_stage.elapsed().as_secs_f64()
            );
        }

        exec.prepare()?;
        Ok(Self {
            vocab: exec.vocab.0,
            cap: exec.cap() as usize,
            pool_pages: exec.pool_pages().0 as usize,
            page_slots: exec.page_slots().0 as usize,
            kv_dim: exec.kv_dim.0,
            pmask_slots: exec.pmask_slots(),
            paged: exec.is_paged(),
            exec,
        })
    }

    /// Kick the process-wide device bring-up (~5.7s) on a background thread so it overlaps host-side
    /// weight load instead of sitting on the critical path inside prepare.
    pub fn prewarm_runtime() {
        crate::sdk_abi::prewarm_runtime();
    }

    /// Whether this bundle's resident cache is PAGED.
    pub fn is_paged(&self) -> bool {
        self.page_slots > 0
    }

    /// Positions a request can hold given the `pages` it was assigned.
    pub fn context_for_pages(&self, pages: usize) -> usize {
        pages * self.page_slots
    }

    /// Declare how many BLOCKS the staged prefix mask has, so the fold can refuse a pass that would
    /// read past it. Takes [`MaskBlocks`], whose only mint is `MaskBlocks::of(MaskPassGrid)` — the
    /// count the fold will walk, never a loose integer.
    ///
    /// PRIVATE, with [`FoldWalker::declare_fold`] as the public door: this number and the page count
    /// below are one decision, and a caller able to send half of it can block the mask by a geometry
    /// the fold does not walk.
    fn set_mask_blocks(&mut self, blocks: MaskBlocks) -> Result<()> {
        self.exec.set_mask_blocks(blocks.get() as u64);
        Ok(())
    }

    /// Declare the host's [`FoldPages`] — the page count the fold must walk, derived ONCE from the
    /// batch's `BatchSlot`. The executor derives the same number itself; declaring it makes a
    /// divergence a refusal rather than a silently wrong fold.
    ///
    /// PRIVATE for the same reason as [`Self::set_mask_blocks`]: see [`FoldWalker::declare_fold`].
    fn set_fold_pages(&mut self, pages: FoldPages) -> Result<()> {
        self.exec.set_fold_pages(pages.get().get() as u64);
        Ok(())
    }

    /// Declare the prefix mask's per-fold-pass stride. A batched decode's mask is one
    /// `[nqh*mq, page]` block per (request, page) pass; only the worker, which stages it, knows how
    /// deep a block is.
    pub fn set_mask_stride(&mut self, stride: MaskRepStride) -> Result<()> {
        self.exec.set_mask_stride(stride.0);
        Ok(())
    }

    /// Declare the per-request row-block stride in the intermediate segment.
    pub fn set_int_stride(&mut self, stride: IntRepStride) -> Result<()> {
        self.exec.set_int_stride(stride.0);
        Ok(())
    }

    /// ⭐⭐⭐⭐⭐ THE GATHER'S ENTRY UNIT — how many index entries one PHYSICAL PAGE spans, i.e. the factor
    /// that turns a block-table page number into the index value the card reads.
    ///
    /// ⛔ NOT A CONSTANT, AND NOT `4096`. An index entry advances one `skip_addr`, which the emitter
    /// declares as ONE STICK BLOCK of the value operand (`PageExtent::of_one_stick()` on the slot axis)
    /// — `hd * ELEMS_PER_STICK` elements, so 4096 at head_dim 64 and **8192 at head_dim 128**. A page
    /// spans every LAYER (`page_stride_bytes = iters * kv_stride`), so the factor carries `iters` too and
    /// is a property of this BUNDLE, not of the pool alone. Every doc comment in this tree that says
    /// "global 4096-element block" is implicitly hd=64.
    ///
    /// ⛔ DERIVED FROM THE SESSION'S OWN `page_stride_bytes`, WHICH IS THE NUMBER `page_base_bytes`
    /// MULTIPLIES. That is the whole point: the host's page address is `phys * page_stride_bytes` and the
    /// card's is `idx * skip_addr + base`, so `idx = phys * (page_stride_bytes / skip_addr_bytes)` is an
    /// identity between two quantities read off the same session — not two derivations that must be kept
    /// in agreement. `zz_the_gather_index_reproduces_the_host_page_address` pins it.
    ///
    /// `None` when the bundle is unpaged, or when a page is not a whole number of entries — in which case
    /// no integer index can name a page boundary and the caller must refuse rather than round.
    ///
    /// The arithmetic itself is [`gather_entries_per_page`]; this hands it the ONE number only the
    /// session knows.
    ///
    /// [`gather_entries_per_page`]: scratchy_subtile::sdsc_abstract::gather_entries_per_page
    pub fn gather_entries_per_page(
        &self,
        pool: scratchy_subtile::sdsc_abstract::PagedKvPool,
    ) -> Option<scratchy_subtile::sdsc_abstract::EntriesPerPage> {
        scratchy_subtile::sdsc_abstract::gather_entries_per_page(
            self.exec.kv_page_stride_bytes(),
            pool,
        )
    }

    /// ⭐⭐⭐⭐⭐ WHICH BODY THE NEXT LAUNCH AT `start` WILL RUN, and what it declares — see
    /// [`superdsc_exec::StepBody`]. The host asks this BEFORE it stages anything, because both the
    /// gather's index table and the prefix mask's block form are decisions about the selected body.
    ///
    /// ⛔ `start` IS THE SAME `start` THE HOST PASSES TO `run_step`, and the selection is derived from it
    /// alone — so the body staged for and the body launched are the same body by construction, not by two
    /// call sites agreeing.
    ///
    /// [`superdsc_exec::StepBody`]: crate::superdsc_exec::StepBody
    pub fn step_body(&mut self, start: usize) -> Result<crate::superdsc_exec::StepBody> {
        self.exec
            .step_body(crate::superdsc_exec::SeqPos(start as i64))
    }

    /// Install a request's page map under the SLOT it occupies in this forward, with its write cursor.
    ///
    /// ⛔ A SLOT, AND NOTHING ELSE. This took a second argument — the KV ROW the request holds for life —
    /// and threaded it down to `SessionKv::kv_rows`, where NOTHING READ IT: every address builder below
    /// says "NO REQUEST TERM", so the row never reached an address. It is deleted rather than kept "for
    /// reporting": a device-layer field named after a forbidden concept is read as evidence the concept
    /// is still load-bearing, and it cost one investigation already.
    ///
    /// What distinguishes two requests down here is their PAGES, installed under the slot they occupy.
    pub fn set_block_table(
        &mut self,
        slot: crate::fold_plan::LaunchSlotIdx,
        pos: i64,
        pages: &[i64],
    ) -> Result<()> {
        if pages.is_empty() {
            bail!("set_block_table: empty page map — a request needs at least one page");
        }
        self.exec
            .set_block_table(
                crate::fold_plan::RowIdx::from_launch_row(slot.get() as u64),
                SeqPos(pos),
                pages,
            )
            .map_err(|e| {
                anyhow!(
                    "{e} (request slot {} at {pos} with {} page(s); pool has {})",
                    slot.get(),
                    pages.len(),
                    self.pool_pages
                )
            })
    }

    /// Run ONE INCREMENTAL forward over `n_new` NEW positions at `[seq_pos..seq_pos+n_new)`.
    /// `activations[i] = (name, f32 data)` are this step's NEW-token sources bound by NAME
    /// (`t{id}` embed + cos/sin), narrowed to f16 at the boundary. The resident pool holds
    /// `[0..seq_pos)`; this writes only the new K/V and attends `[0..seq_pos+i]` causally. Returns
    /// the logits `[vocab]` of the LAST NEW position.
    pub fn run_step(
        &mut self,
        n_new: usize,
        seq_pos: usize,
        activations: &[crate::wiring::Bind],
    ) -> Result<Vec<f32>> {
        Ok(self
            .run_step_inner(n_new, seq_pos, activations, &[], &[], true)?
            .unwrap_or_default())
    }

    /// [`run_step`](Self::run_step) for a forward whose logits are NEVER read — batched prefill.
    /// Skips both the logits D2H and the sen→IEEE convert.
    ///
    /// This matters because the prefill bundle's suffix is the lm_head-less tail (rmsnorm +
    /// scalarmul, zero matmul), so its logits segment is never written — yet every chunk would
    /// otherwise D2H ~3 MB of untouched memory (~1.9 ms measured) into a vocab-sized host buffer.
    pub fn run_step_no_logits(
        &mut self,
        n_new: usize,
        seq_pos: usize,
        activations: &[crate::wiring::Bind],
    ) -> Result<()> {
        self.run_step_inner(n_new, seq_pos, activations, &[], &[], false)?;
        Ok(())
    }

    /// [`run_step_no_logits`](Self::run_step_no_logits) plus activations the caller ALREADY holds as
    /// LE f16 bytes.
    ///
    /// An activation staged as f32 is paid for three times: the host builds it, the bind loop
    /// converts it element by element into a fresh Vec, and only then does it reach the card. For a
    /// decode batch's prefix mask that is the dominant per-step host cost — 93% of every element
    /// bound (2.26 M of 2.42 M at 16 requests, measured), and 0.124·mq² of it a single repeated
    /// constant, because a fold pass masks off every row that is not its own request. The f32 round
    /// trip buys nothing there: the value is `-inf` or `0.0`, both exactly representable, so the
    /// caller can write the f16 bit pattern directly and the card receives the SAME BYTES.
    pub fn run_step_no_logits_f16(
        &mut self,
        n_new: usize,
        seq_pos: usize,
        activations: &[crate::wiring::Bind],
        activations_f16: &[(crate::bundle_code::PlaceId, Vec<u8>)],
    ) -> Result<()> {
        self.run_step_inner(n_new, seq_pos, activations, activations_f16, &[], false)?;
        Ok(())
    }

    /// [`run_step`](Self::run_step) with a pre-narrowed f16 channel.
    pub fn run_step_f16(
        &mut self,
        n_new: usize,
        seq_pos: usize,
        activations: &[crate::wiring::Bind],
        activations_f16: &[(crate::bundle_code::PlaceId, Vec<u8>)],
    ) -> Result<Vec<f32>> {
        Ok(self
            .run_step_inner(n_new, seq_pos, activations, activations_f16, &[], true)?
            .unwrap_or_default())
    }

    /// ⭐⭐⭐ [`run_step_f16`](Self::run_step_f16) plus the INT32 channel — a gather's index table.
    ///
    /// ⛔ A THIRD LIST, NOT A THIRD ENTRY IN THE f16 ONE. `activations_f16` means "bytes that are
    /// already fp16", and every one of them still passes through `ieee_to_sen_bytes` in the shadow
    /// write; int32 entries must NOT (see `Executor::raw_ids`). Two lists that both mean "device
    /// bytes" but need different shadow writes is exactly the collapse this keeps apart.
    pub fn run_step_i32(
        &mut self,
        n_new: usize,
        seq_pos: usize,
        activations: &[crate::wiring::Bind],
        activations_f16: &[(crate::bundle_code::PlaceId, Vec<u8>)],
        activations_i32: &[(crate::bundle_code::PlaceId, Vec<u8>)],
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        self.run_step_inner(
            n_new,
            seq_pos,
            activations,
            activations_f16,
            activations_i32,
            want_logits,
        )
    }

    fn run_step_inner(
        &mut self,
        n_new: usize,
        seq_pos: usize,
        activations: &[crate::wiring::Bind],
        // Already in the device's format — bound as-is, never re-narrowed.
        activations_f16: &[(crate::bundle_code::PlaceId, Vec<u8>)],
        // ⭐ RAW LE int32 bytes — bound through `bind_input_raw`, so the shadow write COPIES them
        // instead of re-encoding them as fp16. A gather's index table, and nothing else today.
        activations_i32: &[(crate::bundle_code::PlaceId, Vec<u8>)],
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        // BIND-LOOP TIMING (SCRATCHY_SDSC_PHASE_TIME, the same flag as the executor's phase
        // breakdown): this loop runs BEFORE predict, so it is invisible to the
        // preamble_h2d/compute/logits_d2h split. For batched prefill it converts ~300k f32
        // (embedding 63k + cos/sin 159k + cmask 63k + consts) — a real per-forward cost the TTFT
        // budget never accounted for.
        let t_bind = std::time::Instant::now();
        let mut bind_elems = 0usize;
        for (id, vals) in activations {
            bind_elems += vals.len();
            self.exec.bind_input(*id, f32_to_f16_le(vals));
        }
        let mut bind_f16_bytes = 0usize;
        for (id, bytes) in activations_f16 {
            bind_f16_bytes += bytes.len();
            self.exec.bind_input(*id, bytes.clone());
        }
        // ⛔ `bind_input_raw`, NOT `bind_input`. See `Executor::raw_ids`: the shadow write applies
        // `ieee_to_sen_bytes` to every ordinary bind, which reads each 4-byte index entry as two fp16
        // values and rewrites both — turning block 37 into a different, valid, arbitrary block.
        // ⛔ AND THE PAYLOAD ARRIVES PAIRED WITH ITS PLACEMENT. `place_raw` is the only constructor of
        // the value `bind_input_raw` takes, so the three ways a raw bind used to go wrong — a byte
        // count that is not whole int32 entries, a payload longer than its own placement, and a name
        // this bundle never placed (silently SKIPPED, and a skipped index reads as entry 0, a real
        // address) — are one refusal at one door instead of a check here, a `bail!` in the refill loop
        // and a `continue` nobody could see.
        for (id, bytes) in activations_i32 {
            let bind = self.exec.place_raw(*id, bytes.clone()).ok_or_else(|| {
                anyhow!(
                    "bind: raw tensor '{id}' ({} B) has no placement in this bundle, does not fit \
                     the one it has, or is not a whole number of int32 entries — a partial trailing \
                     entry is a truncated address, an over-long bind overwrites the next tensor, and \
                     an unplaced one would be skipped in silence (entry 0 is page 0's first block, a \
                     REAL address)",
                    bytes.len()
                )
            })?;
            self.exec.bind_input_raw(bind);
        }
        if std::env::var_os("SCRATCHY_SDSC_PHASE_TIME").is_some() {
            eprintln!(
                "[sdsc-host] bind_loop={:.2} ms  ({} acts, {bind_elems} f32->f16, {} f16 acts, \
                 {bind_f16_bytes} f16 bytes as-is)",
                t_bind.elapsed().as_secs_f64() * 1e3,
                activations.len(),
                activations_f16.len(),
            );
        }
        let mut logits = if want_logits {
            vec![0u8; self.vocab * 2]
        } else {
            Vec::new()
        };
        let out = want_logits.then_some(&mut logits[..]);
        self.exec
            .predict(NewTokens(n_new as i64), SeqPos(seq_pos as i64), out)?;
        Ok(want_logits.then(|| f16_le_to_f32(&logits)))
    }

    // ── DEBUG numeric bisection (SPYRE_SUPERDSC_SELFTEST). NOT a serving path. ──

    /// Bind an activation source (f32 → f16) before a prefix run.
    pub fn bind(&mut self, id: crate::bundle_code::PlaceId, vals: &[f32]) -> Result<()> {
        self.exec.bind_input(id, f32_to_f16_le(vals));
        Ok(())
    }

    /// Run ONLY the prefix program (after binding synthetic activations).
    pub fn run_prefix_only(&mut self) -> Result<()> {
        self.exec.run_prefix_only()
    }

    /// Run the FULL forward (prefix → body×iters → seam → suffix) after binding synthetic
    /// activations, leaving every tid resident. `Ok(false)` if the bundle is not the per-op rolled
    /// path.
    pub fn run_full(&mut self) -> Result<bool> {
        self.exec.run_full()
    }

    /// Read back a tensor (up to `cap` elems) from its resident segment.
    pub fn read_tensor(&mut self, id: crate::bundle_code::PlaceId, cap: usize) -> Result<Vec<f32>> {
        Ok(f16_le_to_f32(&self.exec.read_tensor(id, cap)?))
    }

    /// Zero a whole tensor segment ON THE DEVICE, without a host mirror of it.
    ///
    /// The KV pool must start zeroed (decode computes q·k over a whole page before the validity mask
    /// applies, so uninitialised bytes go through the arithmetic as denormals/NaN). Doing that as
    /// read_seg → write_seg required a host buffer the size of the pool — 960 MB, and 3.84 GB at a
    /// 1024-slot page — to mirror bytes only the device ever reads.
    pub fn zero_seg(&mut self, seg: i64) -> Result<()> {
        let s =
            SegIdx::checked(seg).ok_or_else(|| anyhow!("zero_seg: segment {seg} out of range"))?;
        self.exec.zero_seg(s)
    }

    /// D2H the whole tensor segment as raw sen-fp16 bytes (the prefill→decode KV handoff: read the
    /// PREFILL session's seg2, then [`write_seg`](Self::write_seg) it into the DECODE session's).
    pub fn read_seg(&mut self, seg: i64) -> Result<Vec<u8>> {
        let s =
            SegIdx::checked(seg).ok_or_else(|| anyhow!("read_seg: segment {seg} out of range"))?;
        self.exec.read_seg(s)
    }

    /// Overwrite the FIRST `bytes.len()` bytes of a tensor segment + H2D exactly that window — the
    /// spilled-weight-tail copy, whose source is one bundle's segment and whose destination is
    /// another's of a DIFFERENT extent. See [`Executor::write_seg_prefix`].
    pub fn write_seg_prefix(&mut self, seg: i64, bytes: &[u8]) -> Result<()> {
        let s = SegIdx::checked(seg)
            .ok_or_else(|| anyhow!("write_seg_prefix: segment {seg} out of range"))?;
        self.exec.write_seg_prefix(s, bytes)
    }

    /// Overwrite the whole tensor segment from raw sen-fp16 bytes + H2D it.
    pub fn write_seg(&mut self, seg: i64, bytes: &[u8]) -> Result<()> {
        let s =
            SegIdx::checked(seg).ok_or_else(|| anyhow!("write_seg: segment {seg} out of range"))?;
        // GUARD: this is a RAW whole-segment copy with no layout cross-check. Assert the source byte
        // count == THIS session's segment size. A mismatch means the two independently baked bundles
        // placed the segment differently — which would silently hand wrong-OFFSET data for every
        // tensor past the divergence (semi-coherent garble, no crash).
        //
        // ⛔ THE CAUSE IS NOT ALWAYS A KV ASYMMETRY, which this used to assert outright and cost a
        // misdiagnosis. A whole-segment copy is only meaningful between bundles whose extent for that
        // segment AGREES, and that holds for seg2 (nothing in its size depends on the query-row count)
        // but for very little else: any segment carrying an intermediate COLOR or the logits scales
        // with m, so its total differs per rung by construction — MEASURED, seg6 is 421,257,216 B in
        // the decode bundle and 432,218,112 B in a prefill one, the same 419,430,400 B tail beside a
        // 1,826,816 B and a 12,787,712 B color. For a payload that occupies a PREFIX of the segment
        // (the spilled weight tail) the copy that is actually well-defined is
        // [`SuperDscSession::write_seg_prefix`], not this one.
        let expect = self.exec.seg_bytes(s);
        if bytes.len() != expect {
            bail!(
                "write_seg(seg={seg}): source {} bytes != this session's segment {expect} bytes — \
                 the two bundles place seg{seg} differently. A whole-segment copy needs their \
                 extents to agree; if the payload is a prefix of the segment, copy it with \
                 write_seg_prefix instead",
                bytes.len()
            );
        }
        self.exec.write_seg(s, bytes)
    }

    /// Point THIS session's segment at `owner`'s device region, so the two SHARE it and no copy is
    /// needed. Replaces only this session's descriptor — `owner` is untouched and its behaviour is
    /// bit-for-bit unchanged.
    ///
    /// Used to alias the batched-PREFILL session's seg2 (the resident KV) onto the DECODE session's,
    /// so prefill writes the prompt's KV directly where decode reads it. That deletes the per-prompt
    /// read_seg(2) → write_seg(2) round trip: 188.7 MB of DMA plus two 90 MiB host memcpys and a
    /// 90 MiB zeroed allocation, all serialized around a synchronize (~125–133 ms, prompt-length
    /// independent).
    ///
    /// MUST be called after BOTH sessions have been prepared — prepare zero-inits and H2Ds every
    /// segment, so aliasing earlier would let the owner's prepare wipe the now-shared region.
    pub fn alias_seg_from(&mut self, owner: &SuperDscSession, seg: i64) -> Result<()> {
        let s =
            SegIdx::checked(seg).ok_or_else(|| anyhow!("alias_seg: segment {seg} out of range"))?;
        self.exec.alias_seg_from(&owner.exec, s)
    }
}

/// THE SESSION IS THE THING THAT WALKS THE FOLD, so it is the thing a launch declares its pass geometry
/// to — and the [`DeclaredFold`](scratchy_subtile::sdsc_abstract::DeclaredFold) it hands back is what the
/// prefix-mask staging is blocked by. One call, both numbers, so the count the shim's `fold_plan::reps`
/// checks itself against and the count the mask is sized by are the same declaration.
impl FoldWalker for SuperDscSession {
    type Error = anyhow::Error;

    fn declare_fold(&mut self, pages: FoldPages, blocks: MaskBlocks) -> Result<()> {
        self.set_mask_blocks(blocks)?;
        self.set_fold_pages(pages)
    }
}
