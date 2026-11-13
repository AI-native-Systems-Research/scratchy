// SPDX-License-Identifier: Apache-2.0
//! A loaded model, as metal runs it: [`MetalWeights`] extends the target-neutral
//! [`ScratchyWeights`] with the device-runtime forward surface, and [`MetalArchRegistration`] is
//! the row every metal-compiled arch registers its loaders under. A worker loads through
//! [`try_load`] and holds a `Box<dyn MetalWeights>`.
//!
//! The forward methods take the neutral [`ForwardCtxHandle`] / [`ForwardDeviceHandle`] pointer
//! newtypes; the per-arch impl recovers the concrete `&ForwardCtx` / `&mut GpuDevice` from them.
//! Object-safe (no associated types), so `Box<dyn MetalWeights>` is a plain trait object.

use scratchy_forward_compiler::{
    ArchLoad, ArchRegistration, HfFingerprint, KvCodec, ScratchyWeights, TargetRegistration,
};
use scratchy_tensors::{
    DeviceAllocator, ForwardCtxHandle, ForwardDeviceHandle, LoadStream, MetalDtype, OwnedTensor,
};

/// The objc2 types the emitted forward glue names, so an arch crate needs no direct
/// `objc2`/`objc2_metal` dependency.
pub use objc2::runtime::ProtocolObject;
pub use objc2_metal::{MTL4ComputeCommandEncoder, MTLBuffer};
/// The allocator-erased handle to the worker's per-forward `RuntimeBindings`, which a
/// [`MetalChainBody`] receives.
pub use scratchy_tensors::MetalRuntimeHandle;

/// A loaded model's metal surface. Every metal-compiled arch's `Weights` implements it beside
/// [`ScratchyWeights`].
pub trait MetalWeights: ScratchyWeights {
    /// The loaded variant's `CanonicalParams::KV_CODEC`: how its KV cache is stored, fixed when
    /// it was built.
    fn kv_codec(&self) -> KvCodec;

    /// The loaded variant's `CanonicalParams::KV_CACHE_LAYERS`: the layers that hold KV when not
    /// every layer does (a recurrent hybrid's attention layers). The worker's KV pool holds
    /// tensors for these only and sizes its budget by them. `None`: every layer holds KV.
    fn kv_cache_layers(&self) -> Option<&'static [u32]>;

    /// # Safety
    /// All tensors in `ctx` must be valid GPU memory; `device`
    /// must be the live device. Same invariants as each
    /// per-arch `forward`.
    unsafe fn forward(
        &self,
        ctx: ForwardCtxHandle<'_>,
        device: ForwardDeviceHandle<'_>,
        num_tokens: u64,
    ) -> OwnedTensor;

    /// Backbone-only forward (skips lm_head).
    ///
    /// # Safety
    /// Same as [`Self::forward`] — caller guarantees `ctx`
    /// tensors and `device` outlive the returned `OwnedTensor`
    /// and that kernel launches on `device.compute_stream` have
    /// completed before the output is read on another stream.
    unsafe fn forward_backbone(
        &self,
        ctx: ForwardCtxHandle<'_>,
        device: ForwardDeviceHandle<'_>,
        num_tokens: u64,
    ) -> OwnedTensor;

    /// The kernels this model runs on its logits outside its tape (argmax,
    /// grammar mask, sampler), baked at expansion with its logits width and
    /// dtype: an [`crate::off_tape::OffTapeKernels`], typed `Any`.
    fn metal_off_tape(&self) -> &'static (dyn core::any::Any + Send + Sync);

    /// The model's baked tape rungs, an [`crate::interpreter::metal::MetalRungs`] typed `Any`
    /// (as [`Self::metal_off_tape`]): the worker picks the rung its pool will run on from them
    /// before the pool exists — the KV pool's block-table width, the memory budget.
    fn metal_rungs(&self) -> &'static (dyn core::any::Any + Send + Sync);

    /// What the model's resident pool shares with its guest, a
    /// [`crate::interpreter::metal::HostArena`] typed `Any` (as [`Self::metal_rungs`]): the
    /// arena slots and scratch a model whose forwards never overlap this one's runs in, which the
    /// pool was sized to hold. `None` until this model's first forward builds the pool.
    fn metal_hosted(&self) -> Option<Box<dyn core::any::Any>>;

    /// Make this model's resident pool hold a guest's arena: `guest` a
    /// [`crate::interpreter::metal::GuestArena`] typed `Any` (as [`Self::metal_rungs`]), what a
    /// model whose forwards follow this one's on its command buffer (an MTP head) needs of it.
    /// Set once, before the pool is built at this model's first forward; refused once the pool
    /// exists — it was sized without the guest.
    fn metal_host(&self, guest: Box<dyn core::any::Any>) -> Result<(), String>;

    /// The model's `CanonicalParams::METAL_DTYPE`, so the worker can size
    /// its KV cache without monomorphizing on `W`.
    fn metal_dtype(&self) -> MetalDtype;

    /// Metal forward + caller-supplied follow-on hook chained on
    /// the forward CB's shared event. Used by the executor to
    /// fuse argmax onto the forward CB so there's no host-side
    /// sync between forward and sampling.
    ///
    /// Default impl falls back to plain `forward` and ignores the
    /// hook, for arches that haven't been re-emitted with the
    /// followup glue.
    ///
    /// `Err` only when the forward's command buffer failed on the GPU (an
    /// out-of-memory, say): the executor fails the step and goes on. A
    /// forward that breaks an invariant still panics.
    ///
    /// # Safety
    /// Same as [`Self::forward`].
    unsafe fn forward_with_metal_followup(
        &self,
        ctx: ForwardCtxHandle<'_>,
        device: ForwardDeviceHandle<'_>,
        num_tokens: u64,
        _followup: Option<MetalForwardFollowup<'_>>,
    ) -> Result<OwnedTensor, GpuExecutionFailed> {
        Ok(unsafe { self.forward(ctx, device, num_tokens) })
    }

    /// The rotary cos/sin rows a forward at `positions` (one a token) reads in place of a
    /// position-indexed cache — an MRoPE model's, its positions then each row's own index; `None`
    /// for a model whose rope reads positions.
    fn metal_rope_rows(&self, positions: &[u32]) -> Option<Vec<u8>>;

    /// Encode a forward onto `encoder` — another model's command buffer, after that model's forward,
    /// whose outputs the context's device inputs read — rather than into a command buffer of its
    /// own. The returned forward holds the worker the command buffer reads and writes until it
    /// drops, which must wait until that command buffer is done.
    ///
    /// # Safety
    /// Same as [`Self::forward`].
    unsafe fn metal_forward_onto<'w>(
        &'w self,
        ctx: ForwardCtxHandle<'_>,
        device: ForwardDeviceHandle<'_>,
        num_tokens: u64,
        encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    ) -> Result<Box<dyn MetalForwardOnto + 'w>, String>;

    /// Phase 6 spec-decode K-step chain entry point. Opens ONE MTL4
    /// command buffer on the pool's MTL4 queue and invokes `body`
    /// with a [`ChainStepHandle`] (callable K times to encode a
    /// forward step onto the chain encoder), the worker's
    /// `RuntimeBindings` (for runtime buffer addresses), and the
    /// live compute encoder. Caller is responsible for encoding
    /// per-iter argmax + `chain_advance` dispatches between
    /// forwards. One commit, one host wait for the whole chain.
    ///
    /// Default impl returns `NotImplemented` for arches that
    /// haven't been re-emitted with the Phase 6 glue.
    ///
    /// # Safety
    /// Same as [`Self::forward`]; additionally, `body` must not
    /// retain any references to the encoder past its return.
    unsafe fn metal_chain_with_encoder(
        &self,
        _ctx: ForwardCtxHandle<'_>,
        _device: ForwardDeviceHandle<'_>,
        _num_tokens: u64,
        _body: MetalChainBody<'_>,
    ) -> Result<(), String> {
        Err("metal_chain_with_encoder: not implemented for this arch".into())
    }
}

/// The row every metal-compiled arch registers its loaders under: the shared
/// [`ArchRegistration`], over [`MetalWeights`].
pub struct MetalArchRegistration(pub ArchRegistration<dyn MetalWeights>);

inventory::collect!(MetalArchRegistration);

impl TargetRegistration for MetalArchRegistration {
    type Weights = dyn MetalWeights;
    fn arch(&self) -> &ArchRegistration<dyn MetalWeights> {
        &self.0
    }
}

/// Load the compiled arch that claims `arch_hint` from `gw`, over the metal registry — the shared
/// [`scratchy_forward_compiler::try_load`] walk.
pub fn try_load<A: DeviceAllocator>(
    gw: &mut scratchy_layers::weights::GpuWeights<A>,
    stream: LoadStream,
    arch_hint: &str,
    tp_world_size: u8,
    tp_rank: u8,
    max_model_len: usize,
    hf: HfFingerprint<'_>,
) -> anyhow::Result<ArchLoad<dyn MetalWeights>> {
    scratchy_forward_compiler::try_load::<MetalArchRegistration, A>(
        gw,
        stream,
        arch_hint,
        tp_world_size,
        tp_rank,
        max_model_len,
        hf,
    )
}

/// A metal forward whose command buffer the GPU failed to run (an out-of-memory, say): the one
/// failure `forward_with_metal_followup` returns rather than panics on, so the executor can fail
/// the step and go on. Its text is the pool's error.
#[derive(Debug)]
pub struct GpuExecutionFailed(pub String);

impl std::fmt::Display for GpuExecutionFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuExecutionFailed {}

/// Box for a Metal forward-encoder tail hook. Invoked on the same
/// MTL4 compute encoder used to encode the forward, AFTER the
/// forward dispatches and BEFORE `endEncoding`. The callee can
/// append additional dispatches (e.g. argmax sampling, an MTP head's
/// forward) so they run inside the same command buffer with one
/// commit and one host wait. Receives the MTL4 compute encoder to append dispatches onto and
/// what the forward left ([`FollowupOutputs`]).
///
/// MTL4 only.
pub type MetalForwardFollowup<'a> = Box<
    dyn FnOnce(
            &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
            FollowupOutputs<'_>,
        ) -> Result<(), String>
        + 'a,
>;

/// What a forward leaves its [`MetalForwardFollowup`].
#[derive(Clone, Copy)]
pub struct FollowupOutputs<'a> {
    /// The bucket's terminal arena slot: the logits of the rows the step samples, at its front,
    /// in order.
    pub logits: &'a ProtocolObject<dyn MTLBuffer>,
    /// The final (post-norm) hidden states, `[num_tokens, hidden]` (the bucket's backbone slot).
    pub hidden: &'a ProtocolObject<dyn MTLBuffer>,
    /// The rows a DFlash2 head's target captured, `[num_tokens, slots, hidden]`; `None` for a
    /// model that captures none.
    pub captured: Option<&'a ProtocolObject<dyn MTLBuffer>>,
    /// The sampled rows: the logits' row count.
    pub rows: u32,
    /// The logits' row width.
    pub vocab: u32,
}

/// The forwards [`MetalWeights::metal_forward_onto`] and [`Self::then`] encoded onto another
/// model's command buffer, on one worker. Their outputs are read once that command buffer is done;
/// dropping it returns the worker.
pub trait MetalForwardOnto {
    /// Encode another forward on the same worker onto `encoder`, after what is on it; the outputs
    /// below are then its.
    ///
    /// # Safety
    /// Same as [`MetalWeights::forward`].
    unsafe fn then(
        &mut self,
        ctx: ForwardCtxHandle<'_>,
        num_tokens: u64,
        encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    ) -> Result<(), String>;

    /// The final (post-norm) hidden states, `[num_tokens, hidden]` (the bucket's backbone slot).
    fn hidden(&self) -> &ProtocolObject<dyn MTLBuffer>;

    /// The lm_head output (the bucket's terminal arena slot): the logits of the rows the forward
    /// sampled, at its front, `vocab` wide.
    fn logits(&self) -> &ProtocolObject<dyn MTLBuffer>;

    /// The logits' row width.
    fn vocab(&self) -> u32;

    /// What a DFlash2 drafter's last block draft selected; `None` for any other model.
    fn draft_selection(&self) -> Option<DraftSelection<'_>>;
}

/// A DFlash2 drafter's selection ([`crate::dflash2::DraftOutput`]): its proposals,
/// `[lanes, proposals]` u32; each position's candidates, `[lanes, proposals, candidates]` u32; and
/// a sampling lane's candidate probabilities, `[lanes, proposals, candidates]` f32; each proposal's
/// confidence, `[lanes, proposals]` f32; and the rows of its block each lane's next verify checks,
/// `[lanes]` u32.
#[derive(Clone, Copy)]
pub struct DraftSelection<'a> {
    pub tokens: &'a ProtocolObject<dyn MTLBuffer>,
    pub candidates: &'a ProtocolObject<dyn MTLBuffer>,
    pub q_probs: &'a ProtocolObject<dyn MTLBuffer>,
    pub confidence: &'a ProtocolObject<dyn MTLBuffer>,
    pub verify_rows: &'a ProtocolObject<dyn MTLBuffer>,
}

/// The rows one forward samples (its sampled rows' count): one a sequence, every row of a
/// sequence verifying drafts — the logits it leaves at the front of its output, what a metal
/// tape's runtime gates and sampled-rows grids read.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SampledRowCount(pub u32);

impl SampledRowCount {
    pub fn get(self) -> u32 {
        self.0
    }
}

/// Non-generic view of the emitted MetalWorker for the
/// spec-decode K-step chain body. The body holds an
/// `&dyn ChainStepHandle` and calls `run_forward_step` K times to
/// encode each iter's bucket forward onto the chain encoder.
/// `logits_buf()` returns the bucket's terminal arena slot —
/// the same MTLBuffer is reused across iters, so binding it as
/// argmax's input every iter is correct.
pub trait ChainStepHandle {
    /// Encode one bucket forward dispatch onto the chain encoder, at the tokens its bucket was
    /// chosen for (the bucket is fixed across iters because the K-step decode shape doesn't
    /// change); `sampled_rows` the rows whose logits it leaves, at the front.
    fn run_forward_step(
        &self,
        encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
        num_seqs: u32,
        sampled_rows: SampledRowCount,
    ) -> Result<(), String>;

    /// The bucket's terminal arena slot (lm_head output). Stable
    /// across iters within one chain CB.
    fn logits_buf(&self) -> &ProtocolObject<dyn MTLBuffer>;

    /// Logits column count (= compiled-in `METAL_VOCAB_SIZE`).
    fn vocab(&self) -> u32;
}

/// Box for the Phase 6 chain body. Invoked once per chain
/// invocation; drives K forward dispatches via the handle and
/// encodes argmax + chain_advance dispatches between forwards.
/// All dispatches share one MTL4 compute encoder ⇒ one CB ⇒ one
/// commit ⇒ one host wait.
pub type MetalChainBody<'a> = Box<
    dyn FnOnce(
            &dyn ChainStepHandle,
            MetalRuntimeHandle<'_>,
            &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
        ) -> Result<(), String>
        + 'a,
>;
