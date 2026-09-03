// SPDX-License-Identifier: Apache-2.0
//! Cross-arch dispatcher: the cfg-free `ScratchyWeights` trait + the
//! `inventory`-driven `ScratchyArchRegistration` / `try_load` registry
//! seam every `#[forward] fn <arch>()` registers through.
//!
//! Every `#[forward]` macro invocation auto-emits an
//! `impl ScratchyWeights for Weights` + an `inventory::submit!`
//! registration, so `try_load` discovers every compiled arch
//! without a hand-written central list. Adding a new arch touches
//! only its source file plus a single `pub mod <arch>;` in the
//! calling crate's lib.rs (Rust module system requirement).
//!
//! The load / fingerprint / registry surface names only neutral types
//! (`LoadStream`, `WeightSource`) so it stays cfg-free of cuda/metal and a
//! third target (Spyre) registers through this SAME inventory registry.
//! Device-runtime forward methods (which name `GpuDevice` / `OwnedTensor`
//! / `ForwardCtx`) are internally `#[cfg(any(cuda, metal))]`-gated.

// Backend-neutral device tensor — the `forward*` trait methods return one.
// Lives in `scratchy-tensors` (the cfg-free core the compiler already deps),
// so naming it keeps the trait free of any backend crate.
#[cfg(any(feature = "cuda", feature = "metal"))]
use scratchy_tensors::OwnedTensor;

// Backend-neutral opaque handles for the ambient forward-args bundle
// (`ForwardCtxHandle`) and the live device (`ForwardDeviceHandle`). The
// concrete `ForwardCtx` names the backend-coupled KV/GDN pools and
// `GpuDevice` wraps cublas/streams/allocator, so neither can move into the
// cfg-free dispatcher. The trait names these thin pointer newtypes instead
// (the same neutralization `LoadStream` applies to the load stream); the
// per-arch `impl ScratchyWeights` in `targets/{cuda,metal}` recovers the
// concrete `&ForwardCtx` / `&mut GpuDevice` from them. Object-safe (no
// associated types), so `Box<dyn ScratchyWeights>` and its worker-side
// fat-pointer transmute are unchanged.
#[cfg(any(feature = "cuda", feature = "metal"))]
use scratchy_tensors::{ForwardCtxHandle, ForwardDeviceHandle};

/// One bound weight source: which tensor id it fills, the staged tensor, and
/// whether it is a GEMM operand.
///
/// ⛔ `is_gemm` IS CARRIED, NOT INFERRED. It decides the staged shape's
/// orientation — a GEMM weight sits on disk as `[out, in]` and the bundle wants
/// `[n, k]`; a norm is a flat gain staged `[1, hidden]`. The spyre worker used
/// to read this from a JSON manifest. It cannot be recovered from the tensor's
/// RANK either: a per-channel fp8 `weight_scale` is 2-D on disk (`[N, 1]`) and
/// is NOT a gemm operand, so a rank test would mis-orient every quantized
/// model. The emitter knows which external is the gemm's operand-1, so it says
/// so, once, at bake.
#[derive(Clone, Copy, Debug)]
pub struct BoundWeight {
    pub id: u32,
    pub tensor: scratchy_tensors::GpuTensor,
    pub is_gemm: bool,
}

impl BoundWeight {
    /// The tensor's own `[rows, cols]` — its ON-DISK shape, flattened for the
    /// two ranks that are not a plain matrix.
    ///
    /// ⛔ NOT THE BUNDLE'S `[k, n]` CONVENTION, AND NOT A STAGING DECISION. A
    /// GEMM weight sits on disk as `[out, in]` = `[n, k]` and that is what this
    /// reports. The spyre staging needs the LOGICAL `[k, n]` and reads it from
    /// `Wiring::tensor_shapes` — the generated form of the table it used to get
    /// from the JSON manifest. Deriving it here instead returned the same two
    /// numbers in the other order under the same names, which mis-padded
    /// granite's lm_head; see the note at that call site.
    pub fn staged_shape(&self) -> [usize; 2] {
        match self.tensor.ndim() {
            // A flat gain (norm weight) is one row.
            1 => [1, self.tensor.dim(0)],
            // ⛔ A PER-CHANNEL SCALE IS A COLUMN ON DISK AND A ROW HERE. fp8
            // `weight_scale` ships as `[N, 1]`; staging it verbatim would
            // present N rows of one element where the bundle expects one row of
            // N. Only non-GEMM operands take this arm, so a genuine `[n, 1]`
            // GEMM weight is untouched.
            2 if !self.is_gemm && self.tensor.dim(1) == 1 => [1, self.tensor.dim(0)],
            _ => [self.tensor.dim(0), self.tensor.dim(1)],
        }
    }
}

/// Arch-agnostic handle for a loaded model. Every
/// `#[forward] fn <arch>()` emits an `impl ScratchyWeights` for
/// its per-arch `Weights` type; callers hold
/// `Box<dyn ScratchyWeights>` and never need to know which arch
/// they got.
pub trait ScratchyWeights: Send + Sync {
    fn arch_name(&self) -> &'static str;
    fn num_hidden_layers(&self) -> u64;
    fn hidden_size(&self) -> u64;
    fn intermediate_size(&self) -> u64;
    fn num_attention_heads(&self) -> u64;
    fn num_key_value_heads(&self) -> u64;
    fn head_dim(&self) -> u64;
    fn vocab_size(&self) -> u64;

    /// The embedded KTIR bundle for the host Spyre run path, or `None` for
    /// arches/backends without one. Returned as a neutral
    /// `&'static dyn Any` because this cfg-free trait cannot name
    /// `scratchy_target_spyre::manifest::KtirBundle` (that would make the
    /// compiler depend on a target crate); the spyre worker downcasts it back.
    /// The proc-macro emits a per-arch override under `-Fspyre` returning
    /// `Some(&model::KTIR_BUNDLE)` (the macro-embedded const); the default is
    /// `None` (cuda/metal and arches without a compiled bundle).
    fn ktir_bundle(&self) -> Option<&'static (dyn core::any::Any + Send + Sync)> {
        None
    }

    /// This model's launch sources as `(tensor id, staged tensor)` — the
    /// GENERATED binding of a source id to a `Weights` FIELD.
    ///
    /// ⛔ THIS EXISTS SO NO ONE HAS TO NAME A WEIGHT BY STRING. The spyre
    /// worker previously rebuilt every weight's on-disk key at model load
    /// (`format!("{disk}.weight")`) and re-read it out of the weight store,
    /// because nothing connected a launch source to a struct field. That rule
    /// cannot express a bare `nn.Parameter` (gemma-4's `layer_scalar`, which
    /// has no `.weight`) or a non-default decoder root, and it re-answered a
    /// question the compiler had already settled.
    ///
    /// The proc-macro emits the override per model — it must, because only
    /// generated code can name a generated field. The default is `None`
    /// (cuda/metal, whose forwards read the fields directly).
    ///
    /// ⛔ THE TENSORS BORROW THE WEIGHT STORE. They are views over buffers the
    /// `GpuWeights` allocator owns, so the caller must keep it alive until the
    /// bytes have been staged.
    fn superdsc_weights(&self) -> Option<::anyhow::Result<Vec<BoundWeight>>> {
        None
    }

    /// The embedded **sendnn** bundle (the on-silicon path), peer of
    /// [`Self::ktir_bundle`].
    ///
    /// ⛔ ON THE TRAIT SO ONE MATCHED VARIANT ANSWERS EVERYTHING. It already
    /// exists as a registration fn-pointer, which the worker reached through a
    /// SECOND, independent registry walk (`resolve_sengraph_bundle`) after
    /// loading weights. Two walks means two variant matches, and an arch
    /// registers many base models — so the bundle could come from one variant
    /// while the weights and geometry came from another, with nothing checking
    /// they agreed. Reading it off the loaded model makes that unrepresentable.
    fn sengraph_bundle(&self) -> Option<&'static (dyn core::any::Any + Send + Sync)> {
        None
    }

    /// The per-model launch WIRING — source roles, tensor shapes, per-layer
    /// AttnDecode ids and baked geometry, one set per program.
    ///
    /// ⛔ THIS REPLACES PARSING `manifest.json` AT MODEL LOAD. The macro held
    /// these as typed values, flattened them to JSON strings
    /// (`{"id":7,"role":"prefix_k","layer":3}`), baked the string, and the
    /// worker `serde_json`-parsed it back and matched `role.as_str()` into the
    /// same distinction it started as — a compile-time enum round-tripped
    /// through runtime strings on every load, losing exhaustiveness on the way
    /// out. Neutral `&dyn Any` for the same reason `ktir_bundle` is: this
    /// cfg-free trait cannot name a target crate's type.
    fn superdsc_wiring(&self) -> Option<&'static (dyn core::any::Any + Send + Sync)> {
        None
    }

    /// The embedding table, for the host-side token gather.
    ///
    /// ⛔ IT CANNOT BE RE-READ BY NAME AFTER LOADING. The generated
    /// `Weights::load` TAKES the table out of the weight store, so a worker
    /// that later asks `gw` for `"model.embed_tokens.weight"` finds nothing —
    /// and the hardcoded spelling of that key is itself the gemma-4 bug (its
    /// decoder root is `model.language_model`). The loaded model hands over
    /// its own field instead.
    fn superdsc_embed(&self) -> Option<::anyhow::Result<scratchy_tensors::GpuTensor>> {
        None
    }

    /// Gated-DeltaNet (linear-attention) runtime config for hybrid arches
    /// (Qwen3.5 / Qwen3-Next). Drives the worker's `GdnStatePool` sizing +
    /// allocation. The proc-macro emits a per-arch override returning
    /// `Some(_)` for arches whose forward body contains a `gated_delta_net`
    /// op; the default (non-hybrid arches) returns `None`, so no GDN state
    /// pool is allocated and the `gdn_state` ForwardCtx field stays `None`.
    fn gdn_runtime_config(&self) -> Option<crate::gdn_state_layout::GdnRuntimeConfig> {
        None
    }

    /// Per-layer `kv_heads * head_dim` for hybrid-attention-geometry
    /// arches whose sliding and global classes differ in dims
    /// (Gemma4: sliding 8×256 = 2048 elems/token, global 1×512 =
    /// 512). Drives per-layer KV pool sizing. The proc-macro emits
    /// a per-arch override derived from the unrolled IR (which
    /// layers carry `sliding_attention` vs `attention` tiles) and
    /// the GLOBAL_* CanonicalParams; the default `None` keeps the
    /// uniform pool layout (every same-dims arch, incl. Gemma2/3).
    fn per_layer_kv_token_elems(&self) -> Option<Vec<usize>> {
        None
    }

    /// Spans / position-independent KV caching (`W::ROPE_ON_READ`). When
    /// true, this arch's paged attention + rope_append kernels mask
    /// block_table / slot_mapping **bit 31** and treat it as the
    /// "stored unrotated → rotate on read" flag. The metal worker reads
    /// this to decide whether it is SAFE to OR bit 31 into the block_table
    /// / slot_mapping it uploads (a non-rope-on-read arch's kernels read
    /// those raw, so bit 31 would corrupt the physical block id). Default
    /// `false`; the macro emits `true` for arches with `ROPE_ON_READ`.
    fn rope_on_read(&self) -> bool {
        false
    }

    /// Block-table row stride the metal kernels bake as a function
    /// constant (`CanonicalParams::MAX_BLOCKS_PER_SEQ`). The metal
    /// executor MUST pack host-side `block_table` rows at exactly
    /// this stride — every kernel reads row `seq_idx` at
    /// `block_table + seq_idx * stride`, so a host stride mismatch
    /// corrupts every `seq_idx > 0` (single-seq runs mask it).
    /// Default mirrors the `CanonicalParams` trait default (128);
    /// the proc-macro emits a per-arch override for arches that
    /// set the `max_blocks_per_seq` config key (Gemma4: 2048).
    fn max_blocks_per_seq(&self) -> usize {
        128
    }

    /// # Safety
    /// All tensors in `ctx` must be valid GPU memory; `device`
    /// must be the live CUDA device. Same invariants as each
    /// per-arch `forward`.
    ///
    /// Gated to backends with a device-runtime forward path
    /// (cuda/metal). A target whose forward runs through a
    /// different seam (e.g. Spyre's host-orchestrated bundle FIFO)
    /// still registers + fingerprint-loads through this trait;
    /// it just doesn't carry the device-runtime forward methods.
    #[cfg(any(feature = "cuda", feature = "metal"))]
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
    #[cfg(any(feature = "cuda", feature = "metal"))]
    unsafe fn forward_backbone(
        &self,
        ctx: ForwardCtxHandle<'_>,
        device: ForwardDeviceHandle<'_>,
        num_tokens: u64,
    ) -> OwnedTensor;

    /// Capture this arch's tape into a piecewise CUDA-graph runner.
    /// Used at tp>1 where NCCL inside a monolithic graph fails on
    /// L40S (verified). Runs eager NCCL between captured segments.
    ///
    /// Caller MUST have called
    /// `device.caching.begin_allocate_to_pool()` before invoking
    /// this fn so captured addresses come from a private pool that
    /// stays alive for the runner's lifetime.
    ///
    /// Default impl panics; the proc-macro emits a per-arch
    /// override that delegates to the canonical's
    /// `forward_piecewise_capture` free fn.
    ///
    /// The returned runner is the cuda-only `PiecewiseRunner`, which lives in
    /// `scratchy-target-cuda` (it wraps `CUgraphExec` + the captured private
    /// pool) and so cannot be named by this cfg-free trait. It is boxed as a
    /// neutral `Box<dyn Any + Send>`; the cuda worker downcasts it back to the
    /// concrete `scratchy_target_cuda::piecewise::PiecewiseRunner` (it depends
    /// on that crate directly). Same neutralization the device/ctx handles use.
    ///
    /// # Safety
    /// Same as [`Self::forward`], plus the private-pool contract
    /// above.
    #[cfg(feature = "cuda")]
    unsafe fn forward_piecewise_capture(
        &self,
        ctx: ForwardCtxHandle<'_>,
        device: ForwardDeviceHandle<'_>,
        num_tokens: u64,
    ) -> anyhow::Result<Box<dyn core::any::Any + Send>> {
        let _ = (ctx, device, num_tokens);
        unimplemented!(
            "forward_piecewise_capture: per-arch override not emitted \
             — rebuild scratchy-models with the latest macro"
        )
    }

    /// Per-worker arena peak in bytes (metal only).
    ///
    /// `MetalWorkerPool::for_buckets` derives the per-worker arena
    /// layout as the elementwise-max across [`METAL_BUCKETS`]'
    /// `arena_bytes` rows; the peak resident bytes per worker is
    /// the sum of that elementwise-max. The metal worker reads
    /// this to size `peak_activation_bytes` in
    /// `determine_available_memory`, replacing the 512 MiB
    /// placeholder from Step 3.B.
    ///
    /// Default returns 512 MiB so cuda-backed arches that never
    /// override this still surface a sane placeholder if the
    /// trait method is reached on a non-metal build path.
    #[cfg(feature = "metal")]
    fn metal_arena_peak_bytes(&self) -> u64 {
        512 * 1024 * 1024
    }

    /// `(bucket_m, total_arena_bytes)` for every compiled prefill bucket,
    /// ascending by `bucket_m`. The macro overrides this per-arch from the
    /// canonical's `METAL_BUCKET_ARENA_COSTS`; the load-time selector uses
    /// it to prune the global ladder to what the device can afford. Default
    /// empty → the selector falls back to keeping all compiled buckets.
    #[cfg(feature = "metal")]
    fn metal_bucket_arena_costs(&self) -> &'static [(u32, u64)] {
        &[]
    }

    /// Per-canonical metal dtype. The macro emits an override
    /// returning `<Self as CanonicalParams>::METAL_DTYPE` so the
    /// worker can route argmax / weight-loader / etc. between
    /// the f16 and bf16 paths without monomorphizing on `W`.
    /// Default `Bf16` matches the trait-level default and the
    /// modern HF checkpoint dtype.
    #[cfg(feature = "metal")]
    fn metal_dtype(&self) -> scratchy_tensors::MetalDtype {
        scratchy_tensors::MetalDtype::Bf16
    }

    /// Metal forward + caller-supplied follow-on hook chained on
    /// the forward CB's shared event. Used by the executor to
    /// fuse argmax onto the forward CB so there's no host-side
    /// sync between forward and sampling.
    ///
    /// Default impl falls back to plain `forward` and ignores the
    /// hook — non-metal builds and arches that haven't been
    /// re-emitted with the metal followup glue still build.
    ///
    /// # Safety
    /// Same as [`Self::forward`].
    #[cfg(feature = "metal")]
    unsafe fn forward_with_metal_followup(
        &self,
        ctx: ForwardCtxHandle<'_>,
        device: ForwardDeviceHandle<'_>,
        num_tokens: u64,
        _followup: Option<MetalForwardFollowup<'_>>,
    ) -> OwnedTensor {
        unsafe { self.forward(ctx, device, num_tokens) }
    }

    /// Phase 6 spec-decode K-step chain entry point. Opens ONE MTL4
    /// command buffer on the pool's MTL4 queue and invokes `body`
    /// with a [`ChainStepHandle`] (callable K times to encode a
    /// forward step onto the chain encoder), the worker's
    /// `RuntimeBindings` (for runtime buffer addresses), and the
    /// live compute encoder. Caller is responsible for encoding
    /// per-iter argmax + `chain_advance` dispatches between
    /// forwards. One commit, one host wait for the whole chain.
    ///
    /// Default impl returns `NotImplemented` so non-metal builds
    /// and arches that haven't been re-emitted with the Phase 6
    /// glue still build.
    ///
    /// # Safety
    /// Same as [`Self::forward`]; additionally, `body` must not
    /// retain any references to the encoder past its return.
    #[cfg(feature = "metal")]
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

/// Box for a Metal forward-encoder tail hook. Invoked on the same
/// MTL4 compute encoder used to encode the forward, AFTER the
/// forward dispatches and BEFORE `endEncoding`. The callee can
/// append additional dispatches (e.g. argmax sampling) so they
/// run inside the same command buffer with one commit and one
/// host wait. Receives:
///   - the MTL4 compute encoder to append dispatches onto;
///   - the logits MTLBuffer (the bucket's terminal arena slot);
///   - `total_n` (logits row count) and `vocab` (column count).
///
/// MTL4 only.
#[cfg(feature = "metal")]
pub type MetalForwardFollowup<'a> = Box<
    dyn FnOnce(
            &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
            &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTLBuffer>,
            u32,
            u32,
        ) -> Result<(), String>
        + 'a,
>;

/// Non-generic view of the macro-emitted MetalWorker for the
/// spec-decode K-step chain body. The body holds an
/// `&dyn ChainStepHandle` and calls `run_forward_step` K times to
/// encode each iter's bucket forward onto the chain encoder.
/// `logits_buf()` returns the bucket's terminal arena slot —
/// the same MTLBuffer is reused across iters, so binding it as
/// argmax's input every iter is correct.
#[cfg(feature = "metal")]
pub trait ChainStepHandle {
    /// Encode one bucket forward dispatch onto the chain encoder.
    /// `num_tokens` is real (the bucket is fixed across iters
    /// because the K-step decode shape doesn't change).
    fn run_forward_step(
        &self,
        encoder: &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
        num_tokens: u32,
        num_seqs: u32,
        has_spec_tokens: bool,
    ) -> Result<(), String>;

    /// The bucket's terminal arena slot (lm_head output). Stable
    /// across iters within one chain CB.
    fn logits_buf(&self) -> &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTLBuffer>;

    /// Logits column count (= compiled-in `METAL_VOCAB_SIZE`).
    fn vocab(&self) -> u32;
}

/// Box for the Phase 6 chain body. Invoked once per chain
/// invocation; drives K forward dispatches via the handle and
/// encodes argmax + chain_advance dispatches between forwards.
/// All dispatches share one MTL4 compute encoder ⇒ one CB ⇒ one
/// commit ⇒ one host wait.
#[cfg(feature = "metal")]
pub type MetalChainBody<'a> = Box<
    dyn FnOnce(
            &dyn ChainStepHandle,
            scratchy_tensors::MetalRuntimeHandle<'_>,
            &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
        ) -> Result<(), String>
        + 'a,
>;

/// Minimal HF-config view threaded into `try_load` so per-variant
/// `fingerprint_matches` can disambiguate checkpoints that share
/// on-disk tensor shapes but differ in config-only fields.
/// Phi-3-mini-4k (`max_position_embeddings=4096`, `rope_scaling=null`)
/// and Phi-3.5-mini-128k (`131072`, `{type:"longrope", …}`) have
/// identical weight shapes; without the config view the
/// alphabetically-earlier variant's fingerprint wins and its
/// (manifest-baked) RoPE cache gets used for the wrong model.
///
/// Kept as a struct of plain `Option<primitive>` so scratchy
/// stays decoupled from the caller's full HF-config parser —
/// add fields here only when a future arch truly needs them to
/// disambiguate.
#[derive(Clone, Copy, Debug, Default)]
pub struct HfFingerprint<'a> {
    pub rope_scaling_type: Option<&'a str>,
    /// Deterministic hash of the full `rope_scaling` JSON
    /// subobject (or `None` when the checkpoint has no
    /// rope_scaling). Discriminates checkpoints that share
    /// `rope_scaling_type` but differ in `short_factor` /
    /// `long_factor` / `original_max_position_embeddings`
    /// values — e.g. Phi-3.5-mini vs Phi-3-mini-128k,
    /// Phi-4-mini-instruct vs Phi-4-mini-reasoning. Computed
    /// with [`crate::hash_json_value`].
    ///
    /// `max_position_embeddings` is intentionally not on this
    /// fingerprint. It doesn't affect forward-fn codegen (MPE
    /// is a runtime KV-cache sizing input), and including it
    /// rejected checkpoints whose published `config.json` has
    /// a context window narrower than the upstream base
    /// (e.g. mlx-community 4bit Qwen2.5-1.5B publishes
    /// `max_position_embeddings: 32768` against the upstream
    /// `131072`). The rope-scaling-type and rope-scaling-hash
    /// disambiguation already catches every Phi-3-style
    /// long/short fork — `MPE` was redundant.
    pub rope_scaling_hash: Option<u64>,
}
