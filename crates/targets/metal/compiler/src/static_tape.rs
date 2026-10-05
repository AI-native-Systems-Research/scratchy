//! The macro-side metal tape bake.
//!
//! Runs `scratchy_target_metal::tape::lowering::lower_subtile_tape_to_metal`
//! at expansion, once per rung, and emits the results as `ClassedTape` statics via the generic
//! serializer in [`crate::const_tokens`]. Nothing here re-implements lowering logic —
//! zero per-op surface.
//!
//! Everything that varies is a rung, every fact of it baked; the pool only picks:
//! - **device generation** → one rung per [`GenClass`];
//! - **KV block-table capacity** → one rung per cap of the model's [`kv_cap_ladder`];
//! - **TurboQuant decode heads** (from the device's core count) → one rung per head count the
//!   geometry admits.
//!
//! Rungs that lower identically share one body.

use proc_macro2::TokenStream;
use quote::quote;
use scratchy_target_metal::tape::constants::ConstantValue;
use scratchy_target_metal::tape::ids::{
    AttnSplits, BlockSize, CommandIx, HeadDim, MaxBlocksPerSeq, MaxPositions, NumKvHeads,
    NumQHeads, TqDecodeHeads,
};
use scratchy_target_metal::tape::lowered::{
    Binding, DispatchShape, GatedCommand, GenClass, KernelId, KvAddressing, LoweredCommand,
    LoweredMetalTape, LoweringError, MetalDtype, TapeCommands,
};

/// ⭐ EVERY DISTINCT COMMAND OF ONE MODEL'S BAKED TAPES, SPELLED ONCE.
///
/// A bucket bakes a tape per rung, and the rungs of one bucket — and the buckets of one model —
/// share almost every command: the chunked rung differs in a single `ATTN_BLOCKS_PER_CHUNK`
/// constant on the attention readers, the M5 rung in its GEMMs, a cap rung in its KV readers.
/// Deduping whole bodies only helps when NOTHING differs, so one constant forked a full copy of
/// the tape and rustc's single-threaded front end paid for every token of it.
///
/// Each distinct command is one entry of the model's command table (`__tape_cmds::TABLE`),
/// serialized once; a body lists [`CommandIx`]es into it, so a command every rung shares is stored
/// once and each rung pays two bytes for it. A model with more distinct commands than a
/// [`CommandIx`] indexes refuses the bake.
///
/// So are the bulky parts of the commands: a model's thousands of commands are combinations of a
/// few hundred constant sets, a few hundred dispatch shapes and a few dozen binding lists, each
/// spelled once as its own `const` the commands name.
///
/// So is each distinct kernel of a baked library the commands name for any variant
/// ([`aot::baked_library`]): one entry of the model's kernel table (`__tape_cmds::KERNELS`), naming
/// its [`BakedKernel`](scratchy_target_metal::tape::lowered::BakedKernel) in the build's
/// `__metal_bake` module ([`kernel_ref`]).
#[derive(Default)]
pub struct CommandPool {
    index: std::collections::HashMap<GatedCommand, CommandIx>,
    table: Vec<TokenStream>,
    constants: Parts<&'static [ConstantValue]>,
    dispatches: Parts<DispatchShape>,
    bindings: Parts<&'static [Binding]>,
    named_kernels: std::collections::HashSet<PipelineKey>,
    kernels: Vec<PipelineKey>,
}

/// Each distinct value of one command field, as a `const` of its own.
struct Parts<T> {
    index: std::collections::HashMap<T, proc_macro2::Ident>,
    items: Vec<TokenStream>,
}

impl<T> Default for Parts<T> {
    fn default() -> Self {
        Self {
            index: Default::default(),
            items: Vec::new(),
        }
    }
}

impl<T: std::hash::Hash + Eq + Copy + serde::Serialize> Parts<T> {
    /// The `const` (`<prefix><n>: <ty>`) spelling `value`.
    fn name(&mut self, value: T, prefix: &str, ty: TokenStream) -> Result<TokenStream, BakeDefect> {
        if let Some(id) = self.index.get(&value) {
            return Ok(quote!(#id));
        }
        let id = quote::format_ident!("{prefix}{}", self.items.len());
        let toks = crate::const_tokens::const_tokens(&value)
            .map_err(|e| BakeDefect(format!("serialize {prefix}: {e}")))?;
        self.items
            .push(quote! { pub(super) const #id: #ty = #toks; });
        self.index.insert(value, id.clone());
        Ok(quote!(#id))
    }
}

impl CommandPool {
    fn intern(&mut self, cmd: &GatedCommand) -> Result<TokenStream, BakeDefect> {
        let ix = match self.index.get(cmd) {
            Some(&ix) => ix,
            None => {
                let ix = CommandIx::of(self.table.len()).ok_or_else(|| {
                    BakeDefect(format!(
                        "more than {} distinct commands: a command index cannot address them",
                        self.table.len()
                    ))
                })?;
                let GatedCommand { command, gate } = *cmd;
                let LoweredCommand {
                    kernel,
                    library,
                    function,
                    constants,
                    dispatch,
                    bindings,
                    gemm_dims,
                } = command;
                let constants =
                    (self.constants).name(constants, "CS", quote!(&[__tc::ConstantValue]))?;
                let dispatch = self
                    .dispatches
                    .name(dispatch, "DS", quote!(__tl::DispatchShape))?;
                let bindings = (self.bindings).name(bindings, "BS", quote!(&[__tl::Binding]))?;
                let tok = |r: Result<TokenStream, crate::const_tokens::Error>| {
                    r.map_err(|e| BakeDefect(format!("serialize command: {e}")))
                };
                let kernel = tok(crate::const_tokens::const_tokens(&kernel))?;
                let gemm_dims = tok(crate::const_tokens::const_tokens(&gemm_dims))?;
                let gate = tok(crate::const_tokens::const_tokens(&gate))?;
                let id = quote::format_ident!("C{}", ix.0);
                self.table.push(quote! {
                    pub(super) const #id: __tl::GatedCommand = __tl::GatedCommand {
                        command: __tl::LoweredCommand {
                            kernel: #kernel,
                            library: #library,
                            function: #function,
                            constants: #constants,
                            dispatch: #dispatch,
                            bindings: #bindings,
                            gemm_dims: #gemm_dims,
                        },
                        gate: #gate,
                    };
                });
                self.index.insert(*cmd, ix);
                ix
            }
        };
        let ix = proc_macro2::Literal::u16_unsuffixed(ix.0);
        Ok(quote! { I(#ix) })
    }

    /// Add the baked kernels `commands` name for tape `variant` at activation `dtype` to the
    /// model's kernels, each once.
    fn name_kernels(
        &mut self,
        commands: TapeCommands,
        dtype: MetalDtype,
        variant: TapeVariant,
    ) -> Result<(), BakeDefect> {
        for c in commands.iter() {
            let key = aot::bake_key(&c.command, dtype, variant)
                .map_err(|e| BakeDefect(format!("bake key: {e}")))?;
            let Some(key) = key else { continue };
            if self.named_kernels.insert(key.clone()) {
                self.kernels.push(key);
            }
        }
        Ok(())
    }

    /// The `__tape_cmds` module the model's tape statics reference, and `METAL_KERNELS`, the
    /// model's baked kernels its buckets name. Emit it once, beside them.
    pub fn into_tokens(self) -> TokenStream {
        // A model metal lowers no tape for (a dense MoE has no metal realization) names nothing.
        if self.table.is_empty() && self.kernels.is_empty() {
            return quote! {
                #[cfg(feature = "metal")]
                static METAL_KERNELS: &[::scratchy_target_metal::tape::lowered::BakedKernel] = &[];
            };
        }
        let aliases = crate::const_tokens::alias_preamble();
        let table = self.table;
        let len = proc_macro2::Literal::usize_unsuffixed(table.len());
        let ids = (0..table.len()).map(|i| quote::format_ident!("C{i}"));
        let kernels: Vec<TokenStream> = self.kernels.iter().map(kernel_ref).collect();
        let kernels_len = proc_macro2::Literal::usize_unsuffixed(kernels.len());
        let parts = (self.constants.items.iter())
            .chain(&self.dispatches.items)
            .chain(&self.bindings.items);
        quote! {
            #[cfg(feature = "metal")]
            mod __tape_cmds {
                #aliases
                #(#parts)*
                #(#table)*
                pub(super) static TABLE: [__tl::GatedCommand; #len] = [ #(#ids),* ];
                pub(super) static KERNELS: [__tl::BakedKernel; #kernels_len] = [ #(#kernels),* ];
            }
            #[cfg(feature = "metal")]
            static METAL_KERNELS: &[::scratchy_target_metal::tape::lowered::BakedKernel] =
                &__tape_cmds::KERNELS;
        }
    }
}

/// ⭐ EVERY BAKED KERNEL THE BUILD'S MODELS NAME, BY NAME. A model names each of its kernels by
/// its key as it is emitted ([`kernel_ref`]); [`bake_module`] then bakes them all at once — each
/// distinct key compiled once across the build, each metallib embedded once — as the crate-root
/// `__metal_bake` module the names resolve in.
static NAMED: std::sync::Mutex<std::collections::BTreeMap<String, PipelineKey>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// `key`'s baked kernel, as the path of its `static` in the crate-root `__metal_bake` module
/// ([`bake_module`]).
pub fn kernel_ref(key: &PipelineKey) -> TokenStream {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    key.hash(&mut hasher);
    let name = format!("K_{:016X}", hasher.finish());
    let mut named = NAMED.lock().unwrap();
    let prior = named.entry(name.clone()).or_insert_with(|| key.clone());
    assert!(prior == key, "metal bake: {prior:?} and {key:?} hash alike");
    let id = quote::format_ident!("{name}");
    quote!(crate::__metal_bake::#id)
}

/// The items of the crate-root `__metal_bake` module: every kernel [`kernel_ref`] named, baked
/// ([`aot::bake`]), as a `BakedKernel` static under its name, beside one `static` per metallib —
/// a bake batch's, which its kernels share.
pub fn bake_module() -> TokenStream {
    let named = std::mem::take(&mut *NAMED.lock().unwrap());
    let (names, keys): (Vec<String>, Vec<PipelineKey>) = named.into_iter().unzip();
    let bake = aot::bake(&keys);
    eprintln!(
        "[metal bake] {} kernels, {} compiles here",
        keys.len(),
        bake.compiled
    );
    let mut metallibs = std::collections::HashMap::new();
    let mut statics = Vec::new();
    for ((name, key), kernel) in names.iter().zip(&keys).zip(&bake.kernels) {
        let next = metallibs.len();
        let lib = *(metallibs.entry(std::sync::Arc::as_ptr(&kernel.metallib).cast::<u8>()))
            .or_insert_with(|| {
                let (lib, bytes) = (
                    quote::format_ident!("M{next}"),
                    metallib_file(&kernel.metallib),
                );
                let len = proc_macro2::Literal::usize_unsuffixed(kernel.metallib.len());
                // The bytes ARE the static (an array, not a `&[u8]` to an anonymous allocation):
                // the models' kernel tables copy these kernels, and each codegen unit that holds a
                // copy would otherwise embed its own copy of the anonymous bytes.
                statics.push(quote! { static #lib: [u8; #len] = *#bytes; });
                next
            });
        let (id, metallib) = (
            quote::format_ident!("{name}"),
            quote::format_ident!("M{lib}"),
        );
        let (library, function, entry) = (key.library_name, key.kernel_name, &kernel.entry);
        let constants = crate::const_tokens::const_tokens(&key.constants.as_slice())
            .expect("serialize baked kernel constants");
        statics.push(quote! {
            pub(crate) static #id: __tl::BakedKernel = __tl::BakedKernel {
                library: #library,
                function: #function,
                constants: #constants,
                metallib: &#metallib,
                entry: #entry,
            };
        });
    }
    if statics.is_empty() {
        return TokenStream::new();
    }
    quote! {
        use ::scratchy_target_metal::tape::constants as __tc;
        use ::scratchy_target_metal::tape::lowered as __tl;
        #(#statics)*
    }
}

/// `metallib` as a file under the build's `OUT_DIR`, named by its content (models sharing a batch
/// share the file), and the `include_bytes!` that embeds it: rustc reads the bytes rather than
/// lexing them as a literal.
fn metallib_file(metallib: &[u8]) -> TokenStream {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let out = std::env::var("OUT_DIR").expect("the metal bake writes its metallibs under OUT_DIR");
    let fnv = |h: u64, &b: &u8| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    let hash = metallib.iter().fold(0xcbf2_9ce4_8422_2325, fnv);
    let name = format!("metal-bake/{hash:016x}-{}.metallib", metallib.len());
    let path = std::path::Path::new(&out).join(&name);
    if !path.exists() {
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = path.with_extension(format!("{}.{seq}.tmp", std::process::id()));
        std::fs::create_dir_all(path.parent().expect("metal-bake dir"))
            .and_then(|()| std::fs::write(&tmp, metallib))
            .and_then(|()| std::fs::rename(&tmp, &path))
            .unwrap_or_else(|e| panic!("metal bake: write {}: {e}", path.display()));
    }
    quote! { include_bytes!(concat!(env!("OUT_DIR"), "/", #name)) }
}
use scratchy_target_metal::aot;
use scratchy_target_metal::specialized_pipeline_cache::PipelineKey;
use scratchy_target_metal::tape::constants::TapeVariant;
use scratchy_target_metal::tape::lowering as tl;
use scratchy_target_metal::tape::model_consts::MetalModelConsts;
use scratchy_target_metal::tape::step::{MetalStepTape, RotaryTables};
use scratchy_target_metal::tape::targets::MetalTargetProfile;

/// Why a bucket's tape could not bake — a defect; the caller panics with it.
pub struct BakeDefect(pub String);

impl std::fmt::Display for BakeDefect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// One bucket's lowering input: its step tape and the model's KV cap ladder.
pub struct BucketLowerInput<'a> {
    pub steps: &'a MetalStepTape,
    pub bucket_m: u32,
    pub num_arena_slots: u32,
    /// The model's class rotary tables (rope-on-read), from its source manifest.
    pub rotary: Option<RotaryTables>,
    /// The KV cap rungs the bucket bakes ([`kv_cap_ladder`]).
    pub cap_ladder: &'a [MaxBlocksPerSeq],
}

/// ⭐ THE KV CAP LADDER: the block-table capacities (blocks per sequence) a model's tapes bake,
/// ascending. The decode ladder's shape (`ActiveCap::decode_ladder`): power-of-two rungs — here
/// from one KV chunk ([`BLOCKS_PER_CHUNK`](scratchy_target_metal::BLOCKS_PER_CHUNK) blocks) — under
/// the ceiling, every block `max_positions` fills at `block_size`, which is the top rung. The pool
/// runs on the smallest rung that holds its capacity and refuses a capacity above the top. A model
/// with no KV cache (`None`: a vision tower, an encoder) has no step that reads the cap, so its one
/// rung is the floor.
pub fn kv_cap_ladder(
    max_positions: Option<MaxPositions>,
    block_size: BlockSize,
) -> Vec<MaxBlocksPerSeq> {
    let floor = scratchy_target_metal::BLOCKS_PER_CHUNK;
    let top = max_positions
        .map_or(0, |p| p.get().div_ceil(block_size.get().max(1)))
        .max(floor);
    let rungs = std::iter::successors(Some(floor), |r| r.checked_mul(2));
    let rungs = rungs.take_while(|&r| r < top).chain([top]);
    rungs.map(MaxBlocksPerSeq).collect()
}

/// A bucket's MTL4 barrier flags as a `static`.
pub fn emit_bucket_barriers_static(static_ident: &syn::Ident, barriers: &[bool]) -> TokenStream {
    quote! {
        #[cfg(feature = "metal")]
        static #static_ident: &[bool] = &[ #(#barriers),* ];
    }
}

fn profile_for(class: GenClass) -> MetalTargetProfile {
    use scratchy_target_metal::tape::targets::{M1_MAX, M4_10CORE, M5_10CORE};
    match class {
        GenClass::M1 => M1_MAX,
        GenClass::Mid => M4_10CORE,
        GenClass::M5 => M5_10CORE,
    }
}

/// `f` over `items` on every core, in order.
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = items.len().div_ceil(cores).max(1);
    std::thread::scope(|s| {
        let parts: Vec<_> = (items.chunks(chunk))
            .map(|part| s.spawn(|| part.iter().map(&f).collect::<Vec<R>>()))
            .collect();
        (parts.into_iter())
            .flat_map(|p| p.join().expect("a lowering thread panicked"))
            .collect()
    })
}

/// One rung of a bucket's tape: its class, addressing and KV cap.
struct Rung<'a> {
    profile: &'a MetalTargetProfile,
    addressing: KvAddressing,
    cap: MaxBlocksPerSeq,
}

fn run_lower(
    mc: &MetalModelConsts,
    input: &BucketLowerInput<'_>,
    rung: &Rung<'_>,
) -> Result<LoweredMetalTape, LoweringError> {
    let at = tl::BakePoint {
        chunked: rung.addressing == KvAddressing::Chunked,
        bucket_m: input.bucket_m,
        num_arena_slots: input.num_arena_slots,
        rotary: input.rotary,
        block_cap: rung.cap.get(),
        profile: Some(rung.profile),
    };
    tl::lower_subtile_tape_to_metal(input.steps, mc, at)
}

/// Bake every `(gen class × KV addressing × KV cap × TurboQuant decode heads)` variant of one
/// bucket's tape and emit the `&'static [ClassedTape]` expression. A tape with no TurboQuant decode
/// attention serves every device (`tq_heads: None`); one with it has a variant per head count its
/// geometry admits ([`TqDecodeHeads::candidates`]), which the device picks among. A cap rung whose
/// scratch cannot exist for this bucket ([`LoweringError::ScratchTooLarge`]) ends the bucket's
/// ladder: it and every rung above it are not baked.
///
/// A variant's cap and heads reach its commands only as variant-bound constants
/// ([`ConstantType::KvCap`](scratchy_target_metal::tape::constants::ConstantType::KvCap)), so the
/// variants of a `(class, addressing)` share one hoisted body and each carries only its cap-sized
/// scratch ([`ClassedTape::rung`]); the kernels its values bake join the model's kernel table.
///
/// [`ClassedTape::rung`]: scratchy_target_metal::tape::lowered::ClassedTape::rung
pub fn bake_bucket_tapes(
    mc: &MetalModelConsts,
    input: &BucketLowerInput<'_>,
    pool: &mut CommandPool,
) -> Result<TokenStream, BakeDefect> {
    let classes = [GenClass::M1, GenClass::Mid, GenClass::M5];
    let candidates: Vec<TqDecodeHeads> = TqDecodeHeads::candidates(
        HeadDim(mc.global_head_dim),
        NumQHeads(mc.num_q_heads),
        NumKvHeads(mc.num_global_kv_heads),
    )
    .collect();
    let mut entries: Vec<TokenStream> = Vec::new();
    let mut body_statics: Vec<TokenStream> = Vec::new();
    // Dedupe: identical bodies share ONE hoisted static — every labelled variant would otherwise
    // repeat the full command tape and rustc drowns in tokens (a 10-family build OOM-killed the
    // compiler before this). The cap-sized scratch is the variant's, outside the body.
    //
    // Keyed on the LOWERED VALUES, not on their rendered token text: the tape is megabytes of
    // tokens, so hashing the values beats stringifying every body.
    let mut seen: std::collections::HashMap<LoweredMetalTape, usize> = Default::default();
    let uniq = format!("M{}", input.bucket_m);
    let tok = |what: &str, r: Result<TokenStream, crate::const_tokens::Error>| {
        r.map_err(|e| BakeDefect(format!("serialize {what}: {e}")))
    };
    let defect = |e: LoweringError| BakeDefect(format!("bucket_m={}: {e}", input.bucket_m));
    // Every rung's tapes, lowered in parallel (each lowering is independent); `None` = the rung
    // cannot exist for this bucket.
    let points: Vec<(GenClass, KvAddressing, MaxBlocksPerSeq)> = (classes.iter())
        .flat_map(|&c| [KvAddressing::Direct, KvAddressing::Chunked].map(|a| (c, a)))
        .flat_map(|(c, a)| input.cap_ladder.iter().map(move |&cap| (c, a, cap)))
        .collect();
    let lowered = par_map(&points, |&(class, addressing, cap)| {
        let profile = profile_for(class);
        let rung = Rung {
            profile: &profile,
            addressing,
            cap,
        };
        match run_lower(mc, input, &rung) {
            Err(LoweringError::ScratchTooLarge { .. }) => Ok(None),
            tape => tape.map(Some),
        }
    });
    // The scratch grows with the cap: the first rung of a `(class, addressing)` ladder that cannot
    // exist ends it (`points` holds each ladder contiguous, ascending).
    let mut ended = None;
    for (&(class, addressing, cap), tapes) in points.iter().zip(lowered) {
        if ended == Some((class, addressing)) {
            continue;
        }
        let Some(tape) = tapes.map_err(defect)? else {
            eprintln!(
                "[metal bake] bucket_m={}: ladder ends below KV cap rung {}",
                input.bucket_m,
                cap.get()
            );
            ended = Some((class, addressing));
            continue;
        };
        // The cap-sized scratch is the variant's; the rest of the tape is the body.
        let (roped_k, attn_unfused) = (tape.roped_k_scratch_bytes, tape.attn_unfused_scratch_bytes);
        let body = LoweredMetalTape {
            roped_k_scratch_bytes: 0,
            attn_unfused_scratch_bytes: 0,
            ..tape
        };
        let body_ix = match seen.get(&body).copied() {
            Some(ix) => ix,
            None => {
                let cmd_refs = (body.commands.iter())
                    .map(|c| pool.intern(c))
                    .collect::<Result<Vec<_>, _>>()?;
                let rest = LoweredMetalTape {
                    commands: TapeCommands::EMPTY,
                    ..body
                };
                let rest_toks = tok("tape", crate::const_tokens::const_tokens(&rest))?;
                let ix = body_statics.len();
                let ident = quote::format_ident!("__TAPE_BODY_{uniq}_{ix}");
                body_statics.push(quote! {
                    // A `static`: a `const` holding `&TABLE` would have rustc validate the whole
                    // table once per body.
                    static #ident: __tl::LoweredMetalTape = __tl::LoweredMetalTape {
                        commands: __tl::TapeCommands {
                            table: &__tape_cmds::TABLE,
                            ixs: { use __ti::CommandIx as I; &[ #(#cmd_refs),* ] },
                        },
                        ..#rest_toks
                    };
                });
                seen.insert(body, ix);
                ix
            }
        };
        let ident = quote::format_ident!("__TAPE_BODY_{uniq}_{body_ix}");
        let has = |k: KernelId| tape.commands.iter().any(|c| c.command.kernel == k);
        let heads: Vec<Option<TqDecodeHeads>> = match has(KernelId::AttentionViaCacheTq) {
            true => candidates.iter().copied().map(Some).collect(),
            false => vec![None],
        };
        // A decode attention spreads over the splits the device picks among.
        let splits: Vec<Option<AttnSplits>> = match has(KernelId::AttentionDecodeCombine) {
            true => AttnSplits::CANDIDATES.map(Some).to_vec(),
            false => vec![None],
        };
        for (tq_heads, attn_splits) in
            (heads.iter()).flat_map(|&h| splits.iter().map(move |&s| (h, s)))
        {
            let one = AttnSplits(1);
            let variant = TapeVariant {
                cap,
                tq_heads,
                attn_splits: attn_splits.unwrap_or(one),
            };
            pool.name_kernels(tape.commands, mc.metal_dtype, variant)?;
            let class_toks = tok("class", crate::const_tokens::const_tokens(&class))?;
            let addressing_toks =
                tok("addressing", crate::const_tokens::const_tokens(&addressing))?;
            let cap_toks = tok("cap", crate::const_tokens::const_tokens(&cap))?;
            let tq_toks = tok("tq heads", crate::const_tokens::const_tokens(&tq_heads))?;
            let splits_toks = tok("splits", crate::const_tokens::const_tokens(&attn_splits))?;
            entries.push(quote! {
                __tl::ClassedTape::rung(
                    #ident, #class_toks, #addressing_toks, #cap_toks, #tq_toks, #splits_toks,
                    [#roped_k, #attn_unfused],
                ),
            });
        }
    }
    // ⭐ THE ALIASES THE WHOLE BAKED TAPE RESOLVES THROUGH, opened once for this block. Every
    // literal below names its type through `__tl`/`__tc`/`__ti` instead of spelling
    // `::scratchy_target_metal::tape::<module>::` on the order of a million times — see
    // `const_tokens::alias_preamble`.
    let aliases = crate::const_tokens::alias_preamble();
    Ok(quote! {
        {
            #aliases
            #(#body_statics)*
            &[ #(#entries)* ]
        }
    })
}
