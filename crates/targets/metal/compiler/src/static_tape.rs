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
use scratchy_target_metal::tape::ids::{
    BlockSize, HeadDim, MaxBlocksPerSeq, MaxPositions, NumKvHeads, NumQHeads, SourceIx,
    TqDecodeHeads,
};
use scratchy_target_metal::tape::lowered::{
    CommandTableBuilder, CommandTableParts, GatedCommand, GenClass, KernelId, KvAddressing,
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
/// Each distinct command is one entry of the model's command table (`__tape_cmds::TABLE`, a
/// [`CommandTable`](scratchy_target_metal::tape::lowered::CommandTable)), interned by the same
/// builder a tape lowered at run time is ([`CommandTableBuilder`]); a body lists
/// [`CommandIx`](scratchy_target_metal::tape::ids::CommandIx)es
/// into it, so a command every rung shares is stored once and each rung pays two bytes for it. A
/// model with more distinct commands than a command index addresses refuses the bake.
///
/// So are the bulky parts of the commands: a model's thousands of commands are combinations of a
/// few hundred constant sets, a few hundred dispatch shapes, a few dozen binding lists and kernel
/// symbols, each held once in the table's part arrays and named by a two-byte index — so a
/// command is a few bytes, not the pointers and inline shapes it reads as, and the binary carries
/// no load-time fixup for it.
///
/// So is each distinct kernel of a baked library the commands name for any variant
/// ([`aot::baked_library`]): one entry of the model's kernel table (`__tape_cmds::KERNELS`), naming
/// its [`BakedKernel`](scratchy_target_metal::tape::lowered::BakedKernel) in the build's
/// `__metal_bake` module ([`kernel_ref`]).
#[derive(Default)]
pub struct CommandPool {
    table: CommandTableBuilder,
    named_kernels: std::collections::HashSet<PipelineKey>,
    kernels: Vec<PipelineKey>,
}

impl CommandPool {
    fn intern(&mut self, cmd: GatedCommand) -> Result<TokenStream, BakeDefect> {
        let ix = self.table.intern(cmd).map_err(|e| {
            BakeDefect(format!(
                "more than {} distinct commands: a command index cannot address them",
                e.distinct - 1
            ))
        })?;
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
            let keys = aot::bake_keys(&c.command, dtype, variant)
                .map_err(|e| BakeDefect(format!("bake key: {e}")))?;
            for key in keys {
                if self.named_kernels.insert(key.clone()) {
                    self.kernels.push(key);
                }
            }
        }
        Ok(())
    }

    /// The `__tape_cmds` module the model's tape statics reference, and `METAL_KERNELS`, the
    /// model's baked kernels its buckets name. Emit it once, beside them. Each dense GEMM over a
    /// weight family of `typed` (the families the model's resolver declared a source slot for)
    /// takes its dims from its slot's type, and each binding of a typed family's weight its
    /// index ([`TypedGemms`]).
    pub fn into_tokens(
        self,
        typed: &std::collections::HashSet<SourceIx>,
    ) -> Result<TokenStream, BakeDefect> {
        // A model metal lowers no tape for (a dense MoE has no metal realization) names nothing.
        if self.table.is_empty() && self.kernels.is_empty() {
            return Ok(quote! {
                #[cfg(feature = "metal")]
                static METAL_KERNELS: &[::scratchy_target_metal::tape::lowered::BakedKernel] = &[];
            });
        }
        let parts = self.table.finish();
        let gemms = TypedGemms::of(&parts, typed);
        let aliases = crate::const_tokens::alias_preamble();
        let array = |name: &str, values: Vec<TokenStream>, ty: TokenStream| {
            let (id, len) = (
                quote::format_ident!("{name}"),
                proc_macro2::Literal::usize_unsuffixed(values.len()),
            );
            quote! { static #id: [#ty; #len] = [ #(#values),* ]; }
        };
        fn each<T: serde::Serialize>(values: &[T]) -> Result<Vec<TokenStream>, BakeDefect> {
            (values.iter())
                .map(|v| {
                    crate::const_tokens::const_tokens(v)
                        .map_err(|e| BakeDefect(format!("serialize command table: {e}")))
                })
                .collect()
        }
        let symbols = (parts.symbols.iter())
            .map(|(library, function)| quote! { (#library, #function) })
            .collect();
        // Each command as a call of `command` below: six part indices and its kernel's name, where
        // its literal spells every newtype of seven fields.
        let commands = (parts.commands.iter())
            .map(|p| {
                let kernel = quote::format_ident!("{}", format!("{:?}", p.kernel));
                let [symbol, constants, dispatch, bindings, gemm, gate] = [
                    p.symbol,
                    p.constants,
                    p.dispatch,
                    p.bindings,
                    p.gemm,
                    p.gate,
                ]
                .map(|i| proc_macro2::Literal::u16_unsuffixed(i.0));
                quote!(command(K::#kernel, #symbol, #constants, #dispatch, #bindings, #gemm, #gate))
            })
            .collect();
        let bindings = (parts.bindings.iter())
            .map(|list| gemms.binding_list(list))
            .collect::<Result<Vec<_>, BakeDefect>>()?;
        let gemm_entries = (parts.gemm_entries.iter())
            .map(|g| match gemms.record(*g) {
                Some(record) => Ok(record),
                None => crate::const_tokens::const_tokens(g)
                    .map_err(|e| BakeDefect(format!("serialize command table: {e}"))),
            })
            .collect::<Result<Vec<_>, BakeDefect>>()?;
        let gemm_ats = (parts.gemms.iter())
            .map(|g| match g {
                Some(at) => {
                    let (entry, layer) = (at.entry.0, at.layer.0);
                    quote!(gemm_at(#entry, #layer))
                }
                None => quote!(None),
            })
            .collect();
        let markers = gemms.markers();
        let statics = [
            array("COMMANDS", commands, quote!(__tl::PackedCommand)),
            array("SYMBOLS", symbols, quote!((&str, &str))),
            array(
                "CONSTANTS",
                each(&parts.constants)?,
                quote!(&[__tc::ConstantValue]),
            ),
            array(
                "DISPATCHES",
                each(&parts.dispatches)?,
                quote!(__tl::DispatchShape),
            ),
            array("BINDINGS", bindings, quote!(&[__tl::Binding])),
            array("GEMMS", gemm_ats, quote!(Option<__tl::GemmAt>)),
            array("GEMM_ENTRIES", gemm_entries, quote!(__tl::GemmEntry)),
            array(
                "GATES",
                each(&parts.gates)?,
                quote!(Option<__tl::RuntimeGate>),
            ),
        ];
        let kernels: Vec<TokenStream> = self.kernels.iter().map(kernel_ref).collect();
        let kernels_len = proc_macro2::Literal::usize_unsuffixed(kernels.len());
        Ok(quote! {
            #[cfg(feature = "metal")]
            mod __tape_cmds {
                #aliases
                #markers
                /// The table's GEMM of entry `entry` at layer `layer`: a GEMM of every layer is
                /// one, so each is written as this call, not a literal of its two newtypes.
                const fn gemm_at(entry: u16, layer: u16) -> Option<__tl::GemmAt> {
                    Some(__tl::GemmAt {
                        entry: __ti::PartIx(entry),
                        layer: __tl::PackedLayer(layer),
                    })
                }
                use __tl::KernelId as K;
                /// The table's command of kernel `kernel` and these part indices: a command is
                /// written as this call, not a literal of its seven fields' newtypes.
                const fn command(
                    kernel: K,
                    symbol: u16,
                    constants: u16,
                    dispatch: u16,
                    bindings: u16,
                    gemm: u16,
                    gate: u16,
                ) -> __tl::PackedCommand {
                    __tl::PackedCommand {
                        kernel,
                        symbol: __ti::PartIx(symbol),
                        constants: __ti::PartIx(constants),
                        dispatch: __ti::PartIx(dispatch),
                        bindings: __ti::PartIx(bindings),
                        gemm: __ti::PartIx(gemm),
                        gate: __ti::PartIx(gate),
                    }
                }
                #(#statics)*
                pub(super) static TABLE: __tl::CommandTable = __tl::CommandTable {
                    commands: &COMMANDS,
                    symbols: &SYMBOLS,
                    constants: &CONSTANTS,
                    dispatches: &DISPATCHES,
                    bindings: &BINDINGS,
                    gemms: &GEMMS,
                    gemm_entries: &GEMM_ENTRIES,
                    gates: &GATES,
                };
                pub(super) static KERNELS: [__tl::BakedKernel; #kernels_len] = [ #(#kernels),* ];
            }
            #[cfg(feature = "metal")]
            static METAL_KERNELS: &[::scratchy_target_metal::tape::lowered::BakedKernel] =
                &__tape_cmds::KERNELS;
        })
    }
}

/// ⭐ A MODEL'S DENSE GEMMS, TYPED BY THEIR WEIGHTS.
///
/// A dense GEMM's record (`Gemm`: its dims and the weight it reads, which nothing else of the
/// command names) over a typed family's weight takes that weight's index and dims from ONE const
/// of its slot, `__GEMM<slot>` (`GemmSlot::WEIGHT`, read off the slot's handle) — its rows and
/// layer are its own. And each binding of a typed family's tensor names its slot's index
/// (`__SOURCE<slot>`). Every other record and binding keeps its value; the commands resolve to
/// the same values either way.
///
/// Each typed value is one `const` per slot: the weight's index and dims depend on the slot
/// alone, so a record that named a type of its own would only re-check it.
struct TypedGemms {
    /// The typed families.
    typed: std::collections::HashSet<SourceIx>,
    /// The typed families the GEMM records read, ascending.
    read: std::collections::BTreeSet<u32>,
    /// The typed families the binding lists bind, ascending.
    bound: std::collections::BTreeSet<u32>,
}

impl TypedGemms {
    /// The typed GEMM records and bindings of `parts`, over the families of `typed`.
    fn of(parts: &CommandTableParts, typed: &std::collections::HashSet<SourceIx>) -> Self {
        use scratchy_target_metal::tape::lowered::Binding;
        let read = (parts.gemm_entries.iter())
            .filter(|g| Self::typed_weight(typed, g))
            .map(|g| g.ix.0)
            .collect();
        let bound = (parts.bindings.iter().copied().flatten())
            .filter_map(|b| match b {
                Binding::Source { ix, .. } if typed.contains(ix) => Some(ix.0),
                _ => None,
            })
            .collect();
        Self {
            typed: typed.clone(),
            read,
            bound,
        }
    }

    /// Whether `g` reads a typed family's weight (a router's gate is a bundle tensor, untyped).
    fn typed_weight(
        typed: &std::collections::HashSet<SourceIx>,
        g: &scratchy_target_metal::tape::lowered::GemmEntry,
    ) -> bool {
        use scratchy_target_metal::tape::lowered::WeightTensor;
        typed.contains(&g.ix) && g.which == WeightTensor::Weight
    }

    fn gemm(slot: SourceIx) -> proc_macro2::Ident {
        quote::format_ident!("__GEMM{}", slot.0)
    }

    fn source(slot: SourceIx) -> proc_macro2::Ident {
        quote::format_ident!("__SOURCE{}", slot.0)
    }

    /// GEMM entry `g`'s tokens, for one over a typed weight: its slot's index and dims, and its
    /// rows (its layer is each command's).
    fn record(&self, g: scratchy_target_metal::tape::lowered::GemmEntry) -> Option<TokenStream> {
        if !Self::typed_weight(&self.typed, &g) {
            return None;
        }
        let (w, m) = (Self::gemm(g.ix), g.dims.m);
        Some(quote!(__tl::GemmEntry {
            dims: __tl::GemmDims { m: #m, n: #w.n, k: #w.k },
            ix: #w.ix,
            which: __tl::WeightTensor::Weight,
        }))
    }

    /// `list`, each weight of a typed family named through its slot's index.
    fn binding_list(
        &self,
        list: &[scratchy_target_metal::tape::lowered::Binding],
    ) -> Result<TokenStream, BakeDefect> {
        use scratchy_target_metal::tape::lowered::Binding;
        fn tokens<T: serde::Serialize>(v: &T) -> Result<TokenStream, BakeDefect> {
            crate::const_tokens::const_tokens(v)
                .map_err(|e| BakeDefect(format!("serialize command table: {e}")))
        }
        let items = (list.iter())
            .map(|b| match b {
                Binding::Source {
                    ix,
                    which,
                    layer,
                    binding_index,
                } if self.typed.contains(ix) => {
                    let s = Self::source(*ix);
                    let (which, layer) = (tokens(which)?, tokens(layer)?);
                    let bi = proc_macro2::Literal::u8_suffixed(*binding_index);
                    Ok(quote!(__tl::Binding::Source {
                        ix: #s,
                        which: #which,
                        layer: #layer,
                        binding_index: #bi,
                    }))
                }
                other => tokens(other),
            })
            .collect::<Result<Vec<_>, BakeDefect>>()?;
        Ok(quote!(&[#(#items),*]))
    }

    /// The per-slot consts the table names: each read slot's weight (`GemmSlot::WEIGHT`: its
    /// index and dims, read off its handle), and each bound slot's index (`Source::IX`).
    fn markers(&self) -> TokenStream {
        let tt = quote!(::scratchy_target_metal::tape::typed);
        let weights = self.read.iter().map(|&ix| {
            let ix = SourceIx(ix);
            let (c, s) = (Self::gemm(ix), crate::canonical::slot(ix));
            quote!(const #c: #tt::GemmSlotWeight = <super::#s as #tt::GemmSlot>::WEIGHT;)
        });
        let sources = self.bound.iter().map(|&ix| {
            let ix = SourceIx(ix);
            let (c, s) = (Self::source(ix), crate::canonical::slot(ix));
            quote!(const #c: __ti::SourceIx = <super::#s as #tt::Source>::IX;)
        });
        quote!(#(#weights)* #(#sources)*)
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
    let mut distinct: Vec<&[u8]> = Vec::new();
    let lib_of: Vec<usize> = (bake.kernels.iter())
        .map(|kernel| {
            *(metallibs.entry(std::sync::Arc::as_ptr(&kernel.metallib).cast::<u8>()))
                .or_insert_with(|| {
                    distinct.push(&kernel.metallib);
                    distinct.len() - 1
                })
        })
        .collect();
    if distinct.is_empty() {
        return TokenStream::new();
    }
    let packed = aot::pack(&distinct);
    eprintln!(
        "[metal bake] {} metallibs, {} bytes packed to {} (dictionary {})",
        distinct.len(),
        distinct.iter().map(|m| m.len()).sum::<usize>(),
        packed.frames.iter().map(Vec::len).sum::<usize>() + packed.dictionary.len(),
        packed.dictionary.len(),
    );
    // The bytes ARE the statics (arrays, not `&[u8]`s to anonymous allocations): the models'
    // kernel tables copy these kernels, and each codegen unit that holds a copy would otherwise
    // embed its own copy of the anonymous bytes.
    let (dictionary, dictionary_len) = (
        metal_bake_file(&packed.dictionary, "dict"),
        proc_macro2::Literal::usize_unsuffixed(packed.dictionary.len()),
    );
    let mut statics = vec![quote! {
        static DICTIONARY_BYTES: [u8; #dictionary_len] = *#dictionary;
        static DICTIONARY: __tl::BakedDictionary = __tl::BakedDictionary::new(&DICTIONARY_BYTES);
    }];
    for (i, (metallib, frame)) in distinct.iter().zip(&packed.frames).enumerate() {
        let (lib, frame_id) = (quote::format_ident!("M{i}"), quote::format_ident!("F{i}"));
        let bytes = metal_bake_file(frame, "metallib.zst");
        let frame_len = proc_macro2::Literal::usize_unsuffixed(frame.len());
        let len = proc_macro2::Literal::usize_unsuffixed(metallib.len());
        statics.push(quote! {
            static #frame_id: [u8; #frame_len] = *#bytes;
            static #lib: __tl::BakedMetallib = __tl::BakedMetallib {
                frame: &#frame_id,
                dictionary: &DICTIONARY,
                len: #len,
            };
        });
    }
    for (((name, key), kernel), lib) in names.iter().zip(&keys).zip(&bake.kernels).zip(lib_of) {
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
    quote! {
        use ::scratchy_target_metal::tape::constants as __tc;
        use ::scratchy_target_metal::tape::lowered as __tl;
        #(#statics)*
    }
}

/// `bytes` (a packed metallib's frame or the dictionary) as a file under the build's `OUT_DIR`,
/// named by its content and `extension`, and the `include_bytes!` that embeds it: rustc reads the
/// bytes rather than lexing them as a literal.
fn metal_bake_file(bytes: &[u8], extension: &str) -> TokenStream {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let out = std::env::var("OUT_DIR").expect("the metal bake writes its metallibs under OUT_DIR");
    let fnv = |h: u64, &b: &u8| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325, fnv);
    let name = format!("metal-bake/{hash:016x}-{}.{extension}", bytes.len());
    let path = std::path::Path::new(&out).join(&name);
    if !path.exists() {
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = path.with_extension(format!("{}.{seq}.tmp", std::process::id()));
        std::fs::create_dir_all(path.parent().expect("metal-bake dir"))
            .and_then(|()| std::fs::write(&tmp, bytes))
            .and_then(|()| std::fs::rename(&tmp, &path))
            .unwrap_or_else(|e| panic!("metal bake: write {}: {e}", path.display()));
    }
    quote! { include_bytes!(concat!(env!("OUT_DIR"), "/", #name)) }
}
use scratchy_ir::model_facts::ModelFacts;
use scratchy_target_metal::aot;
use scratchy_target_metal::specialized_pipeline_cache::PipelineKey;
use scratchy_target_metal::tape::constants::TapeVariant;
use scratchy_target_metal::tape::lowering as tl;
use scratchy_target_metal::tape::step::{MetalStepTape, RotaryTables};
use scratchy_target_metal::tape::targets::MetalTargetProfile;

/// Why a bucket's tape could not bake — a defect; the caller panics with it.
pub struct BakeDefect(pub String);

impl std::fmt::Display for BakeDefect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The wide band's share of one bucket's lowering input: its step tape and the arena slot count
/// that tape's colouring allocated — the bands colour differently, so each carries its own.
pub struct WideBand<'a> {
    pub steps: &'a MetalStepTape,
    pub num_arena_slots: u32,
}

/// One bucket's lowering input: its step tape and the model's KV cap ladder.
pub struct BucketLowerInput<'a> {
    pub steps: &'a MetalStepTape,
    /// The wide band's input, when the bands fold differently at this bucket — `None` when they
    /// fold the same, so `steps`/`num_arena_slots` serve every class. The bake lowers
    /// [`GenClass::M5Wide`] under it and every other class under `steps`.
    pub wide_band: Option<WideBand<'a>>,
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
        // The wide class lowers under every M5 fact — the lowering's only device-profile
        // dependence is the generation, which M5Wide shares with M5; it differs in which step
        // tape (the routed fold's band) it bakes, not in any fact the profile carries.
        GenClass::M5 | GenClass::M5Wide => M5_10CORE,
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
    class: GenClass,
    /// The step tape this rung lowers — the wide class's band when the bands fold differently —
    /// and the arena slot count that tape's colouring allocated.
    steps: &'a MetalStepTape,
    num_arena_slots: u32,
    addressing: KvAddressing,
    cap: MaxBlocksPerSeq,
}

fn run_lower(
    mc: &ModelFacts,
    input: &BucketLowerInput<'_>,
    rung: &Rung<'_>,
) -> Result<LoweredMetalTape, LoweringError> {
    let at = tl::BakePoint {
        chunked: rung.addressing == KvAddressing::Chunked,
        bucket_m: input.bucket_m,
        num_arena_slots: rung.num_arena_slots,
        rotary: input.rotary,
        block_cap: rung.cap.get(),
        profile: Some(rung.profile),
        class: rung.class,
    };
    tl::lower_subtile_tape_to_metal(rung.steps, mc, at)
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
    mc: &ModelFacts,
    input: &BucketLowerInput<'_>,
    pool: &mut CommandPool,
) -> Result<TokenStream, BakeDefect> {
    let classes = [GenClass::M1, GenClass::Mid, GenClass::M5, GenClass::M5Wide];
    // The TurboQuant decode's head groups the global class admits; a model without attention
    // decodes no attention, so it bakes none.
    let candidates: Vec<TqDecodeHeads> = match mc.attention() {
        Some(a) => TqDecodeHeads::candidates(
            HeadDim(a.global().head_dim().get()),
            NumQHeads(a.q_heads().get()),
            NumKvHeads(a.global().kv_heads().get()),
        )
        .collect(),
        None => Vec::new(),
    };
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
        let (steps, num_arena_slots) = match (class, input.wide_band.as_ref()) {
            (GenClass::M5Wide, Some(wide)) => (wide.steps, wide.num_arena_slots),
            _ => (input.steps, input.num_arena_slots),
        };
        let rung = Rung {
            profile: &profile,
            class,
            steps,
            num_arena_slots,
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
        let serves_tq =
            (tape.commands.iter()).any(|c| c.command.kernel == KernelId::AttentionViaCacheTq);
        let heads: Vec<Option<TqDecodeHeads>> = match serves_tq {
            true => candidates.iter().copied().map(Some).collect(),
            false => vec![None],
        };
        for tq_heads in heads {
            let variant = TapeVariant { cap, tq_heads };
            pool.name_kernels(tape.commands, mc.metal_dtype(), variant)?;
            let class_toks = tok("class", crate::const_tokens::const_tokens(&class))?;
            let addressing_toks =
                tok("addressing", crate::const_tokens::const_tokens(&addressing))?;
            let cap_toks = tok("cap", crate::const_tokens::const_tokens(&cap))?;
            let tq_toks = tok("tq heads", crate::const_tokens::const_tokens(&tq_heads))?;
            entries.push(quote! {
                __tl::ClassedTape::rung(
                    #ident, #class_toks, #addressing_toks, #cap_toks, #tq_toks,
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
