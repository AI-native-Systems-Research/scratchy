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
    BlockSize, CommandIx, HeadDim, MaxBlocksPerSeq, MaxPositions, NumKvHeads, NumQHeads,
    TqDecodeHeads,
};
use scratchy_target_metal::tape::lowered::{
    GatedCommand, GenClass, KernelId, KvAddressing, LoweredMetalTape, LoweringError, MetalDtype,
    TapeCommands,
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
/// So is each distinct kernel of a baked library the commands name ([`aot::is_baked`]): one
/// `static` [`BakedKernel`](scratchy_target_metal::tape::lowered::BakedKernel), its metallib
/// compiled by [`aot::bake`] when the pool is emitted.
#[derive(Default)]
pub struct CommandPool {
    index: std::collections::HashMap<GatedCommand, CommandIx>,
    table: Vec<TokenStream>,
    kernel_index: std::collections::HashMap<PipelineKey, usize>,
    kernels: Vec<PipelineKey>,
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
                let toks = crate::const_tokens::const_tokens(cmd)
                    .map_err(|e| BakeDefect(format!("serialize command: {e}")))?;
                let id = quote::format_ident!("C{}", ix.0);
                self.table
                    .push(quote! { pub(super) const #id: __tl::GatedCommand = #toks; });
                self.index.insert(*cmd, ix);
                ix
            }
        };
        let ix = proc_macro2::Literal::u16_unsuffixed(ix.0);
        Ok(quote! { I(#ix) })
    }

    /// The baked kernels `commands` name at activation `dtype`, each once.
    fn kernels_of(&mut self, commands: TapeCommands, dtype: MetalDtype) -> Vec<TokenStream> {
        let mut ixs = std::collections::BTreeSet::new();
        for key in commands
            .iter()
            .filter_map(|c| aot::bake_key(&c.command, dtype))
        {
            let next = self.kernels.len();
            ixs.insert(*self.kernel_index.entry(key.clone()).or_insert_with(|| {
                self.kernels.push(key);
                next
            }));
        }
        let ids = ixs.into_iter().map(|ix| quote::format_ident!("K{ix}"));
        ids.map(|id| quote! { &__tape_cmds::#id }).collect()
    }

    /// The `__tape_cmds` module the model's tape statics reference. Emit it once, beside them.
    pub fn into_tokens(self, stem: &str) -> TokenStream {
        // A model metal lowers no tape for (a dense MoE has no metal realization) names nothing.
        if self.table.is_empty() && self.kernels.is_empty() {
            return TokenStream::new();
        }
        let aliases = crate::const_tokens::alias_preamble();
        let table = self.table;
        let len = proc_macro2::Literal::usize_unsuffixed(table.len());
        let ids = (0..table.len()).map(|i| quote::format_ident!("C{i}"));
        let bake = aot::bake(&self.kernels);
        eprintln!(
            "[metal bake] {stem}: {} kernels, {} compiled here, {} shared with an earlier model",
            self.kernels.len(),
            bake.compiled,
            self.kernels.len() - bake.compiled,
        );
        // Kernels whose compiles come out byte-identical (a constant the kernel never reads)
        // share one embedded metallib.
        let mut metallibs = std::collections::HashMap::new();
        let mut metallib_statics = Vec::new();
        let kernels = self.kernels.iter().zip(&bake.metallibs).enumerate();
        let kernels = kernels.map(|(ix, (key, metallib))| {
            let id = quote::format_ident!("K{ix}");
            let (library, function) = (key.library_name, key.kernel_name);
            let constants = crate::const_tokens::const_tokens(&key.constants.as_slice())
                .expect("serialize baked kernel constants");
            let next = metallibs.len();
            let lib = *metallibs.entry(metallib.clone()).or_insert_with(|| {
                let (lib, bytes) = (quote::format_ident!("M{next}"), metallib_file(metallib));
                let len = proc_macro2::Literal::usize_unsuffixed(metallib.len());
                metallib_statics.push(quote! { static #lib: [u8; #len] = *#bytes; });
                next
            });
            let metallib = quote::format_ident!("M{lib}");
            quote! {
                pub(super) static #id: __tl::BakedKernel = __tl::BakedKernel {
                    library: #library,
                    function: #function,
                    constants: #constants,
                    metallib: &#metallib,
                };
            }
        });
        let kernels: Vec<TokenStream> = kernels.collect();
        quote! {
            #[cfg(feature = "metal")]
            mod __tape_cmds {
                #aliases
                #(#table)*
                pub(super) static TABLE: [__tl::GatedCommand; #len] = [ #(#ids),* ];
                #(#metallib_statics)*
                #(#kernels)*
            }
        }
    }
}

/// `metallib` as a file under the build's `OUT_DIR`, named by its content (models sharing a kernel
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

/// One rung of a bucket's tape: its class, addressing, KV cap and TurboQuant decode heads.
struct Rung<'a> {
    profile: &'a MetalTargetProfile,
    addressing: KvAddressing,
    cap: MaxBlocksPerSeq,
    tq_heads: TqDecodeHeads,
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
        tq_heads: rung.tq_heads,
    };
    tl::lower_subtile_tape_to_metal(input.steps, mc, at)
}

/// Bake every `(gen class × KV addressing × KV cap × TurboQuant decode heads)` rung of one bucket's tape
/// and emit the `&'static [ClassedTape]` expression. A tape with no TurboQuant decode attention is
/// baked once for every device (`tq_heads: None`); one with it, once per head count its geometry
/// admits ([`TqDecodeHeads::candidates`]), which the device picks among. A cap rung whose scratch
/// cannot exist for this bucket ([`LoweringError::ScratchTooLarge`]) ends the bucket's ladder: it
/// and every rung above it are not baked. Rungs that lower identically share one hoisted body (see
/// below).
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
    // Dedupe: identical bodies share ONE hoisted static — every labelled rung would otherwise
    // repeat the full command tape and rustc drowns in tokens (a 10-family build OOM-killed the
    // compiler before this).
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
        let rung = |tq_heads| Rung {
            profile: &profile,
            addressing,
            cap,
            tq_heads,
        };
        let one = match run_lower(mc, input, &rung(TqDecodeHeads(1))) {
            Err(LoweringError::ScratchTooLarge { .. }) => return Ok(None),
            one => one?,
        };
        let serves_tq =
            (one.commands.iter()).any(|c| c.command.kernel == KernelId::AttentionViaCacheTq);
        if !serves_tq {
            return Ok(Some(vec![(None, one)]));
        }
        let lower = |h| match h {
            TqDecodeHeads(1) => Ok((Some(h), one)),
            _ => run_lower(mc, input, &rung(h)).map(|t| (Some(h), t)),
        };
        candidates
            .iter()
            .map(|&h| lower(h))
            .collect::<Result<_, _>>()
            .map(Some)
    });
    // The scratch grows with the cap: the first rung of a `(class, addressing)` ladder that cannot
    // exist ends it (`points` holds each ladder contiguous, ascending).
    let mut ended = None;
    for (&(class, addressing, cap), tapes) in points.iter().zip(lowered) {
        if ended == Some((class, addressing)) {
            continue;
        }
        let Some(tapes) = tapes.map_err(defect)? else {
            eprintln!(
                "[metal bake] bucket_m={}: ladder ends below KV cap rung {}",
                input.bucket_m,
                cap.get()
            );
            ended = Some((class, addressing));
            continue;
        };
        for (tq_heads, tape) in tapes {
            let body_ix = match seen.get(&tape).copied() {
                Some(ix) => ix,
                None => {
                    let kernel_refs = pool.kernels_of(tape.commands, mc.metal_dtype);
                    let cmd_refs = (tape.commands.iter())
                        .map(|c| pool.intern(c))
                        .collect::<Result<Vec<_>, _>>()?;
                    let rest = LoweredMetalTape {
                        commands: TapeCommands::EMPTY,
                        ..tape
                    };
                    let rest_toks = tok("tape", crate::const_tokens::const_tokens(&rest))?;
                    let ix = body_statics.len();
                    let ident = quote::format_ident!("__TAPE_BODY_{uniq}_{ix}");
                    body_statics.push(quote! {
                        // A `static`: a `const` holding `&TABLE` would have rustc validate the whole
                        // table once per body.
                        static #ident: __tl::ClassedTape = __tl::ClassedTape {
                            // Placeholder labels; entries override below.
                            gen_class: __tl::GenClass::M1,
                            addressing: __tl::KvAddressing::Direct,
                            cap: __ti::MaxBlocksPerSeq(0),
                            tq_heads: None,
                            tape: __tl::LoweredMetalTape {
                                commands: __tl::TapeCommands {
                                    table: &__tape_cmds::TABLE,
                                    ixs: { use __ti::CommandIx as I; &[ #(#cmd_refs),* ] },
                                },
                                ..#rest_toks
                            },
                            kernels: &[ #(#kernel_refs),* ],
                        };
                    });
                    seen.insert(tape, ix);
                    ix
                }
            };
            let ident = quote::format_ident!("__TAPE_BODY_{uniq}_{body_ix}");
            let class_toks = tok("class", crate::const_tokens::const_tokens(&class))?;
            let addressing_toks =
                tok("addressing", crate::const_tokens::const_tokens(&addressing))?;
            let cap_toks = tok("cap", crate::const_tokens::const_tokens(&cap))?;
            let tq_toks = tok("tq heads", crate::const_tokens::const_tokens(&tq_heads))?;
            entries.push(quote! {
                __tl::ClassedTape {
                    gen_class: #class_toks,
                    addressing: #addressing_toks,
                    cap: #cap_toks,
                    tq_heads: #tq_toks,
                    ..#ident
                },
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
