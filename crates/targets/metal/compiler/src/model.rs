// SPDX-License-Identifier: Apache-2.0
//! METAL'S EMISSION OF ONE CANONICAL MODEL: its buckets, each bucket's baked tapes and arena, the
//! pool that plays them, and the forward entry points over it — beside the weights struct the
//! shared pipeline emits from the tape's accessor set.

use std::collections::{BTreeMap, HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use scratchy_forward_compiler_frontend::classified::{OpKind, Program, UnrollIndex, WeightId};
use scratchy_forward_compiler_macro::codegen::{
    BackboneLayout, CanonicalLowered, DFLASH2_BLOCK_LANES, KvCacheLayers, backbone_layout,
    bound_marker, bounds_for_wp, bucket_static_ident, dim_type, emit_weight_accessors_impl,
    fuf_reads_target_hidden, fuf_uses_rotary, last_non_splice_node, lent_weight_prefix,
};
use scratchy_forward_compiler_macro::fuf::TileId;
use scratchy_forward_compiler_macro::interpreter_codegen::emit_bucket_static_slice;
use scratchy_forward_compiler_macro::target::{ModelCx, ModelEmission};
use scratchy_forward_compiler_macro::weight_vocab::WeightAccessor;
use scratchy_forward_compiler_macro::workload::WorkloadPoint;
use syn::Ident;

/// ⭐ A MODEL'S ARENA TYPES, EACH DECLARED ONCE: every extent its buckets' tile outputs are, spelled
/// at a bucket's rows' parameters, as one marker (`E<n>`); every colour its buckets' colourings
/// make — a slot and the extents it hosts — as one colour type (`K<n>`); and every colouring as one
/// arena of them (`S<n>`). Buckets whose colourings agree share all three, at their own rows
/// (`S0<U8, U128>`, their numbers as `typenum` types).
#[derive(Default)]
struct ArenaTypes {
    /// Each extent's marker, by its spelling.
    extents: BTreeMap<String, String>,
    /// Each arena value's extent's marker: what the bake names a binding of it by.
    values: scratchy_subtile::handoff::ArenaExtents,
    extent_defs: Vec<TokenStream>,
    /// The extents that compute over a bucket's rows ([`computes_over_rows`]).
    computed_defs: Vec<TokenStream>,
    colours: BTreeMap<String, String>,
    colour_defs: Vec<TokenStream>,
    arenas: BTreeMap<Vec<String>, String>,
    arena_defs: Vec<TokenStream>,
}

impl ArenaTypes {
    /// The marker of extent `at` (spelled at the rows' parameters).
    /// `computed`: whether it computes over the rows ([`computes_over_rows`]).
    fn extent(&mut self, (at, computed): &(TokenStream, bool)) -> Ident {
        let n = self.extents.len();
        let defs = match computed {
            true => &mut self.computed_defs,
            false => &mut self.extent_defs,
        };
        let name = (self.extents.entry(at.to_string())).or_insert_with(|| {
            let name = format_ident!("E{n}");
            defs.push(quote!(#name = #at;));
            name.to_string()
        });
        format_ident!("{name}")
    }

    /// Note arena value `value`'s extent, `at`: its marker is the one a binding of it names.
    fn value(&mut self, value: scratchy_subtile::handoff::ArenaValue, at: &(TokenStream, bool)) {
        let marker = self.extent(at).to_string();
        self.values.insert(value, marker);
    }

    /// The colour type of slot `slot` hosting the extents `sized` (counted in its bytes) and
    /// `unsized_`.
    fn colour(&mut self, slot: u32, sized: &[Ident], unsized_: &[Ident]) -> Ident {
        let def = quote!(= #slot [#(#sized),*] [#(#unsized_),*]);
        let n = self.colours.len();
        let defs = &mut self.colour_defs;
        let name = (self.colours.entry(def.to_string())).or_insert_with(|| {
            let name = format_ident!("K{n}");
            defs.push(quote!(#name #def));
            name.to_string()
        });
        format_ident!("{name}")
    }

    /// The arena whose slot `k` is `colours[k]`.
    fn arena(&mut self, colours: &[Ident]) -> Ident {
        let n = self.arenas.len();
        let defs = &mut self.arena_defs;
        let key = colours.iter().map(Ident::to_string).collect();
        let name = (self.arenas.entry(key)).or_insert_with(|| {
            let name = format_ident!("S{n}");
            defs.push(quote!(#name [#(#colours),*]));
            name.to_string()
        });
        format_ident!("{name}")
    }

    /// The declarations, in the model's `__arena`.
    fn tokens(&self) -> TokenStream {
        let (extents, computed) = (&self.extent_defs, &self.computed_defs);
        let (colours, arenas) = (&self.colour_defs, &self.arena_defs);
        quote! {
            ::scratchy_target_metal::arena_extents! { <M, S> [#(#extents)*] [#(#computed)*] }
            ::scratchy_target_metal::arena_colours! { <M, S> #(#colours)* }
            ::scratchy_target_metal::arena_schemas! { <M, S> #(#arenas)* }
        }
    }
}

/// Whether an extent of `shape` computes over a bucket's rows: names `num_tokens` or `sk_bucket`
/// anywhere but as its rows themselves (`Act<M, D>`), so it is an extent only where its dims' laws
/// hold at the bucket's rows.
fn computes_over_rows(shape: &[scratchy_forward_compiler_frontend::dim::Dim]) -> bool {
    use scratchy_forward_compiler_frontend::dim::Dim;
    fn names_rows(dim: &Dim) -> bool {
        match dim {
            Dim::Bound(name) => name == "num_tokens" || name == "sk_bucket",
            Dim::Lit(_) => false,
            Dim::Mul(parts) | Dim::Add(parts) => parts.iter().any(names_rows),
            Dim::Div(n, d) | Dim::FloorDiv(n, d) => names_rows(n) || names_rows(d),
        }
    }
    match shape {
        [] => false,
        [d] => names_rows(d),
        [rows, rest @ ..] => {
            let plain_rows = matches!(rows, Dim::Bound(name) if name == "num_tokens");
            (!plain_rows && names_rows(rows)) || rest.iter().any(names_rows)
        }
    }
}

/// A tile output of `shape` as the arena extent a colour hosts, `Act<rows, columns>`: its first
/// dim its rows, the product of the rest its columns (a rank-1 value is one row). A bucket's own
/// bounds are the bucket's rows' types, `num_tokens` as `m` and `sk_bucket` as `sk` (a colour
/// type's parameters, or one bucket's numbers); any other is the model's `__dims`' marker
/// ([`ModelDims::declare_tile_bounds`]).
fn arena_extent(
    shape: &[scratchy_forward_compiler_frontend::dim::Dim],
    m: &TokenStream,
    sk: &TokenStream,
) -> Result<TokenStream, String> {
    use scratchy_forward_compiler_frontend::dim::Dim;
    let mut bound = |name: &str| -> Result<TokenStream, String> {
        match name {
            "num_tokens" => Ok(m.clone()),
            "sk_bucket" => Ok(sk.clone()),
            _ => {
                let marker = bound_marker(name);
                Ok(quote!(__dims::#marker))
            }
        }
    };
    let (rows, cols) = match shape {
        [] => (Dim::Lit(1), Dim::Lit(1)),
        [d] => (Dim::Lit(1), d.clone()),
        [r, c] => (r.clone(), c.clone()),
        [r, rest @ ..] => (r.clone(), Dim::Mul(rest.to_vec())),
    };
    let (rows, cols) = (dim_type(&rows, &mut bound)?, dim_type(&cols, &mut bound)?);
    // `Act` is in scope wherever an extent is spelled (`scratchy_tensors::typed::Act`).
    Ok(quote!(Act<#rows, #cols>))
}

/// The shape an arena value of `fuf` has: a tile output's, or — the one graph source a step
/// names in the arena, the embedded hidden — its `Embed` tile's.
fn arena_value_shape(
    fuf: &scratchy_forward_compiler_macro::fuf::Fuf,
    value: scratchy_subtile::handoff::ArenaValue,
) -> Result<&scratchy_forward_compiler_frontend::dim::Shape, String> {
    use scratchy_subtile::handoff::ArenaValue;
    let output = |tile: u32, out: u8| {
        let node = fuf.nodes.get(tile as usize);
        node.and_then(|n| n.outputs.get(out as usize))
            .ok_or_else(|| format!("arena value of tile {tile} output {out}, which has none"))
    };
    match value {
        ArenaValue::Tile { tile, out } => output(tile, out),
        ArenaValue::Source(_) => {
            let embed = |n: &&scratchy_forward_compiler_macro::fuf::FufNode| {
                n.op == scratchy_forward_compiler_frontend::classified::OpKind::Embed
            };
            let mut embeds = fuf.nodes.iter().filter(embed);
            match (embeds.next(), embeds.next()) {
                (Some(embed), None) => output(embed.id.0, 0),
                _ => Err("the embedded hidden is not one `Embed` tile's output".to_string()),
            }
        }
    }
}

/// The model's `METAL_OFF_TAPE` static: its argmax, grammar mask and sampler kernels baked for its
/// logits `width` and `dtype`, and — `head`, a multi-token-prediction head's drafts a step and the
/// width of the hidden rows its passes read — its chain between passes
/// (`scratchy_target_metal::off_tape`).
fn metal_off_tape_tokens(
    width: u64,
    mc: &scratchy_ir::model_facts::ModelFacts,
    head: Option<(u32, u32)>,
    vocab_suffix: Option<scratchy_target_metal::off_tape::VocabSuffix>,
) -> TokenStream {
    use scratchy_target_metal::off_tape::{
        ByCodes, Dflash2Role, OffTape, SuffixKernels, TargetSampling, VocabSuffix,
    };
    use scratchy_target_metal::tape::ids::{BlockSize, HiddenSize, LogitsWidth, NumDrafts, RotDim};
    use scratchy_target_metal::tape::kernel_constants::{DraftAcceptConstants, MtpChainConstants};
    let width = u32::try_from(width).expect("logits width fits u32");
    use scratchy_ir::model_facts::DEFAULT_KV_PAGE;
    // The chain advances the full-context group's slots (KV group 0), at the global class's page;
    // a model without attention has no page of its own, and takes the default one.
    let page = mc
        .attention()
        .map_or(DEFAULT_KV_PAGE, |a| a.global().page());
    let block_size = BlockSize(page.get());
    // A draft block: its anchor row and its drafts.
    let block_rows = 1 + mc.spec_drafts().map_or(0, |d| d.get());
    let head = head.map(|(drafts, hidden)| MtpChainConstants {
        hidden: HiddenSize(hidden),
        drafts: NumDrafts(drafts),
        // An MRoPE head reads a cos/sin row a token in place of its position, its rot window
        // wide. The kernel's ABI spells the 1-D positions' path as 0.
        rope: RotDim(
            (mc.attention())
                .filter(|a| a.mrope().is_some())
                .map_or(0, |a| a.base().rot_dim().get()),
        ),
    });
    let keys = OffTape::keys(
        LogitsWidth(width),
        mc.metal_dtype(),
        block_size,
        head,
        mc.spec_drafts().is_some(),
        vocab_suffix,
        // A DFlash2 drafter (its block and the token its proposal rows embed), or a target one
        // drafts for (the rows a lane of its verify holds: the drafter's block).
        match (mc.drafter(), mc.capture_slots()) {
            (Some(drafter), _) => Dflash2Role::Drafter(DraftAcceptConstants {
                rows: block_rows,
                mask: drafter.mask_token().get(),
            }),
            (None, Some(_)) => Dflash2Role::Target { block_rows },
            (None, None) => Dflash2Role::None,
        },
    );
    let ids = keys.map(|key| crate::static_tape::kernel_ref(&key));
    let OffTape {
        vocab,
        argmax,
        argmax_dual_write,
        logprobs,
        grammar_mask,
        sampler,
        chain_advance,
        mtp_chain,
        select_rows,
        vocab_suffix,
        dflash2_accept,
        dflash2_sampling,
    } = ids;
    let dflash2_sampling = dflash2_sampling.map(|TargetSampling { mass, search, draw }| {
        quote!(::scratchy_target_metal::off_tape::TargetSampling {
            mass: #mass,
            search: #search,
            draw: #draw,
        })
    });
    let vocab_suffix = vocab_suffix.map(
        |SuffixKernels {
             suffix,
             qmv,
             argmax,
         }| {
            let VocabSuffix {
                prefix,
                rows,
                hidden,
                bits,
                group,
                scale,
                verify_rows,
                window,
            } = suffix;
            use scratchy_target_metal::tape::quantized::ScaleDtype;
            // Minted in a const context: a width or group no kernel is instantiated for fails the
            // build.
            let (bits, group) = (bits.get(), group.get());
            let scale = match scale {
                ScaleDtype::F16 => {
                    quote!(::scratchy_target_metal::tape::quantized::ScaleDtype::F16)
                }
                ScaleDtype::Bf16 => {
                    quote!(::scratchy_target_metal::tape::quantized::ScaleDtype::Bf16)
                }
            };
            let ByCodes {
                as_written,
                offset8,
            } = qmv;
            quote!(::scratchy_target_metal::off_tape::SuffixKernels {
                suffix: ::scratchy_target_metal::off_tape::VocabSuffix {
                    prefix: #prefix,
                    rows: #rows,
                    hidden: #hidden,
                    bits: const { ::scratchy_target_metal::tape::ids::AffineBits::of(#bits) },
                    group: const { ::scratchy_target_metal::tape::ids::AffineGroupSize::of(#group) },
                    scale: #scale,
                    verify_rows: #verify_rows,
                    window: #window,
                },
                qmv: ::scratchy_target_metal::off_tape::ByCodes {
                    as_written: &[#(#as_written),*],
                    offset8: &[#(#offset8),*],
                },
                argmax: #argmax,
            })
        },
    );
    let vocab = vocab.get();
    let sampler = sampler.iter().map(|row| {
        let row = row.iter();
        quote!([#(#row),*])
    });
    let option = |k: Option<TokenStream>| match k {
        Some(k) => quote!(::core::option::Option::Some(#k)),
        None => quote!(::core::option::Option::None),
    };
    let (mtp_chain, select_rows, vocab_suffix, dflash2_accept, dflash2_sampling) = (
        option(mtp_chain),
        option(select_rows),
        option(vocab_suffix),
        option(dflash2_accept),
        option(dflash2_sampling),
    );
    quote! {
        /// This model's off-tape kernels, baked for its logits.
        #[cfg(scratchy_target = "metal")]
        pub static METAL_OFF_TAPE: ::scratchy_target_metal::off_tape::OffTapeKernels = {
            ::scratchy_target_metal::off_tape::OffTape {
                vocab: ::scratchy_target_metal::tape::ids::LogitsWidth(#vocab),
                argmax: #argmax,
                argmax_dual_write: #argmax_dual_write,
                logprobs: #logprobs,
                grammar_mask: #grammar_mask,
                sampler: [#(#sampler),*],
                chain_advance: #chain_advance,
                mtp_chain: #mtp_chain,
                select_rows: #select_rows,
                vocab_suffix: #vocab_suffix,
                dflash2_accept: #dflash2_accept,
                dflash2_sampling: #dflash2_sampling,
            }
        };
    }
}

/// A multi-token-prediction head drafting over a prefix of its target's vocabulary — its logits
/// `vocab_size` wide, its target's `target_vocab_size` — whose lm_head is MLX-affine: the rest of
/// its target's lm_head (`scratchy_target_metal::off_tape::VocabSuffix`), at the widths its
/// lm_head is quantized to, a pass reading at most its `spec_verify_rows` rows. `None` for any
/// other model; a head whose lm_head is stored otherwise (a dense bf16 build) has no kernel past
/// its prefix, and drafts over the prefix alone.
fn metal_vocab_suffix(
    program: &Program,
    fuf: &scratchy_forward_compiler_macro::fuf::Fuf,
    model: &scratchy_forward_compiler_macro::config::ModelParams,
    mc: &scratchy_ir::model_facts::ModelFacts,
) -> Option<scratchy_target_metal::off_tape::VocabSuffix> {
    let (&prefix, &all) = (
        model.bounds.get("vocab_size")?,
        model.bounds.get("target_vocab_size")?,
    );
    if all <= prefix {
        return None;
    }
    let stem = &model.source_stem;
    let bound = |name: &str| {
        *(model.bounds.get(name)).unwrap_or_else(|| {
            panic!("`{stem}`: a head drafting over a vocabulary prefix declares its `{name}`")
        })
    };
    let (verify_rows, window) = (bound("spec_verify_rows"), bound("spec_wide_window"));
    let lm_head = lent_weight_prefix(
        program,
        fuf,
        model,
        scratchy_forward_compiler::LentWeight::LmHead,
    )?;
    let method = &model.quantization.as_ref()?.method;
    let (group, bits) = scratchy_forward_compiler_macro::quantization::affine_role_widths(
        program, method, &lm_head, None,
    )?;
    let fits = |v: u64| u32::try_from(v).expect("a vocabulary fits u32");
    use scratchy_target_metal::tape::ids::{AffineBits, AffineGroupSize};
    Some(scratchy_target_metal::off_tape::VocabSuffix {
        prefix: fits(prefix),
        rows: fits(all - prefix),
        hidden: fits(mc.hidden().get() as u64),
        bits: AffineBits::new(bits).unwrap_or_else(|| {
            panic!("the lm_head's {bits}-bit width is one no affine kernel is instantiated for")
        }),
        group: AffineGroupSize::new(group).unwrap_or_else(|| {
            panic!("the lm_head's group of {group} is one no affine kernel is instantiated for")
        }),
        scale: mc.scale_dtype(),
        verify_rows: fits(verify_rows),
        window: fits(window),
    })
}

/// The model's `METAL_TQ_CODEBOOK`: its TurboQuant codebook at the global head dim, computed here
/// (`None` for a dense KV cache).
fn metal_tq_codebook_tokens(mc: &scratchy_ir::model_facts::ModelFacts) -> TokenStream {
    use scratchy_target_metal::turboquant::TqCodebook;
    let codebook = match mc.kv_codec() {
        scratchy_forward_compiler::KvCodec::Dense => quote!(::core::option::Option::None),
        scratchy_forward_compiler::KvCodec::TurboQuant(bits) => {
            // The global class's head: the codebook rotates the full layers' keys.
            let attention =
                (mc.attention()).expect("ModelFacts::new: a TurboQuant KV codec is an attention's");
            let cb = TqCodebook::compute(attention.global().head_dim().get() as usize, bits);
            let table = |t: &[f32]| {
                let words = t.iter().map(|v| v.to_bits());
                quote!(&[#(f32::from_bits(#words)),*])
            };
            let (signs, centroids) = (table(cb.signs), table(cb.centroids));
            let (boundaries, bits, packed_dim) = (table(cb.boundaries), bits.get(), cb.packed_dim);
            quote!(::core::option::Option::Some(::scratchy_target_metal::turboquant::TqCodebook {
                bits: ::scratchy_forward_compiler::TqBits::new(#bits),
                packed_dim: #packed_dim,
                signs: #signs,
                centroids: #centroids,
                boundaries: #boundaries,
            }))
        }
    };
    quote! {
        /// This model's TurboQuant codebook (`None`: dense KV cache).
        #[cfg(scratchy_target = "metal")]
        pub static METAL_TQ_CODEBOOK: ::core::option::Option<
            ::scratchy_target_metal::turboquant::TqCodebook,
        > = #codebook;
    }
}

/// One canonical model's emission on metal.
pub(crate) fn emit(cx: &ModelCx<'_>) -> ModelEmission {
    let ModelCx {
        program,
        model,
        fuf,
        workloads,
        inferred,
        tp_world_size,
        canonical_params_impl,
        ..
    } = *cx;
    let model_facts = *cx.facts;
    // A head reading its target's KV (`reads_target_kv`) binds its target's TurboQuant runtime.
    let reads_target_kv = model.arch.reads_target_kv;
    let emit_struct = cx.struct_with;
    let model_dims = cx.dims_of;
    // The struct before the tape's accessors: what a canonical whose lowering refuses keeps.
    let (weights, weights_affine_rows) = cx.struct_of(None);
    let mut tape_accessors_for_struct: Option<Vec<WeightAccessor>> = None;
    // The config params and the typed facts are ONE model — the bridge takes them together.
    let bridge_model = scratchy_forward_compiler_macro::to_wavefront::BridgeModel {
        params: model,
        facts: &model_facts,
    };

    // Group workload points by TAPE fingerprint. Buckets that lower to
    // the same tape produce byte-identical fn bodies, so the canonical
    // is lowered ONCE and duplicates delegate to it through the bucket
    // table. Public API (every `forward_m_<M>[_sk_<SK>]` /
    // `forward_backbone_m_<M>[_sk_<SK>]` name a user might take a
    // fn-pointer to) is preserved. Dedup runs over `(num_tokens,
    // sk_bucket)` 2-D points.
    let bucket_points: Vec<WorkloadPoint> = workloads.iter().copied().collect();
    let bucket_canonical: Vec<WorkloadPoint> = {
        let mut tape_to_canonical: HashMap<String, WorkloadPoint> = HashMap::new();
        let mut bucket_canonical = Vec::with_capacity(bucket_points.len());
        for wp in &bucket_points {
            let bounds = bounds_for_wp(model, *wp, tp_world_size);
            let fp = match scratchy_forward_compiler_macro::to_wavefront::lower_decode_to_wavefront(
                fuf,
                inferred,
                &bounds,
                &bridge_model,
                scratchy_forward_compiler_macro::to_wavefront::PrefixCapacity::new(
                    std::num::NonZeroU32::new(8192).unwrap(),
                ),
                wp.num_tokens as u32,
            ) {
                Ok(l) => {
                    let m = wp.num_tokens as u32;
                    // A canonical realizes its codec guards at its own rows
                    // (`metal_guard_gates`), for every bucket it stands for: buckets whose
                    // realizations differ cannot share one (`lower_canonical` refuses it).
                    let guards = scratchy_target_metal::op_abi::metal_guard_gates(m);
                    let ops = l
                        .input
                        .ops
                        .iter()
                        .map(|od| {
                            // Row counts that track the bucket's `m`
                            // normalize to their EMISSION CLASS (decode
                            // m=1 vs prefill m>1 select different
                            // attention forms); fixed row counts stay
                            // verbatim. Normalizing to one sentinel
                            // collapsed m=1 with m=2 — the drift gate
                            // caught it against the solve's folding.
                            // The metal fold pass folds matvec ends on
                            // every bucket in the qmv matvec band (its
                            // floor: `QMV_MATVEC_BAND_ROWS`) and not
                            // above, so the classes split there too —
                            // a shared canonical would bake one band's
                            // steps at the other's bucket.
                            let band = scratchy_target_metal::tape::quantized::QMV_MATVEC_BAND_ROWS;
                            let mm = if od.m == m {
                                if m == 1 {
                                    u32::MAX
                                } else if m <= band {
                                    u32::MAX - 1
                                } else {
                                    u32::MAX - 2
                                }
                            } else {
                                od.m
                            };
                            format!("{:?}#{mm}#{:?};", od.op, od.inputs)
                        })
                        .collect::<String>();
                    format!("{guards:?}{ops}")
                }
                Err(e) => format!("REFUSED:{e:?}"),
            };
            bucket_canonical.push(*tape_to_canonical.entry(fp).or_insert(*wp));
        }
        bucket_canonical
    };

    // Lower every canonical tape_index once, backbone-shaped: skip the
    // terminal subgraph (the `gemm(<final_norm>, lm_head)` row) and
    // emit it separately as a tiny LM_HEAD slice. forward and
    // forward_backbone share the backbone slice; forward additionally
    // runs LM_HEAD; forward_backbone DtoD-copies the backbone-output
    // slot. No more pair of near-identical full slices per tape_index.
    let mut canonical_lowered: BTreeMap<
        WorkloadPoint,
        (
            CanonicalLowered,
            /* num_slots */ u32,
            /* backbone_slot */ u32,
            /* terminal_slot */ u32,
            /* slots: shared coloring across every tape_index in this
             * canonical's group. Per-tape_index arena_bytes (below in
             * the metal emission loop) computes its OWN sizes from
             * this map + that tape_index's bounds, so each tape_index gets
             * a tight arena sized for its own num_tokens. */
            scratchy_forward_compiler_macro::weight_vocab::SlotMap,
        ),
    > = BTreeMap::new();
    // `last_node_id` must be the lm_head Gemm, not one of the post-Embed `MmEmbedSplice` nodes
    // that sit at fuf-array-tail but semantically belong near the Embed; walking past them with
    // `last_non_splice_node` recovers the real terminal. The metal bucket facts walk the logits
    // chain from it (#784).
    let last_node_id = last_non_splice_node(fuf)
        .expect("non-empty FUF expected")
        .id;
    let layout = backbone_layout(fuf, program);
    // The canonical tuple starts as an empty placeholder and the
    // front-end swap below fills every field from the shared tape (a
    // swap that then refuses panics loudly there — an empty stream
    // cannot ship silently because the bucket table still reads these
    // fields).
    for (i, wp) in bucket_points.iter().enumerate() {
        if bucket_canonical[i] != *wp {
            continue;
        }
        canonical_lowered.insert(
            *wp,
            (
                CanonicalLowered {
                    backbone: scratchy_forward_compiler_macro::interpreter_codegen::LoweredBucket {
                        instances: Vec::new(),
                        weight_slots: Vec::new(),
                        barriers: Vec::new(),
                    },
                    lm_head: scratchy_forward_compiler_macro::interpreter_codegen::LoweredBucket {
                        instances: Vec::new(),
                        weight_slots: Vec::new(),
                        barriers: Vec::new(),
                    },
                },
                0,
                0,
                0,
                scratchy_forward_compiler_macro::weight_vocab::SlotMap::new(),
            ),
        );
    }

    // Per-canonical CanonicalParams impl + Instruction type alias.
    // The alias keeps every static-slice row short instead of
    // repeating `::scratchy_forward_compiler::Instruction::<Weights>::Variant(…)`.
    // One value, two consumers: the impl's `FACTS` (serialized) and the metal bake — the
    // `model_facts` hoisted above the bridge calls, computed once for both.
    // Per-canonical: alias the generic `Instruction<Weights>` for
    // the slice element type AND glob-import the variant
    // constructors so each static-slice row reads `Embed(...)` /
    // `RmsNorm(...)` instead of
    // `::scratchy_forward_compiler::Instruction::<Weights>::Embed(...)`.
    // The `__I` alias and the variant glob-import are backend-agnostic
    // — `Instruction<W>` is defined in `scratchy-forward-compiler` without a
    // backend gate. The per-tape_index static slices reference both, so
    // they compile under every backend feature.
    let instruction_alias = quote! {
        #[allow(non_camel_case_types, dead_code)]
        type __I = ::scratchy_forward_compiler::Instruction;
        use ::scratchy_forward_compiler::Instruction::*;
    };

    // Static slices: BACKBONE_M_<wp> + LM_HEAD_M_<wp> per CANONICAL
    // tape_index only. Non-canonical buckets share their canonical
    // sibling's slices via the FORWARD_TABLE entries below.
    let mut static_slices: Vec<TokenStream> = Vec::new();
    for (i, wp) in bucket_points.iter().enumerate() {
        if bucket_canonical[i] != *wp {
            continue;
        }
        let (lowered, _, _, _, _) = &canonical_lowered[wp];
        let backbone_static_ident = bucket_static_ident("BACKBONE_M", *wp);
        let lm_head_static_ident = bucket_static_ident("LM_HEAD_M", *wp);
        static_slices.push(emit_bucket_static_slice(
            &backbone_static_ident,
            &lowered.backbone.instances,
        ));
        static_slices.push(emit_bucket_static_slice(
            &lm_head_static_ident,
            &lowered.lm_head.instances,
        ));
        // MTL4 barrier flags, computed at FUF/SlotMap level
        // (one bool per `OpInstance`). Length
        // tracks the corresponding instructions static. Runtime
        // consumes via `MetalBucketSpec.{backbone,lm_head}_barriers`.
        {
            let bb_barriers_ident = bucket_static_ident("BACKBONE_BARRIERS_M", *wp);
            let lh_barriers_ident = bucket_static_ident("LM_HEAD_BARRIERS_M", *wp);
            static_slices.push(crate::static_tape::emit_bucket_barriers_static(
                &bb_barriers_ident,
                &lowered.backbone.barriers,
            ));
            static_slices.push(crate::static_tape::emit_bucket_barriers_static(
                &lh_barriers_ident,
                &lowered.lm_head.barriers,
            ));
        }
    }

    // `sk_axis_active`: true when the model declared `sk_buckets`;
    // otherwise every wp has sk_bucket==0 and the table emits
    // `[0, u64::MAX)` for sk on every entry.
    let sk_axis_active = workloads.iter().any(|wp| wp.sk_bucket != 0);
    let num_tokens_points: Vec<u64> = workloads.num_tokens_points();
    let mut sk_by_m: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for wp in workloads.iter() {
        sk_by_m.entry(wp.num_tokens).or_default().push(wp.sk_bucket);
    }
    for v in sk_by_m.values_mut() {
        v.sort();
        v.dedup();
    }

    // (m_min, m_max_excl, sk_min, sk_max_excl) for each workload.
    // M=1 is special-cased: the solver may pick M=1-only kernels
    // (cutlass_gemv) that fail at M>1, so the next tape_index starts at
    // M=2 even if the configured points list `[1, 8, ...]`.
    let m_max_excl_for = |m_idx: usize| -> u64 {
        if m_idx + 1 == num_tokens_points.len() {
            u64::MAX
        } else {
            num_tokens_points[m_idx + 1]
        }
    };
    let m_min_for = |m_idx: usize, m: u64| -> u64 {
        if m_idx > 0 && num_tokens_points[0] == 1 && num_tokens_points[m_idx - 1] == 1 {
            2
        } else {
            m
        }
    };
    let m_idx_of: HashMap<u64, usize> = num_tokens_points
        .iter()
        .enumerate()
        .map(|(i, &m)| (m, i))
        .collect();
    // Bucket-id assignment for the per-arch `WeightAccessors` impl:
    // canonical_lowered.iter() ordering pairs each canonical with
    // tape_index ids `(2*ci, 2*ci+1)` for backbone and lm_head. This
    // mapping must match the iteration order in
    // `emit_weight_accessors_impl` so the match-arm keys line up.
    let canonical_to_bucket_id: HashMap<_, u32> = canonical_lowered
        .iter()
        .enumerate()
        .map(|(ci, (wp, _))| (*wp, (ci as u32) * 2))
        .collect();

    // PD-wavefront macro-emission (env-gated, diagnostic + the const builder).
    // HERE — not in the pre-emit drive — because the decode bucket's
    // `weight_slots` (the real weight-locator source) only exists post-lowering
    // + post-loop-compression, and the `(bucket, op_idx, slot)` triples must
    // match the very match table `emit_weight_accessors_impl` builds from this
    // same `canonical_lowered`.
    // Wavefront-emitting builds only — a bundle target lowers the decode to a
    // single fused bundle. Plain metal dispatches per-op and skips
    // this; the expensive lowering + its hard panic on un-lowerable models
    // must not run for them.

    // THE METAL FRONT END. Each decode canonical is lowered from the SHARED subtile tape by
    // `crate::canonical::lower_canonical` — metal's orchestration lives in
    // the target's compiler crate. This computes the facts only the macro can (the weight table's
    // paths, the embed's storage, the MLP form, the per-module MoE widths), makes the one call, and
    // writes back what the bucket table, the arena statics, the accessor table and the bake read.
    let mut metal_steps: BTreeMap<WorkloadPoint, crate::canonical::ClassedCanonical> =
        BTreeMap::new();
    // THE MODEL'S SOURCE MANIFEST: every weight family its tapes bind, interned as the canonicals
    // lower, plus the class rotary tables rope-on-read binds (the same caches the Weights struct
    // declares: `rotary`, and `rotary_local` on a dual-rotary arch).
    let mut metal_sources = crate::canonical::SourceManifest::default();
    let metal_rotary = fuf_uses_rotary(fuf).then(|| {
        let local = fuf.nodes.iter().any(|n| {
            n.inputs.iter().any(|i| {
                matches!(
                    i,
                    scratchy_forward_compiler_macro::fuf::FufInput::Extern {
                        kind:
                            scratchy_forward_compiler_frontend::classified::ExternKind::RotaryLocal,
                        ..
                    }
                )
            })
        });
        metal_sources.rotary_tables(local)
    });
    for (canonical, (_, ns_field, bb_slot_field, term_field, slots_dec)) in
        canonical_lowered.iter_mut()
    {
        use crate::canonical as mc3;
        use scratchy_forward_compiler_macro::weight_bindings::{
            EmbedStorage, MlpPacking, WeightAbi,
        };
        use scratchy_target_metal::tape::step as st;
        let (m, stem) = (canonical.num_tokens, &model.source_stem);
        let bounds = bounds_for_wp(model, *canonical, tp_world_size);
        let prefix = scratchy_forward_compiler_macro::to_wavefront::PrefixCapacity::new(
            std::num::NonZeroU32::new(8192).unwrap(),
        );
        let l = match scratchy_forward_compiler_macro::to_wavefront::lower_decode_to_wavefront(
            fuf,
            inferred,
            &bounds,
            &bridge_model,
            prefix,
            m as u32,
        ) {
            Ok(l) => l,
            // METAL HAS NO KERNEL (dense MoE) is not a front-end gap: the canonical gets no steps,
            // so the bake bakes NO variants and the pool refuses at load. Anything else has no
            // fallback.
            Err(e) => {
                assert!(
                    e.is_no_metal_realization(),
                    "[m2-flip] {stem} m={m}: the shared tape refused a DECLARED pilot ({e:?}) — \
                     with the solve off there is no fallback",
                );
                eprintln!("[m2-eq] {stem} m={m} wavefront REFUSED: {e}");
                continue;
            }
        };
        // A preset's widths and groups are the kernels' (`quantization`'s preset check refuses
        // any other), so each mints.
        let affine_group = |g: u32| {
            st::AffineGroupSize::new(g).unwrap_or_else(|| {
                panic!("an affine group of {g}, which no kernel is instantiated for")
            })
        };
        let affine_bits = |b: u32| {
            st::AffineBits::new(b).unwrap_or_else(|| {
                panic!("an affine width of {b} bits, which no kernel is instantiated for")
            })
        };
        let embed_quant = fuf.nodes.iter().find_map(|n| {
            match (
                n.op,
                scratchy_forward_compiler_macro::weight_vocab::weight_storage_of(n),
            ) {
                (
                    OpKind::Embed,
                    Some(scratchy_forward_compiler_macro::quantization::StorageFormat::Affine {
                        group_size,
                        bits,
                    }),
                ) => Some((affine_group(*group_size), affine_bits(*bits))),
                _ => None,
            }
        });
        let fused_mlp = l.input.ops.iter().all(|od| {
            !matches!(
                od.op,
                scratchy_subtile::subtile_ir::SubOp::MatmulTile {
                    weight: scratchy_subtile::lower::GemmWeight::Affine { .. },
                    ..
                }
            )
        });
        let weight_paths: Vec<Vec<String>> = (program.weights.ids())
            .map(|id| program.weights.path(id).to_vec())
            .collect();
        let embed_base = weight_paths
            .iter()
            .map(|p| p.join("_"))
            .find(|p| p.contains("embed"));
        let embed_base = embed_base.unwrap_or_else(|| "embed_tokens".into());
        // Each DSL weight whose own subtree declares a `shared_expert`, by id: the bank a MoE op
        // reads is looked up by the weight it reads.
        let shared_expert_banks: std::collections::BTreeSet<u32> = (program.weights.ids())
            .filter(|&id| {
                let path = program.weights.path(id);
                let prefix: Vec<&str> = (path.iter().map(String::as_str))
                    .chain(["shared_expert"])
                    .collect();
                program.weights.has_subtree(&prefix)
            })
            .map(|id| id.0)
            .collect();
        let quant = model.quantization.as_ref();
        // A bank's widths, by the weight its op reads: its DSL path is the resolver's prefix.
        let widths = |bank: u32, layer: st::LayerId, body| {
            let qc = quant.expect("MoE widths are read for a quantized model only");
            let at = Some(UnrollIndex(layer.0 as u64));
            let prefix = program.weights.path(WeightId(bank)).join(".");
            let bits = scratchy_forward_compiler_macro::quantization::affine_moe_expert_bits(
                program, &qc.method, &prefix, at, body,
            );
            bits.map_or([None; 3], |b| b.projections().map(|w| w.map(affine_bits)))
        };
        let facts = mc3::MetalStepFacts {
            weight_paths: &weight_paths,
            shared_expert_banks: &shared_expert_banks,
            embed_base: &embed_base,
            embed_quant,
            mlp: if fused_mlp {
                mc3::MlpForm::Packed
            } else {
                mc3::MlpForm::Split
            },
            moe_expert_bits: quant.map(|_| &widths as &mc3::MoeExpertBits<'_>),
        };
        let mc = &model_facts;
        let buckets: Vec<u64> = (bucket_points.iter().zip(&bucket_canonical))
            .filter(|&(_, c)| c == canonical)
            .map(|(wp, _)| wp.num_tokens)
            .collect();
        let at = mc3::CanonicalAt {
            stem,
            m,
            buckets: &buckets,
            rows: match layout {
                BackboneLayout::Decoder => scratchy_subtile::sample_rows::ResultRows::Sampled,
                BackboneLayout::Encoder => scratchy_subtile::sample_rows::ResultRows::Every,
            },
        };
        let lowered = mc3::lower_canonical_classed(&l, &facts, mc, at, &mut metal_sources)
            .unwrap_or_else(|e| panic!("[m2-flip] {stem} m={m}: {e}"));
        // The narrow band's shared bucket facts — the classed lowering proved the wide band
        // keeps every one of them (only the step tapes may differ).
        *ns_field = lowered.narrow.colours.get();
        *bb_slot_field = lowered.narrow.hidden.0;
        *term_field = lowered.narrow.result.index();
        // Arena sizing evaluates a FUF shape per colour, so colour 0 (the embedded hidden) is
        // keyed by the FUF Embed tile.
        let mut arena = lowered.narrow.arena.clone();
        if let Some(et) = fuf.nodes.iter().find(|nd| nd.op == OpKind::Embed) {
            arena.insert_at(et.id, 0, 0);
        }
        *slots_dec = arena;
        metal_steps.insert(*canonical, lowered);
        // THE WEIGHTS STRUCT, from the tape's bindings. Metal's one ABI fact in it: a gate/up
        // pair is ONE packed `__fused__` buffer.
        let abi = WeightAbi {
            mlp: if fused_mlp {
                MlpPacking::Packed
            } else {
                MlpPacking::Split
            },
            embed: match embed_quant {
                Some(_) => EmbedStorage::AffineQuant,
                None => EmbedStorage::Dense,
            },
        };
        let synth = scratchy_forward_compiler_macro::weight_bindings::from_tape(&l, program, abi)
            .unwrap_or_else(|e| panic!("[weight-bindings] {stem}: {e}"));
        tape_accessors_for_struct = Some(synth.accessors);
    }
    // The `(bucket, op_idx, slot)` accessor table. No stream records a slot, so it is the
    // trait's defaults; on metal the model's tensors resolve through the source manifest's
    // resolver instead.
    let weight_accessors_impl =
        emit_weight_accessors_impl(&canonical_lowered, model.source_stem.as_str());
    let dims = model_dims(tape_accessors_for_struct.as_deref());
    let resolver = metal_sources
        .resolver(&|base| {
            dims.handle(base).map(|h| {
                let weights = &h.weights;
                (h.ty.clone(), quote!(__weights::#weights))
            })
        })
        .unwrap_or_else(|e| panic!("[metal sources] {}: {e:?}", model.source_stem));
    let weight_accessors_impl = {
        let resolver = &resolver.tokens;
        quote! {
            #weight_accessors_impl
            #[cfg(scratchy_target = "metal")]
            #resolver
        }
    };
    // The Weights struct is emitted from the TAPE's accessor set.
    // Re-emitted HERE because the tape accessors only exist after the
    // front-end swap has lowered the canonical.
    let (weights, weights_affine_rows) = match tape_accessors_for_struct.as_ref() {
        Some(tape_accs) => emit_struct(Some(tape_accs), &dims),
        None => (weights, weights_affine_rows),
    };

    let mut bucket_table_entries: Vec<TokenStream> = Vec::new();
    for (i, wp) in bucket_points.iter().enumerate() {
        let canonical = bucket_canonical[i];
        let bb_static = bucket_static_ident("BACKBONE_M", canonical);
        let lm_static = bucket_static_ident("LM_HEAD_M", canonical);
        let bb_bucket_id = *canonical_to_bucket_id
            .get(&canonical)
            .expect("canonical_to_bucket_id missing entry for canonical workload point");
        let bb_bucket_lit = proc_macro2::Literal::u32_unsuffixed(bb_bucket_id);
        let lm_bucket_lit = proc_macro2::Literal::u32_unsuffixed(bb_bucket_id + 1);
        // Per-tape_index slot metadata. The colored slot map is built
        // per workload point (the solver may pick Impls that need
        // different intermediate-tile counts per tape_index — e.g.
        // CutlassGemmAdd fuses the residual into the GEMM at
        // prefill, freeing a slot vs the decode-tape_index's separate
        // Add). The non-canonical buckets share their canonical's
        // slot metadata since they share its static slices.
        let (_, num_slots_b, backbone_slot_b, terminal_slot_b, _) = &canonical_lowered[&canonical];
        let num_slots_lit = proc_macro2::Literal::u32_unsuffixed(*num_slots_b);
        let backbone_slot_lit = proc_macro2::Literal::u32_unsuffixed(*backbone_slot_b);
        let terminal_slot_lit = proc_macro2::Literal::u32_unsuffixed(*terminal_slot_b);
        let m_idx = m_idx_of[&wp.num_tokens];
        let m_min = if wp.num_tokens == 1 {
            1
        } else {
            m_min_for(m_idx, wp.num_tokens)
        };
        let m_max_excl = if wp.num_tokens == 1 {
            2
        } else {
            m_max_excl_for(m_idx)
        };
        let (sk_min, sk_max_excl) = if sk_axis_active {
            let sk_buckets = &sk_by_m[&wp.num_tokens];
            let j = sk_buckets
                .iter()
                .position(|&sk| sk == wp.sk_bucket)
                .unwrap();
            // The first sk tape_index per m group must accept sk < its
            // own configured value — old codegen routed this via a
            // `_ =>` fallback arm onto the smallest sk fn. So emit
            // sk_min=0 for j==0 instead of wp.sk_bucket; without
            // this, find_bucket misses on small max_seqlen_k (the
            // prefill case for short prompts) and the fallback to
            // table[0] silently picks an m=1 row.
            let sk_min = if j == 0 { 0 } else { wp.sk_bucket };
            let sk_max_excl = if j + 1 == sk_buckets.len() {
                u64::MAX
            } else {
                sk_buckets[j + 1]
            };
            (sk_min, sk_max_excl)
        } else {
            (0u64, u64::MAX)
        };
        let m_min_lit = proc_macro2::Literal::u64_unsuffixed(m_min);
        let m_max_lit = if m_max_excl == u64::MAX {
            quote! { u64::MAX }
        } else {
            let v = proc_macro2::Literal::u64_unsuffixed(m_max_excl);
            quote! { #v }
        };
        let sk_min_lit = proc_macro2::Literal::u64_unsuffixed(sk_min);
        let sk_max_lit = if sk_max_excl == u64::MAX {
            quote! { u64::MAX }
        } else {
            let v = proc_macro2::Literal::u64_unsuffixed(sk_max_excl);
            quote! { #v }
        };
        bucket_table_entries.push(quote! {
            __B(
                #m_min_lit, #m_max_lit, #sk_min_lit, #sk_max_lit,
                #bb_static, #lm_static,
                #num_slots_lit, #backbone_slot_lit, #terminal_slot_lit,
                #bb_bucket_lit, #lm_bucket_lit,
            ),
        });
    }

    // FORWARD_TABLE — one row per tape_index. `__B` aliases the
    // `BucketEntry` tuple-struct constructor so each row stays on a
    // single line in expanded source. `scr model info`'s per-variant
    // `dump()` walks it; metal's pool uses `METAL_BUCKETS` for the
    // forward and ignores the runtime fields here.
    let forward_table = quote! {
        #[cfg(scratchy_target = "metal")]
        static FORWARD_TABLE: &[::scratchy_forward_compiler::BucketEntry<__I>] = {
            use ::scratchy_forward_compiler::BucketEntry as __B;
            &[
                #(#bucket_table_entries)*
            ]
        };
    };

    // METAL_BUCKETS — one row per distinct `num_tokens` point per routed fold band: the bands
    // colour differently, so each carries its own slots, terminal and arena (twin rows, sharing
    // everything, when the bands fold the same step tape). The
    // metal pool dispatches on `num_tokens` only (no sk axis); for
    // models that declared `sk_buckets` we pick the `sk_bucket == 0`
    // canonical at each `m`, falling back to whichever wp exists for
    // that `m` if the model elided the sk=0 entry. TinyLlama-class
    // models have no sk axis, so this is a 1:1 enumeration of
    // `num_tokens_points`.
    //
    // Each row's `backbone` / `lm_head` ride on the per-canonical
    // static slices the cuda emission already produced — under both
    // backends the slices are unconditionally emitted (the
    // `instruction_alias` and `Weights` definitions are now
    // mutually-exclusive cfg-gated, but the slice statics
    // themselves are backend-agnostic).
    let mut metal_bucket_entries: Vec<TokenStream> = Vec::new();
    // Each row of `METAL_BUCKETS` as its bucket type, in row order: its rows, band and arena
    // (`baked_buckets!`).
    let mut metal_bucket_types: Vec<TokenStream> = Vec::new();
    let mut metal_arena_bytes_statics: Vec<TokenStream> = Vec::new();
    let mut metal_tape_cmds = crate::static_tape::CommandPool::default();
    // The most arena slots a bucket has: how many the model's arena ranks rank (`RankedSlots`).
    let mut metal_arena_slots: u32 = 0;
    // Every colour and arena the buckets' colourings make, in the model's `__arena`.
    let mut metal_arena_types = ArenaTypes::default();
    // Whether the model's tapes read a paged KV pool. One that reads none (Splash's DFlash2
    // drafter, whose rings are its KV) holds any pool's block count at its ladder's floor.
    let pool_block_cap = match KvCacheLayers::of(fuf).any() {
        true => quote! { ctx.kv_cache.max_blocks_per_seq },
        false => quote! { 0 },
    };
    // The KV cap rungs every bucket bakes, topped by every block the model's positions fill.
    let metal_cap_ladder = {
        use scratchy_target_metal::tape::ids::{BlockSize, MaxPositions};
        let mc = &model_facts;
        let max_pos = KvCacheLayers::of(fuf).any().then(|| {
            let max_pos = model.bounds.get("max_position_embeddings").copied();
            max_pos
                .and_then(|p| u32::try_from(p).ok())
                .map(MaxPositions)
                .unwrap_or_else(|| {
                    panic!(
                        "model `{}`: no u32 max_position_embeddings to top the KV cap ladder",
                        model.source_stem
                    )
                })
        });
        // The pool's page is the base class's; a model without attention reads no pool (its
        // `max_pos` is `None`), and its ladder is the floor whatever the page.
        let page = mc
            .attention()
            .map_or(scratchy_ir::model_facts::DEFAULT_KV_PAGE, |a| {
                a.base().page()
            });
        let ladder = crate::static_tape::kv_cap_ladder(max_pos, BlockSize(page.get()));
        eprintln!(
            "[metal bake] {}: KV cap rungs {ladder:?}",
            model.source_stem
        );
        ladder
    };
    // Splash's two draft graphs (runtime/model/DFlashDraft.cpp `addContextCommit`, `addDecode`)
    // are one tape here, played apart. A DFlash2 drafter's block draft runs a step's lanes'
    // blocks: at most its target's verify rows (`spec_verify_rows`), so at the smallest bucket
    // that holds them (`plays_result_through`). A bucket past it plays the context commit alone,
    // and its arena holds what the commit's tiles (`commit_tiles`) write: the block's — its
    // lm_head's logits, a vocabulary a row, above all — are never written there.
    // The rows are a lane's (`spec_verify_rows`) times the lanes a step drafts
    // (`scratchy_serving_engine::spec_decode::BLOCK_LANES`, Splash's SPLASH_MAXIMUM_BATCH_WIDTH).
    let plays_result_through = match model.bounds.get("selector_rank") {
        Some(&rank) if rank > 0 => {
            (model.bounds.get("spec_verify_rows").copied()).and_then(|rows| {
                let rows = rows * DFLASH2_BLOCK_LANES;
                num_tokens_points
                    .iter()
                    .copied()
                    .filter(|&m| m >= rows)
                    .min()
            })
        }
        _ => None,
    };
    let commit_tiles: HashSet<TileId> = {
        let mut live: HashSet<TileId> = HashSet::new();
        let mut todo: Vec<TileId> = (fuf.nodes.iter())
            .filter(|n| {
                n.op == scratchy_forward_compiler_frontend::classified::OpKind::DraftContextKv
            })
            .map(|n| n.id)
            .collect();
        while let Some(id) = todo.pop() {
            if live.insert(id) {
                todo.extend(fuf.get(id).inputs.iter().filter_map(|i| match *i {
                    scratchy_forward_compiler_macro::fuf::FufInput::Tile { id, .. } => Some(id),
                    _ => None,
                }));
            }
        }
        live
    };
    for &m in &num_tokens_points {
        // Prefer the sk=0 canonical for this `m`; fall back to any wp
        // at `m` if the model never declared sk=0 explicitly.
        let wp = bucket_points
            .iter()
            .find(|wp| wp.num_tokens == m && wp.sk_bucket == 0)
            .copied()
            .or_else(|| bucket_points.iter().find(|wp| wp.num_tokens == m).copied())
            .expect("every num_tokens point has at least one wp");
        let i = bucket_points
            .iter()
            .position(|w| *w == wp)
            .expect("wp came from bucket_points");
        let canonical = bucket_canonical[i];
        let bb_static = bucket_static_ident("BACKBONE_M", canonical);
        let lm_static = bucket_static_ident("LM_HEAD_M", canonical);
        let bb_barriers_static = bucket_static_ident("BACKBONE_BARRIERS_M", canonical);
        let lh_barriers_static = bucket_static_ident("LM_HEAD_BARRIERS_M", canonical);
        let (_, num_slots_b, backbone_slot_b, terminal_slot_b, slots_b) =
            &canonical_lowered[&canonical];
        let bucket_m_lit = proc_macro2::Literal::u32_unsuffixed(m as u32);

        // Per-tape_index arena_bytes: register-coloring tells us which
        // (tile, output_slot) pairs share an arena slot; for THIS
        // tape_index M we evaluate each tile's output shape against this
        // tape_index's bounds and take the per-color max. Each tape_index
        // gets a tight arena sized exactly for its own num_tokens —
        // the worker pool elementwise-maxes across MetalBucketSpec
        // entries once at init time to size the per-worker arena
        // (init-time alloc, reused across every forward).
        let bp_bounds = bounds_for_wp(model, wp, tp_world_size);
        let plays_result = plays_result_through.is_none_or(|through| m <= through);
        // One fold band's bucket facts — its colouring's slot count, terminal, hidden and
        // per-slot bytes; each band's step tape was coloured under its own colouring, so its
        // bucket row carries its own. The logits: the terminal tile and the in-place logit
        // tail under it, back to the lm_head — the terminal slot's chain. A forward leaves only
        // the rows its step samples there, so they take that slot's bytes a row
        // (`MetalBucketSpec::logits_row_bytes`), its rows counted at load; an encoder's
        // terminal rows are every row, as are every row of a backend that samples none (metal
        // alone compacts them). A tile output the coloring did not register (a refused
        // canonical's empty map, an input the tape's arena keys by another tile) is in no
        // slot, so not the terminal's.
        let mut band_facts = |num_slots: u32,
                              terminal_slot: u32,
                              backbone_slot: u32,
                              slots: &scratchy_forward_compiler_macro::weight_vocab::SlotMap|
         -> (u32, u32, u32, u64, (Ident, Vec<String>)) {
            let in_terminal = |id: TileId, slot: u8| {
                (slots.iter()).any(|(at, colour)| at == (id, slot) && colour == terminal_slot)
            };
            let logits_tiles: Vec<(TileId, u8)> = match layout {
                BackboneLayout::Decoder => {
                    let mut chain = vec![(last_node_id, 0u8)];
                    while let Some(&(tile, _)) = chain.last() {
                        let under = fuf.get(tile).inputs.iter().find_map(|i| match *i {
                            scratchy_forward_compiler_macro::fuf::FufInput::Tile { id, slot }
                                if in_terminal(id, slot) =>
                            {
                                Some((id, slot))
                            }
                            _ => None,
                        });
                        match under {
                            Some(under) if !chain.contains(&under) => chain.push(under),
                            _ => break,
                        }
                    }
                    chain
                }
                BackboneLayout::Encoder => Vec::new(),
            };
            let mut logits_row_bytes: u64 = 0;
            // The extents each colour hosts, each once (by its spelling, at the rows' parameters):
            // those its `Colour::BYTES` is the largest of, and those another size covers — the
            // logits rows (`logits_row_bytes`), the tiles a bucket that plays no result never
            // commits, the embedded hidden colour 0 is (`arena_colours!`).
            let no_slots =
                || vec![BTreeMap::<String, (TokenStream, bool)>::new(); num_slots as usize];
            let (mut sized, mut unsized_) = (no_slots(), no_slots());
            let extent = |shape: &scratchy_forward_compiler_frontend::dim::Shape| {
                let at = arena_extent(shape, &quote!(M), &quote!(S))
                    .unwrap_or_else(|e| panic!("[metal arena] {}: m={m}: {e}", model.source_stem));
                (at, computes_over_rows(shape))
            };
            for ((tile_id, out_slot), color) in slots.iter() {
                let node = fuf.get(tile_id);
                let shape = &node.outputs[out_slot as usize];
                let e = extent(shape);
                let value = scratchy_subtile::handoff::ArenaValue::Tile {
                    tile: tile_id.0,
                    out: out_slot,
                };
                metal_arena_types.value(value, &e);
                if !plays_result && !commit_tiles.contains(&tile_id) {
                    unsized_[color as usize].insert(e.0.to_string(), e);
                } else if logits_tiles.contains(&(tile_id, out_slot)) {
                    // f16 = 2 bytes/element; metal kernels are f16-only.
                    let dims = scratchy_forward_compiler_macro::weight_vocab::eval_shape_with(
                        shape, &bp_bounds,
                    )
                    .expect("a tile output names a bound the model lacks — codegen invariant");
                    let row: u64 = dims.iter().skip(1).product();
                    logits_row_bytes = logits_row_bytes.max(row.saturating_mul(2));
                    unsized_[color as usize].insert(e.0.to_string(), e);
                } else {
                    sized[color as usize].insert(e.0.to_string(), e);
                }
            }
            let embedded = scratchy_subtile::handoff::ArenaValue::Source(0);
            if let (Ok(shape), Some(zero)) =
                (arena_value_shape(fuf, embedded), unsized_.first_mut())
            {
                let e = extent(shape);
                metal_arena_types.value(embedded, &e);
                zero.insert(e.0.to_string(), e);
            }
            for (sized, unsized_) in sized.iter().zip(&mut unsized_) {
                unsized_.retain(|e, _| !sized.contains_key(e));
            }
            metal_arena_slots = metal_arena_slots.max(num_slots);
            // The bucket's arena: each slot's colour type, and the arena of them.
            let colours: Vec<Ident> = (sized.iter().zip(&unsized_).enumerate())
                .map(|(slot, (sized, unsized_))| {
                    let mut markers = |extents: &BTreeMap<String, (TokenStream, bool)>| {
                        let at = extents.values();
                        at.map(|at| metal_arena_types.extent(at))
                            .collect::<Vec<_>>()
                    };
                    let (sized, unsized_) = (markers(sized), markers(unsized_));
                    metal_arena_types.colour(slot as u32, &sized, &unsized_)
                })
                .collect();
            let arena = metal_arena_types.arena(&colours);
            let colours = colours.iter().map(Ident::to_string).collect::<Vec<_>>();
            (
                num_slots,
                terminal_slot,
                backbone_slot,
                logits_row_bytes,
                (arena, colours),
            )
        };
        // The narrow band's facts: the shared tuple the metal front-end swap wrote (which the
        // FORWARD_TABLE row reads too).
        let narrow = band_facts(*num_slots_b, *terminal_slot_b, *backbone_slot_b, slots_b);
        // The wide band's: its own canonical's when the bands fold differently at this bucket,
        // the narrow's twin when they fold the same (no routing ops, or the fold refused) —
        // either way the pool picks one row per bucket by its device's band.
        // `None`: the narrow's twin, whose arena it shares.
        let wide = (metal_steps.get(&canonical).and_then(|c| c.wide.as_ref())).map(|wide| {
            band_facts(
                wide.colours.get(),
                wide.result.index(),
                wide.hidden.0,
                &wide.arena,
            )
        });
        // Bake this bucket's tape rungs at expansion — one per device
        // generation, addressing, KV cap and TurboQuant decode heads —
        // every fact baked; the pool picks one. See static_tape.rs.
        let tapes_static_toks = {
            let mc = &model_facts;
            // A canonical metal has no kernel for (`[m2-eq] … REFUSED`: it has no steps) bakes NO
            // variants, so `MetalWorkerPool::for_buckets` refuses at load (`NoRung`), naming
            // the bucket. Baking its empty step tape instead would build VALID empty-command tapes:
            // the forward would no-op and the tail would index the arena out of bounds.
            let tapes_expr = match metal_steps.get(&canonical) {
                None => {
                    eprintln!(
                        "[metal static tape] {}: bucket_m={m}: front-end REFUSED this canonical — \
                         baking an empty variant list so the pool refuses at load",
                        model.source_stem
                    );
                    quote! { &[] }
                }
                Some(classed) => {
                    use crate::static_tape::{BandArena, BucketRows};
                    // Each band's colour types by slot, at the bucket's rows; a wide band that
                    // folds as the narrow one does is its twin.
                    let rows = BucketRows::At {
                        m: m as u32,
                        sk: wp.sk_bucket as u32,
                    };
                    let band_arena = |(_, colours): &(Ident, Vec<String>)| BandArena {
                        colours: colours.clone(),
                        rows,
                    };
                    let narrow_arena = band_arena(&narrow.4);
                    let wide_arena = band_arena(&wide.as_ref().unwrap_or(&narrow).4);
                    let input = crate::static_tape::BucketLowerInput {
                        steps: &classed.narrow.steps,
                        wide_band: classed
                            .wide
                            .as_ref()
                            .map(|wide| crate::static_tape::WideBand {
                                steps: &wide.steps,
                                num_arena_slots: wide.colours.get(),
                                operands: &wide.operands,
                                arena: &wide_arena,
                            }),
                        bucket_m: m as u32,
                        num_arena_slots: *num_slots_b,
                        rotary: metal_rotary,
                        cap_ladder: &metal_cap_ladder,
                        operands: &classed.narrow.operands,
                        arena: &narrow_arena,
                        // What each value its steps name is, as its colour hosts it.
                        extents: &metal_arena_types.values,
                    };
                    crate::static_tape::bake_bucket_tapes(mc, &input, &mut metal_tape_cmds)
                        .unwrap_or_else(|e| {
                            panic!(
                                "[metal static tape] {}: bucket_m={m}: {e}",
                                model.source_stem
                            )
                        })
                }
            };
            let ident = bucket_static_ident("METAL_TAPES_M", wp);
            metal_arena_bytes_statics.push(quote! {
                #[cfg(scratchy_target = "metal")]
                static #ident:
                    &[::scratchy_target_metal::tape::lowered::ClassedTape] = #tapes_expr;
            });
            quote! { #ident }
        };
        // The weight bytes a forward of the bucket reads (`MetalStepTape::weight_bytes`).
        let (fixed, per_row) = (metal_steps.get(&canonical))
            .map(|classed| classed.narrow.steps.weight_bytes())
            .map_or((0, 0), |b| (b.fixed, b.per_row));
        let weight_bytes_toks = {
            let (fixed, per_row) = (
                proc_macro2::Literal::u64_unsuffixed(fixed),
                proc_macro2::Literal::u64_unsuffixed(per_row),
            );
            quote! {
                ::scratchy_target_metal::tape::step::WeightBytes { fixed: #fixed, per_row: #per_row }
            }
        };
        // The model's baked kernels (`CommandPool::into_tokens`).
        let kernels_toks = quote! { METAL_KERNELS };
        // One row per fold band — the wide band's statics under its own names, the tapes and
        // kernels shared: the pick keeps the row whose band the device's class lowers.
        // A wide band that folds as the narrow one does is its twin.
        let band_rows = [
            ("Narrow", &narrow),
            ("Wide", wide.as_ref().unwrap_or(&narrow)),
        ];
        for (band, (num_slots, terminal_slot, backbone_slot, logits_row_bytes, (arena, _))) in
            band_rows
        {
            let num_slots_lit = proc_macro2::Literal::u32_unsuffixed(*num_slots);
            let terminal_slot_lit = proc_macro2::Literal::u32_unsuffixed(*terminal_slot);
            let backbone_slot_lit = proc_macro2::Literal::u32_unsuffixed(*backbone_slot);
            let logits_row_bytes_lit = proc_macro2::Literal::u64_unsuffixed(*logits_row_bytes);
            // The band row's arena at the bucket's rows: each slot's bytes are its colour's.
            let rows = |n: u64| {
                let path = scratchy_tensors::typed::unsigned_alias_path(n);
                (path.parse::<TokenStream>()).unwrap_or_else(|e| {
                    panic!("[metal arena] {}: rows `{path}`: {e}", model.source_stem)
                })
            };
            let (m_rows, sk_rows) = (rows(m), rows(wp.sk_bucket));
            let arena_bytes = quote! {
                <__arena::#arena<#m_rows, #sk_rows> as ::scratchy_target_metal::tape::typed::Arena>
                    ::BYTES
            };
            let band_ident = proc_macro2::Ident::new(band, proc_macro2::Span::call_site());
            let bucket_type = format_ident!("B{}", metal_bucket_types.len());
            metal_bucket_types.push(quote! {
                #bucket_type = (#m_rows, #band_ident, super::__arena::#arena<#m_rows, #sk_rows>)
            });
            metal_bucket_entries.push(quote! {
                ::scratchy_target_metal::interpreter::metal::MetalBucketSpec {
                    bucket_m: #bucket_m_lit,
                    band: ::scratchy_target_metal::interpreter::metal::FoldBand::#band_ident,
                    num_arena_slots: #num_slots_lit,
                    terminal_slot: #terminal_slot_lit,
                    backbone_slot: #backbone_slot_lit,
                    arena_bytes: #arena_bytes,
                    logits_row_bytes: #logits_row_bytes_lit,
                    backbone: #bb_static,
                    lm_head: #lm_static,
                    backbone_barriers: #bb_barriers_static,
                    lm_head_barriers: #lh_barriers_static,
                    tapes: #tapes_static_toks,
                    kernels: #kernels_toks,
                    plays_result: #plays_result,
                    weight_bytes: #weight_bytes_toks,
                },
            });
        }
    }

    // THE ARENA RANKS: the slots by size at their largest bucket, largest first, ties by slot —
    // the order in which another model whose forwards never overlap this one's shares its arena
    // (`ArenaRanks`): every slot as a guest, all but the buckets' final hidden states as a host.
    // Read off `METAL_BUCKETS` as their `static` compiles (`RankedSlots`): each slot's bytes are
    // its colours'.
    let metal_ranked_slots = proc_macro2::Literal::usize_unsuffixed(metal_arena_slots as usize);
    let metal_arena_types = metal_arena_types.tokens();
    metal_arena_bytes_statics.push(quote! {
        /// The model's arena: the colours and arenas its buckets' colourings make, and its slots
        /// ranked by their bytes.
        #[cfg(scratchy_target = "metal")]
        mod __arena {
            use super::__dims;
            use ::scratchy_tensors::typed::Act;
            #metal_arena_types
            /// [`super::METAL_BUCKETS`]' arena slots, ranked by the bytes their colours host.
            pub(super) static RANKED: ::scratchy_target_metal::interpreter::metal::RankedSlots<
                #metal_ranked_slots,
            > = ::scratchy_target_metal::interpreter::metal::RankedSlots::of(super::METAL_BUCKETS);
        }

        /// The model's buckets: each row of [`METAL_BUCKETS`] as a type, and the enum a forward's
        /// bucket is, chosen at the door (`scratchy_target_metal::interpreter::metal::bucket`).
        #[cfg(scratchy_target = "metal")]
        mod __buckets {
            ::scratchy_target_metal::baked_buckets! {
                pub enum Bucket { #(#metal_bucket_types),* }
            }
        }

        #[cfg(scratchy_target = "metal")]
        impl ::scratchy_target_metal::interpreter::metal::BakedBuckets for Weights {
            type Bucket = __buckets::Bucket;
        }
    });

    // Largest `num_tokens` across all buckets — drives runtime
    // buffer sizing in the metal forward body's `RuntimeFactory`.
    let max_bucket_m_lit = {
        let max_m = num_tokens_points
            .iter()
            .copied()
            .max()
            .expect("at least one num_tokens point per canonical");
        proc_macro2::Literal::u32_unsuffixed(max_m as u32)
    };

    // Vision runtime-extern buffer sizes (bytes), baked from the largest
    // bucket + vision geometry. Zero (→ alloc's 16-byte floor) on
    // non-vision arches (the `vision_*` bounds are absent in text
    // configs). `freqs` is f32 `[total_L, vision_head_dim/2]`; `pixels`
    // is the model dtype (bf16) `[num_tokens, vision_in_features]`.
    // An MTP head's body reads its target's final hidden rows (`RowsExtern::TargetHidden`).
    let reads_target_hidden = fuf_reads_target_hidden(fuf);
    // The width of those rows (the `TargetHidden` extern's): its target's hidden size — which a
    // drafter of its own width (Gemma 4's assistant) reads and exports at, its passes chained at.
    let target_hidden_width: u64 = model
        .bounds
        .get("backbone_hidden_size")
        .or_else(|| model.bounds.get("hidden_size"))
        .copied()
        .unwrap_or(0);
    let (
        vision_freqs_bytes_lit,
        vision_pixels_bytes_lit,
        vision_posemb_bytes_lit,
        target_hidden_bytes_lit,
        mm_embeds_bytes_lit,
        mm_dst_rows_bytes_lit,
        mrope_cos_sin_bytes_lit,
    ) = {
        let max_m = num_tokens_points.iter().copied().max().unwrap_or(0);
        let vis_head_dim = model.bounds.get("vision_head_dim").copied().unwrap_or(0);
        let vis_in_features = model.bounds.get("vision_in_features").copied().unwrap_or(0);
        let vis_embed_dim = model.bounds.get("vision_embed_dim").copied().unwrap_or(0);
        let hidden_size = model.bounds.get("hidden_size").copied().unwrap_or(0);
        let freqs_bytes = max_m * (vis_head_dim / 2) * 4;
        let pixels_bytes = max_m * vis_in_features * 2;
        // pos_embeds: model-dtype (bf16) `[num_tokens, vision_embed_dim]`,
        // host-interpolated and added to the patch-embed output.
        let posemb_bytes = max_m * vis_embed_dim * 2;
        // target_hidden (MTP heads): model-dtype `[num_tokens, target_hidden_width]`, the
        // target's final hidden rows. Zero unless the body reads the extern.
        let target_hidden_bytes = match reads_target_hidden {
            true => max_m * target_hidden_width * 2,
            false => 0,
        };
        // mm splice (text decoder): mm_embeds = bf16 `[max_m, hidden]`;
        // mm_dst_rows = u32 `[max_m]`. Zero on non-MM arches.
        let mm_embeds_bytes = max_m * hidden_size * 2;
        let mm_dst_rows_bytes = max_m * 4;
        // MRoPE cos/sin override (MRoPE text decoders only): model-dtype
        // (f16/bf16 → 2 bytes) `[max_m, rot_dim]`. 0 on 1D-rope arches
        // (the `resolve_bindings` swap never fires, so the buffer is
        // never bound — `alloc`'s 16-byte floor covers the placeholder).
        // `rot_dim = partial_rotary_factor * head_dim` (Qwen3.5: 0.25*256
        // = 64), mirroring the rope-cache builder / `ROT_DIM` fn-const.
        let mrope_cos_sin_bytes = if model.mrope_section.is_some() {
            let head_dim = u64::from(model.text.head_dim.get());
            let rot_dim = model
                .scalars
                .get("partial_rotary_factor")
                .copied()
                .filter(|&f| (f - 1.0).abs() > 1e-9)
                .map(|f| (f * head_dim as f64).round() as u64)
                .unwrap_or(head_dim);
            max_m * rot_dim * 2
        } else {
            0
        };
        (
            proc_macro2::Literal::u64_unsuffixed(freqs_bytes),
            proc_macro2::Literal::u64_unsuffixed(pixels_bytes),
            proc_macro2::Literal::u64_unsuffixed(posemb_bytes),
            proc_macro2::Literal::u64_unsuffixed(target_hidden_bytes),
            proc_macro2::Literal::u64_unsuffixed(mm_embeds_bytes),
            proc_macro2::Literal::u64_unsuffixed(mm_dst_rows_bytes),
            proc_macro2::Literal::u64_unsuffixed(mrope_cos_sin_bytes),
        )
    };
    // The rows a target captures for its DFlash2 drafter (model dtype, `[max_m, slots · hidden]`;
    // 0 on a model no drafter reads), and a drafter's block (its drafts and their anchor).
    let (draft_capture_bytes_lit, draft_rows_lit, captured_rows) = {
        let max_m = num_tokens_points.iter().copied().max().unwrap_or(0);
        let hidden_size = model.bounds.get("hidden_size").copied().unwrap_or(0);
        let slots = (model
            .layer_sets
            .get(scratchy_forward_compiler_frontend::draft_capture::DRAFT_CAPTURE_LAYERS))
        .map_or(0, |s| s.members().len() as u64);
        let rows = *model.bounds.get("spec_drafts").unwrap_or(&0) + 1;
        let bytes = max_m * slots * hidden_size * 2;
        // What a step's followup gets of them, from the worker's runtime bindings: they, or none
        // on a model no drafter reads.
        let captured_rows = match bytes > 0 {
            true => (quote! { runtime }, quote! { Some(&*runtime.draft_capture) }),
            false => (quote! { _ }, quote! { None }),
        };
        (
            proc_macro2::Literal::u64_unsuffixed(bytes),
            proc_macro2::Literal::u64_unsuffixed(rows),
            captured_rows,
        )
    };
    let (captured_runtime, captured_rows) = captured_rows;

    // Width of ONE row of the terminal arena slot, baked at
    // macro-expansion time: the metal forward shapes its returned
    // `OwnedTensor` as `[num_tokens, width]` over that slot, and the
    // argmax kernel strides by it.
    //
    // Decoder layouts terminate in the lm_head Gemm, so the width is
    // `vocab_size`. ENCODER layouts (ModernBERT and every
    // `#[vision_forward]` body) have no lm_head — the terminal slot
    // holds hidden states, so the width is `hidden_size`. Using vocab
    // there would stride rows by ~65x the real row and read past the
    // slot. Vision-only encoders carry no vocab at all and would get
    // 0. Cuda builds skip the constant entirely.
    let logits_width = match layout {
        BackboneLayout::Decoder => model.bounds.get("vocab_size").copied().unwrap_or(0),
        BackboneLayout::Encoder => model.bounds.get("hidden_size").copied().unwrap_or(0),
    };
    let vocab_size_lit = proc_macro2::Literal::u64_unsuffixed(logits_width);
    // The rows of the terminal slot a forward leaves for its caller: a decoder's logits of the
    // rows its step samples, at the front; an encoder's hidden state at every row.
    let terminal_rows = match layout {
        BackboneLayout::Decoder => quote! { inputs.sampled_rows.len() },
        BackboneLayout::Encoder => quote! { n },
    };

    // Per-forward MRoPE cos/sin override builder, spliced into the metal
    // `forward` / `forward_with_metal_followup` bodies just before
    // `ForwardInputs`. Only MRoPE text decoders (config carries
    // `rope_scaling.mrope_section`) build the per-token band-split table
    // and switch the rope kernel to identity positions (option (b)):
    // `positions[t] = t` indexes row `t` of the table, so the unmodified
    // 1D rope kernel reads the right (T/H/W band-split) cos/sin without an
    // in-kernel band-split. The worker's `resolve_bindings` redirects the
    // baked `WeightBundleKind::CosSin` pointer to `runtime.mrope_cos_sin`
    // under the same `W::MROPE_SECTION` gate. Every 1D-rope arch keeps
    // `positions` untouched and passes `mrope_cos_sin: None`. The `[3, n]`
    // vs broadcast `[n]` positions are disambiguated at runtime by
    // `ctx.positions` numel (the worker uploads `[3, n]` only when image
    // tokens are present).
    // MRoPE (Qwen3.5-VL): the per-token rotary cos/sin rows a forward reads in place of the
    // position-indexed cache, built on the host from its positions (`metal_rope_rows`); `None` on
    // 1D-rope arches, whose rope reads the positions.
    let metal_rope_rows_fn: proc_macro2::TokenStream = match model.mrope_section {
        Some([t, h, w]) => {
            let head_dim = u64::from(model.text.head_dim.get());
            let rot_dim_val = model
                .scalars
                .get("partial_rotary_factor")
                .copied()
                .filter(|&f| (f - 1.0).abs() > 1e-9)
                .map(|f| (f * head_dim as f64).round() as u32)
                .unwrap_or(head_dim as u32);
            let rope_theta_val: f64 = model.text.rope_theta.get();
            let rope_theta_lit = proc_macro2::Literal::f64_unsuffixed(rope_theta_val);
            let t_lit = proc_macro2::Literal::u32_unsuffixed(t);
            let h_lit = proc_macro2::Literal::u32_unsuffixed(h);
            let w_lit = proc_macro2::Literal::u32_unsuffixed(w);
            let rot_dim_lit2 = proc_macro2::Literal::u32_unsuffixed(rot_dim_val);
            quote! {
                /// The rotary cos/sin rows a forward of `n` tokens at `positions` (`[n]`, or MRoPE's
                /// `[3, n]`) reads in place of the position-indexed cache — its positions then each
                /// row's own index.
                #[cfg(scratchy_target = "metal")]
                pub fn metal_rope_rows(
                    positions: &[u32],
                    n: usize,
                ) -> ::core::option::Option<::std::vec::Vec<u8>> {
                    ::core::option::Option::Some(
                        ::scratchy_target_metal::interpreter::metal::build_mrope_cos_sin_override(
                            positions,
                            n,
                            #rot_dim_lit2 as usize,
                            #rope_theta_lit,
                            [#t_lit, #h_lit, #w_lit],
                            <Weights as ::scratchy_forward_compiler::CanonicalParams>::FACTS.metal_dtype(),
                        ),
                    )
                }
            }
        }
        None => quote! {
            /// None: this model's rope reads the positions.
            #[cfg(scratchy_target = "metal")]
            pub fn metal_rope_rows(
                _positions: &[u32],
                _n: usize,
            ) -> ::core::option::Option<::std::vec::Vec<u8>> {
                ::core::option::Option::None
            }
        },
    };
    let mrope_runtime_block = quote! {
        let mut __mrope_ident_vec: ::std::vec::Vec<u32> = ::std::vec::Vec::new();
        let __rope_rows = if !ctx.positions.as_raw().raw_ptr().is_null() {
            let __pos_all = ::std::slice::from_raw_parts(
                ctx.positions.as_raw().raw_ptr() as *const u32,
                ctx.positions.as_raw().numel(),
            );
            metal_rope_rows(__pos_all, n)
        } else {
            ::core::option::Option::None
        };
        let (positions, mrope_cos_sin): (&[u32], ::core::option::Option<&[u8]>) =
            match &__rope_rows {
                ::core::option::Option::Some(rows) => {
                    __mrope_ident_vec.extend(0..n as u32);
                    (__mrope_ident_vec.as_slice(), ::core::option::Option::Some(rows.as_slice()))
                }
                ::core::option::Option::None => (positions, ::core::option::Option::None),
            };
    };

    {
        let table = metal_tape_cmds
            .into_tokens(&resolver.typed, &resolver.named, &dims.weight_markers)
            .unwrap_or_else(|e| {
                panic!(
                    "[metal static tape] {}: command table: {e}",
                    model.source_stem
                )
            });
        metal_arena_bytes_statics.push(table);
    }
    {
        let mc = &model_facts;
        metal_arena_bytes_statics.push(metal_tq_codebook_tokens(mc));
        // A multi-token-prediction head (its body reads the target's hidden rows) chains its
        // passes on the device.
        let head = reads_target_hidden.then(|| {
            let drafts = model.bounds.get("spec_drafts").copied();
            let drafts =
                u32::try_from(drafts.expect("an MTP head's spec_drafts")).expect("drafts fit u32");
            let hidden = u32::try_from(target_hidden_width).expect("hidden width fits u32");
            (drafts, hidden)
        });
        let vocab_suffix = metal_vocab_suffix(program, fuf, model, mc);
        metal_arena_bytes_statics.push(metal_off_tape_tokens(logits_width, mc, head, vocab_suffix));
    }
    let metal_emission = quote! {
        #(#metal_arena_bytes_statics)*

        /// Per-canonical tape_index plan for the Metal pool. One row per
        /// `num_tokens` point per routed fold band (`FoldBand`), ordered ascending, a
        /// bucket's band rows adjacent. A pool keeps one band's rows (`pick_rung`), so its
        /// bucket index is no index into this table: a forward's slots come from its bucket's row
        /// (`Buckets::row`), the one the door chose.
        #[cfg(scratchy_target = "metal")]
        pub static METAL_BUCKETS:
            &[::scratchy_target_metal::interpreter::metal::MetalBucketSpec]
            = &[
                #(#metal_bucket_entries)*
            ];

        /// [`METAL_BUCKETS`]' baked rungs, which the worker picks the rung a pool runs on from
        /// before the pool exists: the KV pool's block-table width, the memory budget.
        #[cfg(scratchy_target = "metal")]
        pub static METAL_RUNGS: ::scratchy_target_metal::interpreter::metal::MetalRungs =
            ::scratchy_target_metal::interpreter::metal::MetalRungs {
                buckets: METAL_BUCKETS,
                tq: ::scratchy_target_metal::interpreter::metal::TqGroup::of(
                    <Weights as ::scratchy_forward_compiler::CanonicalParams>::FACTS,
                ),
                // Checked against the buckets as this static compiles: every slot of every bucket
                // ranked once, and no bucket's final hidden states among the hosting ranks.
                ranks: __arena::RANKED.ranks(METAL_BUCKETS),
                door: ::scratchy_target_metal::interpreter::metal::bucket_row::<__buckets::Bucket>,
            };

        /// Largest `num_tokens` tape_index across [`METAL_BUCKETS`]. The
        /// metal forward body's `RuntimeFactory` allocates per-worker
        /// runtime buffers (input_ids/positions/slot_mapping/...) at
        /// `METAL_MAX_BUCKET_M * sizeof(u32)`.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_MAX_BUCKET_M: u32 = #max_bucket_m_lit;

        /// Byte size of the vision 2D-RoPE `freqs` runtime buffer
        /// (f32 `[max_bucket_m, vision_head_dim/2]`). 0 on non-vision
        /// arches (`alloc` floors to 16 bytes).
        #[cfg(scratchy_target = "metal")]
        pub const METAL_VISION_FREQS_BYTES: u64 = #vision_freqs_bytes_lit;

        /// Byte size of the vision `pixels` runtime buffer (model dtype
        /// `[max_bucket_m, vision_in_features]`). 0 on non-vision arches.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_VISION_PIXELS_BYTES: u64 = #vision_pixels_bytes_lit;

        /// Byte size of the vision `pos_embeds` runtime buffer (model
        /// dtype `[max_bucket_m, vision_embed_dim]`). 0 on non-vision
        /// arches and on towers without a learned positional embedding.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_VISION_POSEMB_BYTES: u64 = #vision_posemb_bytes_lit;

        /// Byte size of the `target_hidden` runtime buffer (model dtype
        /// `[max_bucket_m, hidden]`). 0 on every arch but an MTP head.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_TARGET_HIDDEN_BYTES: u64 = #target_hidden_bytes_lit;

        /// Byte size of the rows a target captures for its DFlash2 drafter (model dtype
        /// `[max_bucket_m, slots · hidden]`). 0 on a model no drafter reads.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_DRAFT_CAPTURE_BYTES: u64 = #draft_capture_bytes_lit;

        /// A DFlash2 drafter's block: its drafts and their anchor.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_DRAFT_ROWS: u64 = #draft_rows_lit;

        /// Byte size of the `mm_embeds` splice buffer (bf16
        /// `[max_bucket_m, hidden]`) and the `mm_dst_rows` buffer (u32
        /// `[max_bucket_m]`). 0 on arches without the multimodal splice.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_MM_EMBEDS_BYTES: u64 = #mm_embeds_bytes_lit;
        #[cfg(scratchy_target = "metal")]
        pub const METAL_MM_DST_ROWS_BYTES: u64 = #mm_dst_rows_bytes_lit;

        /// Byte size of the MRoPE cos/sin override buffer (model dtype
        /// `[max_bucket_m, rot_dim]`). 0 on 1D-rope arches (the rope
        /// kernel keeps the static cos/sin cache; `alloc` floors to 16
        /// bytes and the buffer is never bound).
        #[cfg(scratchy_target = "metal")]
        pub const METAL_MROPE_COS_SIN_BYTES: u64 = #mrope_cos_sin_bytes_lit;

        /// Width of one row of the terminal arena slot: `vocab_size`
        /// on decoder layouts (the lm_head Gemm's output),
        /// `hidden_size` on encoder layouts (no lm_head — the terminal
        /// IS the hidden state). The metal forward shapes its returned
        /// `OwnedTensor` as `[num_tokens, METAL_VOCAB_SIZE]` f16.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_VOCAB_SIZE: u64 = #vocab_size_lit;

        /// Build a [`MetalWorkerPool`] for this canonical. Thin
        /// wrapper over [`MetalWorkerPool::for_buckets`] that threads
        /// the per-canonical [`METAL_BUCKETS`] static so callers don't
        /// have to construct the tape_index plan by hand. The arena layout
        /// is derived from each tape_index's `arena_bytes` field — taking
        /// the elementwise max so the single per-worker arena fits the
        /// largest activation across every tape_index.
        ///
        /// `weights` is borrowed once: the pool resolves the tapes'
        /// model sources at construction and keeps only their
        /// buffers, so it can live as a field on the `Weights` struct
        /// without an `Arc`-cycle.
        ///
        /// [`MetalWorkerPool`]: ::scratchy_target_metal::interpreter::metal::MetalWorkerPool
        /// [`MetalWorkerPool::for_buckets`]: ::scratchy_target_metal::interpreter::metal::MetalWorkerPool::for_buckets
        #[cfg(scratchy_target = "metal")]
        pub fn metal_pool(
            device: ::std::sync::Arc<
                ::scratchy_target_metal::interpreter::metal::__re::Device,
            >,
            weights: &Weights,
            allocator: ::std::sync::Arc<crate::__gpu::MetalAllocator>,
            runtime_factory: ::scratchy_target_metal::interpreter::metal::RuntimeFactory,
            max_workers: usize,
            // Runtime per-sequence block-table capacity
            // (`KvCachePool::max_blocks_per_seq`); picks the KV cap rung.
            block_cap: usize,
            addressing: ::scratchy_target_metal::tape::lowered::KvAddressing,
        ) -> ::core::result::Result<
            ::scratchy_target_metal::interpreter::metal::MetalWorkerPool<Weights>,
            ::scratchy_target_metal::interpreter::metal::PoolBuildError,
        > {
            ::scratchy_target_metal::interpreter::metal::MetalWorkerPool::for_buckets(
                device,
                weights,
                allocator,
                &METAL_RUNGS,
                runtime_factory,
                max_workers,
                // Standalone factory: no device-budget context here, so keep
                // all buckets. The lazy-init path below passes the real cap.
                None,
                block_cap,
                addressing,
                // Nor the serving engine's batch: every bucket row's decode partials.
                None,
                // Its arena is its own.
                ::scratchy_target_metal::interpreter::metal::ArenaShare::Own,
                // Its callers read the logits on the host.
                ::scratchy_target_metal::interpreter::metal::TerminalReads::Host,
            )
        }

        /// The model's resident metal pool, built on its first forward.
        ///
        /// The factory closure captures (a) the metal device handle
        /// for runtime-buffer allocation, (b) the KV pool's block-table
        /// row stride and the serving batch, which size the block tables,
        /// (c) Arc-handle clones of the per-layer KV
        /// buffers from `ctx.kv_cache` so worker spawns inherit
        /// them without re-allocation. The closure is invoked
        /// once per worker spawn — the pool starts at size 1 so
        /// the first forward triggers the only factory invocation
        /// in single-worker configs. Every forward entry point
        /// reaches the pool through here, whichever runs first.
        #[cfg(scratchy_target = "metal")]
        fn resident_metal_pool<'w>(
            wm: &'w Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &crate::__gpu::GpuDevice,
        ) -> &'w ::scratchy_target_metal::interpreter::metal::MetalWorkerPool<Weights> {
            use ::scratchy_forward_compiler::CanonicalParams as _;
            use ::scratchy_target_metal::interpreter::metal::__re::{Buffer, MTLResourceOptions};

            wm.metal_pool.get_or_init(|| {
                // The RoPE tables the pool binds, for the positions this KV cache's block table
                // reaches.
                wm.build_rope_tables(ctx, device)
                    .expect("Weights::build_rope_tables: RoPE table build failed");
                let num_layers = ctx.kv_cache.num_layers;
                // Reactive (chunked) KV pool: bind the per-layer
                // chunk-address TABLE buffers (device uint64 arrays of
                // chunk gpuAddresses), not the cache data. The KV
                // kernels deref `table[block_id / BLOCKS_PER_CHUNK]`.
                // The table buffer identity is stable across the
                // pool's life (chunk-set growth edits its contents, not
                // its binding), so baking its gpuAddress once is sound.
                let kv_k: ::std::vec::Vec<Buffer> = (0..num_layers)
                    .map(|l| ctx.kv_cache.k_chunk_table_mem(l).buffer().clone())
                    .collect();
                let kv_v: ::std::vec::Vec<Buffer> = (0..num_layers)
                    .map(|l| ctx.kv_cache.v_chunk_table_mem(l).buffer().clone())
                    .collect();
                // KV-cache groups (vLLM hybrid layout). Uniform models have a
                // single group with an all-zero layer→group map (byte-identical
                // to the pre-hybrid single block-table path); gemma4 SWA reports
                // its real group count + per-layer mapping through the pool.
                let num_kv_groups: usize = ctx.kv_cache.num_kv_groups();
                let kv_layer_to_group: ::std::vec::Vec<u32> =
                    ctx.kv_cache.layer_to_group_u32();
                // GDN (Gated-DeltaNet) persistent state buffers, captured per
                // (global) layer from `ctx.gdn_state` for hybrid arches; empty
                // for non-hybrid. Non-linear layers reuse the first linear
                // layer's buffer as a never-bound placeholder (the
                // GatedDeltaNet lowering only emits GdnConvState/GdnSsmState on
                // linear layers, so the placeholder is never read).
                let (gdn_conv, gdn_ssm): (::std::vec::Vec<Buffer>, ::std::vec::Vec<Buffer>) =
                    match ctx.gdn_state {
                        ::core::option::Option::Some(gp) => {
                            match (0..gp.num_layers).find(|&l| gp.is_linear(l)) {
                                ::core::option::Option::Some(f0) => {
                                    let fc = gp.conv_layer_mem(f0).buffer().clone();
                                    let fs = gp.ssm_layer_mem(f0).buffer().clone();
                                    let conv = (0..gp.num_layers)
                                        .map(|l| if gp.is_linear(l) {
                                            gp.conv_layer_mem(l).buffer().clone()
                                        } else { fc.clone() })
                                        .collect();
                                    let ssm = (0..gp.num_layers)
                                        .map(|l| if gp.is_linear(l) {
                                            gp.ssm_layer_mem(l).buffer().clone()
                                        } else { fs.clone() })
                                        .collect();
                                    (conv, ssm)
                                }
                                ::core::option::Option::None => {
                                    (::std::vec::Vec::new(), ::std::vec::Vec::new())
                                }
                            }
                        }
                        ::core::option::Option::None => {
                            (::std::vec::Vec::new(), ::std::vec::Vec::new())
                        }
                    };
                // Its draft lanes, the same way (`GdnStatePool::lanes`). A pool without lanes — a
                // model that drafts nothing, or one whose run drafts with nothing (no head found,
                // `--num-speculative-tokens 0`) — runs only plain steps (its slot allocator builds
                // no other: `GdnStepError::Drafts`), which never touch a lane, so a drafting
                // model's tapes bind the layer's own state buffers there, never read.
                let (gdn_lanes_conv, gdn_lanes_ssm): (
                    ::std::vec::Vec<Buffer>,
                    ::std::vec::Vec<Buffer>,
                ) = match ctx.gdn_state {
                    ::core::option::Option::Some(gp) => {
                        let first = (0..gp.num_layers)
                            .find_map(|l| gp.conv_lanes_mem(l).zip(gp.ssm_lanes_mem(l)));
                        match first {
                            ::core::option::Option::Some((fc, fs)) => (0..gp.num_layers)
                                .map(|l| {
                                    let (c, s) = gp
                                        .conv_lanes_mem(l)
                                        .zip(gp.ssm_lanes_mem(l))
                                        .unwrap_or((fc, fs));
                                    (c.buffer().clone(), s.buffer().clone())
                                })
                                .unzip(),
                            ::core::option::Option::None => (gdn_conv.clone(), gdn_ssm.clone()),
                        }
                    }
                    ::core::option::Option::None => {
                        (::std::vec::Vec::new(), ::std::vec::Vec::new())
                    }
                };
                // A DFlash2 drafter's rings, the pool `ctx` lends (the sibling of `gdn_state`):
                // `Some` exactly for a drafter.
                let __draft_rings: ::core::option::Option<(
                    ::std::vec::Vec<Buffer>,
                    ::std::vec::Vec<Buffer>,
                    Buffer,
                )> = ctx.draft_ring.map(|pool| {
                    let (keys, values) = (0..pool.num_layers())
                        .map(|l| {
                            let (k, v) = pool.layer(l);
                            (k.buffer().clone(), v.buffer().clone())
                        })
                        .unzip();
                    (keys, values, pool.windows().buffer().clone())
                });
                // TurboQuant: captured (Copy) so the 'static factory closure can
                // provision the packed stores + scratch per worker for a model
                // whose `KV_CODEC` is TurboQuant. UNIFORM arches (num_kv_groups
                // == 1): every layer is "global", compressed at the base
                // geometry. HYBRID/SWA arches (gemma4: num_kv_groups > 1)
                // compress ONLY the GLOBAL (full-context, group 0) layers at the
                // GLOBAL geometry; the sliding layers stay fp16.
                // `is_global[L]` = (group_of(L) == 0): group 0 is always the
                // full-context group (see `compute_hybrid_kv_layout`); uniform →
                // all-zero map → all-true. A recurrent hybrid's layers that hold
                // no KV (`FACTS.kv_layers()`) get no store.
                let __tq_nb = ctx.kv_cache.num_blocks;
                let __tq_kv_layers = <Weights as ::scratchy_forward_compiler::CanonicalParams>::FACTS
                    .kv_layers()
                    .map(::scratchy_forward_compiler::model_facts::KvLayers::get);
                // The attention's global class: the full layers' pages and kv heads. `None` for a
                // model without attention, which has neither.
                let __global = <Weights as ::scratchy_forward_compiler::CanonicalParams>::FACTS
                    .attention()
                    .map(|a| a.global());
                let __tq_is_global: ::std::vec::Vec<bool> = ctx
                    .kv_cache
                    .layer_to_group_u32()
                    .iter()
                    .zip(0u32..)
                    .map(|(&g, l)| g == 0 && __tq_kv_layers.is_none_or(|kv| kv.contains(&l)))
                    .collect();
                // The stores register with the device's allocator, whose KV growth maps them.
                let __tq_allocator = device.allocator.clone();
                // A step's block tables: a row per sequence, at most the serving batch (every
                // row of the largest bucket when the worker has not stashed one), each the KV
                // pool's row stride — the KV cap rung it runs on, which the host pads every row
                // to.
                let __bt_stride = ctx.kv_cache.max_blocks_per_seq as u64;
                let __bt_rows = device.max_decode_rows.map_or(METAL_MAX_BUCKET_M, |r| {
                    r.get().min(METAL_MAX_BUCKET_M)
                }) as u64;
                let factory: ::scratchy_target_metal::interpreter::metal::RuntimeFactory =
                    ::scratchy_target_metal::interpreter::metal::RuntimeFactory::new(move |dev, host| {
                        let max_m = METAL_MAX_BUCKET_M as u64;
                        let alloc = |bytes: u64| {
                            use ::scratchy_target_metal::interpreter::metal::__re::MTLDevice as _;
                            dev.newBufferWithLength_options(
                                bytes.max(16) as usize,
                                MTLResourceOptions::StorageModeShared,
                            )
                            .expect("newBufferWithLength_options returned nil")
                        };
                        // Provision at the GLOBAL geometry: uniform arches have
                        // GLOBAL_* == base, so this is identical to the base-geometry
                        // call (every layer global). gemma4 provisions GLOBAL-sized
                        // packed/norms/scratch (head_dim 512, NUM_GLOBAL_KV_HEADS,
                        // GLOBAL_BLOCK_SIZE) for the group-0 layers only.
                        let __tq_prov = METAL_TQ_CODEBOOK.as_ref().map(|codebook| {
                            // A head reading its target's KV binds its target's packed stores.
                            if #reads_target_kv {
                                return host.tq.clone().expect(
                                    "a head reading its target's TurboQuant KV builds its pool in \
                                     its target's, which shares it",
                                );
                            }
                            let global = __global
                                .expect("ModelFacts::new: a TurboQuant KV codec is an attention's");
                            ::scratchy_target_metal::turboquant::build_tq_provision(
                                &__tq_allocator,
                                ::scratchy_target_metal::turboquant::TqPool {
                                    is_global: &__tq_is_global,
                                    num_blocks: __tq_nb,
                                    block_size: global.page().get() as usize,
                                    num_kv_heads: global.kv_heads().get() as usize,
                                    blocks_per_chunk: ::scratchy_target_metal::interpreter::metal::BLOCKS_PER_CHUNK as usize,
                                },
                                codebook,
                                host.tq_scratch.as_ref(),
                            )
                        });
                        ::scratchy_target_metal::interpreter::metal::RuntimeBindings {
                            input_ids: alloc(max_m * 4),
                            positions: alloc(max_m * 4),
                            // One slot_mapping + block_table per KV-cache group
                            // (vLLM hybrid layout). Uniform models have one
                            // group; gemma4 SWA has full + N sliding. The worker
                            // sizes the group count + fills layer_to_group from
                            // the model's HybridKvLayout at runtime (stage 6);
                            // the default single group is byte-identical to the
                            // pre-hybrid path.
                            slot_mappings: (0..num_kv_groups).map(|_| alloc(max_m * 4)).collect(),
                            cu_seqlens_q: alloc((max_m + 1) * 4),
                            seq_used_k: alloc(max_m * 4),
                            // Per-token span label for block-diagonal span
                            // attention, for a step of one sequence: as many as
                            // its full-attention row reaches, the stride's blocks of
                            // `GLOBAL_BLOCK_SIZE` tokens. All-zero unless the
                            // request carries Relocatable spans.
                            span_ids: alloc(
                                __global.map_or(0, |g| __bt_stride * g.page().get() as u64 * 4),
                            ),
                            block_tables: (0..num_kv_groups)
                                .map(|_| alloc(__bt_rows * __bt_stride * 4))
                                .collect(),
                            layer_to_group: kv_layer_to_group.clone(),
                            // TurboQuant: when provisioned, the GLOBAL (group-0)
                            // layers' KV points at the one fp16 scratch (the
                            // injected per-layer dequant/quantize tape ops fill/drain
                            // it); SLIDING layers keep the normal fp16 pool. For
                            // uniform arches every layer is global → all scratch
                            // (byte-identical to the prior all-layers override).
                            // Without tq, every layer uses the fp16 pool.
                            kv_cache_k: __tq_prov
                                .as_ref()
                                .map(|p| {
                                    (0..num_layers)
                                        .map(|l| if __tq_is_global[l] {
                                            p.scratch.k_table.clone()
                                        } else {
                                            kv_k[l].clone()
                                        })
                                        .collect()
                                })
                                .unwrap_or_else(|| kv_k.clone()),
                            kv_cache_v: __tq_prov
                                .as_ref()
                                .map(|p| {
                                    (0..num_layers)
                                        .map(|l| if __tq_is_global[l] {
                                            p.scratch.v_table.clone()
                                        } else {
                                            kv_v[l].clone()
                                        })
                                        .collect()
                                })
                                .unwrap_or_else(|| kv_v.clone()),
                            // Spans rope-on-read per-layer flag mirror.
                            // 16-byte placeholder here; the worker sizes
                            // it to the pool's num_blocks and binds it
                            // only on W::ROPE_ON_READ arches (the metal
                            // lowering omits the binding otherwise, so
                            // this is never read on the non-spans path).
                            block_unrotated_flags: (0..num_layers)
                                .map(|_| alloc(16))
                                .collect(),
                            tq: __tq_prov,
                            // 4 bytes — worker writes the forward's
                            // `num_tokens` here before dispatch: the MoE
                            // sort counts the step's live pairs.
                            num_tokens_u32: alloc(4),
                            // 4 bytes — worker writes the sampled rows'
                            // count here before dispatch so the lm_head
                            // slice's gather / scatter know how many rows
                            // to move.
                            num_sample_rows_u32: alloc(4),
                            // [max_m] u32 — worker writes the sampled
                            // rows here before dispatch.
                            sample_indices: alloc(max_m * 4),
                            // GDN persistent state (per-layer) + per-forward
                            // indices/fresh flags (Shared, overwritten each
                            // forward). Index buffers sized to the max bucket
                            // (num_seqs <= num_tokens <= max_m): a 24-byte
                            // `GdnSeqSlots` record a sequence, a u32 step code.
                            gdn_state_conv: gdn_conv.clone(),
                            gdn_state_ssm: gdn_ssm.clone(),
                            gdn_lanes_conv: gdn_lanes_conv.clone(),
                            gdn_lanes_ssm: gdn_lanes_ssm.clone(),
                            gdn_state_indices: alloc(
                                max_m
                                    * ::std::mem::size_of::<
                                        ::scratchy_target_metal::gdn_state::GdnSeqSlots,
                                    >() as u64,
                            ),
                            gdn_is_fresh: alloc(max_m * 4),
                            // Vision externs: sized from the baked
                            // METAL_VISION_*_BYTES consts (16-byte floor
                            // on non-vision arches). Overwritten per
                            // forward by `write_runtime_inputs`.
                            vision_rope_freqs: alloc(METAL_VISION_FREQS_BYTES),
                            pixels: alloc(METAL_VISION_PIXELS_BYTES),
                            vision_pos_embeds: alloc(METAL_VISION_POSEMB_BYTES),
                            target_hidden: alloc(METAL_TARGET_HIDDEN_BYTES),
                            mm_embeds: alloc(METAL_MM_EMBEDS_BYTES),
                            mm_dst_rows: alloc(METAL_MM_DST_ROWS_BYTES),
                            mrope_cos_sin: alloc(METAL_MROPE_COS_SIN_BYTES),
                            // Qwen2.5-VL windowed-attention externs:
                            // i32/u32 rows bounded by the max bucket.
                            // Tiny — sized unconditionally.
                            vision_cu_seqlens_full: alloc(
                                (METAL_MAX_BUCKET_M as u64 + 8) * 4,
                            ),
                            vision_cu_seqlens_window: alloc(
                                (METAL_MAX_BUCKET_M as u64 + 8) * 4,
                            ),
                            vision_window_index: alloc(
                                (METAL_MAX_BUCKET_M as u64 + 8) * 4,
                            ),
                            vision_reverse_indices: alloc(
                                (METAL_MAX_BUCKET_M as u64 + 8) * 4,
                            ),
                            vision_position_ids: alloc(
                                (METAL_MAX_BUCKET_M as u64 + 8) * 4,
                            ),
                            // The rows a target captures for its DFlash2 drafter: every row of
                            // the largest bucket, a block a captured layer (16-byte floor on a
                            // model no drafter reads).
                            draft_capture: alloc(METAL_DRAFT_CAPTURE_BYTES),
                            // A DFlash2 drafter's buffers: its rings, the pool `ctx` lends (kept
                            // across forwards, pinned by it), and a cycle's per-lane inputs and
                            // outputs, sized for the largest bucket's lanes.
                            draft: __draft_rings.as_ref().map(|(keys, values, windows)| {
                                let lanes = max_m / METAL_DRAFT_ROWS.max(1);
                                ::scratchy_target_metal::interpreter::metal::DraftRuntimeBuffers {
                                    ring_keys: keys.clone(),
                                    ring_values: values.clone(),
                                    ring_slots: alloc(lanes * 4),
                                    windows: windows.clone(),
                                    retained: alloc(lanes * 4),
                                    lane_params: alloc(
                                        lanes
                                            * ::std::mem::size_of::<
                                                ::scratchy_target_metal::dflash2::DraftLaneParams,
                                            >() as u64,
                                    ),
                                    outputs: ::scratchy_target_metal::dflash2::DraftOutput::ALL
                                        .map(|r| alloc(r.bytes(lanes))),
                                }
                            }),
                            written: ::std::default::Default::default(),
                        }
                    });
                ::scratchy_target_metal::interpreter::metal::MetalWorkerPool::for_buckets(
                    device.device.clone(),
                    wm,
                    device.allocator.clone(),
                    &METAL_RUNGS,
                    factory,
                    1,
                    // Target-reactive cap stashed on the device by the worker
                    // after `determine_available_memory`. Prunes the compiled
                    // ladder so the colored arena fits the KV budget.
                    device.metal_bucket_max_m,
                    // Runtime per-sequence block-table capacity, read off the KV
                    // pool: it picks the KV cap rung — none for a model reading no KV.
                    #pool_block_cap,
                    device.kv_addressing,
                    // The serving engine's batch, stashed by the worker: the decode split's
                    // partials are reserved for that many rows.
                    device.max_decode_rows,
                    // How its arena is shared with that of a model whose forwards never overlap
                    // its own: an MTP target holds its head's (a fact of the target, set before
                    // its first forward), a head runs in its target's (its device's).
                    ::scratchy_target_metal::interpreter::metal::ArenaShare::of(
                        wm.metal_guest.get(),
                        device.host.as_deref(),
                    ),
                    // Whether its owner reads the logits on the host (a draft model's seed).
                    device.terminal_reads,
                )
                .expect("MetalWorkerPool::for_buckets: pool init failed")
            })
        }

        /// Per-canonical metal forward dispatch. Lazy-inits
        /// `weights.metal_pool` on the first call (factory closure
        /// captures the per-layer `metal::Buffer` Arc-handles from
        /// `ctx.kv_cache` and the `MAX_BLOCKS_PER_SEQ` block-table
        /// stride from `<Weights as CanonicalParams>`); on every call
        /// reads the runtime input slices off the host-visible
        /// `ctx.<input>` `TensorView`s — under metal those raw_ptrs
        /// are `metal::Buffer.contents()` so the slice borrow lives
        /// as long as the call — hands them to
        /// `MetalWorkerPool::forward`, and copies the tape_index's
        /// terminal arena slot out as a fresh `OwnedTensor` of
        /// `[num_tokens, vocab_size]` f16 logits.
        #[cfg(scratchy_target = "metal")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward(
            wm: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
        ) -> crate::__gpu::OwnedTensor {
            unsafe { forward_with_metal_followup(wm, ctx, device, num_tokens, None) }
                .unwrap_or_else(|e| ::core::panic!("{e}"))
        }

        /// The runtime inputs `ctx` carries for a forward of `num_tokens` tokens, handed to `run`:
        /// they borrow the host slices `ctx`'s views point at and the rows built here. Refused
        /// when they name no rows to sample (`ForwardError::EmptySequence`).
        ///
        /// # Safety
        /// Every non-null view in `ctx` points at host-visible memory of its numel.
        #[cfg(scratchy_target = "metal")]
        unsafe fn with_forward_inputs<R>(
            ctx: &crate::__gpu::ForwardCtx,
            num_tokens: u64,
            run: impl FnOnce(&::scratchy_target_metal::interpreter::metal::ForwardInputs<'_>) -> R,
        ) -> ::core::result::Result<R, ::scratchy_target_metal::interpreter::metal::ForwardError> {
            // ── Read host-visible input slices off ctx ────────────
            //
            // Under metal, every `TensorView` in `ctx` resolves to a
            // pointer inside a `StorageModeShared` `MTLBuffer.contents()`,
            // so reading as `&[u32]` is a direct CPU borrow. The
            // shared buffer keeps backing the slice until the worker
            // memcopies through `write_runtime_inputs`.
            let n = num_tokens as usize;
            // Vision towers carry no token ids / positions — the tape
            // consumes pixels / freqs / cu_seqlens instead, so the macro's
            // ctx leaves these as `null_view`. `from_raw_parts(null, n)` is
            // UB even for an unused read (and `write_runtime_inputs` only
            // copies `len` bytes), so hand the worker an empty slice when
            // the pointer is null — same null-guard shape as the optional
            // decoder inputs (slot_mapping / cu_seqlens_q / …) just below.
            let input_ids: &[u32] = if ctx.input_ids.as_raw().raw_ptr().is_null() {
                &[]
            } else {
                ::std::slice::from_raw_parts(
                    ctx.input_ids.as_raw().raw_ptr() as *const u32,
                    n,
                )
            };
            let positions: &[u32] = if ctx.positions.as_raw().raw_ptr().is_null() {
                &[]
            } else {
                ::std::slice::from_raw_parts(
                    ctx.positions.as_raw().raw_ptr() as *const u32,
                    n,
                )
            };
            let slot_mapping = if !ctx.slot_mapping.as_raw().raw_ptr().is_null() {
                ::std::option::Option::Some(::std::slice::from_raw_parts(
                    ctx.slot_mapping.as_raw().raw_ptr() as *const u32,
                    n,
                ))
            } else {
                ::std::option::Option::None
            };
            let cu_seqlens_q = if !ctx.cu_seqlens_q.as_raw().raw_ptr().is_null() {
                let cu_n = ctx.cu_seqlens_q.as_raw().numel();
                ::std::option::Option::Some(::std::slice::from_raw_parts(
                    ctx.cu_seqlens_q.as_raw().raw_ptr() as *const u32,
                    cu_n,
                ))
            } else {
                ::std::option::Option::None
            };
            let seq_used_k = if !ctx.seqused_k.as_raw().raw_ptr().is_null() {
                let su_n = ctx.seqused_k.as_raw().numel();
                ::std::option::Option::Some(::std::slice::from_raw_parts(
                    ctx.seqused_k.as_raw().raw_ptr() as *const u32,
                    su_n,
                ))
            } else {
                ::std::option::Option::None
            };
            let block_table = if !ctx.block_table.as_raw().raw_ptr().is_null() {
                let bt_n = ctx.block_table.as_raw().numel();
                ::std::option::Option::Some(::std::slice::from_raw_parts(
                    ctx.block_table.as_raw().raw_ptr() as *const u32,
                    bt_n,
                ))
            } else {
                ::std::option::Option::None
            };
            // Sliding KV-cache GROUPS (gemma4 SWA): per sliding group, the
            // per-token slot_mapping + per-block block table. Empty on non-SWA
            // models. Group `s` here is KV-cache group `s + 1` (group 0 = full).
            let sliding_slot_mappings: ::std::vec::Vec<&[u32]> = ctx
                .sliding_slot_mappings
                .iter()
                .map(|v| ::std::slice::from_raw_parts(v.as_raw().raw_ptr() as *const u32, n))
                .collect();
            let sliding_block_tables: ::std::vec::Vec<&[u32]> = ctx
                .sliding_block_tables
                .iter()
                .map(|v| {
                    let sbt_n = v.as_raw().numel();
                    ::std::slice::from_raw_parts(v.as_raw().raw_ptr() as *const u32, sbt_n)
                })
                .collect();

            // The rows the forward samples (`None`: each sequence's last), resolved once here
            // for every reader of the inputs: the lm_head leaves their logits at the front of
            // its output, in order. A sequence of no rows has no last row: refused.
            let last_rows;
            let sampled_rows: &[u32] = match ctx.sampled_rows.as_ref() {
                ::std::option::Option::Some(view) => {
                    let raw = view.as_raw();
                    ::std::slice::from_raw_parts(raw.raw_ptr() as *const u32, raw.numel())
                }
                ::std::option::Option::None => {
                    last_rows = ::scratchy_target_metal::interpreter::metal::ForwardInputs::last_rows(
                        cu_seqlens_q,
                        num_tokens as u32,
                    )
                    .map_err(::scratchy_target_metal::interpreter::metal::ForwardError::EmptySequence)?;
                    last_rows.rows()
                }
            };

            // GDN per-forward indices (hybrid arches): read from ctx's
            // host-visible TensorViews (a `GdnSeqSlots` record — 6 i32s —
            // per seq, a step code per seq). `None` for non-hybrid arches
            // (ctx fields are None).
            let gdn_state_indices = ctx.gdn_state_indices.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const i32,
                    tv.as_raw().numel(),
                )
            });
            let gdn_is_fresh = ctx.gdn_is_fresh.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u32,
                    tv.as_raw().numel(),
                )
            });
            // Vision externs (vision towers): read as raw bytes from the
            // ctx TensorViews. `freqs` is f32, `pixels` is the model
            // dtype; the worker copies bytes verbatim. `None` for text.
            let vision_rope_freqs = ctx.vision_rope_freqs.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            let pixels = ctx.pixels.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            let pos_embeds = ctx.pos_embeds.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            let target_hidden = ctx.target_hidden.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            // Qwen2.5-VL windowed-attention externs (i32/u32 byte
            // views; `None` on non-windowed towers and text bodies).
            let vision_cu_seqlens_full = ctx.vision_cu_seqlens_full.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            let vision_cu_seqlens_window = ctx.vision_cu_seqlens_window.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            let vision_window_index = ctx.vision_window_index.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            let vision_reverse_indices = ctx.vision_reverse_indices.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            let vision_position_ids = ctx.vision_position_ids.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            // Multimodal splice: vision embeddings (bytes) + a per-source-
            // row destination map built from `embed_patches`. Text-only
            // batches leave `embed_patches` empty → both `None` (no-op).
            let mm_embeds = ctx.mm_embeds.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            // ALWAYS materialized (all-`u32::MAX` for text-only batches):
            // the splice command sits in every decoder tape, so the kernel
            // must read `MAX` (= skip) for non-image rows — an unwritten
            // placeholder buffer would scatter garbage into the residual.
            let __mm_dst_rows_vec: ::std::vec::Vec<u32> = {
                let mut __v = ::std::vec![u32::MAX; n];
                let mut __src: usize = 0;
                for __p in ctx.embed_patches {
                    for __k in 0..(__p.length as usize) {
                        if __src < __v.len() {
                            __v[__src] = __p.token_offset + __k as u32;
                        }
                        __src += 1;
                    }
                }
                __v
            };
            let mm_dst_rows = ::core::option::Option::Some(__mm_dst_rows_vec.as_slice());
            // MRoPE (Qwen3.5-VL) only: build the per-token cos/sin override
            // table + identity positions (shadows `positions`); a no-op
            // `let mrope_cos_sin = None;` on 1D-rope arches.
            #mrope_runtime_block
            let inputs = ::scratchy_target_metal::interpreter::metal::ForwardInputs {
                num_tokens: num_tokens as u32,
                input_ids,
                positions,
                // Per-KV-cache-group slot_mappings / block tables (vLLM hybrid
                // layout): group 0 = full, then the sliding group(s). On non-SWA
                // models `sliding_*` are `None`, so this is a single-element vec
                // == the pre-hybrid single table (byte-identical). Stage 6
                // widens `ForwardCtx` to carry all N sliding groups.
                slot_mappings: slot_mapping
                    .into_iter()
                    .chain(sliding_slot_mappings)
                    .collect(),
                cu_seqlens_q,
                seq_used_k,
                // Block-diagonal span labels — wired from the request's block
                // annotations via ForwardCtx; `None` until that path is
                // populated (buffer stays zeroed ⇒ span mask inert).
                span_ids: ctx.span_ids.as_deref(),
                block_tables: block_table
                    .into_iter()
                    .chain(sliding_block_tables)
                    .collect(),
                sampled_rows,
                gdn_state_indices,
                gdn_is_fresh,
                vision_rope_freqs,
                pixels,
                pos_embeds,
                target_hidden,
                vision_cu_seqlens_full,
                vision_cu_seqlens_window,
                vision_window_index,
                vision_reverse_indices,
                vision_position_ids,
                mm_embeds,
                mm_dst_rows,
                mrope_cos_sin,
                draft: ctx.draft,
                device_inputs: ctx.device_inputs,
                deferred: ctx.deferred,
                reads: ctx.reads,
            };
            ::core::result::Result::Ok(run(&inputs))
        }

        #metal_rope_rows_fn

        /// Same as [`forward`] but takes an optional encoder-tail hook
        /// that's invoked on the same MTL4 compute encoder used to
        /// encode the forward, AFTER the bucket dispatches and BEFORE
        /// `endEncoding`. Lets the caller (the executor's argmax
        /// dispatch, today) append its own dispatches onto the same
        /// CB so forward + tail share one commit and one host wait.
        #[cfg(scratchy_target = "metal")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_with_metal_followup(
            wm: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
            followup: ::core::option::Option<::scratchy_target_metal::model::MetalForwardFollowup<'_>>,
        ) -> ::core::result::Result<crate::__gpu::OwnedTensor, ::scratchy_target_metal::model::GpuExecutionFailed> {
            let pool = resident_metal_pool(wm, ctx, device);
            // A guest's slots are its host's, which may be sparse: the host reads none of them.
            assert!(
                ctx.hidden_out.is_none()
                    || pool.role() != ::scratchy_target_metal::interpreter::metal::ArenaRole::Guest,
                "a forward asked the host for hidden rows of a pool that runs in another's arena \
                 (an MTP head's)"
            );
            let n = num_tokens as usize;

            // ── Run forward + copy logits out ─────────────────────
            //
            // The pool picks the tape_index from `num_tokens`, runs
            // `executeCommandsInBuffer` (interleaved with MPS GEMMs
            // on <M4 hardware), waits for completion, then invokes
            // the closure with `&MetalWorker` + `bucket_idx`. The
            // tape_index's terminal arena slot holds the lm_head output —
            // the logits of the rows the step samples, at its front;
            // we wrap that buffer in a fresh `OwnedTensor` (Arc-
            // handle clone — no copy) so the caller can read
            // `[sampled rows, vocab_size]` f16 logits without taking
            // ownership of the worker's arena.
            with_forward_inputs(ctx, num_tokens, |inputs| {
                let rows = #terminal_rows;
                // Adapter: pool's tail receives `(encoder, worker, runtime, bucket)`
                // and returns `Result<(), ForwardError>`. The public hook
                // is `(encoder, FollowupOutputs) -> Result<(), String>`.
                // Resolve the outputs from the worker's arena, its runtime bindings and the
                // canonical's compile-time vocab; map error variants.
                let tail_adapter = followup.map(|f| {
                    move |enc: &::scratchy_target_metal::model::ProtocolObject<
                              dyn ::scratchy_target_metal::model::MTL4ComputeCommandEncoder,
                          >,
                          worker: &::scratchy_target_metal::interpreter::metal::MetalWorker<Weights>,
                          #captured_runtime: &::scratchy_target_metal::interpreter::metal::RuntimeBindings,
                          bucket: &__buckets::Bucket|
                          -> ::core::result::Result<
                        (),
                        ::scratchy_target_metal::interpreter::metal::ForwardError,
                    > {
                        let spec = ::scratchy_target_metal::interpreter::metal::Buckets::row(bucket);
                        let outputs = ::scratchy_target_metal::model::FollowupOutputs {
                            logits: &worker.arena[spec.terminal_slot as usize],
                            hidden: &worker.arena[spec.backbone_slot as usize],
                            // A DFlash2 head's target's captured rows.
                            captured: #captured_rows,
                            rows: rows as u32,
                            vocab: METAL_VOCAB_SIZE as u32,
                        };
                        f(enc, outputs).map_err(
                            ::scratchy_target_metal::interpreter::metal::ForwardError::Followup,
                        )
                    }
                });
                pool.forward_with_tail(
                    inputs,
                    |worker, bucket| {
                        let spec = ::scratchy_target_metal::interpreter::metal::Buckets::row(bucket);
                        use ::scratchy_target_metal::interpreter::metal::__re::MTLBuffer as _;
                        // The caller's rows of the final hidden states (an MTP head's next input),
                        // before the worker goes back to the pool and its arena is reused.
                        if let Some(h) = ctx.hidden_out {
                            let row_bytes =
                                <Weights as ::scratchy_forward_compiler::CanonicalParams>::FACTS.hidden().get() * 2;
                            let hidden = worker.arena[spec.backbone_slot as usize].contents();
                            let mut out = h.out.borrow_mut();
                            out.clear();
                            for &row in h.rows {
                                assert!((row as usize) < n, "hidden row {row} past the forward's {n} rows");
                                // SAFETY: the backbone slot holds `[n, HIDDEN_SIZE]` 2-byte rows (the
                                // bucket's arena is sized for at least `n`), and `row < n`.
                                out.extend_from_slice(unsafe {
                                    ::std::slice::from_raw_parts(
                                        (hidden.as_ptr() as *const u8).add(row as usize * row_bytes),
                                        row_bytes,
                                    )
                                });
                            }
                        }
                        let buf = worker.arena[spec.terminal_slot as usize].clone();
                        let vocab = METAL_VOCAB_SIZE as usize;
                        let shape = [rows, vocab];
                        let bytes = rows * vocab * 2; // bf16/f16 — both 2 bytes
                        let dtype = match <Weights as ::scratchy_forward_compiler::CanonicalParams>::FACTS.metal_dtype() {
                            ::scratchy_target_metal::interpreter::metal::MetalDtype::F16 =>
                                crate::__gpu::dtype::DType::F16,
                            ::scratchy_target_metal::interpreter::metal::MetalDtype::Bf16 =>
                                crate::__gpu::dtype::DType::BF16,
                            ::scratchy_target_metal::interpreter::metal::MetalDtype::Int4 =>
                                ::core::unreachable!("Int4 has no logits dtype"),
                        };
                        // Null where the logits are a sparse (private) slot the host cannot read:
                        // the GPU's argmax and sampler bind the buffer, and a pool whose owner
                        // reads them on the host keeps the slot shared (`TerminalReads::Host`).
                        let inner = crate::__gpu::tensor::GpuTensor::new(
                            ::scratchy_target_metal::sparse_arena::host_ptr(&buf),
                            &shape,
                            dtype,
                        );
                        crate::__gpu::owned_from_metal_buffer(inner, buf, bytes)
                    },
                    tail_adapter,
                )
            })
            .and_then(::core::convert::identity)
            // A command buffer the GPU failed (an out-of-memory, say) fails this step, not the
            // worker thread: the executor fails its requests and schedules again. Any other error
            // is an invariant the forward broke (no bucket fits, a block table of the wrong width,
            // KV not mapped), which still stops the server.
            .map_err(|e| match e {
                ::scratchy_target_metal::interpreter::metal::ForwardError::ExecutionFailed(_) => {
                    ::scratchy_target_metal::model::GpuExecutionFailed(e.to_string())
                }
                // Its `Display` names `MetalWorkerPool::forward` already.
                other => ::core::panic!("{other}"),
            })
        }

        /// One forward encoded onto `encoder` — another model's command buffer, after that model's
        /// forward, whose outputs `ctx.device_inputs` read — with its input writes at its head
        /// (`MetalWorkerPool::forward_onto`). Its worker stays out of the pool until the returned
        /// forward drops, once that command buffer is done.
        ///
        /// # Safety
        /// Same as [`forward`].
        #[cfg(scratchy_target = "metal")]
        pub unsafe fn forward_onto<'w>(
            wm: &'w Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
            encoder: &::scratchy_target_metal::model::ProtocolObject<
                dyn ::scratchy_target_metal::model::MTL4ComputeCommandEncoder,
            >,
        ) -> ::core::result::Result<
            ::std::boxed::Box<dyn ::scratchy_target_metal::model::MetalForwardOnto + 'w>,
            ::std::string::String,
        > {
            struct Onto<'p>(::scratchy_target_metal::interpreter::metal::Onto<'p, Weights>);
            impl ::scratchy_target_metal::model::MetalForwardOnto for Onto<'_> {
                unsafe fn then(
                    &mut self,
                    ctx: crate::__gpu::ForwardCtxHandle<'_>,
                    num_tokens: u64,
                    encoder: &::scratchy_target_metal::model::ProtocolObject<
                        dyn ::scratchy_target_metal::model::MTL4ComputeCommandEncoder,
                    >,
                ) -> ::core::result::Result<(), ::std::string::String> {
                    let ctx: &crate::__gpu::ForwardCtx = unsafe { ctx.as_ref() };
                    with_forward_inputs(ctx, num_tokens, |inputs| self.0.then(inputs, encoder))
                        .and_then(::core::convert::identity)
                        .map_err(|e| format!("Onto::then: {e}"))
                }

                fn hidden(
                    &self,
                ) -> &::scratchy_target_metal::model::ProtocolObject<
                    dyn ::scratchy_target_metal::model::MTLBuffer,
                > {
                    let row = ::scratchy_target_metal::interpreter::metal::Buckets::row(self.0.bucket());
                    &self.0.worker().arena[row.backbone_slot as usize]
                }

                fn logits(
                    &self,
                ) -> &::scratchy_target_metal::model::ProtocolObject<
                    dyn ::scratchy_target_metal::model::MTLBuffer,
                > {
                    let row = ::scratchy_target_metal::interpreter::metal::Buckets::row(self.0.bucket());
                    &self.0.worker().arena[row.terminal_slot as usize]
                }

                fn draft_selection(
                    &self,
                ) -> ::core::option::Option<::scratchy_target_metal::model::DraftSelection<'_>> {
                    use ::scratchy_target_metal::dflash2::DraftOutput as O;
                    (self.0.runtime().draft.as_ref()).map(|d| {
                        ::scratchy_target_metal::model::DraftSelection {
                            tokens: d.output(O::Tokens),
                            candidates: d.output(O::Candidates),
                            q_probs: d.output(O::QProbs),
                            confidence: d.output(O::Confidence),
                            verify_rows: d.output(O::VerifyRows),
                        }
                    })
                }

                fn vocab(&self) -> u32 {
                    METAL_VOCAB_SIZE as u32
                }
            }
            let pool = resident_metal_pool(wm, ctx, device);
            with_forward_inputs(ctx, num_tokens, |inputs| pool.forward_onto(inputs, encoder))
                .and_then(::core::convert::identity)
                .map(|onto| ::std::boxed::Box::new(Onto(onto))
                    as ::std::boxed::Box<dyn ::scratchy_target_metal::model::MetalForwardOnto + 'w>)
                .map_err(|e| format!("MetalWorkerPool::forward_onto: {e}"))
        }

        /// Phase 6 chain entry point. Opens ONE MTL4 cmdbuf on the
        /// pool's MTL4 queue and invokes `body` with a
        /// [`ChainStepHandle`] adapter + the worker's runtime + the
        /// chain encoder. Caller drives K bucket forwards + per-iter
        /// argmax + chain_advance dispatches onto the same encoder;
        /// pool owns CB lifecycle (one commit, one host wait for
        /// the entire chain). Falls back to a clear error string when
        /// the bucket pick fails for the iter shape.
        ///
        /// `num_tokens` is the shape of EACH forward iter (constant
        /// across iters because the K-step decode shape is fixed).
        ///
        /// [`ChainStepHandle`]: ::scratchy_target_metal::model::ChainStepHandle
        #[cfg(scratchy_target = "metal")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_chain_with_encoder(
            wm: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
            body: ::scratchy_target_metal::model::MetalChainBody<'_>,
        ) -> ::core::result::Result<(), ::std::string::String> {
            let pool = resident_metal_pool(wm, ctx, device);
            with_forward_inputs(ctx, num_tokens, |inputs| {
                // The bucket of iter 0's tokens, at the door. The chain's shape is constant across
                // iters, so every iter plays it, at its tokens.
                let bucket = pool
                    .bucket_for(inputs)
                    .map_err(|e| format!("MetalWorkerPool::bucket_for: {e}"))?;
                let terminal_slot =
                    ::scratchy_target_metal::interpreter::metal::Buckets::row(&bucket).terminal_slot
                        as usize;
                let vocab = METAL_VOCAB_SIZE as u32;

                pool.with_chain_encoder(
                    inputs,
                    |worker, runtime, enc| {
                        // Concrete adapter that satisfies the non-generic
                        // `ChainStepHandle` trait. Holds the worker
                        // borrow + the bucket; calls
                        // `worker.run_bucket_mtl4` to encode one iter onto
                        // the encoder.
                        struct Adapter<'a, W: ::scratchy_forward_compiler::CanonicalParams
                            + ::scratchy_target_metal::interpreter::metal::BakedBuckets>
                        {
                            worker: &'a ::scratchy_target_metal::interpreter::metal::MetalWorker<W>,
                            bucket: W::Bucket,
                            terminal_slot: usize,
                            vocab: u32,
                        }
                        impl<W: ::scratchy_forward_compiler::CanonicalParams
                            + ::scratchy_target_metal::interpreter::metal::BakedBuckets>
                            ::scratchy_target_metal::model::ChainStepHandle for Adapter<'_, W>
                        {
                            fn run_forward_step(
                                &self,
                                encoder: &::scratchy_target_metal::model::ProtocolObject<
                                    dyn ::scratchy_target_metal::model::MTL4ComputeCommandEncoder,
                                >,
                                num_seqs: u32,
                                sampled_rows: ::scratchy_target_metal::model::SampledRowCount,
                            ) -> ::core::result::Result<(), ::std::string::String> {
                                self.worker
                                    .run_bucket_mtl4(
                                        &self.bucket,
                                        num_seqs,
                                        sampled_rows,
                                        ::scratchy_target_metal::interpreter::metal::ForwardReads::Outputs,
                                        encoder,
                                    )
                                    .map_err(|e| format!("run_bucket_mtl4: {e:?}"))
                            }

                            fn logits_buf(
                                &self,
                            ) -> &::scratchy_target_metal::model::ProtocolObject<
                                dyn ::scratchy_target_metal::model::MTLBuffer,
                            > {
                                &self.worker.arena[self.terminal_slot]
                            }

                            fn vocab(&self) -> u32 { self.vocab }
                        }
                        let adapter = Adapter {
                            worker,
                            bucket,
                            terminal_slot,
                            vocab,
                        };
                        // Erase the metal-typed `&RuntimeBindings` behind the
                        // neutral handle so `MetalChainBody` (defined in the
                        // target-free compiler) names no metal-target type; the
                        // worker's body closure recovers `&RuntimeBindings`.
                        body(
                            &adapter,
                            ::scratchy_target_metal::model::MetalRuntimeHandle::new(runtime),
                            enc,
                        )
                        .map_err(
                            ::scratchy_target_metal::interpreter::metal::ForwardError::Followup,
                        )
                    },
                )
                .map_err(|e| format!("MetalWorkerPool::with_chain_encoder: {e}"))
            })
            .map_err(|e| format!("forward_chain_with_encoder: {e}"))
            .and_then(::core::convert::identity)
        }
    };

    // Encoder layouts have no lm_head split — `forward_backbone` is
    // semantically identical to `forward`. Decoder layouts return the
    // pre-lm_head activation via a DtoD memcpy of the protected
    // backbone slot. Both shapes are dispatched via the same
    // `FORWARD_TABLE` row.
    let forward_backbone_fn = match layout {
        BackboneLayout::Decoder => quote! {},
        BackboneLayout::Encoder => quote! {},
    };
    let static_slices_emit = quote! { #(#static_slices)* };

    let items = quote! {
        #canonical_params_impl

        #weight_accessors_impl

        #instruction_alias

        #static_slices_emit

        #forward_table

        #metal_emission

        #forward_backbone_fn

        /// Walk `FORWARD_TABLE` and return one [`BucketDump`] per
        /// row, with backbone + lm_head normalized for non-generic
        /// inspection (no `&Weights`, no GPU). Used by
        /// `scr model info` via the inventory registry. Available
        /// under either backend so the CLI subcommand can dump
        /// metal-compiled arches too.
        #[cfg(scratchy_target = "metal")]
        pub fn dump() -> ::std::vec::Vec<::scratchy_forward_compiler::BucketDump> {
            FORWARD_TABLE
                .iter()
                .map(|e| ::scratchy_forward_compiler::BucketDump {
                    m_min: e.0,
                    m_max_excl: e.1,
                    sk_min: e.2,
                    sk_max_excl: e.3,
                    backbone: ::scratchy_forward_compiler::normalize_slice(e.4),
                    lm_head: ::scratchy_forward_compiler::normalize_slice(e.5),
                })
                .collect()
        }
    };
    ModelEmission {
        weights: (weights, weights_affine_rows),
        items,
    }
}
