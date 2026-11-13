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
    Activation, Binding, CommandTableBuilder, CommandTableParts, GatedCommand, Gemm, GenClass,
    KernelId, KvAddressing, LoweredCommand, LoweredMetalTape, LoweringError, MetalDtype,
    PackedKind, TapeCommand, TapeCommands, baked,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::canonical::StepOperands;
use scratchy_subtile::handoff::ArenaExtents;
use scratchy_target_metal::tape::lowering::TapeHalf;

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
    /// Each weight family an MLX-affine matmul of any bucket reads, with every
    /// `(n, k, bits, group)` the bake minted for it.
    affine_reads: AffineReads,
    operands: ArenaOperands,
    /// Each weight family a dense GEMM reads, with the DSL weights the GEMM's ops read (their own
    /// operands', `Recording::read`): what its slot's field must hold.
    gemm_weights: BTreeMap<u32, BTreeSet<u32>>,
    /// Each weight family an MLX-affine matmul reads, with the DSL weights the matmul's op reads:
    /// what its slot's field must hold.
    affine_weights: BTreeMap<u32, BTreeSet<u32>>,
    /// Each weight family any command binds a tensor of, with the DSL weights its rows' ops read.
    sources: SourceReads,
}

/// The DSL weights row `row` of `half`'s op reads through a site of family `ix`.
fn read_through(
    operands: &StepOperands,
    half: TapeHalf,
    row: u32,
    ix: SourceIx,
) -> impl Iterator<Item = u32> + '_ {
    (operands.weights(half, row).iter()).filter_map(move |&(at, id)| (at == ix).then_some(id))
}

/// Each weight family a command binds a tensor of — its weight, an affine part, a bundle's router
/// or expert bank — with the DSL weights the ops of the rows the command was lowered from read
/// through a site of that family (their own operands', `Recording::read`): what its slot's field
/// must hold. And each family a command binds that no row it was lowered from names, with the
/// command's kernel: a typed one refuses the bake.
#[derive(Default)]
struct SourceReads {
    weights: BTreeMap<u32, BTreeSet<u32>>,
    unnamed: BTreeMap<u32, String>,
}

impl SourceReads {
    /// Add the families `cmd` binds, read as its rows (`rows`, of `steps`) read them.
    fn read(
        &mut self,
        cmd: &TapeCommand,
        rows: &tl::CommandRows,
        steps: &MetalStepTape,
        step_operands: &StepOperands,
    ) {
        let bound: Vec<SourceIx> = match cmd {
            TapeCommand::Gemm(g) => vec![g.weight.ix],
            TapeCommand::Dispatch(d) => (d.bindings.iter())
                .filter_map(|b| match *b {
                    Binding::Source { ix, .. } => Some(ix),
                    _ => None,
                })
                .collect(),
        };
        let sources = match rows.half {
            TapeHalf::Backbone => &steps.backbone_sources,
            TapeHalf::LmHead => &steps.lm_head_sources,
        };
        for ix in bound {
            let mut named = false;
            let reads = self.weights.entry(ix.0).or_default();
            for &r in &rows.rows {
                let row = sources.get(r as usize).map_or(&[][..], Vec::as_slice);
                named |= row.iter().any(|s| s.ix == ix);
                reads.extend(read_through(step_operands, rows.half, r, ix));
            }
            if !named {
                (self.unnamed.entry(ix.0)).or_insert_with(|| format!("{:?}", cmd.kernel()));
            }
        }
    }
}

/// ⭐ A MODEL'S ARENA OPERANDS, EACH TYPED ONCE: every `(slot, extents)` its commands bind an arena
/// buffer as — the slot, and the extents the rows a command was lowered from hold there —
/// model-wide, with the colour types that bind it. A command's binding names its operand
/// (`__tape_cmds::O<n>`), whose value is the slot of a colour hosting its extents
/// ([`slot`](scratchy_target_metal::tape::typed::slot)) and which requires every colour type
/// binding it to host them too ([`hosts`](scratchy_target_metal::tape::typed::hosts)). Keyed on
/// the extents' markers, not on the bucket, so the buckets' commands share a binding list
/// wherever their colourings agree.
#[derive(Default)]
struct ArenaOperands {
    ids: HashMap<(u32, BTreeSet<String>), u32>,
    list: Vec<ArenaOperand>,
}

struct ArenaOperand {
    /// Each extent, by its marker (`__arena::E<n>`).
    extents: BTreeSet<String>,
    /// The colour types binding it — each band's colour of its slot — with the rows of the first
    /// band seen binding it through that colour.
    colours: BTreeMap<String, BucketRows>,
}

/// A bucket's rows, the types an extent is spelled at: a colour type's parameters, `M` for
/// `num_tokens` and `S` for `sk_bucket` (`arena_colours!`), or one bucket's numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BucketRows {
    Generic,
    At { m: u32, sk: u32 },
}

impl BucketRows {
    /// `num_tokens`' type and `sk_bucket`'s: a bucket's numbers as [`typenum`] types
    /// (`scratchy_tensors::typed::unsigned_alias_path`).
    ///
    /// [`typenum`]: scratchy_tensors::typed::typenum
    pub fn types(self) -> Result<(TokenStream, TokenStream), String> {
        let number = |n: u32| {
            let path = scratchy_tensors::typed::unsigned_alias_path(u64::from(n));
            path.parse::<TokenStream>()
                .map_err(|e| format!("`{path}`: {e}"))
        };
        match self {
            Self::Generic => Ok((quote!(M), quote!(S))),
            Self::At { m, sk } => Ok((number(m)?, number(sk)?)),
        }
    }
}

/// A band's arena: each slot's colour type (`__arena::K<n>`), and the bucket's rows.
pub struct BandArena {
    pub colours: Vec<String>,
    pub rows: BucketRows,
}

/// Operand `n`'s const, `O<n>` ([`ArenaOperands`]): what a binding list's arena binding names.
fn operand(n: u32) -> proc_macro2::Ident {
    quote::format_ident!("O{n}")
}

/// A GEMM operand, its arena buffer named by its operand.
fn operand_activation(a: Activation) -> Result<TokenStream, BakeDefect> {
    match a {
        Activation::Arena { slot } => {
            let o = operand(slot);
            Ok(quote!(__tl::Activation::Arena { slot: #o }))
        }
        other => crate::const_tokens::const_tokens(&other)
            .map_err(|e| BakeDefect(format!("serialize command table: {e}"))),
    }
}

impl ArenaOperands {
    /// Each operand's const: the slot of the first colour type binding it, as one hosting its first
    /// extent, under the requirement that each colour type binding it hosts each of its extents,
    /// each colour named at the rows of a bucket binding it through that colour.
    fn tokens(&self) -> Result<TokenStream, BakeDefect> {
        if self.list.is_empty() {
            return Ok(TokenStream::new());
        }
        let consts = (self.list.iter().enumerate())
            .map(|(n, op)| {
                let mut uses = (op.colours.iter())
                    .flat_map(|(colour, rows)| op.extents.iter().map(move |e| (colour, rows, e)))
                    .map(|(colour, rows, e)| {
                        let (colour, e) = (
                            quote::format_ident!("{colour}"),
                            quote::format_ident!("{e}"),
                        );
                        let (m, sk) = rows.types().map_err(BakeDefect)?;
                        Ok((quote!(__a::#colour<#m, #sk>), quote!(__a::#e)))
                    });
                let (c, e) = uses.next().ok_or_else(|| {
                    BakeDefect(format!("arena operand {n} is bound by no bucket arena"))
                })??;
                let hosts = uses
                    .map(|u| u.map(|(c, e)| quote!(hosts::<#c, #e>();)))
                    .collect::<Result<Vec<_>, BakeDefect>>()?;
                let name = operand(n as u32);
                Ok(quote! {
                    const #name: u32 = { #(#hosts)* slot::<#c, #e>() };
                })
            })
            .collect::<Result<Vec<_>, BakeDefect>>()?;
        Ok(quote! {
            use super::__arena as __a;
            use ::scratchy_target_metal::tape::typed::{hosts, slot};
            #(#consts)*
        })
    }

    /// The operand `slot` holding `extents` is, bound in `band`'s arena.
    fn id(
        &mut self,
        slot: u32,
        extents: BTreeSet<String>,
        band: &BandArena,
    ) -> Result<u32, BakeDefect> {
        let colour = (band.colours.get(slot as usize))
            .ok_or_else(|| BakeDefect(format!("arena slot {slot} is past the band's colours")))?;
        let key = (slot, extents.clone());
        let list = &mut self.list;
        let id = *self.ids.entry(key).or_insert_with(|| {
            list.push(ArenaOperand {
                extents,
                colours: BTreeMap::new(),
            });
            (list.len() - 1) as u32
        });
        (self.list[id as usize].colours)
            .entry(colour.to_string())
            .or_insert(band.rows);
        Ok(id)
    }
}

/// Each weight family's MLX-affine matmul reads: the `(n, k, bits, group)` the bake minted for
/// it, each once.
type AffineReads =
    std::collections::BTreeMap<u32, std::collections::BTreeSet<(u32, u32, u32, u32)>>;

impl CommandPool {
    /// Add the MLX-affine matmuls of `steps` to the families' reads, each with the linears its
    /// row's op reads (`operands`).
    fn read_affine(
        &mut self,
        steps: &MetalStepTape,
        operands: &StepOperands,
    ) -> Result<(), BakeDefect> {
        let reads = scratchy_target_metal::tape::lowering::affine_matmul_reads(steps)
            .map_err(|e| BakeDefect(format!("an affine matmul's weight: {e}")))?;
        for (half, row, ix, g) in reads {
            let read = (g.n.get(), g.k.get(), g.bits.get(), g.group_size.get());
            self.affine_reads.entry(ix.0).or_default().insert(read);
            let read = read_through(operands, half, row, ix);
            self.affine_weights.entry(ix.0).or_default().extend(read);
        }
        Ok(())
    }

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
    ///
    /// Each arena operand its commands bind is a const of the module, `O<n>` ([`ArenaOperands`]),
    /// its extents spelled by `extent`: the binding lists and GEMM operands name it.
    ///
    /// `weight_markers`: each DSL weight a typed field holds, by `WeightId`, its marker in the
    /// model's `__weights`; each typed GEMM slot is read as its GEMMs' ops' weights
    /// ([`gemm_slot`](scratchy_target_metal::tape::typed::gemm_slot)). `named`: the families the
    /// model names itself ([`ModelNamed`](crate::canonical::ModelNamed)), the only ones bound with
    /// no read naming their weight.
    pub fn into_tokens(
        self,
        typed: &std::collections::HashSet<SourceIx>,
        named: &std::collections::HashSet<SourceIx>,
        weight_markers: &BTreeMap<u32, String>,
    ) -> Result<TokenStream, BakeDefect> {
        // A model metal lowers no tape for (a dense MoE has no metal realization) names nothing.
        if self.table.is_empty() && self.kernels.is_empty() {
            let tokens = quote! {
                #[cfg(scratchy_target = "metal")]
                static METAL_KERNELS: &[::scratchy_target_metal::tape::lowered::BakedKernel] = &[];
            };
            return Ok(tokens);
        }
        let parts = self.table.finish();
        let reads = FamilyReads {
            affine_reads: &self.affine_reads,
            gemm_weights: &self.gemm_weights,
            affine_weights: &self.affine_weights,
            sources: &self.sources,
        };
        let gemms = TypedGemms::of(&parts, typed, named, &reads, weight_markers)?;
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
        // Each command as a call of `gemm` or `command` below: its part indices and a dispatch's
        // kernel's name, where its literal spells every newtype of its fields.
        fn ix<T>(i: scratchy_target_metal::tape::ids::PartIx<T>) -> proc_macro2::Literal {
            proc_macro2::Literal::u16_unsuffixed(i.raw())
        }
        let commands = (parts.commands.iter())
            .map(|p| {
                let gate = ix(p.gate);
                match p.kind {
                    PackedKind::Gemm(g) => {
                        let g = ix(g);
                        quote!(gemm(#g, #gate))
                    }
                    PackedKind::Dispatch {
                        kernel,
                        symbol,
                        constants,
                        dispatch,
                        bindings,
                    } => {
                        let kernel = quote::format_ident!("{}", format!("{kernel:?}"));
                        let (symbol, constants) = (ix(symbol), ix(constants));
                        let (dispatch, bindings) = (ix(dispatch), ix(bindings));
                        quote!(command(K::#kernel, #symbol, #constants, #dispatch, #bindings, #gate))
                    }
                }
            })
            .collect();
        let bindings = (parts.bindings.iter())
            .map(|list| gemms.binding_list(list))
            .collect::<Result<Vec<_>, BakeDefect>>()?;
        let gemm_entries = (parts.gemm_entries.iter())
            .map(|g| match gemms.record(*g)? {
                Some(record) => Ok(record),
                None => crate::const_tokens::const_tokens(g)
                    .map_err(|e| BakeDefect(format!("serialize command table: {e}"))),
            })
            .collect::<Result<Vec<_>, BakeDefect>>()?;
        let gemm_ats = (parts.gemms.iter())
            .map(|at| {
                let (entry, layer, operands) = (at.entry.raw(), at.layer.0, at.operands.raw());
                quote!(gemm_at(#entry, #layer, #operands))
            })
            .collect();
        let markers = gemms.markers();
        let operands = self.operands.tokens()?;
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
            array("GEMMS", gemm_ats, quote!(__tl::GemmAt)),
            array("GEMM_ENTRIES", gemm_entries, quote!(__tl::GemmEntry)),
            array(
                "GEMM_OPERANDS",
                (parts.gemm_operands.iter())
                    .map(|g| {
                        let (out, input) =
                            (operand_activation(g.out)?, operand_activation(g.input)?);
                        Ok(quote!(__tl::GemmOperands { out: #out, input: #input }))
                    })
                    .collect::<Result<Vec<_>, BakeDefect>>()?,
                quote!(__tl::GemmOperands),
            ),
            array(
                "GATES",
                each(&parts.gates)?,
                quote!(Option<__tl::RuntimeGate>),
            ),
        ];
        let kernels: Vec<TokenStream> = self.kernels.iter().map(kernel_ref).collect();
        let kernels_len = proc_macro2::Literal::usize_unsuffixed(kernels.len());
        // Each helper is written only for a table that calls it: a model of affine matmuls alone
        // has no GEMM, so no `gemm_at` or `gemm`.
        let gemm_helpers = (!parts.gemms.is_empty()).then(|| {
            quote! {
                /// The table's GEMM of entry `entry` at layer `layer` over operands `operands`: a
                /// GEMM of every layer is one, so each is written as this call, not a literal of
                /// its three newtypes.
                const fn gemm_at(entry: u16, layer: u16, operands: u16) -> __tl::GemmAt {
                    __tl::GemmAt {
                        entry: __ti::PartIx::at(entry),
                        layer: __tl::PackedLayer(layer),
                        operands: __ti::PartIx::at(operands),
                    }
                }
                /// The table's GEMM `gemm` of [`GEMMS`] under gate `gate`.
                const fn gemm(gemm: u16, gate: u16) -> __tl::PackedCommand {
                    __tl::PackedCommand {
                        kind: __tl::PackedKind::Gemm(__ti::PartIx::at(gemm)),
                        gate: __ti::PartIx::at(gate),
                    }
                }
            }
        });
        let dispatches =
            (parts.commands.iter()).any(|c| matches!(c.kind, PackedKind::Dispatch { .. }));
        let command_helper = dispatches.then(|| {
            quote! {
                use __tl::KernelId as K;
                /// The table's dispatch of kernel `kernel` and these part indices: a command is
                /// written as this call, not a literal of its six fields' newtypes.
                const fn command(
                    kernel: K,
                    symbol: u16,
                    constants: u16,
                    dispatch: u16,
                    bindings: u16,
                    gate: u16,
                ) -> __tl::PackedCommand {
                    __tl::PackedCommand {
                        kind: __tl::PackedKind::Dispatch {
                            kernel,
                            symbol: __ti::PartIx::at(symbol),
                            constants: __ti::PartIx::at(constants),
                            dispatch: __ti::PartIx::at(dispatch),
                            bindings: __ti::PartIx::at(bindings),
                        },
                        gate: __ti::PartIx::at(gate),
                    }
                }
            }
        });
        let tokens = quote! {
            #[cfg(scratchy_target = "metal")]
            mod __tape_cmds {
                #aliases
                #markers
                #operands
                #gemm_helpers
                #command_helper
                #(#statics)*
                pub(super) static TABLE: __tl::CommandTable = __tl::CommandTable {
                    commands: &COMMANDS,
                    symbols: __ti::PartArray(&SYMBOLS),
                    constants: __ti::PartArray(&CONSTANTS),
                    dispatches: __ti::PartArray(&DISPATCHES),
                    bindings: __ti::PartArray(&BINDINGS),
                    gemms: __ti::PartArray(&GEMMS),
                    gemm_entries: __ti::PartArray(&GEMM_ENTRIES),
                    gemm_operands: __ti::PartArray(&GEMM_OPERANDS),
                    gates: __ti::PartArray(&GATES),
                };
                pub(super) static KERNELS: [__tl::BakedKernel; #kernels_len] = [ #(#kernels),* ];
            }
            #[cfg(scratchy_target = "metal")]
            static METAL_KERNELS: &[::scratchy_target_metal::tape::lowered::BakedKernel] =
                &__tape_cmds::KERNELS;
        };
        Ok(tokens)
    }
}

/// ⭐ A MODEL'S DENSE GEMMS, TYPED BY THEIR WEIGHTS.
///
/// A dense GEMM's record (`Gemm`: its dims and the weight it reads, which nothing else of the
/// command names) over a typed family's weight takes that weight's index and dims from ONE const
/// of its slot, `__GEMM<slot>` (`GemmSlot::WEIGHT`, read off the slot's handle) — its rows and
/// layer are its own. Every other record and binding keeps its value; the commands resolve to
/// the same values either way.
///
/// ⭐ AND EVERY BINDING OF A TYPED FAMILY, TYPED BY THE WEIGHTS ITS COMMAND'S OPS READ. A
/// command's binding of a typed family's tensor — its weight, a bundle's router or expert bank, a
/// GEMM of a router's gate — names its slot's index only as the ops of the rows the command was
/// lowered from read it: `__SOURCE<slot>` is `source_slot::<slot, W>()` (and `holds::<slot, W2>()`
/// each other one), `W` the marker of a DSL weight they read through a site of that family
/// ([`SourceReads`]), E0277 where the slot's field holds another weight. A binding of an affine
/// weight's scales, offsets or linear bias names `__SCALES<slot>` instead, the same check through
/// `affine_part`, which only an affine slot has: so a part binds its own weight's slot, and only a
/// quantized one. A typed family a command binds that no row it was lowered from names refuses the
/// bake. Only a family the model names itself ([`ModelNamed`](crate::canonical::ModelNamed): its
/// embedding table, under `embed_base`) is bound by its index alone, its parts by
/// `affine_part_of`; any other typed family whose rows read no weight through it refuses the bake.
///
/// Each typed value is one `const` per slot: the weight's index and dims depend on the slot
/// alone, so a record that named a type of its own would only re-check it.
///
/// ⭐ AND ITS MLX-AFFINE MATMULS, TYPED BY THEIRS. Each typed family an affine matmul reads is a
/// slot whose handle is a quantized weight (`AffineSlot`, else E0277). The bake picked the
/// kernels, named their symbols and compiled their pipelines from the numbers it minted, so each
/// slot's numbers are tied to the bake's in the build:
///
/// - the weight it reads by what its field holds: its record's weight, `__AFFINE<slot>`, is
///   `affine_slot::<slot, W>()`, `W` the marker of a DSL weight its matmuls' ops read (and
///   `holds::<slot, W2>()` each other one), E0277 where the slot's field holds another weight;
/// - its codes by a bound: `affine_codes_agree::<slot, B, G>()`, `B` and `G` the markers of the
///   bits and group the bake minted, is E0271 where they are not the handle's;
/// - an int4-only or matrix-unit kernel's codes by its bound (`int4_codes`, `nax_group`: E0277);
/// - its n and k — and a dense GEMM's — by type equality: `affine_dims_agree::<slot, N, K>()`
///   (`gemm_dims_agree`), `N` and `K` the numbers the bake baked, as types, is E0271 where they
///   are not the numbers of the handle's dims.
///
/// The kernels' K and N constants are values, each set once as on any table: a command whose K or
/// N is not its slot's refuses the bake, so every one is the number the slot's bound ties to its
/// handle.
struct TypedGemms {
    /// The typed families.
    typed: std::collections::HashSet<SourceIx>,
    /// Each typed family's dense GEMM dims `(n, k)` as the bake baked them.
    gemm_dims: std::collections::BTreeSet<(u32, u32, u32)>,
    /// The typed families the binding lists bind a tensor of other than an affine part, and the
    /// GEMM records over a typed family's other tensor (a router's gate) name, ascending — each
    /// with the markers of the DSL weights its commands' rows' ops read through it, if any.
    bound: std::collections::BTreeMap<u32, Option<HeldMarkers>>,
    /// The typed families the binding lists bind an affine part of, likewise.
    parts: std::collections::BTreeMap<u32, Option<HeldMarkers>>,
    /// Each typed family an affine matmul reads, with the numbers the bake minted for it.
    affine: std::collections::BTreeMap<u32, AffineSlotRead>,
    /// Each typed family a dense GEMM reads, with the markers of the DSL weights its GEMMs' ops
    /// read (`__weights::<Marker>`).
    gemm_weights: std::collections::BTreeMap<u32, HeldMarkers>,
}

/// The markers of the DSL weights a slot's field must hold, the first apart ([`held_markers`]).
type HeldMarkers = (proc_macro2::Ident, Vec<proc_macro2::Ident>);

/// Whether `which` is a part of an MLX-affine weight: its scales, offsets or linear bias, which
/// only an affine slot has. Every tensor is placed here by name: a new one is a compile error,
/// not a binding that silently skips the affine check.
fn affine_part(which: scratchy_target_metal::tape::lowered::WeightTensor) -> bool {
    use scratchy_target_metal::tape::lowered::WeightTensor as T;
    match which {
        T::AffineScales | T::AffineBiases | T::AffineLinearBias => true,
        // The weight itself, a dense bias, an NVFP4 linear's scales, and every bundle's own
        // tensors: each its family's, and none an affine linear's part.
        T::Weight
        | T::Bias
        | T::Nvfp4Scales
        | T::MoeRouterGate
        | T::GemmaRouterGate
        | T::GemmaPerExpertScale
        | T::GemmaRouterScale
        | T::MoeExpertGateW
        | T::MoeExpertGateS
        | T::MoeExpertGateB
        | T::MoeExpertUpW
        | T::MoeExpertUpS
        | T::MoeExpertUpB
        | T::MoeExpertDownW
        | T::MoeExpertDownS
        | T::MoeExpertDownB
        | T::MoeSharedGateUpW
        | T::MoeSharedGateUpS
        | T::MoeSharedGateUpB
        | T::MoeSharedDownW
        | T::MoeSharedDownS
        | T::MoeSharedDownB
        | T::MoeSharedExpertGate
        | T::MoeRouterBias
        | T::MoeExpertGateUpLinearBias
        | T::MoeExpertDownLinearBias
        | T::GdnConv1d
        | T::GdnALog
        | T::GdnDtBias
        | T::GdnNorm
        | T::GdnConvBias
        | T::GdnSkip => false,
    }
}

/// A typed family's MLX-affine matmul reads: the one `(n, k, bits, group)` the bake minted for
/// it, and whether an int4-only or a matrix-unit kernel reads it.
struct AffineSlotRead {
    n: u32,
    k: u32,
    bits: u32,
    group: u32,
    int4: bool,
    nax: bool,
    /// The markers of the DSL weights its matmuls' ops read, the first apart ([`held_markers`]).
    weights: (proc_macro2::Ident, Vec<proc_macro2::Ident>),
}

/// The markers (`__weights::<Marker>`) of the DSL weights typed family `ix`'s `what` ops read
/// (`ids`), the first apart: what its slot's field must hold. A family no op of which names a
/// weight, or one naming a weight no typed field holds, refuses the bake.
fn held_markers(
    ix: u32,
    ids: Option<&BTreeSet<u32>>,
    weight_markers: &BTreeMap<u32, String>,
    what: &str,
) -> Result<(proc_macro2::Ident, Vec<proc_macro2::Ident>), BakeDefect> {
    let ids = ids.filter(|ids| !ids.is_empty()).ok_or_else(|| {
        BakeDefect(format!(
            "typed weight family {ix} is read by {what} whose ops read no weight through it"
        ))
    })?;
    let mut markers = (ids.iter())
        .map(|id| {
            let marker = weight_markers.get(id).ok_or_else(|| {
                BakeDefect(format!(
                    "typed weight family {ix}'s {what} read DSL weight {id}, which no typed field \
                     holds"
                ))
            })?;
            Ok(quote::format_ident!("{marker}"))
        })
        .collect::<Result<Vec<_>, BakeDefect>>()?
        .into_iter();
    let first = (markers.next())
        .ok_or_else(|| BakeDefect(format!("typed weight family {ix} has no {what} weight")))?;
    Ok((first, markers.collect()))
}

/// What the pool read of each weight family across the model's tapes: its affine matmuls' numbers,
/// and the DSL weights its dense GEMMs', affine matmuls' and bindings' rows' ops read through it.
struct FamilyReads<'a> {
    affine_reads: &'a AffineReads,
    gemm_weights: &'a BTreeMap<u32, BTreeSet<u32>>,
    affine_weights: &'a BTreeMap<u32, BTreeSet<u32>>,
    sources: &'a SourceReads,
}

impl TypedGemms {
    /// The typed GEMM records, affine matmuls and bindings of `parts`, over the families of
    /// `typed` (`named` those the model names itself), read as `reads`.
    fn of(
        parts: &CommandTableParts,
        typed: &std::collections::HashSet<SourceIx>,
        named: &std::collections::HashSet<SourceIx>,
        reads: &FamilyReads<'_>,
        weight_markers: &BTreeMap<u32, String>,
    ) -> Result<Self, BakeDefect> {
        let FamilyReads {
            affine_reads,
            gemm_weights,
            affine_weights,
            sources,
        } = *reads;
        use scratchy_target_metal::tape::kernel_constants::{
            AFFINE_K, AFFINE_MATMUL_KERNELS, AFFINE_N, INT4_MATMUL_KERNELS, NAX_MATMUL_KERNELS,
        };
        use scratchy_target_metal::tape::lowered::{Binding, WeightTensor};
        let typed_weight =
            |g: &&scratchy_target_metal::tape::lowered::GemmEntry| Self::typed_weight(typed, g);
        let read: std::collections::BTreeSet<u32> =
            (parts.gemm_entries.iter().filter(typed_weight))
                .map(|g| g.ix.0)
                .collect();
        // Each read slot's GEMMs' DSL weights, by marker: a slot no GEMM's op names a weight of
        // refuses the bake.
        let gemm_weights = (read.iter())
            .map(|&ix| {
                let held = held_markers(ix, gemm_weights.get(&ix), weight_markers, "GEMMs")?;
                Ok((ix, held))
            })
            .collect::<Result<_, BakeDefect>>()?;
        let gemm_dims = (parts.gemm_entries.iter().filter(typed_weight))
            .map(|g| (g.ix.0, g.dims.n, g.dims.k))
            .collect();
        // A typed family a command binds and no row it was lowered from names: its rows' ops say
        // nothing of what the slot must hold.
        if let Some((ix, kernel)) =
            (sources.unnamed.iter()).find(|(ix, _)| typed.contains(&SourceIx(**ix)))
        {
            return Err(BakeDefect(format!(
                "a {kernel} command binds typed weight family {ix}, which no row it was lowered \
                 from names"
            )));
        }
        // Each bound typed family's held markers: those of the weights its commands' rows read
        // through it. Only a family the model names itself (its embedding table) is read through
        // no operand; any other one whose rows read no weight through it refuses the bake.
        let held = |ix: u32| -> Result<Option<HeldMarkers>, BakeDefect> {
            match sources.weights.get(&ix).filter(|ids| !ids.is_empty()) {
                Some(ids) => held_markers(ix, Some(ids), weight_markers, "commands").map(Some),
                None if named.contains(&SourceIx(ix)) => Ok(None),
                None => Err(BakeDefect(format!(
                    "typed weight family {ix} is bound by commands whose rows read no weight \
                     through it, and the model does not name it"
                ))),
            }
        };
        let (mut bound, mut affine_parts) = (BTreeMap::new(), BTreeMap::new());
        for b in parts.bindings.iter().copied().flatten() {
            if let Binding::Source { ix, which, .. } = b
                && typed.contains(ix)
            {
                let to = if affine_part(*which) {
                    &mut affine_parts
                } else {
                    &mut bound
                };
                if let std::collections::btree_map::Entry::Vacant(v) = to.entry(ix.0) {
                    v.insert(held(ix.0)?);
                }
            }
        }
        // A GEMM over a typed family's other tensor (a router's gate) names its slot alike.
        for g in &parts.gemm_entries {
            if typed.contains(&g.ix)
                && !Self::typed_weight(typed, g)
                && let std::collections::btree_map::Entry::Vacant(v) = bound.entry(g.ix.0)
            {
                v.insert(held(g.ix.0)?);
            }
        }
        // Each typed family's one affine read: two are no one slot's.
        let mut affine = std::collections::BTreeMap::new();
        for (&ix, reads) in affine_reads {
            if !typed.contains(&SourceIx(ix)) {
                continue;
            }
            let [(n, k, bits, group)] = reads.iter().copied().collect::<Vec<_>>()[..] else {
                return Err(BakeDefect(format!(
                    "typed weight family {ix} is read by affine matmuls of {reads:?}: its slot \
                     states one"
                )));
            };
            let read = AffineSlotRead {
                n,
                k,
                bits,
                group,
                int4: false,
                nax: false,
                weights: held_markers(
                    ix,
                    affine_weights.get(&ix),
                    weight_markers,
                    "affine matmuls",
                )?,
            };
            affine.insert(ix, read);
        }
        // Each affine matmul command over a typed family is baked at its slot's K and N.
        for c in &parts.commands {
            let PackedKind::Dispatch {
                kernel,
                constants: set,
                bindings,
                ..
            } = c.kind
            else {
                continue;
            };
            let weight = (parts.bindings[bindings.get()].iter()).find_map(|b| match b {
                Binding::Source {
                    ix,
                    which: WeightTensor::Weight,
                    ..
                } if typed.contains(ix) => Some(ix.0),
                _ => None,
            });
            match weight {
                Some(ix) if AFFINE_MATMUL_KERNELS.contains(&kernel) => {
                    let Some(read) = affine.get_mut(&ix) else {
                        return Err(BakeDefect(format!(
                            "{kernel:?} reads typed weight family {ix}, which no affine matmul \
                             reads"
                        )));
                    };
                    let value = |at: u16| {
                        (parts.constants[set.get()].iter())
                            .find(|v| v.index == at)
                            .map(|v| v.bits)
                    };
                    let (k, n) = (value(AFFINE_K.0), value(AFFINE_N.0));
                    if (k, n) != (Some(read.k), Some(read.n)) {
                        return Err(BakeDefect(format!(
                            "{kernel:?} over typed weight family {ix} is baked at k {k:?}, n \
                             {n:?}; its slot's are {}, {}",
                            read.k, read.n
                        )));
                    }
                    read.int4 |= INT4_MATMUL_KERNELS.contains(&kernel);
                    read.nax |= NAX_MATMUL_KERNELS.contains(&kernel);
                }
                _ if AFFINE_MATMUL_KERNELS.contains(&kernel) => {
                    return Err(BakeDefect(format!(
                        "{kernel:?} reads a weight no typed slot holds: every affine matmul's \
                         weight is a typed handle's"
                    )));
                }
                _ => {}
            }
        }
        Ok(Self {
            typed: typed.clone(),
            gemm_dims,
            bound,
            parts: affine_parts,
            affine,
            gemm_weights,
        })
    }

    fn part(slot: SourceIx) -> proc_macro2::Ident {
        quote::format_ident!("__SCALES{}", slot.0)
    }

    fn affine_slot(slot: u32) -> proc_macro2::Ident {
        quote::format_ident!("__AFFINE{slot}")
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
    fn record(
        &self,
        g: scratchy_target_metal::tape::lowered::GemmEntry,
    ) -> Result<Option<TokenStream>, BakeDefect> {
        if !self.typed.contains(&g.ix) {
            return Ok(None);
        }
        if !Self::typed_weight(&self.typed, &g) {
            // A typed family's other tensor (a router's gate): its slot as its rows read it, its
            // dims and tensor their own.
            let s = Self::source(g.ix);
            let (dims, which) = (
                crate::const_tokens::const_tokens(&g.dims),
                crate::const_tokens::const_tokens(&g.which),
            );
            let tokens = |e| BakeDefect(format!("serialize command table: {e}"));
            let (dims, which) = (dims.map_err(tokens)?, which.map_err(tokens)?);
            return Ok(Some(
                quote!(__tl::GemmEntry { dims: #dims, ix: #s, which: #which }),
            ));
        }
        let (w, m) = (Self::gemm(g.ix), g.dims.m);
        Ok(Some(quote!(__tl::GemmEntry {
            dims: __tl::GemmDims { m: #m, n: #w.n, k: #w.k },
            ix: #w.ix,
            which: __tl::WeightTensor::Weight,
        })))
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
                    let s = match affine_part(*which) {
                        true => Self::part(*ix),
                        false => Self::source(*ix),
                    };
                    let (which, layer) = (tokens(which)?, tokens(layer)?);
                    let bi = proc_macro2::Literal::u8_suffixed(*binding_index);
                    Ok(quote!(__tl::Binding::Source {
                        ix: #s,
                        which: #which,
                        layer: #layer,
                        binding_index: #bi,
                    }))
                }
                Binding::ArenaSlot {
                    slot,
                    binding_index,
                } => {
                    let o = operand(*slot);
                    let bi = proc_macro2::Literal::u8_suffixed(*binding_index);
                    Ok(quote!(__tl::Binding::ArenaSlot { slot: #o, binding_index: #bi }))
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
        // Each read slot's weight, as its GEMMs' ops' DSL weights read it: only a slot whose field
        // holds each of them has one (`gemm_slot`, `holds`).
        let weights = self.gemm_weights.iter().map(|(&ix, (first, rest))| {
            let (c, s) = (
                Self::gemm(SourceIx(ix)),
                crate::canonical::slot(SourceIx(ix)),
            );
            quote!(const #c: #tt::GemmSlotWeight = {
                #(#tt::holds::<super::#s, super::__weights::#rest>();)*
                #tt::gemm_slot::<super::#s, super::__weights::#first>()
            };)
        });
        let gemm_dims = self.gemm_dims.iter().map(|&(ix, n, k)| {
            let s = crate::canonical::slot(SourceIx(ix));
            let (n, k) = (unsigned(n), unsigned(k));
            quote!(const _: () = #tt::gemm_dims_agree::<super::#s, #n, #k>();)
        });
        // Each bound slot's index, as its commands' rows' ops read it (`source_slot`, `holds`); a
        // slot read through no operand by its index alone.
        let sources = self.bound.iter().map(|(&ix, held)| {
            let ix = SourceIx(ix);
            let (c, s) = (Self::source(ix), crate::canonical::slot(ix));
            match held {
                Some((first, rest)) => quote!(const #c: __ti::SourceIx = {
                    #(#tt::holds::<super::#s, super::__weights::#rest>();)*
                    #tt::source_slot::<super::#s, super::__weights::#first>()
                };),
                None => quote!(const #c: __ti::SourceIx = <super::#s as #tt::Source>::IX;),
            }
        });
        // Each slot an affine part of which is bound: likewise, and only an affine slot has one
        // (`affine_part`, `affine_part_of`).
        let parts = self.parts.iter().map(|(&ix, held)| {
            let ix = SourceIx(ix);
            let (c, s) = (Self::part(ix), crate::canonical::slot(ix));
            match held {
                Some((first, rest)) => quote!(const #c: __ti::SourceIx = {
                    #(#tt::holds::<super::#s, super::__weights::#rest>();)*
                    #tt::affine_part::<super::#s, super::__weights::#first>()
                };),
                None => quote!(const #c: __ti::SourceIx = #tt::affine_part_of::<super::#s>();),
            }
        });
        let affine = self.affine.iter().map(|(&ix, r)| {
            let (c, s) = (Self::affine_slot(ix), crate::canonical::slot(SourceIx(ix)));
            let w = quote!(::scratchy_tensors::typed);
            let (b, g) = (
                quote::format_ident!("B{}", r.bits),
                quote::format_ident!("G{}", r.group),
            );
            let (n, k) = (unsigned(r.n), unsigned(r.k));
            let int4 = r
                .int4
                .then(|| quote!(const _: () = #tt::int4_codes::<super::#s>();));
            let nax = r
                .nax
                .then(|| quote!(const _: () = #tt::nax_group::<super::#s>();));
            // The slot's weight, as its matmuls' ops' DSL weights read it (`affine_slot`).
            let (first, rest) = &r.weights;
            quote! {
                const #c: #tt::AffineSlotWeight = {
                    #(#tt::holds::<super::#s, super::__weights::#rest>();)*
                    #tt::affine_slot::<super::#s, super::__weights::#first>()
                };
                const _: () = #tt::affine_dims_agree::<super::#s, #n, #k>();
                const _: () = #tt::affine_codes_agree::<super::#s, #w::#b, #w::#g>();
                #int4
                #nax
            }
        });
        quote!(#(#weights)* #(#gemm_dims)* #(#sources)* #(#parts)* #(#affine)*)
    }
}

/// `n` as its type-level number (`scratchy_tensors::typed::unsigned_path`).
fn unsigned(n: u32) -> TokenStream {
    (scratchy_tensors::typed::unsigned_path(u64::from(n)).parse())
        .expect("a typenum path is Rust tokens")
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
    /// Its rows' arena operands, and its arena.
    pub operands: &'a StepOperands,
    pub arena: &'a BandArena,
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
    /// `steps`' rows' arena operands, and its arena.
    pub operands: &'a StepOperands,
    pub arena: &'a BandArena,
    /// Each arena value's extent, by its marker in the model's `__arena` (`arena_extents!`).
    pub extents: &'a ArenaExtents,
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
        #[cfg(scratchy_target = "metal")]
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
) -> Result<(LoweredMetalTape, Vec<tl::CommandRows>), LoweringError> {
    let at = tl::BakePoint {
        chunked: rung.addressing == KvAddressing::Chunked,
        bucket_m: input.bucket_m,
        num_arena_slots: rung.num_arena_slots,
        rotary: input.rotary,
        block_cap: rung.cap.get(),
        profile: Some(rung.profile),
        class: rung.class,
    };
    tl::lower_with_rows(rung.steps, mc, at)
}

/// `tape`'s commands with each arena operand named by its model-wide operand
/// ([`ArenaOperands`]): the slot it binds, held as the extents the rows the command was lowered
/// from (`rows`) hold there, in `band`'s colouring. A command binding a slot none of its rows
/// names refuses the bake. Each weight family a command binds joins `sources` with the weights its
/// rows' ops read ([`SourceReads`]).
fn name_operands(
    tape: &LoweredMetalTape,
    rows: &[tl::CommandRows],
    band: (&MetalStepTape, &StepOperands, &BandArena),
    extents: &ArenaExtents,
    operands: &mut ArenaOperands,
    gemm_weights: &mut BTreeMap<u32, BTreeSet<u32>>,
    sources: &mut SourceReads,
) -> Result<Vec<GatedCommand>, BakeDefect> {
    let (steps, step_operands, arena) = band;
    (tape.commands.iter().zip(rows).enumerate())
        .map(|(c, (cmd, rows))| {
            sources.read(&cmd.command, rows, steps, step_operands);
            let mut named = |slot: u32| -> Result<u32, BakeDefect> {
                let held = (rows.rows.iter())
                    .flat_map(|&r| step_operands.row(rows.half, r))
                    .filter(|o| o.slot.get() == slot)
                    .map(|o| {
                        let marker = extents.of(o.value).ok_or_else(|| {
                            BakeDefect(format!("arena value {:?} is hosted by no colour", o.value))
                        })?;
                        Ok(marker.to_string())
                    })
                    .collect::<Result<BTreeSet<_>, BakeDefect>>()?;
                if held.is_empty() {
                    return Err(BakeDefect(format!(
                        "command {c} ({:?}) binds arena slot {slot}, which no row it was lowered \
                         from names",
                        cmd.command.kernel()
                    )));
                }
                operands.id(slot, held, arena)
            };
            let mut activation = |a: Activation| match a {
                Activation::Arena { slot } => Ok(Activation::Arena { slot: named(slot)? }),
                other => Ok(other),
            };
            // A dense GEMM's weight: the weights its rows' ops read through its family.
            if let TapeCommand::Gemm(g) = cmd.command
                && g.weight.which == scratchy_target_metal::tape::lowered::WeightTensor::Weight
            {
                let ix = g.weight.ix;
                let read =
                    (rows.rows.iter()).flat_map(|&r| read_through(step_operands, rows.half, r, ix));
                gemm_weights.entry(ix.0).or_default().extend(read);
            }
            let command = match cmd.command {
                TapeCommand::Gemm(g) => TapeCommand::Gemm(Gemm {
                    out: activation(g.out)?,
                    input: activation(g.input)?,
                    ..g
                }),
                TapeCommand::Dispatch(d) => {
                    let bindings = (d.bindings.iter())
                        .map(|b| match *b {
                            Binding::ArenaSlot {
                                slot,
                                binding_index,
                            } => Ok(Binding::ArenaSlot {
                                slot: named(slot)?,
                                binding_index,
                            }),
                            other => Ok(other),
                        })
                        .collect::<Result<Vec<_>, BakeDefect>>()?;
                    TapeCommand::Dispatch(LoweredCommand {
                        bindings: baked(bindings),
                        ..d
                    })
                }
            };
            Ok(GatedCommand { command, ..cmd })
        })
        .collect()
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
    pool.read_affine(input.steps, input.operands)?;
    if let Some(wide) = &input.wide_band {
        pool.read_affine(wide.steps, wide.operands)?;
    }
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
    // Keyed by the band too: a body's arena operands are named under its band's colouring.
    let mut seen: std::collections::HashMap<(bool, LoweredMetalTape), usize> = Default::default();
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
        let Some((tape, rows)) = tapes.map_err(defect)? else {
            eprintln!(
                "[metal bake] bucket_m={}: ladder ends below KV cap rung {}",
                input.bucket_m,
                cap.get()
            );
            ended = Some((class, addressing));
            continue;
        };
        // The band the rung lowered under, whose colouring its arena operands are named in.
        let (wide, band) = match (class, input.wide_band.as_ref()) {
            (GenClass::M5Wide, Some(wide)) => (true, (wide.steps, wide.operands, wide.arena)),
            _ => (false, (input.steps, input.operands, input.arena)),
        };
        // The cap-sized scratch is the variant's; the rest of the tape is the body.
        let (roped_k, attn_unfused) = (tape.roped_k_scratch_bytes, tape.attn_unfused_scratch_bytes);
        let body = LoweredMetalTape {
            roped_k_scratch_bytes: 0,
            attn_unfused_scratch_bytes: 0,
            ..tape
        };
        let body_ix = match seen.get(&(wide, body)).copied() {
            Some(ix) => ix,
            None => {
                // Its commands as the table holds them: each arena operand named by its typed
                // operand.
                let named = name_operands(
                    &body,
                    &rows,
                    band,
                    input.extents,
                    &mut pool.operands,
                    &mut pool.gemm_weights,
                    &mut pool.sources,
                )?;
                let cmd_refs = (named.into_iter())
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
                seen.insert((wide, body), ix);
                ix
            }
        };
        let ident = quote::format_ident!("__TAPE_BODY_{uniq}_{body_ix}");
        let serves_tq = (tape.commands.iter())
            .any(|c| c.command.kernel() == Some(KernelId::AttentionViaCacheTq));
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
