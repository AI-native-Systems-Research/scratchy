// SPDX-License-Identifier: Apache-2.0
//! A spyre model's weights struct, from the weight binding its decode tape carries: the shared
//! accessor set (`weight_bindings::from_tape`) feeds the shared struct emission, and beside it
//! the id → FIELD binding (`superdsc_weights`) the worker stages each launch source through.

use quote::quote;
use scratchy_forward_compiler_frontend::classified::{OpKind, Program, WeightId};
use scratchy_forward_compiler_macro::target::{StructCx, WeightsStruct};
use scratchy_forward_compiler_macro::to_wavefront::{
    self, BridgeModel, LoweredDecode, PrefixCapacity, SourceBinding,
};
use scratchy_forward_compiler_macro::weight_bindings::{
    self, EmbedStorage, MlpPacking, TapeWeights, WeightAbi,
};
use scratchy_forward_compiler_macro::weight_vocab::WeightKind;
use scratchy_subtile::lower::InputRef;
use scratchy_subtile::subtile_ir::SubOp;

/// THE MODEL'S WEIGHT BINDING, from the tape both backends lower.
///
/// `from_tape` reads only `LoweredDecode.bindings` + the op that consumes each one, so it is a
/// fact about the MODEL. The struct it feeds is target-neutral (`crate::__gpu::layers::*`, where
/// `__gpu` is the target's runtime crate), so the same fields compile and load on any backend.
pub(crate) fn weights_struct(cx: &StructCx<'_>) -> WeightsStruct {
    let StructCx {
        program,
        model,
        facts,
        fuf,
        inferred,
        decode_point,
        struct_of,
    } = *cx;
    let bridge_model = BridgeModel {
        params: model,
        facts,
    };
    let lowered = decode_point.and_then(|(bounds, num_tokens)| {
        to_wavefront::lower_decode_to_wavefront(
            fuf,
            inferred,
            bounds,
            &bridge_model,
            PrefixCapacity::new(std::num::NonZeroU32::new(8192).expect("8192 != 0")),
            num_tokens,
        )
        .ok()
    });
    let tw = lowered.as_ref().map(|l| {
        weight_bindings::from_tape(
            l,
            program,
            WeightAbi {
                // Two matmuls plus a SiluMul — no packed-buffer kernel to read one.
                mlp: MlpPacking::Split,
                // ⛔ DERIVED, NOT ASSUMED. Read from the FUF, exactly as the subtile bridge reads
                // it, so the two cannot disagree about one model.
                embed: match fuf.nodes.iter().find_map(|n| {
                    (n.op == OpKind::Embed)
                        .then(|| {
                            scratchy_forward_compiler_macro::weight_vocab::weight_storage_of(n)
                        })
                        .flatten()
                }) {
                    Some(
                        scratchy_forward_compiler_macro::quantization::StorageFormat::Affine {
                            ..
                        },
                    ) => EmbedStorage::AffineQuant,
                    _ => EmbedStorage::Dense,
                },
            },
        )
        .unwrap_or_else(|e| panic!("[weight-bindings] {}: {e}", model.source_stem))
    });
    eprintln!(
        "[weight-bindings] {}: {} weight accessors from the tape",
        model.source_stem,
        tw.as_ref().map(|t| t.accessors.len()).unwrap_or(0),
    );
    let (struct_tokens, struct_affine_rows) =
        struct_of(tw.as_ref().map(|t| t.accessors.as_slice()));
    // The id → FIELD binding, emitted INTO THE SAME MODULE as the struct so
    // its arms can name `w.self_attn_q_proj[3]` directly. Only generated code
    // can name a generated field.
    let bindings = match (lowered.as_ref(), tw.as_ref()) {
        (Some(l), Some(t)) => emit_weight_bindings(l, program, t)
            .unwrap_or_else(|e| panic!("[weight-bindings] {}: {e}", model.source_stem)),
        // ⛔ THE FN IS ALWAYS EMITTED, AND IT REFUSES. The loaded model's
        // `superdsc_weights` names it per variant, so omitting it
        // is a confusing compile error in generated code — but emitting an
        // EMPTY binding would be worse: a model that silently loads zero
        // weights and produces garbage. A model whose decode point did not
        // lower has no binding to give, and says so.
        _ => {
            let stem = model.source_stem.as_str();
            quote! {
                pub fn superdsc_weights(
                    _w: &Weights,
                ) -> ::anyhow::Result<::std::vec::Vec<::scratchy_target_spyre::BoundWeight>> {
                    ::anyhow::bail!(
                        "{}: no weight binding was generated — the decode point did not lower, \
                         so this model cannot be launched on spyre",
                        #stem,
                    )
                }
            }
        }
    };
    (quote! { #struct_tokens #bindings }, struct_affine_rows)
}

/// Emit `superdsc_weights(&Weights) -> Vec<(u32, GpuTensor)>` — the id → FIELD
/// binding, as generated code.
///
/// ⛔ THIS IS THE PIECE THAT LETS THE STRING PATH DIE. The worker's
/// `load_weights` re-found every tensor at runtime by building
/// `"{disk}.weight"` and asking the weight store for it, because nothing told
/// it which struct field a source id meant. Only GENERATED code can name a
/// GENERATED field, so this is emitted per model — the same reason the GPU target's
/// forward reads `wm.q_proj` rather than looking a name up.
///
/// ⛔ AND IT IS A `match` OVER REAL FIELDS, NOT A TABLE OF NAMES. A name here
/// would just be the disk-key rule again in another spelling, with the same
/// inability to express a bare `nn.Parameter` or a non-default decoder root.
/// Because the arms are field accesses, a weight the struct does not have is a
/// COMPILE error in generated code, not a `weight not found` at model load.
fn emit_weight_bindings(
    lowered: &LoweredDecode,
    program: &Program,
    tw: &TapeWeights,
) -> Result<proc_macro2::TokenStream, String> {
    use quote::quote;
    // Function-scoped because every user is in this spyre-gated body: a
    // module-level import would be dead outside spyre (b76516bab).
    use scratchy_subtile::handoff::BundleTensor;
    // Accessor name → how to reach that field on `w`. The three group shapes
    // are the struct emitter's own (`Unindexed` ⇒ `w.base`,
    // `LayeredContiguous` ⇒ `w.base[L]`, `LayeredSparse` ⇒ `w.base_L`), keyed
    // by accessor name so one map covers all three.
    let groups = scratchy_forward_compiler_macro::codegen::group_accessors_by_base(&tw.accessors);
    let mut access: std::collections::BTreeMap<String, proc_macro2::TokenStream> =
        Default::default();
    for g in &groups {
        let base = syn::Ident::new(&g.base, proc_macro2::Span::call_site());
        match g.kind {
            scratchy_forward_compiler_macro::codegen::AccessorGroupKind::Unindexed => {
                for (_, acc) in &g.entries {
                    access.insert(acc.name.to_string(), quote! { w.#base });
                }
            }
            scratchy_forward_compiler_macro::codegen::AccessorGroupKind::LayeredContiguous => {
                for (idx, acc) in &g.entries {
                    let l = proc_macro2::Literal::usize_unsuffixed(
                        idx.map(|u| u.0 as usize).unwrap_or(0),
                    );
                    access.insert(acc.name.to_string(), quote! { w.#base[#l] });
                }
            }
            scratchy_forward_compiler_macro::codegen::AccessorGroupKind::LayeredSparse => {
                for (_, acc) in &g.entries {
                    let f = &acc.name;
                    access.insert(acc.name.to_string(), quote! { w.#f });
                }
            }
        }
    }

    // Which externals are a GEMM's operand-1 — the fact that decides staged
    // shape orientation. Computed HERE, from the tape, because it cannot be
    // recovered downstream: a per-channel fp8 `weight_scale` is 2-D on disk
    // and is NOT a gemm operand, so a rank test would mis-orient it.
    let gemm_operands: std::collections::HashSet<usize> = lowered
        .input
        .ops
        .iter()
        .filter(|od| matches!(od.op, SubOp::MatmulTile { .. }))
        .filter_map(|od| match od.inputs.get(1) {
            Some(InputRef::Ext(e)) => Some(*e),
            _ => None,
        })
        .collect();

    let mut arms: Vec<proc_macro2::TokenStream> = Vec::new();
    for (i, b) in lowered.bindings().iter().enumerate() {
        // Which tensor inside the field the source binds — `None` for the field's own
        // primary tensor, `Some(..)` for a tensor of a MULTI-TENSOR bundle.
        let mut bundle_tensor = None;
        let (id, index, is_scale) = match b {
            SourceBinding::Weight { id, index } => (WeightId(*id), *index, false),
            SourceBinding::WeightScale { id, index } => (WeightId(*id), *index, true),
            SourceBinding::BundleTensor { id, index, tensor } => {
                bundle_tensor = Some(*tensor);
                (WeightId(*id), *index, false)
            }
            _ => continue,
        };
        let name = scratchy_forward_compiler_macro::emit::weight_field_name(program, id, index)
            .to_string();
        let Some(expr) = access.get(&name) else {
            return Err(format!(
                "weight_bindings: source {i} binds `{name}`, which the Weights struct has no field \
                 for — the accessor walk and the binding walk disagree"
            ));
        };
        let kind = tw
            .kind_of_field
            .get(&name)
            .ok_or_else(|| format!("weight_bindings: no kind recorded for field `{name}`"))?;
        // How to reach the TENSOR inside the field. Per-kind, because the field
        // types differ — a norm holds one tensor, a quantized linear holds a
        // code tensor and a scale under a variant.
        let tensor = match (kind, is_scale) {
            (WeightKind::Fp8, false) => {
                quote! { ::scratchy_target_spyre::wiring::fp8_weight(&#expr) }
            }
            (WeightKind::Fp8, true) => {
                quote! { ::scratchy_target_spyre::wiring::fp8_scale(&#expr) }
            }
            (WeightKind::Linear, _) => {
                quote! { ::scratchy_target_spyre::wiring::linear_weight(&#expr)? }
            }
            (WeightKind::RmsNorm | WeightKind::LayerNorm | WeightKind::Embedding, _) => {
                quote! { #expr.weight }
            }
            // A MoE bundle's tensor: `bundle_tensor` names WHICH one (the source's own
            // `BundleTensor`), and the wiring accessors are per-(kind, tensor) — one free
            // function per combination, so an unmodelled combination is a MISSING arm here
            // rather than a wrong-tensor guess.
            (WeightKind::GemmaRouter, false) => match bundle_tensor {
                Some(BundleTensor::RouterGate) => {
                    quote! { ::scratchy_target_spyre::wiring::gemma_router_gate(&#expr) }
                }
                Some(BundleTensor::RouterPerExpertScale) => {
                    quote! { ::scratchy_target_spyre::wiring::gemma_router_per_expert_scale(&#expr) }
                }
                Some(BundleTensor::RouterScale) => {
                    quote! { ::scratchy_target_spyre::wiring::gemma_router_scale(&#expr) }
                }
                other => {
                    return Err(format!(
                        "weight_bindings: GemmaRouter field `{name}` binds bundle tensor \
                         {other:?}, which is not a router tensor — the source and the field \
                         kind disagree"
                    ));
                }
            },
            (WeightKind::GemmaSwitchGlu, false) => match bundle_tensor {
                Some(BundleTensor::ExpertGateW) => {
                    quote! { ::scratchy_target_spyre::wiring::switch_glu_gate_w(&#expr) }
                }
                Some(BundleTensor::ExpertGateS) => {
                    quote! { ::scratchy_target_spyre::wiring::switch_glu_gate_s(&#expr) }
                }
                Some(BundleTensor::ExpertUpW) => {
                    quote! { ::scratchy_target_spyre::wiring::switch_glu_up_w(&#expr) }
                }
                Some(BundleTensor::ExpertUpS) => {
                    quote! { ::scratchy_target_spyre::wiring::switch_glu_up_s(&#expr) }
                }
                Some(BundleTensor::ExpertDownW) => {
                    quote! { ::scratchy_target_spyre::wiring::switch_glu_down_w(&#expr) }
                }
                Some(BundleTensor::ExpertDownS) => {
                    quote! { ::scratchy_target_spyre::wiring::switch_glu_down_s(&#expr) }
                }
                other => {
                    return Err(format!(
                        "weight_bindings: SwitchGluExperts field `{name}` binds bundle tensor \
                         {other:?}, which is not an expert tensor — the source and the field \
                         kind disagree"
                    ));
                }
            },
            (other, _) => {
                return Err(format!(
                    "weight_bindings: no staged-tensor accessor for weight kind {other:?} (field \
                     `{name}`) — add its arm rather than letting the source bind nothing"
                ));
            }
        };
        // ⭐ A BUNDLE TENSOR'S GEMM-NESS IS THE TENSOR'S OWN FACT. `gemm_operands` reads
        // `MatmulTile` operand-1 positions, which an expansion op never occupies — its matmuls are
        // `ExpertMatmul`/`RouterLogits`, and their weight operands sit at other positions. A gemm
        // bundle tensor (the router's `[E,hidden]` projection, the experts' `[E·out,in]` codes) is
        // staged by the SAME on-disk-`[n,k]`-verbatim + device-width contract every other gemm
        // weight is, so it must carry the flag its own `BundleTensor::is_gemm_operand()` states —
        // the same fact the JSON manifest emits for the same source, so the two cannot disagree.
        let is_gemm = match bundle_tensor {
            Some(t) => t.is_gemm_operand(),
            None => gemm_operands.contains(&i),
        };
        let i = proc_macro2::Literal::u32_unsuffixed(i as u32);
        arms.push(quote! {
            out.push(::scratchy_target_spyre::BoundWeight {
                id: #i,
                tensor: #tensor,
                is_gemm: #is_gemm,
            });
        });
    }

    // ── The embedding table ──
    //
    // ⛔ NOT ONE OF THE SOURCES ABOVE. The tape starts at the ALREADY-EMBEDDED
    // hidden row, so the gather `embed_tokens[token]` is the host's job and the
    // table binds to no launch source. The worker still needs it, and after
    // `try_load` it lives in `Weights` (the generated loader took it out of the
    // weight store), so it cannot be re-read by name from `gw` afterwards — it
    // has to be handed over from the field.
    let embed_field = tw
        .kind_of_field
        .iter()
        .find(|(_, k)| matches!(k, WeightKind::Embedding))
        .map(|(name, _)| syn::Ident::new(name, proc_macro2::Span::call_site()));
    let embed_fn = match embed_field {
        Some(f) => quote! {
            /// The embedding table, for the host-side token gather.
            pub fn superdsc_embed(w: &Weights) -> ::anyhow::Result<::scratchy_target_spyre::GpuTensor> {
                Ok(w.#f.weight)
            }
        },
        None => {
            let stem = "this model";
            quote! {
                pub fn superdsc_embed(_w: &Weights) -> ::anyhow::Result<::scratchy_target_spyre::GpuTensor> {
                    ::anyhow::bail!("{}: no embedding table in the generated Weights", #stem)
                }
            }
        }
    };

    Ok(quote! {
        #embed_fn

        /// Every weight source of this model, bound to a `Weights` FIELD.
        ///
        /// Generated: each entry NAMES a field, so a source the struct cannot
        /// satisfy fails to compile here rather than failing to resolve a
        /// string at model load.
        #[allow(clippy::vec_init_then_push)]
        pub fn superdsc_weights(
            w: &Weights,
        ) -> ::anyhow::Result<::std::vec::Vec<::scratchy_target_spyre::BoundWeight>> {
            let mut out = ::std::vec::Vec::new();
            #(#arms)*
            Ok(out)
        }
    })
}
