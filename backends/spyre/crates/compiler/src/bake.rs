// SPDX-License-Identifier: Apache-2.0
//! The per-model spyre bake: the decode and prefill ladders lowered through the shared
//! `SubtileTape` to KTIR and SuperDSC, the launch wiring and the sharded worlds, emitted as the
//! model module's `KTIR_BUNDLE` / `SENGRAPH_BUNDLE` / `SUPERDSC_*` items.

use std::collections::BTreeMap;

use quote::quote;
use scratchy_forward_compiler_frontend::classified::{OpKind, Program, WeightId};
use scratchy_forward_compiler_macro::codegen::tie_word_embeddings;
use scratchy_forward_compiler_macro::config::ModelParams;
use scratchy_forward_compiler_macro::fuf::{Fuf, FufInput, TileId};

/// PD-wavefront macro-emission (env-gated, diagnostic): route the solved
/// decode FUF through the FULL compile-time wavefront pipeline — bridge
/// (`to_wavefront`) → silu·mul fusion → region N-block tiling — with REAL
/// per-weight [`WeightLoc`]s recovered from this decode bucket's
/// `weight_slots`, and dump the result.
///
/// Lives here (not the pre-emit drive) because `weight_slots` — the real
/// locator source — only exists after lowering + loop compression, and the
/// `(bucket, op_idx, slot)` triples it produces must match the very match
/// table [`emit_weight_accessors_impl`] builds from the same
/// `canonical_lowered`. Every fallible step logs and returns; it never gates
/// a build.
///
/// Under `-Fspyre`, fills `ktir_bundle_out` with the embedded KTIR/SuperDSC
/// bundle for this model. A no-op otherwise: a program with placeholder
/// locators (e.g. dense gate/up fused under one accessor) is not runnable.
/// Stable content fingerprint of a sengraph JSON — the AoT g2 cache key shared
/// between this proc-macro (which dumps `<fp>.sengraph.json`) and
/// `scratchy-builder-spyre` (which compiles + bakes `<fp>.g2.sen`). FNV-1a 64-bit
/// over the bytes, lowercase hex; deterministic, no deps. MUST stay byte-identical
/// to `scratchy_builder_spyre::sengraph_fingerprint` (kept in sync by hand — a
/// drift just misses the cache and falls back to CompileGraph, never miscompiles).
/// Emit the per-model [`scratchy_target_spyre::wiring::Wiring`] value.
///
/// ⛔ EVERY DERIVATION HERE USED TO RUN IN THE WORKER, AT MODEL LOAD, AGAINST
/// PARSED JSON. Doing it at expansion is not an optimisation — it is the
/// difference between a fact and a re-derivation:
///
/// - ROLES were `match s.role.as_str()` over `"embed"`/`"cos"`/`"prefix_k"`;
///   here they are the `SourceBinding` enum the compiler already has, mapped
///   to spyre's own enum, so an unhandled role is a build error not a `_ => {}`.
/// - LAYER WIRING was recovered by scanning every node for one mentioning the
///   mask tensor and then taking *the argument positionally after* the prefix-K
///   arg as `new_k`. That positional inference is done here instead, once,
///   where the emitter's own arg order is authoritative.
/// - GEOMETRY was re-read from `config.json` behind `unwrap_or` defaults — a
///   second source of truth against the bounds the tape was lowered at. It is
///   taken from those same bounds here, so the two cannot disagree.
///
/// ⭐⭐⭐ THE DIFFERENTIAL — the wiring must say EXACTLY what the manifest said.
///
/// `wiring_to_parsed` replaced `parse_bundle`: the worker used to `serde_json` the manifest and
/// recover the per-layer AttnDecode wiring by a POSITIONAL SCAN of the node list, and now reads an
/// emitted array instead. Those two answers have to be the same array, and until this function
/// nothing checked that they were — the replacement was verified by reading it.
///
/// They are computed from one `GraphKtir`, so a disagreement is a bug in one of the two scans, and
/// the consequence is not subtle: the worker indexes `layers[i]` for layer `i`, so a single
/// dropped or reordered entry binds one layer's K/V cache to another layer's tids. On a device
/// with no stack to attach to that is a wrong address, which is a hang or silent corruption.
///
/// The manifest is still emitted (the KTIR emulator's session takes it), so both answers exist at
/// bake and this costs one parse per model.
/// ⭐ THE ROW A TIED DECODE STEP GATHERS — the `RowGather` the sharding analysis reads for a
/// forward whose input row the chained head fills: a tied model's decode, its embedded-hidden
/// source read on device out of the vocabulary projection's own table at the token the previous
/// step drew. `mint_embed_chain` builds that gather on the lowered graph; the analysis needs only
/// the fact, and places the row's all-reduce where it is first read whole when the table splits.
/// A prefill forward's rows are the prompt's, staged whole by the host, and an untied model's
/// decode does not chain: both state none. `decode` is the predicate the chain's mint runs under
/// (a decode rung, never a prefill one), so the two cannot disagree about which bakes chain.
/// ⭐ A RUNG OF THE SHARDED WORLD READS THE DECODE'S STAGED WEIGHTS — one copy per rank, cut by the
/// decode's slice laws — so a rung whose own analysis cuts a staged source differently would read
/// the wrong slice. That is a BUILD error, naming the bundle, the binding and both cuts
/// (`tensor_parallel::staged_cut_apart`).
fn refuse_if_staged_cut_apart(
    stem: &str,
    bundle: &str,
    decode: (
        &[scratchy_subtile::handoff::SourceBinding],
        &[scratchy_subtile::tensor_parallel::SourceShard],
    ),
    rung: (
        &[scratchy_subtile::handoff::SourceBinding],
        &[scratchy_subtile::tensor_parallel::SourceShard],
    ),
) {
    if let Some((binding, decode_cut, rung_cut)) =
        scratchy_subtile::tensor_parallel::staged_cut_apart(decode, rung)
    {
        panic!(
            "[spyre-tp] {stem}: {bundle} cuts {binding:?} as {rung_cut:?}, but the world's decode \
             cuts it as {decode_cut:?} — one staged copy per rank cannot be both"
        );
    }
}

/// A spyre bake's tape: a decoder's, whose embed row is a fact
/// ([`scratchy_subtile::handoff::EmbedRow`]) — every read of it below takes the row, never looks
/// for it.
type BakeTape = scratchy_subtile::handoff::LoweredDecode<scratchy_subtile::handoff::EmbedRow>;

/// The bridge's tape as a spyre bake's ([`BakeTape`]): a decoder gathers an embed row, and a tape
/// that gathers none is no decoder this bake serves — a build error naming the model.
fn bake_tape(
    stem: &str,
    l: scratchy_forward_compiler_macro::to_wavefront::LoweredDecode,
) -> BakeTape {
    l.into_decode().unwrap_or_else(|scratchy_subtile::handoff::NoEmbedRow| {
        panic!(
            "[spyre-superdsc] {stem}: the lowered tape gathers no embed row — a spyre bake lowers \
             a decoder, whose host gather lands in its one EmbeddedHidden source"
        )
    })
}

fn chained_row_gathers(
    lwd: &BakeTape,
    model: &ModelParams,
    decode: bool,
) -> Vec<scratchy_subtile::tensor_parallel::RowGather> {
    if !(decode && tie_word_embeddings(model)) {
        return Vec::new();
    }
    match lwd.input.vocab_table() {
        Some(table) => vec![scratchy_subtile::tensor_parallel::RowGather {
            row: lwd.embed_row().index(),
            table,
        }],
        None => Vec::new(),
    }
}

// ── The rotary-class VALUE law, shared by the wiring's tokens and the gate ──
// The wiring bakes `RotaryKind` TOKENS (`rot_kind` in `emit_superdsc_wiring`
// re-spells this value); the bake-time numeric gate's prefill probe stages
// `rope_cos_sin_kind` rows by the SAME value (`PrefillGate`). One law, two
// consumers — the baked table and the gate's staged rows cannot drift apart.

/// The model's attention head dim — the config's, typed at load ([`scratchy_forward_compiler_macro::config::HeadDim`]: stated,
/// or the loader's one `hidden/heads` derivation).
fn superdsc_model_head_dim(model: &ModelParams) -> u64 {
    u64::from(model.text.head_dim.get())
}

/// The rotary class a `local`-flagged source serves, as a VALUE: sliding
/// sources take the local class's base frequency (the config's stated
/// `rope_local_base_freq`, typed at load — a body with a local source and no
/// stated base is a build error, [`scratchy_forward_compiler_macro::config::TextGeometry::rope_local_for`]); global sources take
/// proportional partial rotary when the config declares it on a WIDER global
/// head dim (gemma-4), else the base class.
fn superdsc_rotary_kind(
    model: &ModelParams,
    hd: u64,
    theta: f64,
    local: bool,
) -> scratchy_target_spyre::wiring::RotaryKind {
    let theta_bits = (theta as f32).to_bits();
    if local {
        let local_theta = model.text.rope_local_for(&model.source_stem).get();
        scratchy_target_spyre::wiring::RotaryKind::Default {
            head_dim: hd as u32,
            theta_bits: (local_theta as f32).to_bits(),
        }
    } else {
        let g_hd = model
            .bounds
            .get("global_head_dim")
            .map(|v| *v as u32)
            .unwrap_or(hd as u32);
        match model.scalars.get("global_partial_rotary_factor").copied() {
            Some(f) if g_hd != hd as u32 && (f - 1.0).abs() > 1e-9 => {
                scratchy_target_spyre::wiring::RotaryKind::Proportional {
                    head_dim: g_hd,
                    theta_bits,
                    rotated: (f * g_hd as f64).round() as u32,
                }
            }
            _ => scratchy_target_spyre::wiring::RotaryKind::Default {
                // Uniform: the class is the base head dim and θ (g_hd == hd
                // here whenever the factor is absent).
                head_dim: g_hd,
                theta_bits,
            },
        }
    }
}

/// One decode batch rung as a bake commits it: its request count, its swept extent, its body
/// fingerprint and its own staged constants.
type DecodeRungEntry = (
    u32,
    u32,
    String,
    Vec<(u32, scratchy_target_spyre::wiring::LoadConst)>,
);

/// ⭐ ONE WORLD'S THREE LADDERS, AS TOKENS — `(prefill_rungs, decode_rungs, prefill_prefix)` for a
/// [`SengraphBundleGroup`](scratchy_target_spyre::manifest::SengraphBundleGroup) or a
/// [`TpForward`](scratchy_target_spyre::wiring::TpForward), from that world's committed rung cells.
/// One renderer for every world, so a sharded world's ladders are spelled exactly as the one-rank
/// world's.
///
/// The prefix-FREE prefill ladder and the decode batch ladder are emitted ONLY when more than one
/// rung baked: a single rung is exactly the world's own `prefill`/`decode` slot, so the empty slice
/// keeps the one-bundle path byte-identical instead of making the worker special-case a one-rung
/// ladder. The prefix-capable ladder (start>0 chunks) ships every width that baked, a single rung
/// included: no other slot carries it.
fn ladder_lits(
    prefill_rungs: &[(u32, String)],
    decode_rungs: &[DecodeRungEntry],
    prefill_prefix_rungs: &[(u32, String)],
) -> (
    Vec<proc_macro2::TokenStream>,
    Vec<proc_macro2::TokenStream>,
    Vec<proc_macro2::TokenStream>,
) {
    let sentinel = |fp: &str| proc_macro2::Literal::string(&format!("SUPERDSC_BUNDLE:{fp}"));
    let mq_rungs = |rungs: &[(u32, String)]| -> Vec<proc_macro2::TokenStream> {
        let mut rungs = rungs.to_vec();
        rungs.sort_by_key(|(mq, _)| *mq);
        rungs
            .iter()
            .map(|(mq, fp)| {
                let m = proc_macro2::Literal::u32_unsuffixed(*mq);
                let s = sentinel(fp);
                quote! { (#m, #s) }
            })
            .collect()
    };
    let prefix = mq_rungs(prefill_prefix_rungs);
    let prefill = if prefill_rungs.len() > 1 {
        mq_rungs(prefill_rungs)
    } else {
        Vec::new()
    };
    let decode = {
        let mut rungs = decode_rungs.to_vec();
        rungs.sort_by_key(|(n, _, _, _)| *n);
        rungs.dedup_by_key(|(n, _, _, _)| *n);
        if rungs.len() > 1 {
            rungs
                .iter()
                .map(|(n, swept, fp, consts)| {
                    // ⛔ `RungSeqs`, NOT a bare u32: the decode ladder is keyed by `decode_rows` (the
                    // BATCH WIDTH), while the sibling `sk_bucket_rungs` list is keyed by `active_cap`
                    // (the SWEEP EXTENT). Both were `(u32, String)`, and the worker selects over this
                    // one with `>= live`. Naming the quantity here is what stops the two ladders being
                    // interchangeable at the call site.
                    let n = proc_macro2::Literal::u32_unsuffixed(*n);
                    let sw = proc_macro2::Literal::u32_unsuffixed(*swept);
                    let sfp = sentinel(fp);
                    // ⭐ THE RUNG'S OWN KINDS — staged by the worker into THIS rung's session, never
                    // the single-request wiring's list: the ten uniform `[rows, 1]` tables' stick
                    // heads are per-width (Finding 1's compounding half). Rendered by the ONE
                    // load-const renderer the wiring static itself uses, so the spellings cannot
                    // drift.
                    let consts = consts
                        .iter()
                        .map(load_const_entry_tokens)
                        .collect::<Vec<_>>();
                    quote! {
                        (
                            ::scratchy_target_spyre::manifest::RungSeqs::new(#n),
                            ::scratchy_target_spyre::manifest::SweptCols::new(#sw),
                            ::scratchy_target_spyre::manifest::BundleSentinel::new(#sfp),
                            &[#(#consts),*],
                        )
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    };
    (prefill, decode, prefix)
}

/// One load-const entry `(tid, kind)` as a token stream — the ONE renderer,
/// shared by the wiring static (`emit_superdsc_wiring`) and every decode
/// batch rung's OWN consts in the manifest (`decode_rung_lits`), so the two
/// can never disagree about how a kind is spelled.
fn load_const_entry_tokens(
    entry: &(u32, scratchy_target_spyre::wiring::LoadConst),
) -> proc_macro2::TokenStream {
    let &(tid, ref kind) = entry;
    let tid = proc_macro2::Literal::u32_unsuffixed(tid);
    let kind = match *kind {
        scratchy_target_spyre::wiring::LoadConst::SampleTable {
            vocab,
            rows,
            operand,
            arrangement,
        } => {
            let vocab = proc_macro2::Literal::u32_unsuffixed(vocab);
            let rows = proc_macro2::Literal::u32_unsuffixed(rows);
            let operand = proc_macro2::Literal::u8_unsuffixed(operand);
            let arrangement = match arrangement {
                scratchy_target_spyre::wiring::ElementArrangement::Standard => {
                    quote! {
                        ::scratchy_target_spyre::wiring::ElementArrangement::Standard
                    }
                }
                scratchy_target_spyre::wiring::ElementArrangement::Dl16ToFp32 => {
                    quote! {
                        ::scratchy_target_spyre::wiring::ElementArrangement::Dl16ToFp32
                    }
                }
            };
            quote! {
                ::scratchy_target_spyre::wiring::LoadConst::SampleTable {
                    vocab: #vocab,
                    rows: #rows,
                    operand: #operand,
                    arrangement: #arrangement,
                }
            }
        }
        scratchy_target_spyre::wiring::LoadConst::ChainStageTable {
            pad_vocab,
            k,
            hidden,
            block,
            operand,
        } => {
            let pad_vocab = proc_macro2::Literal::u32_unsuffixed(pad_vocab);
            let k = proc_macro2::Literal::u32_unsuffixed(k);
            let hidden = proc_macro2::Literal::u32_unsuffixed(hidden);
            let block = proc_macro2::Literal::u32_unsuffixed(block);
            let operand = proc_macro2::Literal::u8_unsuffixed(operand);
            quote! {
                ::scratchy_target_spyre::wiring::LoadConst::ChainStageTable {
                    pad_vocab: #pad_vocab,
                    k: #k,
                    hidden: #hidden,
                    block: #block,
                    operand: #operand,
                }
            }
        }
    };
    quote! { (#tid, #kind) }
}

fn emit_superdsc_wiring(
    gk: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::BundleWiring,
    lwd: &BakeTape,
    model: &ModelParams,
    decode_position: u32,
) -> proc_macro2::TokenStream {
    use scratchy_forward_compiler_macro::to_wavefront::SourceBinding;
    let u32l = proc_macro2::Literal::u32_unsuffixed;

    // ── The DERIVED ANSWERS, resolved at bake ──
    //
    // ⛔ NO ROLE ARRAY IS EMITTED. The worker used to receive `&[Source]` and
    // SCAN it at every model load for the embed row and the cos/sin sources —
    // a compile-time fact re-derived at runtime, which is the same tell as the
    // JSON `role` strings it replaced, only better typed. The exhaustive match
    // over `SourceBinding` still happens, HERE, where a new variant is a build
    // error; what crosses to the runtime is its ANSWER.
    //
    // The embed row is the tape's own ([`BakeTape`]: a decoder's always gathers one); the rotary
    // lookup below REFUSES rather than defaulting: a wiring that carries no rotary is not a wiring.
    let embed_src = lwd.embed_row().index() as u32;
    // ── Geometry, from the bounds the tape was lowered at ──
    let b = |k: &str| model.bounds.get(k).copied().unwrap_or(0);
    let hidden = u32l(b("hidden_size") as u32);
    let vocab = u32l(b("vocab_size") as u32);
    let n_layers_expected = model.text.layers.get();
    let n_layers = u32l(n_layers_expected as u32);
    // head_dim is declared by most arches and derived by the rest — the same
    // `hidden/heads` fallback the emitter itself uses, resolved HERE so the
    // worker never re-derives it (that re-derivation, against a config whose
    // `num_attention_heads` disagreed with the baked bundle, is a known
    // gemma-4-class failure).
    let heads = b("num_attention_heads").max(1);
    let hd = superdsc_model_head_dim(model);
    let head_dim = u32l(hd as u32);
    // ⭐⭐⭐⭐⭐ THE HEAD COUNT ITSELF, NOT THE `hidden/hd` QUOTIENT. Baking `heads`
    // here is what retires the host-side `hidden / head_dim` derivation everywhere:
    // that quotient is the head count only for a SQUARE Q projection, and gemma-4's
    // is not (hidden 3840 against nqh·hd 4096/8192). A worker staging the causal or
    // prefix mask with the quotient head count binds only that many of the baked
    // `nqh·mq` rows — the shortfall is silent (the bind guard refuses over-binds
    // only) and the unstaged tail of an ADDITIVE mask reads as ZERO, i.e. as no
    // mask. MEASURED on gemma-4 tiny-allglobal: heads 2-3 ran unmasked, every layer.
    let nqh = u32l(heads as u32);
    let kv_dim = u32l((b("num_key_value_heads").max(1) * hd) as u32);
    // θ is the config's, typed at load (stated, or its arch's own default), the same value the
    // numeric gate's prefill probe stages its rows by.
    let theta = model.text.rope_theta.get() as f32;
    let theta_bits = u32l(theta.to_bits());

    // cos/sin carry their FULL column width: GQA gives more than one (Q vs K),
    // each filled by tiling the per-position rotary row across heads.
    //
    // ⭐ AND EACH CARRIES ITS ROTARY CLASS. A hybrid-attention arch (gemma-4)
    // gives sliding and global layers DIFFERENT tables (hd/θ/partial-rotary all
    // differ), and the binding's own `local` flag says which class a source
    // serves: `local` = the sliding class's `rotary_local` table, else the
    // global/base class. The class FACTS come from the same bounds the tape was
    // lowered at, so the baked table and the bundle cannot disagree. A uniform
    // model has `local: false` everywhere and `global_head_dim == head_dim`
    // with no partial factor, which folds to the base `Default` class.
    // The class VALUE is one law (`superdsc_rotary_kind` above), shared with
    // the bake-time numeric gate's prefill probe; the tokens below are its
    // spelling — derived, never re-derived, so the wiring's baked tables and
    // the gate's staged rows cannot disagree.
    let rot_kind = |local: bool| -> proc_macro2::TokenStream {
        match superdsc_rotary_kind(model, hd, f64::from(theta), local) {
            scratchy_target_spyre::wiring::RotaryKind::Default {
                head_dim,
                theta_bits,
            } => {
                let hd_lit = u32l(head_dim);
                let tb = u32l(theta_bits);
                quote! {
                    ::scratchy_target_spyre::wiring::RotaryKind::Default {
                        head_dim: #hd_lit, theta_bits: #tb,
                    }
                }
            }
            scratchy_target_spyre::wiring::RotaryKind::Proportional {
                head_dim,
                theta_bits,
                rotated,
            } => {
                let hd_lit = u32l(head_dim);
                let tb = u32l(theta_bits);
                let r = u32l(rotated);
                quote! {
                    ::scratchy_target_spyre::wiring::RotaryKind::Proportional {
                        head_dim: #hd_lit, theta_bits: #tb, rotated: #r,
                    }
                }
            }
        }
    };
    let rot_srcs = |want_cos: bool| -> Vec<(u32, u32, proc_macro2::TokenStream)> {
        lwd.bindings()
            .iter()
            .enumerate()
            .filter(|(_, b)| match b {
                SourceBinding::Cos { .. } => want_cos,
                SourceBinding::Sin { .. } => !want_cos,
                _ => false,
            })
            .map(|(i, b)| {
                let local = match b {
                    SourceBinding::Cos { local } | SourceBinding::Sin { local } => *local,
                    _ => false,
                };
                (i as u32, gk.tensor_shapes[i].1, rot_kind(local))
            })
            .collect()
    };
    let (cos_v, sin_v) = (rot_srcs(true), rot_srcs(false));
    if cos_v.is_empty() || sin_v.is_empty() {
        panic!(
            "[superdsc-wiring] {}: no cos/sin source — refusing to bake a wiring with no rotary",
            model.source_stem
        );
    }
    let pair_toks = |v: &[(u32, u32, proc_macro2::TokenStream)]| -> Vec<proc_macro2::TokenStream> {
        v.iter()
            .map(|(i, w, k)| {
                let (i, w) = (u32l(*i), u32l(*w));
                quote! { (#i, #w, #k) }
            })
            .collect()
    };
    let (cos_toks, sin_toks) = (pair_toks(&cos_v), pair_toks(&sin_v));
    let embed_src_lit = u32l(embed_src);

    let shapes = gk.tensor_shapes.iter().map(|(r, c)| {
        let (r, c) = (u32l(*r), u32l(*c));
        quote! { (#r, #c) }
    });

    // ── Per-layer pools ──
    // Every paged attention program binds `[q, out, k pool, v pool, page table, mask]`: the node
    // that reads the mask names its layer's K pool (a `PrefixK` source) and V pool (a `PrefixV`
    // source). Nothing else of the layer leaves the device — the step's K/V are scattered into the
    // pools by the layer's own write — so the pools are the whole per-layer wiring.
    let mut layers: Vec<(u64, u32, u32)> = Vec::new();
    if let Some(mask) = gk.attn_mask {
        let pk: std::collections::HashMap<u32, u64> = lwd
            .bindings()
            .iter()
            .enumerate()
            .filter_map(|(i, b)| match b {
                SourceBinding::PrefixK { layer } => Some((i as u32, *layer)),
                _ => None,
            })
            .collect();
        let pv: std::collections::HashSet<u32> = lwd
            .bindings()
            .iter()
            .enumerate()
            .filter_map(|(i, b)| match b {
                SourceBinding::PrefixV { .. } => Some(i as u32),
                _ => None,
            })
            .collect();
        for node in &gk.nodes {
            if !node.args.iter().any(|a| *a as u32 == mask) {
                continue;
            }
            let k_pool = node
                .args
                .iter()
                .map(|a| *a as u32)
                .find(|a| pk.contains_key(a));
            let v_pool = node.args.iter().map(|a| *a as u32).find(|a| pv.contains(a));
            if let (Some(ks), Some(vs)) = (k_pool, v_pool) {
                layers.push((pk[&ks], ks, vs));
            }
        }
        layers.sort_by_key(|x| x.0);
    }
    // ══════════════════════════════════════════════════════════════════════════════════════════
    //  ⛔⛔⛔ THE COMPLETENESS LOCK — a wiring that does not fill every caller-filled source
    //  MAY NOT COMPILE.
    //
    //  `num_sources` says ids `0..num_sources` are FILLED BY THE HOST every forward. Four things
    //  fill them: the generated weight loader (`Weight`/`WeightScale`), the forward tape's embed
    //  gather (`EmbeddedHidden`) and rotary kernels (`Cos`/`Sin`), and the resident KV pool
    //  (`PrefixK`/`PrefixV`, bound per layer through `LayerWiring`). A source that NONE of them
    //  covers is never written — and the device does not report that. It waits, or it reads
    //  whatever the segment held.
    //
    //  Every check below is a hole that existed and was silent:
    //
    //    • `layers` is built by POSITIONAL SCAN over each AttnDecode node's args, and its
    //      `if let (Some, Some, Some, Some, Some)` DROPPED a layer whose pattern did not match.
    //      Forty layers in, thirty-nine wired, no message.
    //    • `embed_src` is `.position(..)` — the FIRST `EmbeddedHidden`. A second one is never
    //      filled by anything.
    //    • `cos_srcs`/`sin_srcs` are collected by filter from the same bindings, so they cannot
    //      miss one, and nothing here re-checks it.
    //
    //  This is the sibling of `superdsc_bake.rs`'s no-device-compiler refusal, and for the same
    //  reason it gives: the binary would link, every session would report ready, and the failure
    //  would surface as a hang or as empty output with no error anywhere.
    {
        use scratchy_forward_compiler_macro::to_wavefront::SourceBinding as SB;

        let mut fail: Vec<String> = Vec::new();

        // ── every layer's prefix KV is wired ──
        if layers.len() != n_layers_expected {
            fail.push(format!(
                "{} of {n_layers_expected} layers have prefix-KV wiring. The rest were dropped by \
                 the positional arg scan; their K/V cache would never be bound",
                layers.len(),
            ));
        }
        let mut seen_layers: Vec<u64> = layers.iter().map(|x| x.0).collect();
        seen_layers.sort_unstable();
        seen_layers.dedup();
        if seen_layers.len() != layers.len() {
            fail.push(format!(
                "two AttnDecode nodes claim the same layer index — one layer's cache would be \
                 bound twice and another's not at all ({:?})",
                layers.iter().map(|x| x.0).collect::<Vec<_>>()
            ));
        }

        // ── every caller-filled source is covered by exactly one filler ──
        // ⭐ THE SAMPLER'S PER-STEP SEAT — ITS OWN CLASS AT THIS GATE. The seat's
        // five tids (the seed quadruple's four word tables and inv_t) are minted
        // below the `LoweredDecode` binding layer (the mint's own law), so they
        // carry no `SourceBinding`, and the CLASS declares their filler: the
        // worker binds every one FRESH EVERY DECODE STEP, and the runtime launch
        // gate refuses a decode without a fresh bind of all of them — the
        // PrefixMask precedent, never a held default, because a held seed row is
        // a held TOKEN. This block is the class's declaration at the expansion
        // gate (the loop below counts the seat's tids as covered BY CLASS); the
        // per-step enforcement is the runtime gate's own arm. Everything else
        // about the seat is typed here, where the wiring is in hand:
        if let Some(seat) = &gk.sample_seat {
            let mut seat_tids: Vec<(String, u32)> = seat
                .seed_words
                .iter()
                .enumerate()
                .map(|(k, &tid)| (format!("seed word {k}"), tid))
                .collect();
            seat_tids.push(("invt".to_string(), seat.invt));
            for (name, tid) in seat_tids {
                if tid >= gk.num_sources {
                    fail.push(format!(
                        "the sample seat's {name} source t{tid} is not inside `num_sources` \
                         ({}) — nothing can ever bind it",
                        gk.num_sources
                    ));
                }
                if gk.load_consts.iter().any(|&(ltid, _)| ltid == tid) {
                    fail.push(format!(
                        "the sample seat's {name} source t{tid} is ALSO a load-time const — a \
                         tensor cannot both stage once at load and be the step's own freshness"
                    ));
                }
            }
            let seat_all: std::collections::HashSet<u32> =
                seat.seed_words.iter().copied().chain([seat.invt]).collect();
            if seat_all.len() != 1 + seat.seed_words.len() {
                fail.push(
                    "the sample seat names one tid twice — the mint mints one source per seat \
                     operand, so a shared tid means two mints disagree"
                        .to_string(),
                );
            }
            if !matches!(
                gk.result_dtype,
                scratchy_target_spyre::DType::I32 | scratchy_target_spyre::DType::U32
            ) {
                fail.push(format!(
                    "the rung carries the sample seat but its result dtype is {:?}, not an \
                     index — the mint re-points the result to the idx stick, so a non-integer \
                     result means the wiring and the mint disagree",
                    gk.result_dtype
                ));
            }
        }
        // ⭐ THE CHAINED HEAD'S STICKS — the seat's own class law, stated for the
        // chain: the two persistent `[1, CHAIN_IDX_BLOCK]` u32 sources are minted
        // below the `LoweredDecode` binding layer, so they carry no
        // `SourceBinding`, and the CLASS declares their filler — the tape's
        // ChainSeed steps bind both (the s stick's lane pattern, the l stick's
        // ramp), fresh every step, idempotent with the device's own ChainStage
        // writes. The two decompose TABLES ride `load_consts` like the sampler's
        // nine (staged once at load) and need no class of their own.
        let chain_sticks: Option<(u32, u32)> = match &gk.chain {
            scratchy_target_spyre::wiring::ChainFact::Chained { chain } => {
                for (name, tid) in [("s", chain.s_src), ("l", chain.l_src)] {
                    if tid >= gk.num_sources {
                        fail.push(format!(
                            "the chain's {name} stick t{tid} is not inside `num_sources` ({}) — \
                             nothing can ever bind it",
                            gk.num_sources
                        ));
                    }
                    if gk.load_consts.iter().any(|&(ltid, _)| ltid == tid) {
                        fail.push(format!(
                            "the chain's {name} stick t{tid} is ALSO a load-time const — the \
                             sticks are per-step state the device rewrites, never load-staged"
                        ));
                    }
                }
                if chain.s_src == chain.l_src {
                    fail.push(
                        "the chain names one tid as both the s and the l stick — the mint mints \
                         two sources, so a shared tid means two mints disagree"
                            .to_string(),
                    );
                }
                Some((chain.s_src, chain.l_src))
            }
            scratchy_target_spyre::wiring::ChainFact::Unchained => None,
        };
        let mut embeds = 0usize;
        for id in 0..gk.num_sources {
            let Some(b) = lwd.bindings().get(id as usize) else {
                // ⭐ A LOAD-CONST SOURCE — minted below the `LoweredDecode`
                // binding layer, so it carries NO `SourceBinding`: the wiring's
                // `load_consts` list names it (BY MEMBERSHIP in the generic
                // list, never by the op that minted it), and the worker stages
                // its bytes once at load through the kind's builder — the rope
                // tables' own law. This arm is that list's declaration; every
                // other binding-less source is still the
                // hang-or-silent-wrong refusal below.
                if gk.load_consts.iter().any(|&(tid, _)| tid == id) {
                    continue;
                }
                // ⭐ THE SEAT'S OWN ARM — the per-step class the block above declares:
                // covered BY CLASS (the worker's fresh-every-step bind), which is why
                // this is a `continue` and not a refusal. The loud refusal lives at the
                // RUNTIME gate, per launch, where "this step" is a fact — an expansion
                // cannot know it.
                if let Some(seat) = &gk.sample_seat
                    && (seat.seed_words.contains(&id) || seat.invt == id)
                {
                    continue;
                }
                // ⭐ THE CHAIN'S OWN ARM — the per-step class the block above
                // declares: covered BY CLASS (the tape's ChainSeed steps bind
                // both sticks every step), which is why this is a `continue`
                // and not a refusal.
                if let Some((s_src, l_src)) = chain_sticks
                    && (s_src == id || l_src == id)
                {
                    continue;
                }
                fail.push(format!(
                    "source t{id} is inside `num_sources` but has NO binding — nothing fills it"
                ));
                continue;
            };
            // ⛔ EXHAUSTIVE, NO CATCH-ALL. A new role must be taught a filler here, at build time.
            match b {
                // The generated weight loader binds these once, before prepare.
                SB::Weight { .. } | SB::WeightScale { .. } | SB::BundleTensor { .. } => {}
                SB::EmbeddedHidden => {
                    // ⭐ THE CHAIN'S RESIDENT EXCUSE — the replaced row's host
                    // filler is gone (the tape's ChainSeed steps replaced the
                    // embed-row push), but the row is not unbound: the device's
                    // own EmbedRowGather writes it every step. The chain names
                    // the replaced id at the mint; this gate reads the same
                    // statement, never a re-derivation.
                    if let scratchy_target_spyre::wiring::ChainFact::Chained { chain } = &gk.chain
                        && id == chain.replaces
                    {
                        continue;
                    }
                    embeds += 1;
                    if id != embed_src {
                        fail.push(format!(
                            "source t{id} is an EmbeddedHidden the tape does not gather \
                             (it gathers t{embed_src}) — t{id} is never written"
                        ));
                    }
                }
                // `cos_srcs`/`sin_srcs` ARE every `Cos`/`Sin` binding (`rot_srcs` collects them
                // from these bindings), so each is in its list by construction.
                SB::Cos { .. } | SB::Sin { .. } => {}
                // Device-resident, bound per layer through `LayerWiring` — covered by the layer
                // count check above.
                SB::PrefixK { .. } | SB::PrefixV { .. } => {}
            }
        }
        // A chained decode carries its EmbeddedHidden row as DEVICE-WRITTEN
        // (excused above), so the tape-side count is legitimately zero there.
        let embed_floor = if chain_sticks.is_some() { 0 } else { 1 };
        if embeds < embed_floor {
            fail.push(format!(
                "{embeds} EmbeddedHidden sources; the tape gathers exactly one row"
            ));
        }

        if !fail.is_empty() {
            panic!(
                "\n⛔ [superdsc-wiring] {}: THE LAUNCH WIRING DOES NOT FILL EVERY CALLER-FILLED \
                 SOURCE. Refusing to bake.\n\n{}\n\nA binary built from this wiring LINKS AND \
                 SERVES: the session reports ready and the launch is issued, but a source nobody \
                 writes leaves the device reading memory the host never filled — which surfaces as \
                 a HANG or as fluent-but-wrong output, with no error anywhere. Same reason \
                 `superdsc_bake` refuses a plan with no device programs.\n",
                model.source_stem,
                fail.iter()
                    .map(|f| format!("  • {f}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════
    //  ⛔⛔⛔ A GEMM WEIGHT'S DECLARED SHAPE IS `[k, n]`, AND THE CONSUMING OP SAYS WHICH IS WHICH.
    //
    //  `tensor_shapes[w]` is the pair the host stages from: it decides the orientation guard, the
    //  device-width pad, the K-split gate and the emitted shape. It is `[rows = k, cols = n]` — the
    //  LOGICAL `[in, out]` — and the ONLY evidence of that was a comment.
    //
    //  Reading it as `[n, k]` is not an error, it is a plausible answer, and it shipped: the pad
    //  became `for_output(1, k, n)`, a NO-OP for granite's lm_head (k = 2048 is already 32 sticks)
    //  where its n (49155 → 769 PRIME sticks) needs the full-occupancy bump to 51200. The host
    //  staged 49155 unpadded columns into a placement sized for 51200.
    //
    //  The consuming `Gemm` carries `n`. So the orientation is checkable, and this checks it.
    // ══════════════════════════════════════════════════════════════════════════════════════════
    {
        use scratchy_forward_compiler_macro::to_wavefront::SourceBinding as SB;
        use scratchy_subtile::lower::InputRef;
        use scratchy_subtile::subtile_ir::SubOp;
        let mut bad: Vec<String> = Vec::new();
        for od in &lwd.input.ops {
            let SubOp::MatmulTile { n, .. } = od.op else {
                continue;
            };
            let Some(InputRef::Ext(e)) = od.inputs.get(1) else {
                continue;
            };
            // Only a real weight has a staged shape; a fp8 scale is a different tensor.
            if !matches!(lwd.bindings().get(*e), Some(SB::Weight { .. })) {
                continue;
            }
            let Some(&(rows, cols)) = gk.tensor_shapes.get(*e) else {
                bad.push(format!("gemm weight t{e} has no entry in tensor_shapes"));
                continue;
            };
            if cols != n {
                bad.push(format!(
                    "gemm weight t{e}: the op produces n={n}, but tensor_shapes says \
                     [rows={rows}, cols={cols}] — cols is supposed to BE n. If these are \
                     transposed, every host consumer of this pair pads, gates and shapes the \
                     wrong axis"
                ));
            }
        }
        if !bad.is_empty() {
            panic!(
                "\n⛔ [superdsc-wiring] {}: A GEMM WEIGHT'S DECLARED SHAPE DISAGREES WITH ITS \
                 CONSUMING OP. Refusing to bake.\n\n{}\n",
                model.source_stem,
                bad.iter()
                    .map(|b| format!("  • {b}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
    }

    let layer_toks = layers.iter().map(|(_, ks, vs)| {
        let (ks, vs) = (u32l(*ks), u32l(*vs));
        quote! {
            ::scratchy_target_spyre::wiring::LayerWiring {
                prefix_k: #ks, prefix_v: #vs,
            }
        }
    });

    // ── Geometry, from the bounds the tape was lowered at ──
    // (Moved above the rotary-source emission: `hd` and `theta` are the base
    // class's facts, and each cos/sin source's `RotaryKind` reads them.)
    let result = u32l(gk.result_tensor);
    // ⭐ THE RESULT'S OWN DTYPE — straight off the bake's one derivation
    // (`result_dtype_of_ir`, the result producer's own kernel signature), never
    // recomputed here: it is the fact that tells the worker whether the rung's
    // answer is the f16 row (convert and sample on the host) or the i32 idx
    // stick (read raw — the device already drew the token).
    // `scratchy-tensors` is only a dev-dependency where this expands (the
    // models crate), so the literal rides the spyre crate's own root re-export
    // — the same `::scratchy_target_spyre::` prefix every other wiring type
    // in this emission uses.
    let result_dtype = match gk.result_dtype {
        scratchy_target_spyre::DType::F16 => quote! { ::scratchy_target_spyre::DType::F16 },
        scratchy_target_spyre::DType::BF16 => quote! { ::scratchy_target_spyre::DType::BF16 },
        scratchy_target_spyre::DType::F32 => quote! { ::scratchy_target_spyre::DType::F32 },
        scratchy_target_spyre::DType::U32 => quote! { ::scratchy_target_spyre::DType::U32 },
        scratchy_target_spyre::DType::I32 => quote! { ::scratchy_target_spyre::DType::I32 },
        // `result_dtype_of_ir` refuses every dtype the readback cannot name, so
        // this arm is unreachable by construction — spelled out so a new DType
        // variant fails HERE rather than silently lower in the wiring.
        other => panic!(
            "[superdsc-wiring] {}: the result's dtype {other:?} has no wiring literal — the \
             readback cannot read it back",
            model.source_stem
        ),
    };
    let num_sources = u32l(gk.num_sources);
    let dp = u32l(decode_position);
    let mask = match gk.attn_mask {
        Some(m) => {
            let m = u32l(m);
            quote! { ::core::option::Option::Some(#m) }
        }
        None => quote! { ::core::option::Option::None },
    };
    // ⭐ THE PAGED STEP, as the bake derived it (`paged_step_of_ir`) — the worker stages its three
    // sources every step against this pool.
    let paged = match gk.paged {
        Some(p) => {
            let (m, t, w) = (u32l(p.mask), u32l(p.page_table), u32l(p.write_rows));
            let (q, kvp) = (u32l(p.query_rows), u32l(p.kv_row_pool));
            let (rr, cp, cm) = (u32l(p.rep_rows), u32l(p.chunk_pages), u32l(p.chunk_masks));
            let zero = u32l(p.int32_zero);
            let (pages, slots) = (u32l(p.pages), u32l(p.page_slots));
            let (rows, nkvh, walk) = (u32l(p.rows), u32l(p.nkvh), u32l(p.walk));
            let (q_width, head_size) = (u32l(p.q_width), u32l(p.head_size));
            // ⭐ THE EXTENT PAIR, re-minted in the wiring's own static by the ONE mint
            // (`QueryExtent::of_baked_rung`, the same constructor the bake's codegen minted
            // the pair by) — const, so the static carries it. Every `None` arm names a pair
            // that does not describe a rung, which the bake's own mint already refused, so it
            // is a build failure here and never a runtime fact (const panic takes a literal, so
            // the messages are). The types ride the spyre crate's own wiring re-export (the
            // models crate does not name `scratchy-subtile` directly).
            let (ew, es) = (
                u32l(p.extent.per_request().get()),
                u32l(p.extent.requests().get()),
            );
            let extent = quote! {
                match ::scratchy_target_spyre::wiring::QueryExtent::of_baked_rung(
                    match ::scratchy_target_spyre::wiring::RungWidth::of_baked_rows(#ew) {
                        ::core::option::Option::Some(w) => w,
                        ::core::option::Option::None => ::core::panic!(
                            "the wiring's per-request width does not name a baked rung — a bake \
                             bug this static cannot carry"
                        ),
                    },
                    match ::scratchy_target_spyre::wiring::RungCap::of_baked_seqs(#es) {
                        ::core::option::Option::Some(s) => s,
                        ::core::option::Option::None => ::core::panic!(
                            "the wiring's request count does not name a baked rung — a bake bug \
                             this static cannot carry"
                        ),
                    },
                ) {
                    ::core::option::Option::Some(e) => e,
                    ::core::option::Option::None => ::core::panic!(
                        "the wiring's query extent does not name a baked rung — a bake bug this \
                         static cannot carry"
                    ),
                }
            };
            let paged_rows = match p.paged_rows {
                scratchy_target_spyre::paged_step::PagedRows::Batch => {
                    quote! { ::scratchy_target_spyre::paged_step::PagedRows::Batch }
                }
                scratchy_target_spyre::paged_step::PagedRows::Sequence => {
                    quote! { ::scratchy_target_spyre::paged_step::PagedRows::Sequence }
                }
            };
            quote! {
                ::core::option::Option::Some(::scratchy_target_spyre::wiring::PagedStep {
                    mask: #m, page_table: #t, write_rows: #w, query_rows: #q, kv_row_pool: #kvp,
                    rep_rows: #rr, chunk_pages: #cp, chunk_masks: #cm, int32_zero: #zero,
                    pages: #pages, page_slots: #slots, rows: #rows, nkvh: #nkvh, q_width: #q_width,
                    head_size: #head_size, walk: #walk,
                    paged_rows: #paged_rows,
                    extent: #extent,
                })
            }
        }
        None => quote! { ::core::option::Option::None },
    };

    // ⭐ THE KERNEL-TABLE CLASS SETS, FROM THE BAKE — NOT THE TABLES THEMSELVES. The tables are
    // pure functions of each class's head dim, but baking them as literals was the gemma-4 defect
    // in two ways at once: the literals were staged at the BASE `head_dim` while the placement
    // pass sized the tids at the MAX over the tape's classes (a `[64,64]` table in a `[128,128]`
    // placement — the wide class's rope rotated pairs at half=32 and its GQA krep read slab 1 as
    // zeros, dump-proven), and a hybrid's widest class is `hd = 512`, whose 512² table is 262,144
    // f32 literals for one field — the expansion-cost rule broken at its most expensive.
    //
    // So the bake carries the class HEAD DIMS (the same lists `gk` derived from the same `ir` the
    // placement pass read — one source of truth), and the worker's `synthetic_constants` builds
    // each class's table at load through the SAME shared helpers the emission used
    // (`stage_kernel_table` + `rope_p_entry`), with `Wiring::verify_class_placements`
    // cross-checking the sets against the baked placements. Class 0 of each set is the widest and
    // keeps the pre-class sentinel tid, so a uniform model's binds are byte-identical.
    let rope_class_hds: Vec<proc_macro2::Literal> = gk
        .rope_class_hds
        .iter()
        .map(|v| proc_macro2::Literal::u32_unsuffixed(*v))
        .collect();
    let attn_class_hds: Vec<proc_macro2::Literal> = gk
        .attn_class_hds
        .iter()
        .map(|v| proc_macro2::Literal::u32_unsuffixed(*v))
        .collect();
    // `1/hidden` over one stick — the mq>1 sum-based amax pre-scale.
    let rms_invcols_lits: Vec<proc_macro2::Literal> = {
        let inv = 1.0f32 / (b("hidden_size").max(1) as f32);
        (0..64)
            .map(|_| proc_macro2::Literal::f32_suffixed(inv))
            .collect()
    };
    // ⭐ EVERY COMPILE-TIME SCALAR THE KTIR READS, from the bake's own registry — not recomputed
    // here from the config. The programs bake the INDEX into themselves, so the list
    // the worker binds has to be the same list, in the same order, that the lowering indexed.
    // ⭐ AS BITS: the registry holds the page walk's `-inf` seed, which no float literal spells.
    let scalarmul_scale_lits: Vec<proc_macro2::TokenStream> =
        gk.scalarmul_scales.iter().map(|v| f32_bits(*v)).collect();
    // The ScalarMul weight folds, straight off the bake's recognition — the worker scales these
    // staged weights at load, so the list it reads has to be the one the lowering recognized.
    let weight_scale_fold_lits: Vec<proc_macro2::TokenStream> = gk
        .weight_scale_folds
        .iter()
        .map(|(tid, scale)| {
            let tid = proc_macro2::Literal::u32_suffixed(*tid);
            let scale = proc_macro2::Literal::f32_suffixed(*scale);
            quote! { (#tid, #scale) }
        })
        .collect();

    // The VALUE, not a `static` — the caller composes decode + prefill into one
    // `Wirings` const, because a bundle's tensor ids are per-PROGRAM and the
    // prefill program has its own.
    // ⭐ THE DENSE fp8 K-PADS — same rule, same tape, as the retile's own `fp8_k_pad` (the
    // worker rebuilds the on-disk `[n, k]` buffer's inner axis to `k_pad` so the staged bytes
    // match the packed retile). Computed HERE off `gk`/`lwd.input.ops` because the macro is
    // where the wiring literal is rendered; the orientation (which axis is k) comes from the
    // CONSUMING op, the same evidence the retile uses.
    let fp8_k_pads: Vec<(u32, u32)> = {
        use scratchy_subtile::lower::InputRef;
        use scratchy_subtile::subtile_ir::SubOp;
        let mut pads: Vec<(u32, u32)> = Vec::new();
        for od in &lwd.input.ops {
            let SubOp::MatmulTile { .. } = od.op else {
                continue;
            };
            if od.inputs.len() != 3 {
                continue;
            }
            let Some(InputRef::Ext(e)) = od.inputs.get(1) else {
                continue;
            };
            let Some(&(rows, _cols)) = gk.tensor_shapes.get(*e) else {
                continue;
            };
            let k_pad = scratchy_target_spyre::lower_subtile_tape_to_superdsc::fp8_k_pad(rows);
            if k_pad > rows {
                pads.push((*e as u32, k_pad));
            }
        }
        pads.sort();
        pads.dedup();
        pads
    };
    let fp8_k_pad_toks = fp8_k_pads
        .iter()
        .map(|&(id, k)| {
            let (id, k) = (u32l(id), u32l(k));
            quote! { (#id, #k) }
        })
        .collect::<Vec<_>>();
    // ⭐ THE LOAD-TIME CONSTANTS — straight off the bake's own one derivation
    // (`load_consts_of_ir`, the same list the numeric gate stages), never
    // recomputed here: each entry is `(tid, kind)` and the KIND names the
    // shared builder that makes the bytes, so the worker and the gate stage
    // identical bytes by construction. The sampler's TWENTY staged tables
    // ride this list (the selection's f32, the noise section's i32/f32/fp16
    // — each entry's own dtype); its four seed word tables and inv_t do NOT —
    // they are per-step seats the worker binds fresh every decode, and a
    // load-time constant staged over them would freeze one draw's seed into
    // every step.
    let load_const_toks = gk
        .load_consts
        .iter()
        .map(load_const_entry_tokens)
        .collect::<Vec<_>>();
    // ⭐ THE SAMPLER'S PER-STEP SEAT — the named class of the five sources the
    // token draw binds FRESH EVERY DECODE STEP (the seed quadruple's four
    // `[rows, 1]` word tables and inv_t), straight off the bake's one
    // derivation (`sample_seat_of_ir`, the mint's own facts), never
    // recomputed here. `None` on every rung that mints no draw; `Some` names
    // the seed word and inv_t tids the worker must bind before EVERY decode
    // launch — the class the source gate counts as filled only when bound
    // THIS step.
    let sample_seat_tok = match &gk.sample_seat {
        Some(seat) => {
            let seed_words = seat.seed_words.iter().map(|&t| u32l(t)).collect::<Vec<_>>();
            let invt = u32l(seat.invt);
            let rows = u32l(seat.rows);
            let row = u32l(seat.row);
            let vocab = u32l(seat.vocab);
            let pad_vocab = u32l(seat.pad_vocab);
            let seed_row_words = u32l(seat.seed_row_words);
            let gate_row_words = u32l(seat.gate_row_words);
            let idx_row_words = u32l(seat.idx_row_words);
            quote! {
                ::core::option::Option::Some(
                    ::scratchy_target_spyre::wiring::SampleSeat {
                        seed_words: [#(#seed_words),*],
                        invt: #invt,
                        rows: #rows,
                        row: #row,
                        vocab: #vocab,
                        pad_vocab: #pad_vocab,
                        seed_row_words: #seed_row_words,
                        gate_row_words: #gate_row_words,
                        idx_row_words: #idx_row_words,
                    }
                )
            }
        }
        None => quote! { ::core::option::Option::None },
    };
    // ⭐ THE FORWARD'S OWN PHASE — a projection of the wiring's own chain
    // fact, never a second derivation: a rung the mint chained is a DECODE
    // rung (the mint runs on decode slots only, and it refuses — panics — a
    // decode slot it cannot chain), so `Chained` ⇒ `Decode { chain }` and
    // `Unchained` ⇒ `Prefill`, byte-identical to the pre-mint behavior.
    let phase_tok = match &gk.chain {
        scratchy_target_spyre::wiring::ChainFact::Chained { chain } => {
            let s_src = u32l(chain.s_src);
            let l_src = u32l(chain.l_src);
            let k = u32l(chain.k);
            let hidden = u32l(chain.hidden);
            let replaces = u32l(chain.replaces);
            let block = u32l(chain.block);
            quote! {
                ::scratchy_target_spyre::forward_tape::ForwardPhase::Decode {
                    chain: ::scratchy_target_spyre::wiring::ChainWiring {
                        s_src: #s_src,
                        l_src: #l_src,
                        k: #k,
                        hidden: #hidden,
                        replaces: #replaces,
                        block: #block,
                    },
                }
            }
        }
        scratchy_target_spyre::wiring::ChainFact::Unchained => {
            quote! { ::scratchy_target_spyre::forward_tape::ForwardPhase::Prefill }
        }
    };
    quote! {
        ::scratchy_target_spyre::wiring::Wiring {
            result: #result,
            result_dtype: #result_dtype,
            num_sources: #num_sources,
            attn_mask: #mask,
            paged: #paged,
            decode_position: #dp,
            tensor_shapes: &[#(#shapes),*],
            embed_src: #embed_src_lit,
            cos_srcs: &[#(#cos_toks),*],
            sin_srcs: &[#(#sin_toks),*],
            layers: &[#(#layer_toks),*],
            geometry: ::scratchy_target_spyre::wiring::Geometry {
                hidden: #hidden,
                kv_dim: #kv_dim,
                vocab: #vocab,
                layers: #n_layers,
                head_dim: #head_dim,
                nqh: #nqh,
                rope_theta_bits: #theta_bits,
            },
            rope_class_hds: &[#(#rope_class_hds),*],
            attn_class_hds: &[#(#attn_class_hds),*],
            rms_invcols: &[#(#rms_invcols_lits),*],
            scalarmul_scales: &[#(#scalarmul_scale_lits),*],
            weight_scale_folds: &[#(#weight_scale_fold_lits),*],
            fp8_k_pads: &[#(#fp8_k_pad_toks),*],
            load_consts: &[#(#load_const_toks),*],
            sample_seat: #sample_seat_tok,
            phase: #phase_tok,
        }
    }
}

/// The SuperDSC AoT cache root (peer of `sengraphforge_cache_dir`). The
/// proc-macro DROPS one bundle dir `<fp>/` (sdsc_{i}.json + bundle.mlir) per
/// runnable graph here when `SCRATCHY_SENDNN_MODE=superdsc`; build.rs will
/// `dxp_standalone --bundle -d <fp>` each into a device program. Fixed path so
/// both sides agree without a build-graph edge — same contract as sengraphforge.
/// ⭐⭐⭐⭐⭐ TURN THE EMITTED BUNDLES INTO RUST STRUCTS — the whole of getting a SuperDSC bundle into
/// the binary.
///
/// The emitter hands over a `Vec<BundleCode>` (`drain_emitted_bundles`) and this renders each as a
/// `const`-constructed value inside `inventory::submit!`, which the runtime looks up by fingerprint.
/// Every quantity stays in its own type from the lowering that computed it to the launch that reads it.
///
/// ## The device image is `include_bytes!`, packed
///
/// One model's rung ladder is ~100-290 MB of device code, and that many tokens would not survive the
/// macro, let alone `prettyplease` — so the images are written to `OUT_DIR` and referenced by
/// `include_bytes!`. That is a mechanism for getting BYTES into a binary, not a serialization format:
/// nothing parses it, at build time or after. `OUT_DIR` is cargo's own per-build directory for the crate
/// being expanded, and every arch crate has a `build.rs`, so it is set here.
///
/// The bytes are the metal AOT precedent's packed form (`crates/targets/metal/src/aot.rs`, `pack`):
/// ONE dictionary trained on every image the build bakes, each DISTINCT image one zstd frame against
/// it, all frames in one blob. Measured on granite-3.1-8b fp8, the images are ~97% redundant —
/// 156 MB of distinct device code packs to single-digit MB. A launch group's token names its frame's
/// ROW in the one `BakedImages` static; the card crate decompresses at the door that uploads it, so
/// the card receives byte-identical bytes and nothing between here and there parses anything.
#[cfg_attr(not(feature = "spyre-lower"), allow(unused_variables))]
fn superdsc_bundle_tokens(model: &ModelParams) -> proc_macro2::TokenStream {
    // ⭐⭐⭐ WHAT IS INVENTORIED DEPENDS ON THE DEVICE, AND ONLY ON THE DEVICE.
    //
    // On a CARD the launch names a resident device image by VA, so that image is compiled from this
    // bundle's KTIR at bake time and the emit must drain the compiler before it can read the
    // artifacts back. That is this barrier, and its refusal on a host without one is the right
    // answer: a bundle whose programs never compiled would serve, launch nothing, and return empty
    // completions with no error anywhere.
    //
    // ⛔ ON THE EMULATOR THERE IS NOTHING TO COMPILE. The device runs the KTIR itself, so the
    // programs the group already carries ARE what gets inventoried — nothing is submitted to that
    // queue and nothing reads a result from it. Draining it would refuse a build that has every
    // program it needs.
    #[cfg(feature = "spyre-hw")]
    match scratchy_target_spyre::superdsc_bake::finish_global() {
        Ok(Some(stats)) => eprintln!(
            "[spyre] the device compiler built {} launch group(s) inline during the emit ({:.1} MB \
             of device code); {} skipped as byte-identical to one already compiled; {} kept from \
             an earlier build; peak staging {:.1} MB",
            stats.groups,
            stats.device_bytes as f64 / 1048576.0,
            stats.memo_hits,
            stats.disk_hits,
            stats.peak_staged_bytes as f64 / 1048576.0,
        ),
        Ok(None) => {}
        Err(e) => panic!("[spyre] {e}"),
    }
    // ⭐ EVERY WORLD'S BUNDLES, EACH AS ITS WORLD'S TYPE: the one-rank forward's and, when the build
    // names a sharded world, `Build`'s. A bundle is rendered with its world spelled in its type
    // (`BundleCode::<'static, W>`), so the registry downcast finds it in that world alone.
    use scratchy_target_spyre::bundle_code::collective::{Build, TpWorld, World};
    let alone = match scratchy_target_spyre::lower_subtile_tape_to_superdsc::drain_emitted_bundles::<
        TpWorld<1>,
    >() {
        Ok(b) => b,
        Err(e) => panic!("[spyre-superdsc] {e}"),
    };
    let sharded = if <Build as World>::SHARDED {
        match scratchy_target_spyre::lower_subtile_tape_to_superdsc::drain_emitted_bundles::<Build>(
        ) {
            Ok(b) => b,
            Err(e) => panic!("[spyre-superdsc] {e}"),
        }
    } else {
        Vec::new()
    };
    if alone.is_empty() && sharded.is_empty() {
        return proc_macro2::TokenStream::new();
    }
    let images = match std::env::var("OUT_DIR") {
        Ok(d) => std::path::PathBuf::from(d).join("superdsc-code"),
        // No OUT_DIR means we are not inside a build script's expansion, so there is nowhere to put
        // the images and nothing that could `include_bytes!` them.
        Err(_) => return proc_macro2::TokenStream::new(),
    };
    let mut out = proc_macro2::TokenStream::new();
    let (mut n_groups, mut n_bytes) = (0usize, 0usize);
    // ⭐ ONE INTERNER FOR THE WHOLE EMISSION — every bundle's launches reach the same 33-ish
    // programs, so they share one set of `const` items rather than each carrying its own copy.
    let mut interner = crate::ktir_tokens::ProgramInterner::default();
    let mut places_interner = PlacesInterner::default();
    // ⭐ ONE IMAGE INTERNER FOR THE WHOLE EMISSION — both worlds' groups share one blob, one
    // dictionary, one row table, exactly as they share the programs and placement tables above.
    let mut image_interner = ImageInterner::default();
    let n_bundles = alone.len() + sharded.len();
    let world_alone = quote! { ::scratchy_target_spyre::bundle_code::collective::TpWorld<1> };
    // The binary's own `Build` — `SUPERDSC_TP`'s assertion holds it to the world baked here.
    let world_build = quote! { ::scratchy_target_spyre::bundle_code::collective::Build };
    bundle_items(
        &alone,
        &world_alone,
        &mut interner,
        &mut places_interner,
        &mut image_interner,
        &mut out,
        (&mut n_groups, &mut n_bytes),
    );
    bundle_items(
        &sharded,
        &world_build,
        &mut interner,
        &mut places_interner,
        &mut image_interner,
        &mut out,
        (&mut n_groups, &mut n_bytes),
    );
    // ── THE BAKE BUDGET, PER MODEL (see `scratchy_target_spyre::superdsc_bake::BakeBudget`) ──
    //    Launch-group COUNT is the dxp floor's multiplication lever: the floor is total compile
    //    cost ÷ the queue's width, so a ladder that doubles its rungs doubles the floor no
    //    parallelism recovers. Checked at the drain — one `#[forward]` expansion, i.e. one
    //    model's whole ladder — and scaled per LAYER, because the models in scope span the
    //    measured 25 groups/layer (granite-2b, 40 layers) to 73 (gemma-4-12b, 48): a flat
    //    ceiling tight enough to hold the 2b refuses the 12b. Card-only: on the emulator the
    //    drained groups are one-per-op KTIR programs, a different unit this budget never sees.
    #[cfg(feature = "spyre-lower")]
    {
        let layers = model.text.layers.get();
        let per_layer = scratchy_target_spyre::superdsc_bake::BakeBudget::MAX_GROUPS_PER_LAYER;
        let ceiling = layers * per_layer;
        if n_groups > ceiling {
            panic!(
                "[spyre-superdsc] {}: {n_groups} launch group(s) across {layers} layer(s) is \
                 PAST THE BAKE BUDGET of {per_layer} groups per layer ({ceiling} total). Group \
                 count is the dxp floor's multiplication lever — a ladder this wide is a bake \
                 that grew, not one a wider queue absorbs. Moving the budget is a measured \
                 decision that names the bake; this refusal is the measurement.",
                model.source_stem
            );
        }
    }
    // The programs are `const` items and the placement tables `static`s; both precede the
    // `inventory::submit!`s that name them. The packed images are one more `static` of the same
    // kind: `__SDSC_IMAGES`, the one table every launch group's `BakedImage::Baked` names.
    let programs = interner.items();
    let tables = places_interner.items();
    let (images_static, packed) = image_interner.pack(&images);
    let preamble = crate::ktir_tokens::preamble();
    out = quote! { #preamble #(#programs)* #(#tables)* #images_static #out };
    let packed_note = if packed.frames > 0 {
        format!(
            " packed to {:.1} MB in {} zstd frame(s) against a {:.0} KB dictionary",
            packed.blob_bytes as f64 / 1048576.0,
            packed.frames,
            packed.dict_bytes as f64 / 1024.0,
        )
    } else {
        String::new()
    };
    eprintln!(
        "[spyre-superdsc] baked {} bundle(s) / {n_groups} launch group(s) / {} distinct program(s) \
         / {:.1} MB of device code{packed_note} into the binary as Rust structs",
        n_bundles,
        interner.items().len(),
        n_bytes as f64 / 1048576.0,
    );
    out
}

/// Render one world's drained bundles as `inventory` registrations — each `BundleCode::<'static, W>`
/// (`world` is `W`'s path), registered type-erased so a lookup of world `W` finds it by downcast.
fn bundle_items<T: scratchy_target_spyre::bundle_code::collective::Tp>(
    bundles: &[scratchy_target_spyre::bundle_code::BundleCode<'static, T>],
    world: &proc_macro2::TokenStream,
    interner: &mut crate::ktir_tokens::ProgramInterner,
    places_interner: &mut PlacesInterner,
    image_interner: &mut ImageInterner,
    out: &mut proc_macro2::TokenStream,
    (n_groups, n_bytes): (&mut usize, &mut usize),
) {
    for b in bundles {
        let fp = proc_macro2::Literal::string(&b.fp);
        let layout = layout_tokens(&b.layout, places_interner);
        let groups = b
            .groups
            .iter()
            .enumerate()
            .map(|(i, g)| {
                *n_groups += 1;
                *n_bytes += g.init_binary.len();
                launch_tokens(g, i, interner, image_interner)
            })
            .collect::<Vec<_>>();
        let reroll = match &b.reroll {
            Some(m) => {
                let m = reroll_tokens(m);
                quote! { ::core::option::Option::Some(#m) }
            }
            None => quote! { ::core::option::Option::None },
        };
        out.extend(quote! {
            ::scratchy_target_spyre::bundle_code::inventory::submit! {
                ::scratchy_target_spyre::bundle_code::RegisteredBundle {
                    code: &::scratchy_target_spyre::bundle_code::BundleCode::<'static, #world> {
                        fp: ::std::borrow::Cow::Borrowed(#fp),
                        layout: #layout,
                        groups: ::std::borrow::Cow::Borrowed(&[#(#groups),*]),
                        reroll: #reroll,
                    },
                }
            }
        });
    }
}

/// A `Cow::Borrowed` string literal — every `Cow<'static, str>` field in the baked artifact.
fn cow_str(s: &str) -> proc_macro2::TokenStream {
    let lit = proc_macro2::Literal::string(s);
    quote! { ::std::borrow::Cow::Borrowed(#lit) }
}

/// A `Cow::Borrowed` slice of `(u64, u64)` pairs — the walk's ONE (extent, stride) slice.
fn cow_axis_pairs(v: &[(u64, u64)]) -> proc_macro2::TokenStream {
    let lits = v.iter().map(|&(e, s)| {
        let (e, s) = (
            proc_macro2::Literal::u64_unsuffixed(e),
            proc_macro2::Literal::u64_unsuffixed(s),
        );
        quote! { (#e, #s) }
    });
    quote! { ::std::borrow::Cow::Borrowed(&[#(#lits),*]) }
}

/// A `Cow::Borrowed` slice of unsuffixed `u8` literals — the bake's host-scan axis order.
fn cow_u8s(v: &[u8]) -> proc_macro2::TokenStream {
    let lits = v.iter().map(|x| proc_macro2::Literal::u8_unsuffixed(*x));
    quote! { ::std::borrow::Cow::Borrowed(&[#(#lits),*]) }
}

/// One [`bundle::Placement`].
///
/// ⛔ THE DESTRUCTURES IN THIS FAMILY OF FUNCTIONS ARE THE GUARD, in both directions: a field ADDED to
/// the artifact type leaves the `quote!` below incomplete (a missing-field error in the expanded code),
/// and a field REMOVED leaves the destructure naming something that no longer exists. So the baked value
/// cannot drift from the type it is a value of.
/// One [`bundle::SynthRole`] as tokens.
///
/// ⛔ AN EXHAUSTIVE MATCH, NOT A STRING. A synthetic's role used to be a name SUFFIX
/// (`format!("{out}_rot")`) built at the emit site and matched at the consume site. Here the
/// bake holds the value and the macro emits the value; adding a role breaks THIS match at build
/// time, which is the whole reason it is an enum.
fn synth_role_tokens(
    r: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::SynthRole,
) -> proc_macro2::TokenStream {
    use scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::SynthRole as R;
    let b = quote! { ::scratchy_target_spyre::bundle_code::SynthRole };
    match r {
        R::Rot => quote! { #b::Rot },
        // The chained head's synthetics (Front C): the per-request slabs the
        // gather walks, their leg-2 transposes with the sentinel's reserved
        // zero row, and the request-major leg-3 rows.
        R::ChainSlab => quote! { #b::ChainSlab },
        R::ChainSlabT => quote! { #b::ChainSlabT },
        R::ChainRow => quote! { #b::ChainRow },
        R::Xc => quote! { #b::Xc },
        R::Rs => quote! { #b::Rs },
        R::Silu => quote! { #b::Silu },
        R::Qs => quote! { #b::Qs },
        R::KRep => quote! { #b::KRep },
        R::VRep => quote! { #b::VRep },
        R::NewKRep => quote! { #b::NewKRep },
        R::NewVRep => quote! { #b::NewVRep },
        R::NewKScaled => quote! { #b::NewKScaled },
        R::FqAbsX => quote! { #b::FqAbsX },
        R::FqAfp8 => quote! { #b::FqAfp8 },
        R::FqAmax => quote! { #b::FqAmax },
        R::FqAmaxFl => quote! { #b::FqAmaxFl },
        R::FqAscale => quote! { #b::FqAscale },
        R::FqChi => quote! { #b::FqChi },
        R::FqCl => quote! { #b::FqCl },
        R::FqInvS => quote! { #b::FqInvS },
        R::FqSc => quote! { #b::FqSc },
        R::FqDqA => quote! { #b::FqDqA },
        R::FqMm => quote! { #b::FqMm },
        R::FqRaw => quote! { #b::FqRaw },
        R::FqAmaxP(j) => {
            let j = proc_macro2::Literal::u32_unsuffixed(*j);
            quote! { #b::FqAmaxP(#j) }
        }
        // The long-reduce partial slot — indexed like `FqAmaxP`, same reason.
        R::ReducePartial(j) => {
            let j = proc_macro2::Literal::u32_unsuffixed(*j);
            quote! { #b::ReducePartial(#j) }
        }
        // The per-head windowed-norm synthetics (the `FqAmaxP` shape: one per head,
        // all at offset zero).
        R::HeadMean(h) => {
            let h = proc_macro2::Literal::u32_unsuffixed(*h);
            quote! { #b::HeadMean(#h) }
        }
        R::HeadMeps(h) => {
            let h = proc_macro2::Literal::u32_unsuffixed(*h);
            quote! { #b::HeadMeps(#h) }
        }
        R::HeadRinv(h) => {
            let h = proc_macro2::Literal::u32_unsuffixed(*h);
            quote! { #b::HeadRinv(#h) }
        }
        R::Sq16 => quote! { #b::Sq16 },
        R::Mean => quote! { #b::Mean },
        R::Meps => quote! { #b::Meps },
        R::Rinv => quote! { #b::Rinv },
        R::Xn => quote! { #b::Xn },
        // The OnePlusScale gain row — `gamma + GAIN_OFFSET` as one `[1, cols]` buffer the
        // door fills ahead of the gain multiply (the stored gamma row rides untouched).
        R::Gw => quote! { #b::Gw },
        R::Tanhc => quote! { #b::Tanhc },
        R::GatherKt => quote! { #b::GatherKt },
        R::GatherV => quote! { #b::GatherV },
        R::ChunkSc => quote! { #b::ChunkSc },
        R::ChunkESub => quote! { #b::ChunkESub },
        R::ChunkExp => quote! { #b::ChunkExp },
        R::ChunkQGroup => quote! { #b::ChunkQGroup },
        R::ChunkRed(i) => {
            let i = proc_macro2::Literal::u32_unsuffixed(*i);
            quote! { #b::ChunkRed(#i) }
        }
        R::NewChunkSc => quote! { #b::NewChunkSc },
        R::NewChunkESub => quote! { #b::NewChunkESub },
        R::NewChunkExp => quote! { #b::NewChunkExp },
        R::NewKt => quote! { #b::NewKt },
        R::Sc => quote! { #b::Sc },
        R::BMax => quote! { #b::BMax },
        R::NewM => quote! { #b::NewM },
        R::Corr => quote! { #b::Corr },
        R::CorrSubT => quote! { #b::CorrSubT },
        R::ExpB => quote! { #b::ExpB },
        R::ESubT => quote! { #b::ESubT },
        R::BSum => quote! { #b::BSum },
        R::OTmp => quote! { #b::OTmp },
        R::LTmp => quote! { #b::LTmp },
        R::RunM => quote! { #b::RunM },
        R::RunL => quote! { #b::RunL },
        R::RunO => quote! { #b::RunO },
        R::Blk(i) => {
            let i = proc_macro2::Literal::u32_unsuffixed(*i);
            quote! { #b::Blk(#i) }
        }
        R::Acc(i) => {
            let i = proc_macro2::Literal::u32_unsuffixed(*i);
            quote! { #b::Acc(#i) }
        }
        R::LBlk(i) => {
            let i = proc_macro2::Literal::u32_unsuffixed(*i);
            quote! { #b::LBlk(#i) }
        }
        R::LAcc(i) => {
            let i = proc_macro2::Literal::u32_unsuffixed(*i);
            quote! { #b::LAcc(#i) }
        }
        R::GatherKtLx(i) => {
            let i = proc_macro2::Literal::u32_unsuffixed(*i);
            quote! { #b::GatherKtLx(#i) }
        }
        R::GatherVLx(h) => {
            let h = proc_macro2::Literal::u32_unsuffixed(*h);
            quote! { #b::GatherVLx(#h) }
        }
    }
}

/// One [`bundle::PlaceId`] as tokens — the identity a placement is keyed by.
fn place_id_tokens(
    id: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::PlaceId,
) -> proc_macro2::TokenStream {
    use scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::PlaceId as P;
    match id {
        P::Act(t) => {
            let t = proc_macro2::Literal::u32_unsuffixed(*t);
            quote! { ::scratchy_target_spyre::bundle_code::PlaceId::Act(#t) }
        }
        P::Synth { of, role } => {
            let of = proc_macro2::Literal::u32_unsuffixed(*of);
            let role = synth_role_tokens(role);
            quote! {
                ::scratchy_target_spyre::bundle_code::PlaceId::Synth { of: #of, role: #role }
            }
        }
        P::OpLocal { of, role } => {
            let of = proc_macro2::Literal::u32_unsuffixed(*of);
            let role = op_local_role_tokens(role);
            quote! {
                ::scratchy_target_spyre::bundle_code::PlaceId::OpLocal { of: #of, role: #role }
            }
        }
    }
}

/// One op-local pin role as tokens — its named constant. Exhaustive over the kinds, so a new
/// op-local pin breaks THIS match at build, as a new [`SynthRole`] breaks `synth_role_tokens`.
pub(crate) fn op_local_role_tokens(
    r: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::OpLocalRole,
) -> proc_macro2::TokenStream {
    use scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::OpLocalKind as K;
    let b = quote! { ::scratchy_target_spyre::bundle_code::OpLocalRole };
    match r.kind() {
        K::Rinv => quote! { #b::RINV },
    }
}

fn placement_tokens(
    p: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::Placement,
) -> proc_macro2::TokenStream {
    let scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::Placement {
        id,
        segment,
        bank,
        offset,
        size,
        is_logits,
    } = p;
    let id = place_id_tokens(id);
    let (segment, bank, offset, size) = (
        proc_macro2::Literal::u32_unsuffixed(*segment),
        proc_macro2::Literal::u32_unsuffixed(*bank),
        proc_macro2::Literal::u64_unsuffixed(*offset),
        proc_macro2::Literal::u64_unsuffixed(*size),
    );
    quote! {
        ::scratchy_target_spyre::bundle_code::Placement {
            id: #id, segment: #segment, bank: #bank, offset: #offset, size: #size,
            is_logits: #is_logits,
        }
    }
}

/// One [`bundle::KernelWeight`].
fn kernel_weight_tokens(
    k: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::KernelWeight<'_>,
) -> proc_macro2::TokenStream {
    use scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle;
    let bundle::KernelWeight {
        id,
        axes,
        arm,
        stick_size,
        word_length,
    } = k;
    let id = place_id_tokens(id);
    // ⭐ AN EXHAUSTIVE MATCH, NOT A DERIVATION: the bake holds the arm verdict and the macro
    // emits it verbatim — a variant ADDED to `WalkArm` breaks THIS match at build time, and
    // the boot receives the statement, never a re-analysis of the axes. The permutation arm
    // carries its own order, so the macro cannot emit an arm and an order that disagree.
    let arm = match arm {
        bundle::WalkArm::Permutation { order } => {
            let order = cow_u8s(order);
            quote! {
                ::scratchy_target_spyre::bundle_code::WalkArm::Permutation { order: #order }
            }
        }
        bundle::WalkArm::InnerContig => {
            quote! { ::scratchy_target_spyre::bundle_code::WalkArm::InnerContig }
        }
        bundle::WalkArm::PerElement => {
            quote! { ::scratchy_target_spyre::bundle_code::WalkArm::PerElement }
        }
    };
    let axes = cow_axis_pairs(axes);
    let (stick_size, word_length) = (
        proc_macro2::Literal::u32_unsuffixed(*stick_size),
        proc_macro2::Literal::u32_unsuffixed(*word_length),
    );
    quote! {
        ::scratchy_target_spyre::bundle_code::KernelWeight {
            id: #id, axes: #axes, arm: #arm,
            stick_size: #stick_size, word_length: #word_length,
        }
    }
}

/// ⭐ ONE `static` PER DISTINCT PLACEMENT TABLE — the rungs of a ladder share one resident layout, so
/// most bundles' tables are the same table. MEASURED on granite-2b: the 223 bundles carried 223 tables
/// (~1,550 placements each, 76 MB of the 100 MB generated source) and only 102 were distinct.
///
/// Keyed by the table's rendered tokens — which ARE its value — so two tables share a `static` only
/// when every placement in them is the same.
#[derive(Default)]
struct PlacesInterner {
    seen: std::collections::HashMap<String, proc_macro2::Ident>,
    items: Vec<proc_macro2::TokenStream>,
}

impl PlacesInterner {
    /// The `static` naming `places`, defined on first sight.
    fn ident(
        &mut self,
        places: &[scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::Placement],
    ) -> proc_macro2::Ident {
        let rows: Vec<proc_macro2::TokenStream> = places.iter().map(placement_tokens).collect();
        let body = quote! { [#(#rows),*] };
        let key = body.to_string();
        if let Some(id) = self.seen.get(&key) {
            return id.clone();
        }
        let id = proc_macro2::Ident::new(
            &format!("__SDSC_PLACES_{}", self.items.len()),
            proc_macro2::Span::call_site(),
        );
        let n = proc_macro2::Literal::usize_unsuffixed(places.len());
        self.items.push(quote! {
            static #id: [::scratchy_target_spyre::bundle_code::Placement; #n] = #body;
        });
        self.seen.insert(key, id.clone());
        id
    }

    /// The `static` items to place ahead of the bundles that name them.
    fn items(&self) -> &[proc_macro2::TokenStream] {
        &self.items
    }
}

/// The whole memory plan.
/// ⭐ THE TENSOR-PARALLEL WORLD THIS BUILD BAKES — `scratchy_tp_world::Build`, the one `tp/N` the
/// build names (one rank when it names none). Its rank count, for the messages and the bundle names
/// that spell it; every TYPE the bake instantiates names `Build` itself.
const TP_WORLD: u32 =
    <scratchy_target_spyre::bundle_code::collective::Build as scratchy_target_spyre::bundle_code::collective::World>::RANKS;

fn layout_tokens(
    l: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::BundleLayout<'_>,
    places_interner: &mut PlacesInterner,
) -> proc_macro2::TokenStream {
    let scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::BundleLayout {
        segment_bytes,
        weight_bank_bytes,
        weight_tail,
        places,
        kernel_weights,
        scalarmul_scales,
        weight_scale_folds,
        kv_request_stride_bytes,
        lx_op_local_bytes,
        lx_prob_pin_bytes,
        lx_pin_window_bytes,
        num_sources,
        resident_source_tids,
    } = l;
    let segs = segment_bytes
        .iter()
        .map(|b| proc_macro2::Literal::u64_unsuffixed(*b));
    let wbanks = weight_bank_bytes
        .iter()
        .map(|b| proc_macro2::Literal::u64_unsuffixed(*b));
    let places = places_interner.ident(places);
    let kws = kernel_weights.iter().map(kernel_weight_tokens);
    // The exact value, as its BITS — these are the granite ScalarMul multipliers (a rounded one is a
    // quietly wrong model) and the page walk's `-inf` seed (which no float literal spells).
    let scales = scalarmul_scales.iter().map(|s| f32_bits(*s));
    // The fold list, same exact-value rule: a rounded multiplier is a quietly wrong weight scale.
    let folds = weight_scale_folds.iter().map(|(tid, scale)| {
        let tid = proc_macro2::Literal::u32_unsuffixed(*tid);
        let scale = proc_macro2::Literal::f32_suffixed(*scale);
        quote! { (#tid, #scale) }
    });
    let krs = proc_macro2::Literal::u64_unsuffixed(*kv_request_stride_bytes);
    // The rung's op-local LX pin slots — the host's term of the one `lx_window_fits` budget. The
    // pin PLACEMENTS stay emit-only (compiled sdsc allocates); the SUM is baked here so the host
    // budgets against the same pins the bake did. Same number for every bundle of the rung.
    let lxol = proc_macro2::Literal::u64_unsuffixed(*lx_op_local_bytes);
    // The chunk form's probability-part pins — the host's SECOND core-local term of the same
    // budget, summed beside the gathers at the bake's `lx_form` admission. 0 for a bundle whose
    // chunk form mints none.
    let lxpp = proc_macro2::Literal::u64_unsuffixed(*lx_prob_pin_bytes);
    // ⭐ THE WHOLE PIN WINDOW — core-local pins + op-local slot classes, the budget the bake's
    // `lx_window_fits` guard proved. Carried as an Option: `None` is "no window was ever proven",
    // which the resident fused path refuses at load rather than charging a silent zero.
    let lxpw = match lx_pin_window_bytes {
        Some(b) => {
            let lit = proc_macro2::Literal::u64_unsuffixed(*b);
            quote! { ::core::option::Option::Some(#lit) }
        }
        None => quote! { ::core::option::Option::None },
    };
    // The graph's own wiring facts — the source count and the resident ids, so every session
    // reads them off the bundle it runs rather than off another graph's wiring (the mint's
    // renumbering makes the two differ; see the layout field's own doc).
    let nsrc = proc_macro2::Literal::u32_unsuffixed(*num_sources);
    let res = resident_source_tids
        .iter()
        .map(|t| proc_macro2::Literal::u32_unsuffixed(*t));
    // The segment budget's verdict, as a variant: every reader matches on it.
    let tail = match weight_tail {
        scratchy_target_spyre::bundle_code::WeightTail::InPlace => {
            quote! { ::scratchy_target_spyre::bundle_code::WeightTail::InPlace }
        }
        scratchy_target_spyre::bundle_code::WeightTail::Spilled { segment, bytes } => {
            let segment = proc_macro2::Literal::u32_unsuffixed(*segment);
            let bytes = proc_macro2::Literal::u64_unsuffixed(*bytes);
            quote! {
                ::scratchy_target_spyre::bundle_code::WeightTail::Spilled {
                    segment: #segment,
                    bytes: #bytes,
                }
            }
        }
    };
    quote! {
        ::scratchy_target_spyre::bundle_code::BundleLayout {
            segment_bytes: [#(#segs),*],
            weight_bank_bytes: ::std::borrow::Cow::Borrowed(&[#(#wbanks),*]),
            weight_tail: #tail,
            places: ::std::borrow::Cow::Borrowed(&#places),
            kernel_weights: ::std::borrow::Cow::Borrowed(&[#(#kws),*]),
            scalarmul_scales: ::std::borrow::Cow::Borrowed(&[#(#scales),*]),
            weight_scale_folds: ::std::borrow::Cow::Borrowed(&[#(#folds),*]),
            kv_request_stride_bytes: #krs,
            lx_op_local_bytes: #lxol,
            lx_prob_pin_bytes: #lxpp,
            lx_pin_window_bytes: #lxpw,
            num_sources: #nsrc,
            resident_source_tids: ::std::borrow::Cow::Borrowed(&[#(#res),*]),
        }
    }
}

/// ONE LAUNCH: its address shifts and its compiled program, as a single value. The device image is
/// a row of the emission's one packed image table (see [`ImageInterner`]); everything else is a
/// literal.
fn launch_tokens<T: scratchy_target_spyre::bundle_code::collective::Tp>(
    // `'static` because a baked program's operations live in the arena, which outlives the whole
    // expansion — the tokens borrow it rather than copying it.
    g: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::LaunchGroup<'static, T>,
    index: usize,
    interner: &mut crate::ktir_tokens::ProgramInterner,
    image_interner: &mut ImageInterner,
) -> proc_macro2::TokenStream {
    let scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::LaunchGroup {
        kv,
        programs,
        init_binary,
        job_bin_ptr,
        correction,
    } = g;
    let scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::KvShifts {
        page_fold,
        gathered,
        walk,
        tp,
    } = kv;
    // A collective's consume: its kind, the operand it sends, and the fragment — re-minted through
    // `Fragment::baked`, a `const fn`, so the static this lands in re-proves the fragment's law.
    let tp = match tp {
        None => quote! { ::core::option::Option::None },
        Some(c) => {
            use scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::collective::CollectiveKind as K;
            let kind = match c.kind {
                K::AllReduce => {
                    quote! { ::scratchy_target_spyre::bundle_code::collective::CollectiveKind::AllReduce }
                }
                K::AllGather => {
                    quote! { ::scratchy_target_spyre::bundle_code::collective::CollectiveKind::AllGather }
                }
            };
            let operand = place_id_tokens(&c.operand);
            let [first, sticks, rows] = [
                c.fragment.first_stick(),
                c.fragment.sticks(),
                c.fragment.rows(),
            ]
            .map(proc_macro2::Literal::u32_unsuffixed);
            // The consume's rank, re-minted through `Rank::baked`, the const fn that refuses a rank
            // past its world — the world being the bundle's type, inferred from the static.
            let r = proc_macro2::Literal::u32_unsuffixed(c.plays_at.get());
            let plays_at = quote! {
                ::scratchy_target_spyre::bundle_code::collective::Rank::baked(#r)
            };
            // The window cut, re-minted through the const fn that refuses a cut the window
            // does not have.
            let cut = proc_macro2::Literal::u32_unsuffixed(c.cut.index() as u32);
            quote! {
                ::core::option::Option::Some(::scratchy_target_spyre::bundle_code::collective::TpConsume {
                    kind: #kind,
                    operand: #operand,
                    fragment: ::scratchy_target_spyre::bundle_code::collective::Fragment::baked(
                        #first, #sticks, #rows,
                    ),
                    plays_at: #plays_at,
                    cut: ::scratchy_target_spyre::bundle_code::collective::WindowCut::baked(#cut),
                })
            }
        }
    };
    let walk = match walk {
        Some(scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::Walk {
            trips,
            step_bytes,
        }) => {
            let (t, s) = (
                proc_macro2::Literal::u32_unsuffixed(*trips),
                proc_macro2::Literal::u32_unsuffixed(*step_bytes),
            );
            quote! {
                ::std::option::Option::Some(::scratchy_target_spyre::bundle_code::Walk {
                    trips: #t,
                    step_bytes: #s,
                })
            }
        }
        None => quote! { ::std::option::Option::None },
    };
    let jbp = proc_macro2::Literal::u64_unsuffixed(*job_bin_ptr);
    // ⭐ THE IMAGE, AS ITS BUILD'S ONE PACKED TABLE — metal's AOT precedent: one dictionary over the
    // whole build, one frame per DISTINCT image, the row minted here and the frame written at the
    // splice by `ImageInterner::pack`. An empty image (dxp compiled a group to a job plan alone)
    // carries no row: it stays raw bytes, so a build with no images gets no blob, no dictionary,
    // and no static.
    let image = match init_binary {
        scratchy_target_spyre::bundle_code::BakedImage::Raw(c) if c.is_empty() => quote! {
            ::scratchy_target_spyre::bundle_code::BakedImage::Raw(
                ::std::borrow::Cow::Borrowed(&[])
            )
        },
        scratchy_target_spyre::bundle_code::BakedImage::Raw(c) => {
            let row = image_interner.image(c.as_ref());
            let row = proc_macro2::Literal::u32_unsuffixed(row);
            quote! {
                ::scratchy_target_spyre::bundle_code::BakedImage::Baked {
                    images: &__SDSC_IMAGES,
                    row: #row,
                }
            }
        }
        // The arena's groups are built by `into_code` from dxp's output, raw — a `Baked` image here
        // would mean an already-emitted group was re-emitted, which the interners exist to prevent.
        // This names it at the bake rather than emitting a frame of a frame.
        scratchy_target_spyre::bundle_code::BakedImage::Baked { .. } => {
            panic!("[spyre-superdsc] launch group {index}: re-emitting an already-baked image")
        }
    };
    // The correction flits are kilobytes at most (a 128-byte flit per symbolic address), so they go out
    // as a byte literal rather than a second file.
    let corr = correction
        .iter()
        .map(|b| proc_macro2::Literal::u8_unsuffixed(*b));
    // ⭐ THE PROGRAMS THEMSELVES. A launch group IS its KTIR, so this is what `inventory::submit!`
    // carries: the constructed functions, rendered as const data. On the emulator the device runs
    // exactly these; on a card the image above is compiled FROM them.
    let progs: Vec<_> = programs
        .iter()
        .map(|p| crate::ktir_tokens::program_tokens(p, interner))
        .collect();
    quote! {
        ::scratchy_target_spyre::bundle_code::LaunchGroup {
            kv: ::scratchy_target_spyre::bundle_code::KvShifts {
                page_fold: #page_fold,
                gathered: #gathered,
                walk: #walk,
                tp: #tp,
            },
            programs: ::std::borrow::Cow::Borrowed(&[#(#progs),*]),
            init_binary: #image,
            job_bin_ptr: #jbp,
            correction: ::std::borrow::Cow::Borrowed(&[#(#corr),*]),
        }
    }
}

/// An `f32` as the expression of its exact BITS — `f32::from_bits(0x…)`, a const expression that
/// spells every value, the infinities included, which `Literal::f32_*` refuses.
fn f32_bits(v: f32) -> proc_macro2::TokenStream {
    let bits = proc_macro2::Literal::u32_unsuffixed(v.to_bits());
    quote! { f32::from_bits(#bits) }
}

/// ⭐ THE EMISSION'S ONE IMAGE TABLE — the interner for device images, as [`PlacesInterner`] is for
/// placement tables: minted row-by-row during the walk, materialized once at the splice. The packing
/// is the metal AOT precedent (`crates/targets/metal/src/aot.rs`, `pack`): one dictionary trained on
/// every image the build bakes — near-copies share their bytes through it — and each distinct image
/// one zstd frame against it, all frames in one `include_bytes!`'d blob.
#[derive(Default)]
struct ImageInterner {
    /// DISTINCT images by their exact bytes — the map IS the store, and the value is the row a
    /// launch group's token names. Content addressing is the whole point: bundles dxp compiled to
    /// the same image share one frame (measured on granite-3.1-8b fp8: 456 groups, 356 distinct
    /// images), exactly as bundles with one placement table share one static. The bytes are copied
    /// out of the arena here — the one cost of content addressing — and live only for the expansion.
    index: std::collections::HashMap<Vec<u8>, u32>,
}

/// What `ImageInterner::pack` wrote, for the bake's own report.
struct PackedImages {
    frames: usize,
    blob_bytes: u64,
    dict_bytes: u64,
}

impl ImageInterner {
    /// The row for `bytes`, minted on first sight.
    fn image(&mut self, bytes: &[u8]) -> u32 {
        if let Some(row) = self.index.get(bytes) {
            return *row;
        }
        let row = self.index.len() as u32;
        self.index.insert(bytes.to_vec(), row);
        row
    }

    /// Materialize the table: train the dictionary over every distinct image, compress each to a
    /// frame, write the blob and the dictionary under `dir`, and return the one `static` every
    /// `BakedImage::Baked` names. No images, no output: an emulator build emits nothing.
    fn pack(self, dir: &std::path::Path) -> (proc_macro2::TokenStream, PackedImages) {
        if self.index.is_empty() {
            return (
                proc_macro2::TokenStream::new(),
                PackedImages { frames: 0, blob_bytes: 0, dict_bytes: 0 },
            );
        }
        // First-seen order: the row a launch group's token already names is this position.
        let mut rows: Vec<(&Vec<u8>, &u32)> = self.index.iter().collect();
        rows.sort_by_key(|(_, row)| **row);
        // ── Metal's `pack` constants, verbatim in meaning ──
        // The dictionary holds at most this much: the bytes a build's images share.
        const DICTIONARY: usize = 1 << 20;
        // The samples the dictionary is trained on are the images cut to this size, so one image
        // alone is samples enough.
        const SAMPLE: usize = 4 << 10;
        // The level to compress at: the dictionary carries the shared bytes, and higher levels save
        // under a fifth more for several times the build time.
        const LEVEL: i32 = 3;
        let all: Vec<u8> = rows.iter().flat_map(|(b, _)| b.iter().copied()).collect();
        let samples: Vec<usize> = all.chunks(SAMPLE).map(<[u8]>::len).collect();
        // Training on too few bytes for a dictionary leaves it empty: each frame then stands alone.
        let dictionary = zstd::dict::from_continuous(&all, &samples, DICTIONARY).unwrap_or_default();
        use rayon::prelude::*;
        let frames: Vec<(Vec<u8>, usize)> = rows
            .par_iter()
            .map(|(image, _)| {
                zstd::bulk::Compressor::with_dictionary(LEVEL, &dictionary)
                    .and_then(|mut c| c.compress(image))
                    .unwrap_or_else(|e| panic!("[spyre-superdsc] compress a device image: {e}"))
            })
            .zip(rows.par_iter().map(|(image, _)| image.len()))
            .collect();
        let mut blob: Vec<u8> = Vec::with_capacity(frames.iter().map(|(f, _)| f.len()).sum());
        let row_tokens: Vec<proc_macro2::TokenStream> = frames
            .iter()
            .map(|(frame, len)| {
                let offset = proc_macro2::Literal::u32_unsuffixed(blob.len() as u32);
                blob.extend_from_slice(frame);
                let frame_len = proc_macro2::Literal::u32_unsuffixed(frame.len() as u32);
                let len = proc_macro2::Literal::u32_unsuffixed(*len as u32);
                quote! {
                    ::scratchy_target_spyre::bundle_code::BakedFrame {
                        offset: #offset,
                        frame_len: #frame_len,
                        len: #len,
                    }
                }
            })
            .collect();
        let write = |name: &str, bytes: &[u8]| {
            // The directory is ours to make: nothing else writes under it anymore (the per-group
            // files that used to create it are gone), and a first build has no OUT_DIR subtree.
            if let Err(e) = std::fs::create_dir_all(dir) {
                panic!(
                    "[spyre-superdsc] creating the packed image directory {}: {e}",
                    dir.display()
                );
            }
            let dst = dir.join(name);
            if let Err(e) = std::fs::write(&dst, bytes) {
                panic!(
                    "[spyre-superdsc] writing the packed device images to {}: {e} — they are \
                     `include_bytes!`d into the binary, so there is no fallback",
                    dst.display()
                );
            }
        };
        write("images.bin", &blob);
        write("images.dict", &dictionary);
        let stats = PackedImages {
            frames: frames.len(),
            blob_bytes: blob.len() as u64,
            dict_bytes: dictionary.len() as u64,
        };
        let tokens = quote! {
            static __SDSC_IMAGES: ::scratchy_target_spyre::bundle_code::BakedImages =
                ::scratchy_target_spyre::bundle_code::BakedImages {
                    blob: ::core::include_bytes!(::core::concat!(
                        ::core::env!("OUT_DIR"), "/superdsc-code/images.bin"
                    )) as &[u8],
                    dictionary: ::core::include_bytes!(::core::concat!(
                        ::core::env!("OUT_DIR"), "/superdsc-code/images.dict"
                    )) as &[u8],
                    frames: &[#(#row_tokens),*],
                };
        };
        (tokens, stats)
    }
}

/// The re-rolled layer loop, with every sibling it names.
fn reroll_tokens(
    m: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::RerollMeta<'_>,
) -> proc_macro2::TokenStream {
    let scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::RerollMeta {
        prefix,
        suffix,
        bodies,
        bodies_fused,
        rungs,
        fallback_rung,
        launches,
        kv_page_stride,
        prefix_weight_bank,
        suffix_weight_bank,
    } = m;
    let fallback_rung = proc_macro2::Literal::u32_unsuffixed(*fallback_rung);
    let sib = |s: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::SiblingFp<'_>| {
        let inner = cow_str(s.as_str());
        quote! { ::scratchy_target_spyre::bundle_code::SiblingFp(#inner) }
    };
    let (prefix, suffix) = (sib(prefix), sib(suffix));
    let bodies = bodies.iter().map(sib);
    let bodies_fused = bodies_fused.iter().map(sib);
    let rungs = rungs.iter().map(|r| {
        let scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::LadderRung {
            active_cap,
            bodies,
            bodies_fused,
        } = r;
        let cap = proc_macro2::Literal::u32_unsuffixed(active_cap.get());
        let bodies = bodies.iter().map(sib);
        let bodies_fused = bodies_fused.iter().map(sib);
        quote! {
            ::scratchy_target_spyre::bundle_code::LadderRung {
                active_cap: ::scratchy_target_spyre::bundle_code::SweptCols::new(#cap),
                bodies: ::std::borrow::Cow::Borrowed(&[#(#bodies),*]),
                bodies_fused: ::std::borrow::Cow::Borrowed(&[#(#bodies_fused),*]),
            }
        }
    });
    // ⭐ THE LAUNCH TABLE, ONE ROW PER LAYER, IN TRUE LAYER ORDER — the executor runs exactly this
    // sequence; there is no per-layer stride arithmetic left to bake differently.
    let launches = launches.iter().map(|l| {
        let scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::LayerLaunch {
            body,
            iter,
            w_bank,
            w_off,
            kv_off,
        } = l;
        let (b, it, wb) = (
            proc_macro2::Literal::u8_unsuffixed(*body),
            proc_macro2::Literal::u32_unsuffixed(*iter),
            proc_macro2::Literal::u8_unsuffixed(*w_bank),
        );
        let (wo, ko) = (
            proc_macro2::Literal::u64_unsuffixed(*w_off),
            proc_macro2::Literal::u64_unsuffixed(*kv_off),
        );
        quote! {
            ::scratchy_target_spyre::bundle_code::LayerLaunch {
                body: #b,
                iter: #it,
                w_bank: #wb,
                w_off: #wo,
                kv_off: #ko,
            }
        }
    });
    let (kps, pwb, swb) = (
        proc_macro2::Literal::u64_unsuffixed(*kv_page_stride),
        proc_macro2::Literal::u32_unsuffixed(*prefix_weight_bank),
        proc_macro2::Literal::u32_unsuffixed(*suffix_weight_bank),
    );
    quote! {
        ::scratchy_target_spyre::bundle_code::RerollMeta {
            prefix: #prefix,
            suffix: #suffix,
            bodies: ::std::borrow::Cow::Borrowed(&[#(#bodies),*]),
            bodies_fused: ::std::borrow::Cow::Borrowed(&[#(#bodies_fused),*]),
            rungs: ::std::borrow::Cow::Borrowed(&[#(#rungs),*]),
            fallback_rung: #fallback_rung,
            launches: ::std::borrow::Cow::Borrowed(&[#(#launches),*]),
            kv_page_stride: #kps,
            prefix_weight_bank: #pwb,
            suffix_weight_bank: #swb,
        }
    }
}

// The emu's ONE width is LX-derived per model (`emu_prefill_lx_width`
// below): the widest ladder rung whose fp8 gemms' INTRA-NODE tiles fit a
// core's 2 MiB LX. This constant is now only the COLLISION-FREE CEILING
// on that derivation — the largest width the emu will ever bake.
//
// CRITICAL: a by-value symbolization consumer (every concrete dim ==
// prefill_m → `s0`) must not see the width equal any structural model dim,
// else a non-seq dim is wrongly symbolized (the measured refusal was the
// m=32 == head_dim/2 collision: the RoPE rotate-half StridedSlice output
// `[rows, heads, 32]` became `[s0, heads, s0]`, which flex rejected at
// graphOptimizer.cpp:2865). That consumer is HISTORICAL — it lived in the
// deleted sengraph/offline_decoder path; in this tree `seq_sym`'s value
// half is constructed and never destructured, so the emu's KTIR tape is
// fully CONCRETE and PREFILL_LX_SAFE stands as the collision-free ceiling
// for whichever symbolic consumer returns. 31 is prime + ≤32 (LX-safe) +
// collides with no common dim (head_dim 64/128, half 32/64, n_heads,
// hidden, intermediate, block_size 64). `KTIR_PREFILL_LEN` overrides (must
// also be collision-free). Read by
// dump_wavefront_mega and its pinned test.
const PREFILL_LX_SAFE: u32 = 31;

/// SuperDSC PREFILL WIDTH LADDER, ASCENDING. Each rung is baked as its own prefill bundle and
/// the worker runs a chunk on the NARROWEST rung that holds it: per-chunk cost is
/// `fixed + slope·mq` with the fixed term dominant (granite-3.1-2b, measured: m=16 → 53.1 ms,
/// m=31 → 69 ms ⇒ ~42 ms + ~0.7 ms/row), so one width is wrong both ways — too narrow re-pays
/// the fixed term on every extra chunk, too wide computes padding rows that carry no tokens.
///
/// `2^k − 1` by construction: a by-value symbolization of the tape width would wrongly
/// symbolize an equal NON-seq dim (see [`PREFILL_LX_SAFE`]'s note on m=32 == head_dim/2).
/// These values collide with none of the power-of-two dims (head_dim, block_size, cap,
/// hidden, intermediate).
///
/// The 63-row ceiling is GONE (2026-07-29): the new-token block is now folded as `mq_pad/64`
/// stick-wide sub-blocks, so a chunk may span more than one stick. 96 is HARDWARE-PROVEN.
/// That matters more than padding efficiency, because a chunk costs ~41 ms REGARDLESS of how
/// few rows it carries (measured: rung 15 = 55 ms, rung 31 = 70 ms, rung 63 = 88 ms, i.e.
/// 3.7 / 2.3 / 1.4 ms per row) — so a prompt one token past the ceiling used to pay a whole
/// extra chunk. A 65-row prompt split 63+2 and the 2-token tail alone cost 56.8 ms.
/// 80 sits between 63 and 96 to halve the worst-case padding in that gap.
///
/// `2^k − 1` is no longer required: this ladder is the SuperDSC bake's alone, and that prefill
/// tape is CONCRETE (`seq_sym` is forced to `None` below), so nothing symbolizes it by value.
/// That is why 128 may equal granite-8b's head_dim here.
///
/// Rungs that fail to lower for a given model FAIL THE BUILD, so this list is a promise:
/// every admitted rung bakes or nothing does.
/// DENSER AT THE LOW END on purpose: padding cost is `(rung − real) × slope`, so what hurts is
/// the PROPORTION of the chunk that is padding, and that is worst for short prompts. A measured
/// 19-token prompt landed on rung 31 and spent 12 of its 31 rows (39%) computing nothing — about
/// 11 ms.
///
/// GAPS HALVED (2026-07-30) now that the slope is pinned rather than estimated. Fitting the
/// three measured prefill/decode `compute` points gives
///     compute ≈ 16.3 ms + 0.65 µs·trips + **1.24 ms·row**
/// so a padding row is not a rounding error — it costs the same 1.24 ms a real token does,
/// because the bundle is baked at `mq` and computes every row whether or not it holds a token.
/// A 19-token prompt on rung 23 threw away 4 × 1.24 ≈ 5 ms, comparable to the entire RoPE
/// head-merge win. Gaps are 2 through the short-prompt band, 4 to 47, then 8 — so the
/// worst-case overshoot where interactive chat actually lands is ONE row, ~1.2 ms. The extra
/// rungs cost bake time, startup and program memory, but NOT another weight copy (rungs borrow
/// from the top rung — `3623c76b`) and NOT decode throughput (the placement cost is non-linear
/// in session count and was already paid at the 1→2 step).
///
/// The LOW end was the worst served and matters most, because what hurts is the PROPORTION of
/// the chunk that is padding. The ladder used to start at 15, so a 3-token "hi" ran 15 rows
/// and discarded 12 of them (~15 ms) — a third of a prefill on a prompt that needs almost
/// none of it. 7 and 11 cover that. Measured mid-range case: a 20-token prompt sat on rung 23
/// (`n_new=23` in the phase log) and wasted 3 rows; 21 takes that to one.
///
/// ⭐ THE CEILING IS 128, AND 127 IS WHY IT WAS 96. Chunk count dominates TTFT (measured
/// 2026-07-30: splitting a ~23-token prompt 15+8 instead of one chunk cost **+48 ms**, since a
/// forward's ~40-45 ms fixed term is paid PER CHUNK), so taller rungs are exactly the right
/// lever. Adding 127 failed the BUILD, not the ladder:
///   time_tile_for_lx('out'): per-core tile does not fit the 1,677,721-B usable LX scratchpad
///   even tiled to single 64-elem sticks (resident 1,923,712 B) — needs K-time PSUM
///   accumulation (Stage 2), which the frontend does not emit.
/// That is down_proj (K=8192) — but at 127 rows, which is PRIME: `matmul_split_plan` splits `mb`
/// only over divisors of M, so every core kept all 127 rows. 128 splits, and every granite-2b
/// projection fits LX at one stick. MEASURED, granite-3.1-2b fp16 on card: a 10k-token prompt's
/// TTFT 25.65 s at 96 rows → 17.07 s at 128, 3/3 byte-identical; 1k 1.67 → 1.30 s. And 128 divides
/// the page, so no chunk steps back (see `PagedKvPool::PREFILL_CHUNK_SLOTS`).
///
/// ⛔ An LX overflow still PANICS the proc-macro rather than shortening the ladder — the skip above
/// covers a `lower_decode_to_wavefront` Err, and an overflow raises inside `emit_bundle`. That
/// panic is deliberate (a swallowed eprintln once shipped a zero-group bundle that only crashed
/// at pod load), so a rung added here must be proven to lower on every spyre model first.
///
/// ⭐ ABOVE [`SUB_BLOCKED_PREFILL_CEILING`] A RUNG IS BAKED ONLY WHERE ITS CAUSAL BLOCK IS CHUNK-FORMED
/// (`NewBlockForm::of(rung, head_dim, false) == Chunk`: whole sticks at head dim 64 or 128). A wider chunk
/// re-streams the weights half as often, but sub-blocked per query head its causal block grows with
/// the width and gives that back. MEASURED, granite-3.1-2b fp16, one card, ABAB:
///   10k TTFT  sub-blocked: 13.6 s at 128 rows → 13.0 s at 256 (and ~1k got SLOWER, 0.90 → 0.94 s)
///             chunk-formed: 12.8 s at 128 → **11.3 s at 256**; ~1k 0.73 s, ~400 tokens 0.27 s
/// So granite-2b and granite-8b (hd=128, cut per query head) get 192 through 512, and a model whose
/// causal block stays sub-blocked (the gemma bakes, hd ≥ 256) keeps its 128-row ladder byte for byte.
///
/// ⭐ 384 AND 512 SPAN TWO PAGES. Their cache write is cut into page parts at emission (each part
/// aimed at its own page through the block table — `EmittedOp::kv_slot_offset`), which is what
/// lets a chunk outgrow the page: the weights are re-streamed once per chunk, so a 10k prompt's
/// 40 chunks of 256 become 20 of 512.
// Spyre-only, as every reader is: the spyre bake's ladder, `emu_prefill_lx_width`, and the
// shape asserts below.
const PREFILL_RUNGS: [u32; 27] = [
    7, 11, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 39, 43, 47, 55, 63, 71, 80, 88, 96, 112, 128,
    192, 256, 384, 512,
];

/// The widest rung a SUB-BLOCKED causal block is baked at (see [`PREFILL_RUNGS`]). Read by the
/// ladder in `dump_wavefront_mega`.
const SUB_BLOCKED_PREFILL_CEILING: u32 = 128;

// ⛔ THE POOL SIZES ITS SLACK FROM THIS CEILING, so the two cannot be allowed to drift.
//
// A continuation chunk's cache write covers the PREFIX BUNDLE'S BAKED ROW COUNT — this ceiling —
// not the chunk's real length, and `PagedKvPool::WRITE_SLACK` is exactly the amount by which that
// write may overrun a page. Raise the ceiling without raising the slack and the overrun lands on
// the next kv head's keys again: a correct first token, then fluent garbage, for every prompt
// needing a third chunk. A `const` assertion makes that a `cargo build` error instead, in every
// build that has the pool (`spyre`: its `sdsc_abstract` is the Spyre target's).
const _: () = assert!(
    PREFILL_RUNGS[PREFILL_RUNGS.len() - 1] as usize
        == ktir_superdsc::sdsc_abstract::PagedKvPool::PREFILL_CHUNK_SLOTS,
    "PagedKvPool::PREFILL_CHUNK_SLOTS must equal the PREFILL_RUNGS ceiling — the pool's \
     WRITE_SLACK is derived from it, and a padded chunk write that overruns by more than the \
     slack corrupts the next kv head's keys"
);

// ⛔ THE LADDER'S OWN SHAPE LAWS (the table is POLICY — measured cost-law discretization,
// see the rung notes above — so what is checked is its structure, not its values):
// strictly ascending (the bake and the worker's rung choice both walk it in order); the
// top rung is the pool's whole-page window (PREFILL_WINDOW_PAGES · POOL_PAGE_SLOTS, itself
// PREFILL_CHUNK_SLOTS); the TOP THREE rungs are whole-page or half-page spans (a chunk
// at them starts at a page or half-page boundary, so no padded write straddles a page);
// every rung positive.
const _: () = assert!(
    {
        use ktir_superdsc::sdsc_abstract::PagedKvPool;
        let page = PagedKvPool::POOL_PAGE_SLOTS.get();
        let mut ok = true;
        let mut i = 0;
        while i < PREFILL_RUNGS.len() {
            if PREFILL_RUNGS[i] == 0 || (i > 0 && PREFILL_RUNGS[i - 1] >= PREFILL_RUNGS[i]) {
                ok = false;
            }
            i += 1;
        }
        let n = PREFILL_RUNGS.len();
        let top = PREFILL_RUNGS[n - 1];
        let half_page = page / 2;
        ok && top as usize
            == PagedKvPool::PREFILL_WINDOW_PAGES * PagedKvPool::POOL_PAGE_SLOTS.as_usize()
            && top as usize == PagedKvPool::PREFILL_CHUNK_SLOTS
            && PREFILL_RUNGS[n - 1].is_multiple_of(half_page)
            && PREFILL_RUNGS[n - 2].is_multiple_of(half_page)
            && PREFILL_RUNGS[n - 3].is_multiple_of(half_page)
    },
    "PREFILL_RUNGS must be strictly ascending, positive, topped at the pool's whole-page \
     window (PREFILL_WINDOW_PAGES · POOL_PAGE_SLOTS == PREFILL_CHUNK_SLOTS), with its top three \
     rungs whole-page or half-page spans"
);

// ── THE EMU'S ONE BAKED PREFILL WIDTH, DERIVED FROM THE MODEL'S OWN fp8 GEMMS ──
//
// ktir-emulator runs each fused segment on ONE core against a real 2 MiB LX,
// and an fp8-dynamic gemm's minted program holds, at its epilogue peak,
// `FP8_EPILOGUE_LIVE_TILES` `[m, N]` f16 tiles — the dot result `p`, the
// broadcast's shape-defining `tensor.empty`, and the broadcast replica of the
// per-channel scale row — plus the collapsed `[N]` scale row itself; the
// count and the peak-bytes formula are the SPLICE'S OWN (minted beside the
// program they describe, `scratchy_triton_splice::fp8_epilogue_peak_bytes`).
// MEASURED (isolated single-node
// probe through the production session entry, K=3840/N=15360): the refusal
// `1935360 + 952320 > 2097152` is exactly p + empty + scale-row + the
// charging broadcast; the zero-init splat is freed by the resident
// scheduler's dies-at reclaim right after the dot, and the activation/weight
// loads are uncharged views. The fp8 form CANNOT be N-blocked down to fit —
// `verify_canonical_fp8_matmul_kernel` refuses `BLOCK_N < N` by name — so the
// only lever that shrinks the epilogue is the baked WIDTH. The old constant
// 31 was derived from llama-1b's DENSE 8192-wide FFN and predates the fp8
// splice: on gemma-4-12b (N=15360) rung 31 needed 2,887,680 B and the first
// chat refused ~40 s in. Flip boundary pinned by the same probe: m=21 fits,
// m=23 refuses — exactly at `3·m·N·2 + N·2` vs the 2 MiB capacity.
//
// DENSE gemms do not constrain the width and are not counted: `matmul_tile`'s
// `n_block` law blocks them to a ≤1 MiB accumulator pair whenever `m·N` is
// large (measured at m=71, N=15360 — clean), and its un-blocked arm only
// fires when `m·N ≤ 90_000` elements, whose whole tile is ~180 KB.
//
// The budget is the ONE const, read from ktir-superdsc beside the LX capacity
// it derives from (`LX_FUSION_BUDGET_BYTES` = 7/8 of the 2 MiB LX — headroom
// for the planner's model-vs-mechanism gap, which is exactly what the old
// constant lacked); the tile count and the peak-bytes formula are the
// SPLICE'S OWN, minted beside the fp8-dynamic program they describe
// (`FP8_EPILOGUE_LIVE_TILES`, `fp8_epilogue_peak_bytes`). Gemms whose rows are a concrete
// bound (not the seq dim) don't scale with the baked width and are skipped;
// the lm_head is skipped because the prefill fold runs it at m=1 (its
// `[1, vocab]` peak is the decode path's own, already proven) — identified by
// the SPLICE'S OWN LAW (width match + feeds-the-result-through-row-local-ops,
// the walk that survived the gemma-4 parity fixture's vocab-512/global-q
// width collision), mirrored here at FUF level. Rung table this yields:
// 12b (N=15360) → 19; 31b (N=21504) → 11; granite fp8 (N=8192) → 31 (the
// collision-free cap binds first); dense-only models → 31 (no fp8 gemm
// constrains). Pinned by `emu_prefill_lx_width_pins_the_gemma4_rungs`.

// Gated on `spyre`, with the one-home budget it reads: `ktir_superdsc::superdsc_opspec`
// is re-exported only under `spyre`, and the fn's one caller is the spyre bake block.
fn emu_prefill_lx_width(fuf: &Fuf, model: &ModelParams, prefix_len: u32) -> u32 {
    use scratchy_forward_compiler_macro::shape::eval_closed_dim;
    // The result tail: the LAST tile nothing consumes — the bridge's own law
    // (`to_wavefront` takes the last value-producing op as the result), which
    // in the pre-bridge FUF's program order is the last consumer-free node.
    let mut has_consumer = vec![false; fuf.nodes.len()];
    for n in &fuf.nodes {
        for inp in &n.inputs {
            if let FufInput::Tile { id, .. } = inp {
                has_consumer[id.0 as usize] = true;
            }
        }
    }
    let result = fuf
        .nodes
        .iter()
        .rposition(|n| !has_consumer[n.id.0 as usize])
        .expect("the FUF has a result tail");
    let result_width = fuf.nodes[result]
        .outputs
        .first()
        .and_then(|s| s.last())
        .and_then(|d| eval_closed_dim(d, &model.bounds));
    // The N of every fp8-dynamic gemm that runs at the baked width.
    let mut fp8_ns: Vec<u64> = Vec::new();
    for node in &fuf.nodes {
        if node.op != OpKind::Gemm {
            continue;
        }
        let Some(out) = node.outputs.first() else {
            continue;
        };
        if out.len() != 2
            // Fixed rows: does not scale with the baked width.
            || eval_closed_dim(&out[0], &model.bounds).is_some()
        {
            continue;
        }
        let Some(n) = eval_closed_dim(&out[1], &model.bounds) else {
            panic!(
                "[wavefront] {}: the emu prefill-width derivation needs every \
                 seq-scaled gemm's width closed under the model's bounds; gemm tile {} \
                 states N = {:?} — a symbolic output width cannot be LX-planned",
                model.source_stem, node.id.0, out[1]
            );
        };
        let is_lm_head_tail = result_width == Some(n)
            && fuf_feeds_result_through_row_local_ops(fuf, node.id, fuf.nodes[result].id);
        if is_lm_head_tail {
            continue; // the prefill lm-head fold runs it at m=1
        }
        if matches!(
            scratchy_forward_compiler_macro::weight_vocab::weight_storage_of(node),
            Some(scratchy_forward_compiler_macro::quantization::StorageFormat::Fp8 { .. })
        ) {
            fp8_ns.push(n);
        }
    }
    // Descending rung search: the widest ladder rung (capped at the
    // collision-free `PREFILL_LX_SAFE`) at which EVERY fp8 gemm's intra-node
    // footprint fits the budget at the width the bake would actually use
    // (`min(prefix_len, rung)` — a short prefix checks its own width, not
    // the rung's).
    for &r in PREFILL_RUNGS.iter().rev() {
        if r > PREFILL_LX_SAFE {
            continue;
        }
        let w = prefix_len.min(r);
        if fp8_ns.iter().all(|&n| {
            scratchy_triton_splice::fp8_epilogue_peak_bytes(w, n)
                <= u128::from(ktir_superdsc::superdsc_opspec::LX_FUSION_BUDGET_BYTES)
        }) {
            return w;
        }
    }
    // No width fits — refuse at BUILD with the numbers (the card-failure
    // law: fail at `cargo build`, not 40 s into a chat).
    let w = prefix_len.min(PREFILL_RUNGS[0]);
    let n = fp8_ns.iter().copied().max().unwrap_or(0);
    panic!(
        "[wavefront] {}: no emu prefill width fits a core's LX — the widest fp8 gemm \
         (N={n}) needs {} B of intra-node tiles at m={} against the {} B budget ({} live \
         [m, N] f16 tiles + the [N] scale row); the fp8 kernel cannot be N-blocked \
         (verify_canonical_fp8_matmul_kernel refuses BLOCK_N < N)",
        model.source_stem,
        scratchy_triton_splice::fp8_epilogue_peak_bytes(w, n),
        w,
        u128::from(ktir_superdsc::superdsc_opspec::LX_FUSION_BUDGET_BYTES),
        scratchy_triton_splice::FP8_EPILOGUE_LIVE_TILES,
    );
}

/// The splice's lm-head-tail law (`is_prefill_lm_head_tail`'s dataflow half),
/// mirrored at FUF level: does tile `from` feed the `result` tile through
/// row-local consumers only (or is it the result)? Every consumer must be
/// row-local AND keep the property — one row-mixing reader anywhere
/// (attention, rope, an MoE block) makes the gemm's other rows live and the
/// m=1 fold unsound on it. Mid-layer gemms are refused by their own layer's
/// rope/attention, exactly as at subtile level.
fn fuf_feeds_result_through_row_local_ops(fuf: &Fuf, from: TileId, result: TileId) -> bool {
    if from == result {
        return true;
    }
    let mut any_reader = false;
    for n in &fuf.nodes {
        if !n
            .inputs
            .iter()
            .any(|i| matches!(i, FufInput::Tile { id, .. } if *id == from))
        {
            continue;
        }
        any_reader = true;
        if !fuf_row_local_tail_op(n.op)
            || !fuf_feeds_result_through_row_local_ops(fuf, n.id, result)
        {
            return false;
        }
    }
    any_reader
}

/// The FUF-level mirror of the splice's `row_local_tail_op` set: ops whose
/// output row `i` reads only input row `i` (matmuls, last-dim
/// norms/reductions, elementwise). RoPE is excluded deliberately — it also
/// writes the paged KV cache, a consumer the prefill graph cannot see.
fn fuf_row_local_tail_op(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::Gemm
            | OpKind::RmsNorm
            | OpKind::RmsNormUnit
            | OpKind::Mean
            | OpKind::Silu
            | OpKind::Gelu
            | OpKind::QuickGelu
            | OpKind::GeluErf
            | OpKind::Relu2
            | OpKind::TanhSoftCap
            | OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::BiasAdd
            | OpKind::ScalarWeightMul
    )
}

#[allow(clippy::too_many_arguments)]
// `program` and `ktir_bundle_out` are only
// read inside the spyre cfg block below; without that feature the
// function still lowers + reports stats but consumes none of them.
pub(crate) fn dump_wavefront_mega(
    program: &Program,
    model: &ModelParams,
    // The model's facts (#1083): the one source of the attention geometry and scale this
    // bake lowers by — resolved once below, never re-read from a config or a bound.
    facts: &scratchy_ir::model_facts::ModelFacts,
    fuf: &Fuf,
    inferred: &scratchy_forward_compiler_macro::shape::Inferred,
    decode_bounds: &BTreeMap<String, u64>,
    // Out: under `-Fspyre` the embedded `pub const KTIR_BUNDLE` token stream
    // (spliced into the per-model module); empty under every other target.
    ktir_bundle_out: &mut proc_macro2::TokenStream,
    // Where the bake writes each bundle's dxp input, when the build asked (`superdsc-dump`): the
    // build script's own output directory, passed down from `compile_carrier`'s caller.
    dump_dir: Option<&std::path::Path>,
) {
    use scratchy_forward_compiler_macro::to_wavefront;
    let stem = model.source_stem.as_str();
    let dump = match dump_dir {
        Some(d) => scratchy_target_spyre::lower_subtile_tape_to_superdsc::DumpRoot::under(d),
        None => scratchy_target_spyre::lower_subtile_tape_to_superdsc::DumpRoot::NONE,
    };
    // ⭐ THE MODEL'S ATTENTION FACTS ride the ops themselves now, minted once at the
    // bridge: every consumer below (the rolls, the wiring, the SuperDSC lowering) reads
    // the op's own value. A canonical stating no attention facts (a vision tower) holds
    // no op that reads them — the bridge's one presence check refuses any that appear —
    // so this bake needs no facts of its own.
    // The config params and the typed facts are ONE model — the bridge takes them together.
    let bridge = to_wavefront::BridgeModel {
        params: model,
        facts,
    };
    // Structural prefix-KV-cache rows (== modeled valid_len). MUST be >= 1:
    // with 0 the cache Source is [0, kvdim] and the AttnDecode's prefix read
    // (`rows..1`) overruns it — `SubtileIR rejected by validate(): node N input
    // 1 region exceeds tensor 7` for EVERY model. The real decode length binds
    // at run time: cuda via the `DecodePosition` kernel arg (its .cu is
    // independent of this value), KTIR via the AttnDecode runtime length mask
    // (which reads this as the prefix CAPACITY and masks past `decode_position`
    // — see `lower_subtile_tape_to_ktir`).
    //
    // ⭐ THE KV POOL IS THE MODEL'S OWN CONTEXT. A KTIR bundle is a static-shape graph, so the
    // resident cache is a baked dimension (the GPU target takes `decode_position` as a
    // runtime arg over a host-allocated cache instead): every layer's K and V pools hold the
    // model's `max_position_embeddings` token slots — vLLM's own default `max_model_len` — in whole
    // scheduler pages, as much of it as one device region holds (below). Nothing else caps it and
    // nothing in the environment moves it: the page walk reads
    // only the pages a step's context reaches (its walk rungs, spyre-inference's `num_blocks`
    // buckets), so the pool's size is memory the steps touch, never time.
    // Resident prefix-cache capacity as `NonZeroU32` end-to-end: the 0-row case
    // (an invalid SubtileIR prefix `Source`) is excluded by the TYPE, not a
    // runtime `.filter`, and the value is wrapped in `to_wavefront::PrefixCapacity`
    // before every `lower_decode_to_wavefront` call so decode + prefill can never
    // be baked at diverging capacities.
    //
    // The page is the one the forward's attention walks (`paged_step::SchedulerBlockTokens::REFERENCE.get()`).
    let page_slots = scratchy_target_spyre::paged_step::SchedulerBlockTokens::REFERENCE.get();
    let prefix_len: std::num::NonZeroU32 = model
        .bounds
        .get("max_position_embeddings")
        .map(|&v| (v.min(u32::MAX as u64) as u32).next_multiple_of(page_slots))
        .and_then(std::num::NonZeroU32::new)
        .unwrap_or_else(|| {
            panic!(
                "[spyre] {stem}: the config states no max_position_embeddings — the KV pool is \
                 the model's own context"
            )
        });
    // Primary bundle row count: 1 = decode (default), >1 = the mq golden-gate
    // knob. The worker's batched-prefill bundle is emitted separately below
    // (default m = prefix_len; `KTIR_PREFILL_LEN` overrides).
    let primary_m: u32 = std::env::var("KTIR_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(1);
    let lower_at = |cap: std::num::NonZeroU32| {
        scratchy_forward_compiler_macro::to_wavefront::lower_decode_to_wavefront(
            fuf,
            inferred,
            decode_bounds,
            &bridge,
            scratchy_forward_compiler_macro::to_wavefront::PrefixCapacity::new(cap),
            primary_m,
        )
    };
    let mut first = lower_at(prefix_len);
    // ⭐ AND THE POOL IS WHAT ONE DEVICE REGION HOLDS OF IT (`paged_step::pool_slots`): the KV
    // segment is one allocation, so a model whose every-layer K and V of its whole context outgrow
    // one region (granite-3.1-8b: 20 GiB) gets the whole pages that fit, and serves a context up to
    // them. The bytes a slot costs are the lowered tape's own prefix K and V sources.
    let prefix_len = {
        let kv_slot_bytes: Option<u64> = first.as_ref().ok().map(|l| {
            l.bindings()
                .iter()
                .zip(&l.input.sources)
                .filter(|(b, _)| {
                    matches!(
                        b,
                        scratchy_forward_compiler_macro::to_wavefront::SourceBinding::PrefixK { .. }
                            | scratchy_forward_compiler_macro::to_wavefront::SourceBinding::PrefixV { .. }
                    )
                })
                .map(|(_, s)| u64::from(s.cols) * 2)
                .sum()
        });
        match kv_slot_bytes {
            Some(bytes) => {
                let pool = scratchy_target_spyre::paged_step::pool_slots(prefix_len.get(), bytes)
                    .unwrap_or_else(|| {
                        panic!(
                            "[spyre] {stem}: one position's K and V across every layer ({bytes} B) \
                             leave not one {page_slots}-slot page inside a device region"
                        )
                    });
                if pool < prefix_len {
                    eprintln!(
                        "[spyre] {stem}: the KV pool holds {pool} of the model's {prefix_len} \
                         positions — {bytes} B a position across every layer, one device region \
                         in all"
                    );
                    first = lower_at(pool);
                }
                pool
            }
            None => prefix_len,
        }
    };
    let lowered = match first {
        // A quant weight scheme the wavefront/superdsc lowering does not emit (int4/ggml, or fp8 until
        // wired) genuinely has NO superdsc bundle — skip THIS preset (return no bundle). This
        // is NOT the silent-green bug the panic below guards against: DENSE models never hit
        // UnsupportedWeight, so a dense regression still panics; only quant presets are skipped.
        // The bridge now lowers MLX-affine gemms (for the GPU target's static-tape
        // path); the superdsc emitter here still has no affine
        // realization — skip the preset exactly as the bridge's refusal used
        // to, rather than letting a packed weight flow into a dense-only
        // emitter.
        Ok(l)
            if l.input.ops.iter().any(|od| {
                matches!(
                    od.op,
                    scratchy_subtile::subtile_ir::SubOp::MatmulTile {
                        weight: scratchy_subtile::lower::GemmWeight::Affine { .. },
                        ..
                    }
                )
            }) =>
        {
            eprintln!(
                "[wavefront] {stem}: no superdsc bundle — affine-quantized \
                 weights have no realization on this path; preset skipped"
            );
            return;
        }
        Ok(l) => bake_tape(stem, l),
        Err(to_wavefront::BridgeError::UnsupportedWeight { detail, .. }) => {
            eprintln!(
                "[wavefront] {stem}: no superdsc bundle — weight scheme unsupported here \
                 ({detail}); preset skipped"
            );
            return;
        }
        Err(e) => {
            // If you asked for superdsc and a DENSE model's decode can't be lowered, that is a
            // HARD ERROR, not a silent fall-back. A swallowed `eprintln` here is precisely how "no model
            // produces a superdsc bundle" shipped green for every model at once.
            panic!("[wavefront] {stem}: superdsc decode NOT LOWERED — {e}");
        }
    };
    let st = to_wavefront::stats(fuf, &lowered);
    eprintln!(
        "[wavefront] {stem}: {} fuf tiles → {} sources ({} weights, \
         {} prefix-kv), {} ops; ops {:?}",
        st.fuf_tiles, st.sources, st.weight_sources, st.prefix_sources, st.ops, st.op_histogram,
    );

    // Spyre/KTIR (-Fspyre): emit the REAL model's per-op KTIR from the solver-produced
    // graph — every dim, rope form, GQA ratio, and eps is scratchy's, not a hand-built
    // stand-in. Drop a per-model KTIR bundle.
    {
        use std::num::NonZeroU32;
        let knb = NonZeroU32::new(8192).expect("8192 != 0");
        let stem_id = stem.replace(['-', '.'], "_");
        // Opt-in `--device sdsc` (real-silicon) dump: emit the SAME real-model
        // `krg` SubtileIR to a sendnn op-graph JSON the DeepTools `compile_graph`
        // toolchain consumes. Off by default; writes one `<base>.sengraph.json`
        // ── PAGED revival (opt-in): `SCRATCHY_SENDNN_MODE=paged` emits the vLLM-Spyre
        //    offline_decoder [prefill,decode] PAGED bundle (PagedAttnStore/Compute,
        //    resident KV, symbolic s0/s1) — prefill = SDPA+Store, decode = Store+Compute,
        //    distinct graphs sharing the cache by node name. UNSET (the build default)
        //    keeps the dd2-PROVEN static-SDPA host-grown-KV single graph so the decode
        //    sentinel never regresses. PROVEN viable on dd2 SENTIENT 2026-06-23.
        let sendnn_paged = std::env::var("SCRATCHY_SENDNN_MODE").as_deref() == Ok("paged");
        // ── SuperDSC PERF path: the `superdsc` CARGO FEATURE (`--features superdsc`,
        //    NOT an env var) walks the SAME `krg` SubtileIR straight to a TILE-LEVEL,
        //    WORK-DIVIDED SuperDSC bundle (`lower_graph_to_superdsc`) that OWNS the
        //    32-core split — the fix for the sengraph path leaving 31 cores idle
        //    (SENCORES 1→14.4ms vs 32→9.9ms, only 1.45×). Drops the bundle into the
        //    per-process scratch tree, where the bake queue dxp-compiles it in place. Only the
        //    runnable m=1 DECODE tape (seq_sym=None, !is_prefill) is walked; the
        //    symbolic prefill/CB tapes are skipped (emulator-only). An op the emitter
        //    can't lower yet is a HARD error here (build-time guard), never a silent
        //    skip. (A Cargo feature, not an env trigger.)
        // Every spyre build emits the bundle: the lowering has one target, and it is KTIR.
        let sendnn_superdsc = true;

        // Emit ONE KTIR bundle (decode or prefill) from a lowered graph into
        // `dir`. Factored so we drop BOTH a cheap m=1 decode bundle (fast
        // per-token generation) AND an m=M batched-prefill bundle — the worker
        // uses the right one per phase, so decode never pays the prefill tax.
        // Returns (whole_graph_sengraph_json, group_graphs) where group_graphs[i] =
        // (group_idx, input_tensor_id, result_tensor_id, group_graph_json) — the
        // per-layer-group split graphs.
        type GroupGraphs = Vec<(u32, u32, u32, String)>;
        // SuperDSC bundle fingerprint (set by the m=1 decode walk under
        // `SCRATCHY_SENDNN_MODE=superdsc`). When present, the SengraphBundle's
        // (single) group has its prefill+decode `graph_json` STAMPED to
        // `SUPERDSC_BUNDLE:{fp}` so the worker resolves the baked dxp bundle dir
        // (via `scratchy_builder_spyre::baked_superdsc_dir`) instead of compiling a
        // sengraph. A `RefCell` because the closure is `Fn` (called twice) and must
        // write the fp out without taking ownership.
        let superdsc_fp: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
        // THE TYPED LAUNCH WIRING (`scratchy_target_spyre::wiring::Wiring`),
        // emitted by the DECODE walk — the same facts `manifest.json` carries,
        // as spyre's own types instead of a string the worker re-parses. A
        // `RefCell` for the same reason `superdsc_fp` is one: the closure is
        // `Fn` and runs twice (decode, prefill), and only the decode walk's
        // wiring is kept (both agree on every field below — the prefill bundle
        // differs only in row count, which is not a term here).
        #[allow(clippy::type_complexity)]
        let wiring_tokens: std::cell::RefCell<(
            Option<proc_macro2::TokenStream>,
            Option<proc_macro2::TokenStream>,
        )> = std::cell::RefCell::new((None, None));
        // (label, fold set sorted by tid) of the first program that walked — every
        // later program must reproduce it exactly. See the census doc below.
        type WeightFoldCensus = Option<(String, Vec<(u32, f32)>)>;
        // ⛔ CROSS-PROGRAM FOLD CENSUS (guard-every-crash-at-build-time). Every program this
        // model bakes — decode, each cap bucket, each batch rung, each prefill width — runs
        // over ONE set of staged weight bytes, and a ScalarMul weight fold rewrites those
        // bytes at load. So the fold set is not a per-program fact: a program whose
        // recognition folds a multiplier another program still runs as an op would read
        // pre-scaled weights and scale them AGAIN (or route the op to the host and never
        // apply it). The first walk's set is the reference; every later walk must equal it,
        // sorted by tid, or the build refuses naming both programs.
        let weight_fold_census: std::cell::RefCell<WeightFoldCensus> =
            std::cell::RefCell::new(None);
        // SEPARATE fp for the m=N batched-PREFILL reroll bundle ("prefill-all-but-last"). The
        // prefill `emit_bundle` (is_prefill=true; under superdsc its seq_sym is forced to None so
        // the tape is CONCRETE at m=prefill_m) bakes its OWN prefix/body/suffix — with the lm_head
        // SKIPPED (lower_one_node: is_lm_head && m>1) so the vocab-wide lm_head never time-tiles —
        // and stamps HERE, so it does NOT clobber the decode body_fp. Threaded into the
        // SengraphBundle's PREFILL graph slot (C4) so the worker resolves a DISTINCT prefill dxp
        // bundle (prefill:Some) and runs the prompt in one m=N forward, then decodes the last token.
        let superdsc_prefill_fp: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
        // PREFILL WIDTH LADDER `[(mq, body_fp)]` — every prefill reroll that bakes stamps its own
        // width here, so the ladder is whatever actually LOWERED, never an assumed list. The top rung
        // is `prefill_m` (also stamped into `superdsc_prefill_fp`, keeping the single-bundle path and
        // every downstream guard unchanged); interior rungs are extra bakes over the SAME resident
        // weights+KV. See `manifest::SengraphBundleGroup::prefill_rungs`.
        let superdsc_prefill_rungs: std::cell::RefCell<Vec<(u32, String)>> =
            std::cell::RefCell::new(Vec::new());
        // DECODE BATCH LADDER `[(seqs, body_fp)]` — one entry per width that actually LOWERED, never
        // an assumed list. The one-request rung is the bundle already stamped into `superdsc_fp`, so
        // every path that predates the ladder is unchanged; wider rungs only ever add entries.
        // (seqs, active_cap, body_fp, OWN load_consts) — the swept extent rides along so the worker
        // can tighten its mask-coverage guard from `POOL_PAGE_SLOTS * pages` to the real
        // `active_cap * pages`, and the consts are THIS RUNG'S OWN KINDS (staged at the rung's own
        // `rows` — the ten uniform `[rows, 1]` tables' stick heads are per-width, Finding 1's
        // compounding half: the single-request wiring's rows=1 kinds staged into a rows=N rung
        // left every row past 0 reading zero).
        let superdsc_decode_rungs: std::cell::RefCell<Vec<DecodeRungEntry>> =
            std::cell::RefCell::new(Vec::new());
        // The PREFIX-CAPABLE ladder `[(mq, body_fp)]` — the prefill bundles baked with a full
        // resident-prefix sweep. Every prefix-free rung is baked ActiveCap::NONE because a start==0 chunk
        // has nothing resident to attend, which drops 308 of 710 ops/layer. A CONTINUATION chunk (start>0)
        // does have resident KV, so it runs on the smallest of these that holds it. See
        // `manifest::SengraphBundleGroup::prefill_prefix`.
        let superdsc_prefill_prefix_rungs: std::cell::RefCell<Vec<(u32, String)>> =
            std::cell::RefCell::new(Vec::new());
        // ⭐⭐⭐ THE MODEL'S TIED LM-HEAD TABLE, AS THE FIRST BAKE RECOGNIZED IT — the
        // ⭐ THE WEIGHT-SEGMENT PARITY LEDGER — one `(bundle label, bank0 bytes, extra
        // banks)` per superdsc bundle this model baked (decode ceiling, every ladder
        // rung, every batch rung, every prefill width), fed by `BakeEffect::WeightSeg`
        // and judged ONCE after the last bake by
        // `lower_subtile_tape_to_superdsc::weight_segment_parity` — the build-time half
        // of the runtime's `alias_seg` size guard, so the next bundle skew is a BUILD
        // error, never a card-load refusal.
        // ⭐ THE REGISTRY LEDGER — every bake's scalar registry, by phase, beside the registry of
        // the wiring that phase's launches bind (`registry_bound`: decode, prefill). The worker binds
        // ONE wiring per phase for every bundle of it (a prompt chunk's rung, a decode batch rung),
        // so a bundle placing a different registry reads its constants by the wrong index. Judged
        // once, after the last bake.
        let registry_ledger: std::cell::RefCell<Vec<(bool, String, Vec<f32>)>> =
            std::cell::RefCell::new(Vec::new());
        /// `(decode, prefill)`: the registry each phase's bound wiring places.
        type PhaseRegistries = (Option<Vec<f32>>, Option<Vec<f32>>);
        let registry_bound: std::cell::RefCell<PhaseRegistries> =
            std::cell::RefCell::new((None, None));
        let weight_seg_ledger: std::cell::RefCell<Vec<(String, u64, Vec<u64>)>> =
            std::cell::RefCell::new(Vec::new());
        // ⭐ THE STATE ONE TENSOR-PARALLEL WORLD'S BAKES COMMIT INTO. Every cell above is the
        // one-rank world's; a sharded world bakes its own forward (rank-local weights, its own
        // fingerprints, its own wiring) and commits into cells of its own — its weight segment is
        // not the one-rank world's, so the parity law judges its ledger apart, and its wiring must
        // not overwrite the one-rank decode slot. ONE commit (`commit_bake`) serves every world;
        // the world is which cells it is handed.
        struct WorldBake<'w> {
            fp: &'w std::cell::RefCell<Option<String>>,
            prefill_fp: &'w std::cell::RefCell<Option<String>>,
            prefill_rungs: &'w std::cell::RefCell<Vec<(u32, String)>>,
            prefill_prefix_rungs: &'w std::cell::RefCell<Vec<(u32, String)>>,
            decode_rungs: &'w std::cell::RefCell<Vec<DecodeRungEntry>>,
            wiring: &'w std::cell::RefCell<WiringPair>,
            fold_census: &'w std::cell::RefCell<WeightFoldCensus>,
            weight_seg_ledger: &'w std::cell::RefCell<Vec<(String, u64, Vec<u64>)>>,
        }
        type WiringPair = (
            Option<proc_macro2::TokenStream>,
            Option<proc_macro2::TokenStream>,
        );
        let one_rank = WorldBake {
            fp: &superdsc_fp,
            prefill_fp: &superdsc_prefill_fp,
            prefill_rungs: &superdsc_prefill_rungs,
            prefill_prefix_rungs: &superdsc_prefill_prefix_rungs,
            decode_rungs: &superdsc_decode_rungs,
            wiring: &wiring_tokens,
            fold_census: &weight_fold_census,
            weight_seg_ledger: &weight_seg_ledger,
        };
        // The sharded world's cells (`TP_WORLD`; see the sharded bake below).
        let tp_fp = std::cell::RefCell::new(None);
        let tp_prefill_fp = std::cell::RefCell::new(None);
        let tp_prefill_rungs = std::cell::RefCell::new(Vec::new());
        let tp_prefill_prefix_rungs = std::cell::RefCell::new(Vec::new());
        let tp_decode_rungs = std::cell::RefCell::new(Vec::new());
        let tp_wiring: std::cell::RefCell<WiringPair> = std::cell::RefCell::new((None, None));
        let tp_fold_census = std::cell::RefCell::new(None);
        let tp_weight_seg_ledger = std::cell::RefCell::new(Vec::new());
        let sharded_world = WorldBake {
            fp: &tp_fp,
            prefill_fp: &tp_prefill_fp,
            prefill_rungs: &tp_prefill_rungs,
            prefill_prefix_rungs: &tp_prefill_prefix_rungs,
            decode_rungs: &tp_decode_rungs,
            wiring: &tp_wiring,
            fold_census: &tp_fold_census,
            weight_seg_ledger: &tp_weight_seg_ledger,
        };
        // ⭐ WHAT ONE BAKE DOES TO THE EXPANSION'S SHARED STATE, recorded in the order it happens and
        // applied by `commit_bake` — so the heavy part of a bake (`emit_bundle_deferred`) touches no
        // shared state at all, and can run on any thread, while the expansion still sees exactly the
        // writes, in exactly the order, that a serial loop made.
        enum BakeEffect {
            PrefillPrefixRung {
                mq: u32,
                fp: String,
            },
            PrefillRung {
                mq: u32,
                fp: String,
            },
            PrefillFp(String),
            // ⭐ THE RUNG'S OWN KINDS ride the effect: the decode batch rung's
            // load_consts at ITS OWN `rows`, from THIS bake's wiring (`gk`) —
            // the one derivation, carried to the manifest so the worker never
            // stages the single-request wiring's rows=1 kinds into a wider
            // rung (Finding 1's compounding half).
            DecodeRung {
                rows: u32,
                cap: u32,
                fp: String,
                consts: Vec<(u32, scratchy_target_spyre::wiring::LoadConst)>,
            },
            DecodeFp(String),
            FoldCensus {
                base: String,
                folds: Vec<(u32, f32)>,
            },
            Wiring {
                slot: Option<WiringSlot>,
                // Boxed for the enum's variant-size budget only — `BakeEffect` rides a Vec
                // through every bake, and this one field is five sixths of the largest
                // variant; the box changes no ownership the arm below does not already
                // borrow (`&gk` derefs through it where the wiring is built).
                gk: Box<scratchy_target_spyre::lower_subtile_tape_to_superdsc::BundleWiring>,
                decode_position: u32,
            },
            // ⭐ ONE WEIGHT-SEGMENT PARITY LEDGER ENTRY — this bundle's packed weight
            // segment (bank 0 bytes + extra banks), exactly as the runtime's alias guard
            // reads it, labelled by the bake that produced it. Judged once, after the
            // last bake, by `weight_segment_parity`.
            WeightSeg {
                base: String,
                bank0: u64,
                banks: Vec<u64>,
            },
            // ⭐ THIS BAKE'S SCALAR REGISTRY, by phase — the registry ledger's entry.
            Registry {
                prefill: bool,
                base: String,
                scales: Vec<f32>,
            },
        }
        /// What one bake returns: its sengraph json and launch groups.
        type BakeOut = (String, GroupGraphs);
        /// Which half of `wiring_tokens` a bake's wiring lands in, if either.
        enum WiringSlot {
            Decode,
            Prefill,
        }
        // `prefill_mq`: Some(mq) marks the PREFILL path AND carries the concrete query-row count the
        // tape was lowered at — the ladder rung this bake belongs to. `None` = decode.
        // ⭐ ONE BAKE BODY, STAMPED PER WORLD. A bake's ops, launch groups and bundle are its
        // world's types and a closure cannot be generic, so the body is stated ONCE here and
        // instantiated for each world this expansion bakes: the one-rank forward's (`TpWorld<1>`)
        // and the build's one `tp/N` (`Build`). `$W` is that world inside the body.
        macro_rules! world_bake {
            ($W:ty) => {
            |lwd: &BakeTape,
             base: String,
             // The capacity THIS bundle is baked for. Decode cap buckets vary it;
             // every other call site passes the full `prefix_len`.
             cap: u32,
             seq_sym: Option<(&'static str, u32)>,
             prefill_mq: Option<u32>,
             // The resident-prefix sweep this bundle is baked for. Decode always FULL.
             // A PREFILL chunk with start==0 has no resident prefix, so it bakes NONE
             // (nb=0) and skips 308 of 710 ops/layer; continuation chunks need FULL.
             active_cap: scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap,
             // The row count THIS bundle is baked at. Not `primary_m`: a batch-ladder
             // rung is a wider bake of the same tape, and reading the env knob here
             // would tell every rung it was one row.
             decode_rows: u32,
             // The slice law of every source when `lwd` is one rank of a sharded world — the
             // numeric gate runs such a bundle on both ranks at once. `None` for a world of one.
             shards: Option<&[scratchy_subtile::tensor_parallel::SourceShard]>|
             -> (BakeOut, Vec<BakeEffect>) {
                // Only the emulator's numeric gate reads the slice laws.
                #[cfg(not(feature = "spyre-emu"))]
                let _ = shards;
                let mut fx: Vec<BakeEffect> = Vec::new();
                let is_prefill = prefill_mq.is_some();
                // A prefill chunk's rows are one prompt's positions; a decode bundle's are one token
                // each of the running requests, at any width — one row is one request's token, and
                // spyre-inference decodes it with the batched kernel at one sequence (its sequence
                // buckets start at 1). Every other reader of the fact short-circuits at one row.
                let rows_are_requests = !is_prefill;
                // ⭐ THE QUERY'S EXTENT PAIR, MINTED ONCE FROM THE RUNG — the rung's own
                // `(per-request width, requests)`, never a division of the graph's rows. A
                // prefill chunk is ONE request however many positions it spans (its pair is
                // `(mq, ONE)`); a decode rung's every row is a request's one token (its pair
                // is `(1 row, decode_rows)`). The attention splice spends the pair as the
                // kernel's two extents and checks its held product against the graph it
                // lowers — the one place the mint and the graph are proven to agree.
                // ONE match on the one fact (`prefill_mq`'s presence IS `is_prefill`), one
                // named bake panic for the whole mint: a half that does not name a baked
                // rung — a zero or overflowing count, or a product past u32 — is refused
                // here, never defaulted. (The u32 parameters `prefill_mq` / `decode_rows`
                // are the missing source type: the rung walks should hand the closure a
                // `RungWidth` / `RungCap` directly, follow-up.)
                //
                // ⭐ A SYMBOLIC BATCH TAPE states its width through `seq_sym`: the CB decode
                // emission lowers at `CB_BATCH_TEMPLATE` rows (a batch SYMBOL the DEM
                // resolves at runtime) while passing `decode_rows = 1` on purpose — its
                // `DecodeRung` effect and slot discriminators read the CONCRETE rung
                // count, which the symbolic tape does not have. Each arm says which
                // source states the width for which kind of tape: a prefill — symbolic
                // or concrete — takes its width from `prefill_mq` (one request however
                // many positions), a symbolic decode batch from its symbol's template,
                // a concrete decode rung from `decode_rows`.
                let (per_request, requests) = match (prefill_mq, seq_sym) {
                    (Some(mq), _) => (mq, 1u32),
                    (None, Some((_, width))) => (1u32, width),
                    (None, None) => (1u32, decode_rows),
                };
                let query_rows = (|| {
                    ktir_superdsc::sdsc_abstract::QueryExtent::of_baked_rung(
                        ktir_superdsc::sdsc_abstract::RungWidth::of_baked_rows(per_request)?,
                        ktir_superdsc::sdsc_abstract::RungCap::of_baked_seqs(requests)?,
                    )
                })()
                .unwrap_or_else(|| {
                    panic!(
                        "{base}: the rung's query extents (per-request {per_request} row(s), \
                         {requests} request(s)) do not name a baked rung — a zero or \
                         overflowing count is a bake bug, named here at the mint"
                    )
                });
                // ⛔ THE WIRING MUST NOT DESCRIBE A PROGRAM THAT DID NOT SHIP. A prefill
                // rung whose reroll fails PANICS (no rung may fail to lower), but this
                // closure's KTIR emission and wiring stamp run AFTER the reroll and
                // OUTLIVE its failure — so an
                // unqualified `slot.1 = Some(w)` shipped a wiring promising an m=31
                // batched prefill no bundle carried. The worker keys `prefill_prog` on
                // the wiring alone (`spyre_load.rs`, `pparsed.is_some() ⇒ Some(0)`) and
                // pushes the prefill program only when the bundle has one, so the
                // mismatch routed the prompt into DECODE program 0 and staged `m_cap`
                // rows into an m=1 program — ten unchecked set_source OVERFLOW clobbers,
                // silently wrong output (MEASURED: gemma-4-26b and the tiny MoE fixture
                // on the emulator, t0/t10-13/t187-190/t2405 at x31; the dense granite,
                // whose m=31 rung DID bake, staged clean). This flag is the per-call
                // record of "this prefill bundle actually baked", set at the only two
                // sites that stamp a prefill fp and read at the wiring stamp below — a
                // rung that did not bake leaves the wiring unstamped, `pparsed` parses
                // nothing, and the worker falls to its sequential per-token prefill
                // (slower, never wrong).
                let mut prefill_fp_stamped = false;
                let fused = scratchy_subtile::lower::fuse_silu_mul(&lwd.input);
                let mut krg = scratchy_subtile::subtile_ir::lower_region(&fused, knb);
                // ⭐⭐⭐ THE CHAINED HEAD'S WIDENING PASS — ABOVE every mint, tie-gated
                // at the model: the tied embed weight's STATED cols widen to the pick's
                // pad (`sample_pad_vocab`), so the gather's stride-`pad` walk and the
                // decompose's clamp boundary are ONE width by construction. It must run
                // BEFORE the mints — both passes walk the result tail back to the
                // lm-head, and after the draw the result IS the idx stick, a tail the
                // widen walk cannot read. Non-admitting tails return None and touch
                // nothing; every admitting bundle of a tied model reserves the same
                // widened bytes (the weight-segment parity ledger's law).
                let tied = tie_word_embeddings(model);
                let widen = if tied {
                    scratchy_subtile::subtile_ir::widen_tied_embed_weight(&mut krg)
                } else {
                    None
                };
                if let Some(w) = &widen {
                    eprintln!(
                        "[spyre-superdsc] {base}: the tied embed weight widened — vocab {} \
                         (stated cols {}), reservation grown, FLOPs untouched (the matmul's \
                         region stays 0..vocab)",
                        w.vocab, w.pad_vocab
                    );
                }
                // ⭐⭐⭐ THE TOKEN DRAW'S MINT — EVERY rung this closure bakes: the
                // solo m=1 bake, every emulator cap bucket, every batch-ladder rung,
                // AND every prefill rung (the general mint — no prefill-specific
                // copy, no `one_request` gate, no lm-head admission walk here: #813
                // minted only the one-request rung whose lm-head admitted, and what
                // survived that front is the LAW, not the gate — a rung without the
                // draw is a rung whose answer must still come back as logits for a
                // host argmax, which is exactly the readback this front deletes).
                // The splice is the kernel's ONLY entry — the mint appends the node,
                // its per-row per-step seat (the seed quadruple's four word tables,
                // inv_t) and its twenty staged-table
                // sources to the graph, and everything downstream (the splice's row,
                // the door's whole-function walk, the placement pass, the wiring's
                // sampler facts) is the same machinery every other Triton kernel
                // takes. The mint's own tail walk IS the admission: a graph whose
                // result tail is not the lm-head chain it knows PANICS at expansion —
                // a build error naming the op, never a silent fall back to the host
                // readback. A prefill rung's result is the FOLDED lm-head rows (the
                // emitter folds the tail to the requests' last baked rows): its
                // prompt-final chunk's picks are the idx stick, read back exactly
                // like decode's; a middle chunk's suffix never runs, so its draw
                // is never taken.
                {
                    let mint = scratchy_subtile::subtile_ir::mint_sample(
                        &mut krg,
                        if is_prefill {
                            // The Folded draw serves the rung's OWN request
                            // capacity — the same extent pair's `RungCap` the
                            // attention and the lm-head fold read (one mint,
                            // never a division of the graph's rows): the fold
                            // writes S logits rows and the draw serves exactly
                            // them.
                            scratchy_subtile::subtile_ir::SampleRows::Folded {
                                requests: query_rows.requests().nonzero(),
                            }
                        } else {
                            scratchy_subtile::subtile_ir::SampleRows::All
                        },
                    );
                    eprintln!(
                        "[spyre-superdsc] {base}: the token draw minted — vocab {} (padded row \
                         {}), rows {}, the per-row seat (seed words t{}..t{}, inv_t t{}) + \
                         twenty staged tables appended, and the rung's RESULT re-pointed to the \
                         idx stick (the row stays placed and readable by tid){}",
                        mint.vocab,
                        mint.pad_vocab,
                        mint.rows,
                        mint.seed_words[0].index(),
                        mint.seed_words[3].index(),
                        mint.invt.index(),
                        if is_prefill { " (prefill rung)" } else { "" },
                    );
                    // ⭐⭐⭐ THE CHAINED HEAD'S MINT (#603) — the second splice onto the
                    // same DECODE rung (prefill keeps the host-staged embed rows — its
                    // input row is prompt tokens the host owns, never a device pick):
                    // the EmbedRowGather at the entry (every whole-tensor consumer of
                    // the embed row re-pointed onto its output), the ChainStage at the
                    // tail (the pick decomposed into the persistent s/l sticks through
                    // the two staged tables), the four-source insert below the
                    // SourceBinding layer. THE PICK IS THE DRAW'S OWN OUTPUT — the
                    // by-kind find the A-seam refused on: `mint_sample` re-pointed the
                    // rung's result to the idx stick, so `krg.result` IS the device
                    // pick, and the vocab the tables decompose over is the mint's own
                    // emitted vocab. GATED ADDITIONALLY ON THE TIE (the gather's matrix
                    // operand is the lm-head's weight source, the embedding matrix IFF
                    // the model ties): an untied decode rung REFUSES here — the chained
                    // head is the only decode path, and there is no host-pick arm to
                    // fall to. Every width mints (the solo rung, every batch-ladder
                    // rung, the symbolic CB batch alike): the chain is the decode path,
                    // not a wide-batch extra.
                    // ⛔ A PARTIAL ROW IS A CHAIN'S, AND A PREFILL HAS NO CHAIN. A source the
                    // analysis cut as a gathered row's partial sum is filled by the chain's
                    // per-rank gather; a prefill rung stages its rows whole from the host, so the
                    // all-reduce would sum the prompt's rows once per rank.
                    if is_prefill
                        && let Some(e) = shards.and_then(|s| {
                            s.iter().position(|c| {
                                *c == scratchy_subtile::tensor_parallel::SourceShard::Partial
                            })
                        })
                    {
                        panic!(
                            "[spyre-tp] {base}: source t{e} is a gathered row's partial sum, but a \
                             prefill rung mints no chained head to gather it — the forward was \
                             sharded with a decode's row gather"
                        );
                    }
                    if !is_prefill {
                        if !tied {
                            panic!(
                                "[spyre-superdsc] {base}: the decode rung's chained head \
                                 requires a tied embedding head — this model does not tie \
                                 word embeddings. The chained head is the only decode path \
                                 (there is no host-pick arm to fall to); the untied \
                                 admission is not implemented."
                            );
                        }
                        let embed_row = lwd.embed_row().index();
                        let pick = krg.result;
                        let chain = scratchy_subtile::subtile_ir::mint_embed_chain(
                            &mut krg,
                            scratchy_subtile::subtile_ir::TensorId::from_index(embed_row),
                            pick,
                            mint.vocab,
                        );
                        eprintln!(
                            "[spyre-superdsc] {base}: the chained head minted — the embed row \
                             gather spliced over source t{embed_row} (hidden {}, vocab {}), the \
                             persistent s/l sticks at t{}/t{}, the two decompose tables staged \
                             once at load, and the pick decomposed into the sticks at the tail",
                            chain.hidden,
                            chain.vocab,
                            chain.s_stick.index(),
                            chain.l_stick.index()
                        );
                    }
                }
                if let Err(e) = scratchy_subtile::subtile_ir::ValidatedGraph::new(&krg) {
                    eprintln!("[spyre-ktir] {stem}: SubtileIR rejected by validate() — {e:?}");
                    return ((String::new(), Vec::new()), fx);
                }
                // `--target sendnn`: lower the SAME real-model `krg` to a sendnn
                // op-graph JSON (the DeepTools toolchain compiles + runs it on the
                // AIU). Baked into the binary below as `SENGRAPH_BUNDLE`; dumped to
                // disk too when `SCRATCHY_SENGRAPH_DUMP` is set.
                // Only symbolic bundles (seq_sym=Some: prefill prompt, CB decode batch)
                // are real `--target sendnn` decoder graphs. The plain m=1 KTIR decode
                // (seq_sym=None) is emulator-only — skip its sengraph so it never trips
                // the decoder-contract guards as build noise.
                // DEFAULT-MODE single graph: lower the m=1 decode tape to the static-SDPA
                // host-grown-KV sengraph and bake it as group 0 (the only group). The
                // offline_decoder layer-group split was removed (it never cleared the
                // dd2 per-job flit cap); the default-mode graph partitions itself via the
                // explicit host Reshape/Transpose breaks. `group_graphs[i]` =
                // (group, input_tensor, result_tensor, json).
                let group_graphs: Vec<(u32, u32, u32, String)> = Vec::new();
                // WEIGHT IDS for the fold pass: the sengraph source tensor id `t{i}` is a
                // model WEIGHT iff `lwd.bindings()[i]` is `SourceBinding::Weight` (the
                // binding index == the sengraph source id — both from the same
                // `lwd.input`). Runtime activations (embed/cos/sin/prefix-kv) stay
                // PrimaryInput. `fold_weights_into_model` reclassifies the weights to
                // `ModelInput` so the DEM folds them into the resident model_tensor, off
                // the per-job I/O descriptor table (the flit-cap fix).
                let weight_ids: std::collections::HashSet<u32> = {
                    use scratchy_forward_compiler_macro::to_wavefront::SourceBinding;
                    lwd.bindings()
                        .iter()
                        .enumerate()
                        .filter_map(|(i, b)| match b {
                            // The fp8 per-channel weight_scale is a STATIC resident tensor (loaded once,
                            // like a weight) — not a per-step activation — so it joins seg1 (SegRole::Weight).
                            SourceBinding::Weight { .. }
                            | SourceBinding::WeightScale { .. }
                            | SourceBinding::BundleTensor { .. } => Some(i as u32),
                            _ => None,
                        })
                        .collect()
                };
                // ── SuperDSC PERF emission (opt-in `SCRATCHY_SENDNN_MODE=superdsc`) ──
                // Walk the SAME `krg` straight to a work-divided SuperDSC bundle and
                // write it to the per-process scratch tree, dxp-compiled in place. Only
                // the runnable m=1 decode tape (seq_sym=None, !is_prefill) — the symbolic
                // prefill/CB tapes are emulator-only. An unlowered op is a HARD error
                // (the worklist signal: it names the exact next op to emit), never a
                // silent skip (which would mean silently-wrong on-card output).
                // superdsc bakes BOTH the m=1 decode reroll AND the m=N prefill reroll (C1 forces the
                // prefill's seq_sym to None so it is CONCRETE here). `seq_sym.is_none()` still excludes
                // the emulator-only symbolic CB-decode tape (Some(("s0",…))). is_prefill selects which fp
                // to stamp (superdsc_prefill_fp vs superdsc_fp) at the bake site below.
                // COMPILE-gated on `superdsc`: the sdsc emitter module only exists under that feature, so a
                // `spyre`-only (ktir) build must not compile this block at all (`sendnn_superdsc` alone is a
                // runtime bool and would still require the module to resolve).
                // ⭐ THE WIRING COMES OFF THE LOWERING THAT EMITS THE PROGRAMS. Its parameter
                // pairings are the ones the bundle's launches bind, so there is one lowering and
                // the wiring cannot describe a different program than the one that ships.
                // ⭐ HOISTED ABOVE THE REROLL BAKE: the decode batch rung's effect (pushed inside
                // that bake) carries THIS bake's own `load_consts` — the kinds at the rung's own
                // `rows`, never the single-request wiring's (Finding 1's compounding half) — so
                // the derivation must exist before the push. `krg` is minted and validated above
                // and only ever borrowed below, so computing the wiring here is the same value at
                // an earlier point, nothing more.
                let gk = match scratchy_target_spyre::lower_subtile_tape_to_superdsc::graph_wiring::<$W, _>(
                    &krg,
                    &weight_ids,
                    active_cap,
                    scratchy_target_spyre::paged_step::PagedRows::of_requests(rows_are_requests),
                    query_rows,
                ) {
                    Ok(w) => w,
                    Err(e) => panic!("[spyre] {base}: wiring not lowerable — {}", e.0),
                };
                if sendnn_superdsc && seq_sym.is_none() {
                    use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;
                    // ── GATE-1 de-risk: when `SCRATCHY_SUPERDSC_GATE1=1`, also drop a
                    //    SINGLE self-contained matmul bundle (no dangling tensor refs)
                    //    so a pod `sendnn` build's dxp-bake can prove dxp INGESTS
                    //    scratchy's emitted SuperDSC (encoding + build.rs arm + init.txt
                    //    end-to-end) BEFORE the full decode bundle is complete. Content-
                    //    keyed write → idempotent across emit_bundle calls.
                    if std::env::var("SCRATCHY_SUPERDSC_GATE1").as_deref() == Ok("1") {
                        // SMALL LX-fitting matmul (64×64×64, batch 1): per-core tile
                        // (kernel ~8KB) is well under the LX scratchpad, so it compiles
                        // WITHOUT the frontend coarse-tile (time-tiling) pass — isolating
                        // "is the typed emitter's format dxp-valid?" from "does it tile
                        // large tiles?" (the latter = the coarse_tile port, next).
                        use ktir_superdsc::sdsc_abstract::{
                            KernelTag, RowBlockedTag, StickLayout, Stk,
                        };
                        // Typed matmul operands (compile-time addressing): activation/output RowBlocked, weight Kernel.
                        let g1_a =
                            Stk::<RowBlockedTag>::new("g1_a", StickLayout::row_blocked(64, 64))
                                .unwrap();
                        let g1_w = Stk::<KernelTag>::kernel(64, 64, "g1_w");
                        let g1_o =
                            Stk::<RowBlockedTag>::new("g1_o", StickLayout::row_blocked(64, 64))
                                .unwrap();
                        let g1 = superdsc::assemble_matmul(
                            "gate1_mm", 64, 64, 64, 1, &g1_a, &g1_w, &g1_o, None,
                        );
                        // A hand-authored matmul, so there is no attention node for the four
                        // attention facts to be facts of — main's own `false`/absent at such a site.
                        match g1.map_err(|e| e.to_string()).and_then(|g1| {
                            superdsc::emit_bundle::<$W>(
                                &[g1],
                                None,
                                superdsc::FoldGrouping::Split,
                                None,
                                dump,
                            )
                            .map_err(|e| e.to_string())
                        }) {
                            Ok(fp) => eprintln!(
                                "[spyre-superdsc] GATE-1 single-matmul bundle emitted (fp {fp}) — \
                                 dxp compiles it inline"
                            ),
                            Err(e) => eprintln!("[spyre-superdsc] GATE-1 bundle emit failed: {e}"),
                        }
                    }
                    // RE-ROLL the layer loop (the 40-min-compile fix): instead of lowering
                    // the 30×-UNROLLED `krg` to one 2047-op bundle (dxp super-linear → 40 min),
                    // re-roll the tape (mirror the tk_tape path, codegen ~8807) and lower the
                    // ONE layer BODY (~68 ops) → a SMALL bundle dxp compiles in SECONDS.
                    // STAGE 1: bake the BODY bundle (proves the fast compile). STAGE 2 adds the
                    // prefix(embed)/suffix(lm_head) bundles + the executor loop over `iters`
                    // layers (per_layer weight/KV/hidden ivar threading).
                    use scratchy_subtile::subtile_ir::ValidatedGraph;
                    use scratchy_subtile::subtile_tape::{
                        lower_dag_to_tape, reroll_layer_classes, reroll_subtile_tape,
                    };
                    match ValidatedGraph::new(&krg) {
                        Ok(valid) => {
                            let tape_unrolled = lower_dag_to_tape(&valid);
                            // ⭐ THE CLASS-SPLIT ROLL FIRST. A hybrid-attention model (gemma-4:
                            // alternating sliding/global layers) has TWO distinct layer shapes, and
                            // the welded roll's smallest repeat unit is the whole 6-layer cell — a
                            // bake-unit ~6× larger than either class's body. `reroll_layer_classes`
                            // splits it into one sibling loop per class; `reroll_subtile_tape`
                            // remains the single-class roll every other model takes, byte-identical.
                            let tape_rolled = reroll_layer_classes(&tape_unrolled, &krg)
                                .unwrap_or_else(|| reroll_subtile_tape(&tape_unrolled, &krg));
                            match superdsc::lower_subtile_tape_to_superdsc::<$W, _>(
                                &tape_rolled,
                                &krg,
                                &weight_ids,
                                active_cap, // decode: FULL (sk_bucket rungs land below). prefill: NONE for a
                                // start==0 chunk, FULL for the continuation bundle.
                                rows_are_requests,
                                query_rows,
                            ) {
                                Ok(rolled) => {
                                    // ⭐ AND THE PARITY LEDGER'S ENTRY FOR THIS BUNDLE — the
                                    // ceiling's packed weight segment, as the runtime's alias
                                    // guard reads it. One entry per baked bundle; the ledger is
                                    // judged once, after the last bake.
                                    {
                                        let (bank0, banks) =
                                            superdsc::weight_seg_record(&rolled.layout);
                                        fx.push(BakeEffect::WeightSeg {
                                            base: format!("{base} (ceiling)"),
                                            bank0,
                                            banks,
                                        });
                                    }
                                    // `iters` counts LOOP ITERATIONS, not layers: a welded body can
                                    // be a multi-layer cell (tiny26's (S,F) pair, ×3 = all 6
                                    // layers), and the runtime load line reports this same sum as
                                    // "launch(es)" — one launch per (body, iteration).
                                    let launches: u32 = rolled.bodies.iter().map(|b| b.iters).sum();
                                    eprintln!(
                                        "[spyre-superdsc] {base}: RE-ROLLED — prefix {} / {} body \
                                         class(es) ({} ops each, ×{} launch(es) total) / suffix {} ops \
                                         ({} per-layer tids). Baking the body bundle(s) (STAGE 1).",
                                        rolled.prefix.len(),
                                        rolled.bodies.len(),
                                        rolled.bodies.first().map(|b| b.ops.len()).unwrap_or(0),
                                        launches,
                                        rolled.suffix.len(),
                                        rolled.per_layer.len(),
                                    );
                                    // STAGE 2: bake ALL THREE small bundles (prefix=embed,
                                    // body=ONE layer, suffix=lm_head) + drop `reroll_meta.json`
                                    // in the body dir (3 fps + iters + per-layer seg strides) so
                                    // the worker runs prefix → body×iters (seg-base advance by
                                    // v·stride) → suffix. The sentinel stays `SUPERDSC_BUNDLE:{body_fp}`;
                                    // the worker detects rolled mode by the presence of reroll_meta.json.
                                    // ⭐⭐⭐ WHICH OPS THE ONE PATH GETS IS A PROPERTY OF THE DEVICE.
                                    //
                                    // ⛔ AND THE UNROLL'S "COSTS ALMOST NOTHING" ARGUMENT IS TRUE FOR
                                    // THE EMULATOR AND FALSE FOR THE CARD. A KTIR launch binds a
                                    // TENSOR, and KTIR programs are INTERNED, so `iters` copies of a
                                    // body really are `iters` arg-lists over one shared program —
                                    // free. Nothing is interned at DESCRIPTOR level: on `-Fspyre-hw`
                                    // every unrolled layer's op becomes its own `SdscOp` and its own
                                    // dxp group compile. MEASURED on granite-3.1-2b: the unrolled form
                                    // is 690 KTIR programs → 8744 descriptors / 161 groups, where this
                                    // same site's ROLLED sibling bake (`&b.ops`, the fused twin
                                    // below) is 137-164 descriptors / 3 groups. That 40x is the whole
                                    // bake-time story, and it ends in `dxp refused …: LLVM ERROR:
                                    // pthread_create failed`.
                                    //
                                    // ⭐ SO THE CARD BAKES THE ROLLED BODY and its executor loops it —
                                    // `superdsc_exec::load_rolled` advances the resident segment base
                                    // by `v·stride` per iteration, which is exactly what
                                    // `RerollMeta`'s strides are for and why they are real HERE even
                                    // though a KTIR launch has no segment base of its own: on this
                                    // path addresses come from `BundleLayout.places`, not from
                                    // `LaunchProgram::args` (which `ktir_groups_via_superdsc` leaves
                                    // empty for exactly that reason).
                                    //
                                    // ⛔ THIS IS NOT A SECOND PATH. Both devices still go
                                    // `SubtileIR → KTIR → SuperDSC → this bake`; only the op LIST
                                    // differs, the same way `ktir_optimizer::matmul_tile` differs by
                                    // device without forking the lowering. It is the same correction
                                    // the matmul pre-tiling needed: an emulator-motivated transform
                                    // applied upstream of both consumers, moved to the consumer that
                                    // wants it.
                                    let unrolled = superdsc::unroll_layers::<$W>(&rolled);
                                    let card = cfg!(feature = "spyre-lower");
                                    let unrolled_params = rolled.attn_params;
                                    let pre = if card {
                                        superdsc::emit_bundle::<$W>(
                                            &rolled.prefix,
                                            Some(&rolled.layout),
                                            superdsc::FoldGrouping::Split,
                                            rolled.attn_params,
                                            dump,
                                        )
                                    } else {
                                        Ok(String::new())
                                    };
                                    // ⭐ ONE BODY, PLAYED `iters` TIMES. The old ladder baked one
                                    // bundle per attention class; this ladder's rolled contract is
                                    // the single `body` (the welded cell for a hybrid — see the roll
                                    // call above), and every launch row below replays it with the
                                    // per-layer segment shifts.
                                    // ⛔ A FAILED BODY IS A BUILD ERROR, NOT A SKIP — EXCEPT
                                    // ON THE BATCH LADDER, WHERE THE RUNG CONTRACT IS THE SKIP. An
                                    // empty fp here rides into `RerollMeta::bodies` as a sibling
                                    // nobody emitted: the worker's body selector finds no bundle and
                                    // the first signal is a launch-time refusal with none of this
                                    // context. This is the site the oversized-dxp-group guard
                                    // (`DxGroupCeiling`) reaches — log-and-skip here is exactly the
                                    // swallow that would have turned the guard into an eprintln, so
                                    // an emit failure is FATAL for every rung alike: primary body,
                                    // decode batch rung, prefill rung.
                                    // ONE BUNDLE PER CLASS, class 0 first — class 0 is the ANCHOR
                                    // (`RerollMeta::bodies[0]` names THIS bundle), and every other
                                    // class becomes a sibling fingerprint. The emulator's single
                                    // unrolled bundle stays ONE bundle: `unroll_layers` already
                                    // walks the launch sequence across classes.
                                    let mut body_fps: Vec<String> = Vec::new();
                                    for (ci, b) in rolled.bodies.iter().enumerate() {
                                        let r = superdsc::emit_bundle::<$W>(
                                            if card { &b.ops } else { &unrolled },
                                            Some(&rolled.layout),
                                            superdsc::FoldGrouping::Split,
                                            if card {
                                                rolled.attn_params
                                            } else {
                                                unrolled_params
                                            },
                                            dump,
                                        );
                                        match r {
                                            Ok(fp) => body_fps.push(fp),
                                            Err(e) => panic!(
                                                "[spyre-superdsc] {base}: body class {ci} emit \
                                                 failed at decode_rows={decode_rows} \
                                                 prefill_mq={prefill_mq:?} — {e}. A rung the bake \
                                                 was asked for must ship its bundle; a skip here \
                                                 ships a binary whose batch silently runs elsewhere"
                                            ),
                                        }
                                    }
                                    let bod: Result<String, std::io::Error> =
                                        Ok(body_fps.first().cloned().unwrap_or_default());
                                    // The SAME body, with the per-page fold fused back into the
                                    // surrounding work instead of standing alone. Splitting the fold
                                    // costs one extra launch per layer (measured: 5 groups vs the
                                    // baseline's 4 = 40 more launches per token on a launch-bound path),
                                    // and that is only worth paying when the fold is actually
                                    // re-launched. The runtime picks this one whenever the context fits
                                    // a single page, which is the case that has to stay at the
                                    // baseline's cost.
                                    // ⭐ THE TWIN IS A LAUNCH-*GROUPING* VARIANT, SO IT IS THE CARD'S.
                                    // Off-card `ktir_groups` takes `_fold` and never reads it — one
                                    // launch group per op, always — so an emulator twin is a
                                    // byte-identical copy of its split under a different name, and its
                                    // only reader (`superdsc_exec::select_body_paged`) is
                                    // `spyre-hw`-gated. Same rule as the prefix/suffix below: bake it
                                    // where it is read.
                                    // One fold-fused twin PER CLASS, same order as the bodies. An
                                    // emit failure is logged and left `none()` — silent at runtime
                                    // only in the sense the selector documented: it falls through
                                    // to the split body, one launch per layer dearer.
                                    // ⛔ AND ONLY WHERE THERE IS A FOLD TO FUSE
                                    // (`superdsc::has_fused_twin`). A body with none — every paged
                                    // step's, whose walk is never fused away — has a "twin" that is
                                    // its split program again under a second name: a second dxp
                                    // compile and a second copy in the card image, for every rung.
                                    let bodies_fused: Vec<superdsc::bundle::SiblingFp<'static>> =
                                        if card {
                                            rolled
                                                .bodies
                                                .iter()
                                                .map(|b| {
                                                    if !superdsc::has_fused_twin::<$W>(&b.ops) {
                                                        return superdsc::bundle::SiblingFp::none();
                                                    }
                                                    superdsc::emit_bundle::<$W>(
                                                        &b.ops,
                                                        Some(&rolled.layout),
                                                        superdsc::FoldGrouping::Fused,
                                                        rolled.attn_params,
                                                        dump,
                                                    )
                                                    .map(superdsc::bundle::SiblingFp::from)
                                                    .unwrap_or_else(|e| {
                                                        panic!(
                                                            "[spyre-superdsc] {base}: fold-fused \
                                                             body twin does not emit: {e}"
                                                        )
                                                    })
                                                })
                                                .collect()
                                        } else {
                                            vec![
                                                superdsc::bundle::SiblingFp::none();
                                                rolled.bodies.len()
                                            ]
                                        };
                                    // ⭐ THE SUFFIX IS ITS OWN BUNDLE ON THE CARD. Rolling the body
                                    // means the lm-head tail can no longer ride inside it: the body is
                                    // ONE layer now, run `iters` times. The emulator keeps the single
                                    // unrolled bundle, where the suffix really is part of it.
                                    let suf = if card {
                                        superdsc::emit_bundle::<$W>(
                                            &rolled.suffix,
                                            Some(&rolled.layout),
                                            superdsc::FoldGrouping::Split,
                                            rolled.attn_params,
                                            dump,
                                        )
                                    } else {
                                        Ok(String::new())
                                    };
                                    match (pre, bod, suf) {
                                        (Ok(pfp), Ok(bfp), Ok(sfp)) => {
                                            // ── sk_bucket LADDER (decode only) ── The full-cap body just
                                            // baked (`bfp`) is the CEILING rung. Additionally bake the decode
                                            // BODY at each INTERIOR rung (active_cap < cap): same prefix/
                                            // suffix/layout, only the attention sweep extent shrinks, so all
                                            // rungs ride the SAME resident weights+KV. `decode_rungs` =
                                            // [(active_cap, body_fp)] ascending, top = (cap, bfp); the worker
                                            // loads all N bodies into ONE session + picks the smallest rung ≥
                                            // the live KV length each step. Only the m=1 DECODE reroll ladders
                                            // (prefill sweeps its whole chunk); a rung that fails to bake is
                                            // logged + skipped (the runtime falls back to the next rung up).
                                            let cap = prefix_len.get();
                                            // ⭐ EACH RUNG NAMES ITS OWN FUSED TWINS, rather than the
                                            // runtime reconstructing names by appending "f" — a
                                            // convention whose miss is silently "this rung has no fused
                                            // twin", and one extra launch per layer.
                                            let sib = |fp: Option<String>| match fp {
                                                Some(fp) => superdsc::bundle::SiblingFp::from(fp),
                                                None => superdsc::bundle::SiblingFp::none(),
                                            };
                                            // ⭐ ONE BODY FINGERPRINT per rung — the same list the
                                            // ceiling rung's are the bodies just baked; the
                                            // primary fp is the anchor `bfp`.
                                            let rung_fps = |fps: &[String]| {
                                                std::borrow::Cow::Owned(
                                                    fps.iter()
                                                        .map(|fp| sib(Some(fp.clone())))
                                                        .collect::<Vec<_>>(),
                                                )
                                            };
                                            // ⭐ THE PAGE WALK'S RUNGS — spyre-inference's
                                            // `num_blocks` buckets over this bundle's pool
                                            // (`scratchy_target_spyre::paged_step::walk_buckets`). The body just
                                            // baked walks `ceiling_walk` pages — the whole pool for a
                                            // decode bake, the chunk's own pages for a prompt chunk
                                            // baked with no resident prefix — and every other bucket
                                            // is a rung, so each step walks the fewest pages that
                                            // reach its context.
                                            let pool_pages =
                                                cap / scratchy_target_spyre::paged_step::SchedulerBlockTokens::REFERENCE.get();
                                            let ceiling_walk =
                                                scratchy_target_spyre::paged_step::walk_pages(
                                                    pool_pages,
                                                    prefill_mq.unwrap_or(decode_rows),
                                                    active_cap,
                                                );
                                            let mut decode_rungs: Vec<
                                                superdsc::bundle::LadderRung<'static>,
                                            > = vec![superdsc::bundle::LadderRung {
                                                // A chunk body sweeps its whole chunk, so its rung
                                                // is the chunk's slots PER REQUEST — what
                                                // `select_body` compares the live context against.
                                                // A page walk's rung is the slots its walk reaches.
                                                active_cap: superdsc::bundle::SweptCols::new(
                                                    ceiling_walk * scratchy_target_spyre::paged_step::SchedulerBlockTokens::REFERENCE.get(),
                                                ),
                                                bodies: rung_fps(&body_fps),
                                                bodies_fused: std::borrow::Cow::Owned(
                                                    bodies_fused.clone(),
                                                ),
                                            }];
                                            // ⛔ WHAT EACH BAKED RUNG SERVES PAST ITS OWN SWEEP, beside
                                            // `decode_rungs` and pushed exactly where it is — read off
                                            // the EMITTED bundle's launch groups, the `page_fold` the
                                            // runtime loads (the program it was emitted from carries no
                                            // fold flag: the attention expands inside `emit_bundle`).
                                            // Proven below.
                                            let folds_of = |fp: &str| {
                                                superdsc::emitted_bundle_folds::<$W>(fp).unwrap_or_else(|| {
                                                    panic!(
                                                        "[spyre-superdsc] {base}: ladder body {fp} was named but never emitted"
                                                    )
                                                })
                                            };
                                            let mut rung_reach: Vec<(
                                                superdsc::bundle::SweptCols,
                                                scratchy_target_spyre::fold_plan::RungReach,
                                            )> = vec![(
                                                decode_rungs[0].active_cap,
                                                scratchy_target_spyre::fold_plan::RungReach {
                                                    // EVERY class body: a rung serves any length only
                                                    // if each of its classes' attentions folds.
                                                    folds: body_fps.iter().all(|fp| folds_of(fp)),
                                                    fused_twin: decode_rungs[0]
                                                        .bodies_fused
                                                        .iter()
                                                        .any(|f| !f.is_none()),
                                                },
                                            )];
                                            // The walk rungs: every bucket below a whole-pool walk — a
                                            // decode body's, and a prefix-capable prompt chunk's on the
                                            // card, which runs every chunk past a prompt's first — and,
                                            // off the card, where a prompt chunk past its first page
                                            // runs on the prefill ladder, every bucket above a
                                            // prefix-free chunk's own pages.
                                            let prefix_free_chunk = is_prefill
                                                && active_cap
                                                    == scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::NONE;
                                            let sweep = scratchy_target_spyre::paged_step::walk_buckets(pool_pages)
                                                .into_iter()
                                                .filter(|&b| {
                                                    if prefix_free_chunk {
                                                        !card && b > ceiling_walk
                                                    } else {
                                                        b < ceiling_walk
                                                    }
                                                })
                                                .map(|b| {
                                                    let slots = b * scratchy_target_spyre::paged_step::SchedulerBlockTokens::REFERENCE.get();
                                                    (
                                                        scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::new(slots),
                                                        slots,
                                                    )
                                                });
                                            let ladder: Vec<(
                                                scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap,
                                                u32,
                                            )> = sweep.collect();
                                            if !ladder.is_empty() {
                                                // ⭐ EVERY RUNG IS ITS OWN LOWERING of the same tape at
                                                // its own sweep/chunk, reading nothing another rung
                                                // writes — so they lower in parallel and are recorded in
                                                // ladder order (the vectors are sorted below either way).
                                                // The rung's own layout rides back with its record: the
                                                // one-layout check reads the ceiling's layout, which
                                                // holds interior cells and stays on this thread.
                                                type RungRecord = (
                                                    superdsc::bundle::LadderRung<'static>,
                                                    (
                                                        superdsc::bundle::SweptCols,
                                                        scratchy_target_spyre::fold_plan::RungReach,
                                                    ),
                                                    (
                                                        scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap,
                                                        superdsc::BundleLayout,
                                                    ),
                                                );
                                                let per_rung: Vec<Option<RungRecord>> = {
                                                    use rayon::prelude::*;
                                                    ladder
                                                        .into_par_iter()
                                                        .map(|(rung, rung_swept)| -> Option<RungRecord> {
                                                    match superdsc::lower_subtile_tape_to_superdsc::<$W, _>(
                                                        &tape_rolled,
                                                        &krg,
                                                        &weight_ids,
                                                        rung,
                                                        rows_are_requests,
                                                        query_rows,
                                                    ) {
                                                        // Same device split as the ceiling rung above: the
                                                        // card bakes the ROLLED body, the emulator the
                                                        // single unrolled one. Without this each of the
                                                        // three interior rungs pays the same 40x.
                                                        Ok(rr) => {
                                                            let rung_unrolled = if card {
                                                                Vec::new()
                                                            } else {
                                                                superdsc::unroll_layers::<$W>(&rr)
                                                            };
                                                            let rung_params = rr.attn_params;
                                                            // ONE FP per body class; a failed body emit
                                                            // leaves the rung unbaked — the runtime falls
                                                            // back to the next rung up.
                                                            let mut rfps: Vec<String> = Vec::new();
                                                            let mut rfused: Vec<
                                                                superdsc::bundle::SiblingFp<
                                                                    'static,
                                                                >,
                                                            > = Vec::new();
                                                            for (ci, b) in
                                                                rr.bodies.iter().enumerate()
                                                            {
                                                                match superdsc::emit_bundle::<$W>(
                                                                    if card {
                                                                        &b.ops
                                                                    } else {
                                                                        &rung_unrolled
                                                                    },
                                                                    Some(&rr.layout),
                                                                    superdsc::FoldGrouping::Split,
                                                                    // THIS rung's own params — `rr` was
                                                                    // lowered at `rung`, so its swept
                                                                    // extent is `rung`'s and not the
                                                                    // ceiling bundle's.
                                                                    if card {
                                                                        rr.attn_params
                                                                    } else {
                                                                        rung_params
                                                                    },
                                                                    dump,
                                                                ) {
                                                                    Ok(rfp) => {
                                                                        rfps.push(rfp);
                                                                        // The rung's fold-fused
                                                                        // twin too: the rung bounds
                                                                        // the sweep, this bounds the
                                                                        // launch count, and a short
                                                                        // context needs BOTH. ON THE
                                                                        // CARD, for the ceiling twin's
                                                                        // reason above — and here it
                                                                        // also keeps the twin's NAME
                                                                        // lawful. `bundle_fp` says a
                                                                        // twin is `<split fp>f`, so it
                                                                        // must hash the ops the split
                                                                        // hashed; off-card the split
                                                                        // is `&rung_unrolled` while
                                                                        // this is `&b.ops`, so the
                                                                        // "twin" was named after a
                                                                        // bundle nothing else emitted. And
                                                                        // only where there is a fold to fuse,
                                                                        // as for the ceiling's.
                                                                        rfused.push(if card
                                                                            && superdsc::has_fused_twin::<$W>(&b.ops)
                                                                        {
                                                                        superdsc::emit_bundle::<$W>(
                                                                            &b.ops,
                                                                            Some(&rr.layout),
                                                                            superdsc::FoldGrouping::Fused,
                                                                            rr.attn_params,
                                                                            dump,
                                                                        )
                                                                        .map(
                                                                            superdsc::bundle::SiblingFp::from,
                                                                        )
                                                                        .unwrap_or_else(|e| {
                                                                            panic!(
                                                                                "[spyre-superdsc] {base}: rung {} \
                                                                                 fused twin does not emit: {e}",
                                                                                rung.get()
                                                                            )
                                                                        })
                                                                    } else {
                                                                        superdsc::bundle::SiblingFp::none()
                                                                    });
                                                                    }
                                                                    Err(e) => panic!(
                                                                        "[spyre-superdsc] {base}: ladder rung {} body \
                                                                         class {ci} does not emit: {e}",
                                                                        rung.get()
                                                                    ),
                                                                }
                                                            }
                                                            eprintln!(
                                                                "[spyre-superdsc] {base}: ladder rung \
                                                             active_cap={} baked",
                                                                rung.get()
                                                            );
                                                            let rung_folds = rfps
                                                                .iter()
                                                                .all(|fp| folds_of(fp));
                                                            let rec = superdsc::bundle::LadderRung {
                                                                active_cap: superdsc::bundle::SweptCols::new(rung_swept),
                                                                bodies: rung_fps(&rfps),
                                                                bodies_fused: std::borrow::Cow::Owned(rfused),
                                                            };
                                                            // Read off the rung record itself: the
                                                            // record the runtime loads — any class's
                                                            // twin counts.
                                                            let fused_twin =
                                                                rec.bodies_fused.iter().any(|f| !f.is_none());
                                                            Some((
                                                                rec,
                                                                (
                                                                    superdsc::bundle::SweptCols::new(rung_swept),
                                                                    scratchy_target_spyre::fold_plan::RungReach {
                                                                        folds: rung_folds,
                                                                        fused_twin,
                                                                    },
                                                                ),
                                                                (rung, rr.layout),
                                                            ))
                                                        }
                                                        // ⛔ A RUNG THE LADDER ASKED FOR MUST LOWER (#1023). Dropping it
                                                        // served its positions on the next rung up with no sign
                                                        // anywhere (a proc macro's stderr is swallowed on success),
                                                        // which hid a broken one-chunk lowering for a whole bisect.
                                                        // A ladder that wants fewer rungs states so where it is built.
                                                        Err(e) => panic!(
                                                            "[spyre-superdsc] {base}: ladder rung {} does not lower: {e}",
                                                            rung.get()
                                                        ),
                                                    }
                                                        })
                                                        .collect()
                                                };
                                                for (rec, reach, (rung, layout)) in
                                                    per_rung.into_iter().flatten()
                                                {
                                                    // ⛔ ONE SESSION STAGES ONE LAYOUT: a rung reading a
                                                    // staged input from elsewhere is a build failure,
                                                    // never a wrong constant.
                                                    if let Err(e) =
                                                        superdsc::ladder_stages_one_layout(
                                                            &rolled.layout,
                                                            &layout,
                                                        )
                                                    {
                                                        panic!(
                                                            "[spyre-superdsc] {base}: ladder rung active_cap={}: {e}",
                                                            rung.get()
                                                        );
                                                    }
                                                    // ⭐ THE PARITY LEDGER'S ENTRY FOR THIS RUNG —
                                                    // one entry per baked bundle, labelled by its
                                                    // sweep, so a rung that packs its weight segment
                                                    // apart from the ceiling's is a build error here,
                                                    // not an `alias_seg` refusal on the card.
                                                    {
                                                        let (bank0, banks) =
                                                            superdsc::weight_seg_record(&layout);
                                                        fx.push(BakeEffect::WeightSeg {
                                                            base: format!(
                                                                "{base} (rung cap {})",
                                                                rung.get()
                                                            ),
                                                            bank0,
                                                            banks,
                                                        });
                                                    }
                                                    decode_rungs.push(rec);
                                                    rung_reach.push(reach);
                                                }
                                            }
                                            decode_rungs.sort_by_key(|r| r.active_cap);
                                            rung_reach.sort_by_key(|(cap, _)| *cap);
                                            // The rung a context past every sweep runs — the
                                            // runtime reads the index baked here
                                            // (`RerollMeta::fallback_rung`).
                                            let reach: Vec<
                                                scratchy_target_spyre::fold_plan::RungReach,
                                            > = rung_reach.iter().map(|(_, r)| *r).collect();
                                            let fallback_rung =
                                                scratchy_target_spyre::fold_plan::fallback_rung(
                                                    &reach,
                                                )
                                                    as u32;
                                            // ⭐ THE LAYER LOOP, ATTACHED TO THE BODY IT DESCRIBES.
                                            // Every fingerprint in it is a sibling the registry
                                            // resolves at load, and `RerollMeta::siblings` walks the
                                            // fields themselves — so there is no second list of
                                            // "which fields hold a fingerprint" to fall out of step.
                                            //
                                            // ⛔⛔⛔ THE CARD PLAYS THE LAUNCH TABLE; THE EMULATOR
                                            // PLAYS ONE UNROLLED BUNDLE. `bod` above is `&unrolled`
                                            // when `!card`, so the emulator's single bundle really is
                                            // the whole program and its launch table is ONE row —
                                            // which is the truth about that bundle, not a stub (the
                                            // same fact the old `iters: 1` carried, as the ONE row
                                            // the old `for v in 0..1` ran). The CARD's bodies are
                                            // ONE layer each, played `launches.len()` times with the
                                            // per-segment shifts each row names.
                                            //
                                            // MEASURED, granite-3.1-2b, the defect this guards
                                            // against repeating: the runtime reported `iters=1
                                            // wstride=0 kvstride=0` where main reported `iters=40
                                            // wstride=121643008 kvstride=786432`, and the card ran
                                            // ONE of forty layers per token — 3.1 ms ITL against
                                            // main's 42 ms — answering with a single repeated token,
                                            // while the log beside it printed ×40. The launch table
                                            // is now ONE artifact both the bake and the runtime
                                            // read, so the two cannot disagree about the count.
                                            let baked_launches: Vec<superdsc::bundle::LayerLaunch> =
                                                if card {
                                                    // ⭐ ZIPPED FROM THE WALK'S OWN TABLES —
                                                    // `launches` (the true layer order), `launch_w`
                                                    // and `launch_kv` (the per-launch segment
                                                    // shifts, read off the placements and PROVEN
                                                    // against them in the walk) — so the table the
                                                    // runtime plays cannot disagree with the
                                                    // addresses the bake placed.
                                                    rolled
                                                        .launches
                                                        .iter()
                                                        .zip(&rolled.launch_w)
                                                        .zip(&rolled.launch_kv)
                                                        .map(|((&(b, it), &(wb, wo)), &ko)| {
                                                            superdsc::bundle::LayerLaunch {
                                                                body: b,
                                                                iter: it,
                                                                w_bank: wb,
                                                                w_off: wo,
                                                                kv_off: ko,
                                                            }
                                                        })
                                                        .collect()
                                                } else {
                                                    vec![superdsc::bundle::LayerLaunch {
                                                        body: 0,
                                                        iter: 0,
                                                        w_bank: 0,
                                                        w_off: 0,
                                                        kv_off: 0,
                                                    }]
                                                };
                                            let meta = superdsc::bundle::RerollMeta {
                                                prefix: sib(Some(pfp.clone())),
                                                suffix: sib(Some(sfp.clone())),
                                                // ⭐ THE ONE BODY = the ANCHOR (`bfp`, the
                                                // bundle this meta attaches to). The
                                                // emulator's unrolled bundle has no siblings —
                                                // one row, one body, run once.
                                                bodies: std::borrow::Cow::Owned(
                                                    (if card {
                                                        body_fps.clone()
                                                    } else {
                                                        vec![bfp.clone()]
                                                    })
                                                    .iter()
                                                    .map(|fp| sib(Some(fp.clone())))
                                                    .collect(),
                                                ),
                                                bodies_fused: std::borrow::Cow::Owned(if card {
                                                    bodies_fused.clone()
                                                } else {
                                                    vec![superdsc::bundle::SiblingFp::none()]
                                                }),
                                                // ⛔ `rungs` IS KEYED BY ACTIVE_CAP — the columns one
                                                // fold pass sweeps — NOT by the batch width. Two other
                                                // lists of `(u32, fingerprint)` in this file are keyed
                                                // by `seqs`, and reading the wrong one selects a body
                                                // baked for a 64-column sweep because four requests are
                                                // live.
                                                rungs: std::borrow::Cow::Owned(decode_rungs),
                                                fallback_rung,
                                                launches: std::borrow::Cow::Owned(baked_launches),
                                                kv_page_stride: if card {
                                                    // One pool page = EVERY layer of EVERY
                                                    // class — the walk's own max-end over the
                                                    // per-class last-layer KV placements (0
                                                    // when the bundle has no paged KV).
                                                    rolled.kv_page_stride
                                                } else {
                                                    0
                                                },
                                                prefix_weight_bank: if card {
                                                    rolled.prefix_weight_bank
                                                } else {
                                                    0
                                                },
                                                suffix_weight_bank: if card {
                                                    rolled.suffix_weight_bank
                                                } else {
                                                    0
                                                },
                                            };
                                            // ⛔⛔ AND EVERY POSITION THE CARD SERVES ON IT STAGES A WALK
                                            // — proven over the ladder the executor reads and the
                                            // selection it runs (`paged_step::prove_ladder_walks`),
                                            // so a ladder read as no walk fails here, not on a card.
                                            if card
                                                && let Some(step) = gk.paged
                                                && let Err(e) =
                                                    scratchy_target_spyre::paged_step::prove_ladder_walks(
                                                        &meta, step,
                                                    )
                                            {
                                                panic!("[spyre-superdsc] {base}: {e}");
                                            }
                                            superdsc::attach_reroll::<$W>(&bfp, meta);
                                            // ⛔ AND THE SESSION THIS BODY OPENS ALLOCATES FOR EVERY
                                            // SIBLING IT LAUNCHES, not for itself alone: a rung whose
                                            // scratch outgrows the ceiling's addresses past its segment.
                                            if let Err(e) = superdsc::cover_ladder_allocation::<$W>(&bfp)
                                            {
                                                panic!("[spyre-superdsc] {base}: {e}");
                                            }
                                            // The prefill probe's rotary classes, from the SAME
                                            // bindings the wiring bakes its classes from — one
                                            // value law (`superdsc_rotary_kind`), two consumers
                                            // (the wiring's baked tokens, the gate's staged rows).
                                            // Carries the rung's mq so the gate's PrefillGate and
                                            // the DBG golden below share one construction.
                                            // (mq, cos rows, sin rows) — each row list naming
                                            // (source row, rotary kind).
                                            type PrefillRotaryClasses = (
                                                u32,
                                                Vec<(
                                                    u32,
                                                    scratchy_target_spyre::wiring::RotaryKind,
                                                )>,
                                                Vec<(
                                                    u32,
                                                    scratchy_target_spyre::wiring::RotaryKind,
                                                )>,
                                            );
                                            let prefill_classes: Option<PrefillRotaryClasses> =
                                                prefill_mq.map(|mq| {
                                                let hd = superdsc_model_head_dim(model);
                                                let theta = model.text.rope_theta.get();
                                                let mut rotary_cos = Vec::new();
                                                let mut rotary_sin = Vec::new();
                                                for (i, b) in lwd.bindings().iter().enumerate() {
                                                    let (want_cos, local) = match b {
                                                        scratchy_forward_compiler_macro::to_wavefront::SourceBinding::Cos { local } => {
                                                            (true, *local)
                                                        }
                                                        scratchy_forward_compiler_macro::to_wavefront::SourceBinding::Sin { local } => {
                                                            (false, *local)
                                                        }
                                                        _ => continue,
                                                    };
                                                    let kind =
                                                        superdsc_rotary_kind(model, hd, theta, local);
                                                    let pair = (i as u32, kind);
                                                    if want_cos {
                                                        rotary_cos.push(pair)
                                                    } else {
                                                        rotary_sin.push(pair)
                                                    }
                                                }
                                                (mq, rotary_cos, rotary_sin)
                                            });
                                            // ── NUMERIC BISECTION oracle (opt-in SCRATCHY_SUPERDSC_DBG):
                                            //    eval_dag golden + synthetic source shapes so
                                            //    SPYRE_SUPERDSC_SELFTEST localizes the first divergent op.
                                            //    `dbg_golden_dir` is the ONE statement of where it lives,
                                            //    called by both the writer here and the reader there. ──
                                            if let Some(dbg) =
                                                superdsc::bundle::dbg_golden_dir(&bfp)
                                            {
                                                // The DBG golden stages the SAME rope rows the
                                                // gate does (`staged_rope_rows` — one law), so
                                                // the bisection oracle cannot drift from the
                                                // gate's own golden. Decode: an empty map, the
                                                // period-half contract.
                                                let staged_rope = prefill_classes
                                                    .as_ref()
                                                    .map(|(_, cs, sn)| {
                                                        superdsc::staged_rope_rows(&krg, cs, sn)
                                                            .unwrap_or_default()
                                                    })
                                                    .unwrap_or_default();
                                                match superdsc::write_eval_golden(
                                                    &krg,
                                                    &weight_ids,
                                                    &dbg,
                                                    &staged_rope,
                                                ) {
                                                    Ok(()) => eprintln!(
                                                        "[spyre-superdsc] {base}: DBG golden written → {}/dbg",
                                                        dbg.display()
                                                    ),
                                                    Err(e) => eprintln!(
                                                        "[spyre-superdsc] {base}: DBG golden write failed: {e}"
                                                    ),
                                                }
                                            }
                                            // ⭐⭐⭐ THE NUMERIC GATE — "if there is garbage, it
                                            //    cannot compile." Every DECODE bundle AND every
                                            //    PREFIX-FREE PREFILL RUNG (ActiveCap::NONE — the
                                            //    first-chunk programs) this bake produced runs on
                                            //    the emulator and is compared against the
                                            //    reference semantics of the same dsl_math
                                            //    (`eval_golden_buffers`), the prefill arm staging
                                            //    the runtime's OWN first-chunk law (the [mq, mq]
                                            //    clamped triangle, per-position rope rows, the
                                            //    dead-prefix view); divergence beyond the quant
                                            //    floor is a PANIC — the build refuses to ship
                                            //    the bundle. No env var: under `-Fspyre-emu`
                                            //    the gate IS the bake's contract. A PREFIX-CAPABLE
                                            //    prefill bundle (ActiveCap::FULL — the
                                            //    continuation chunks) still skips loudly: its
                                            //    resident-prefix mask law is the next rung.
                                            //    ⭐ THE CLASS-COVERING TWIN: the gate judges the
                                            //    smallest prefix of launches that covers every
                                            //    body class AND every class transition (floor 2)
                                            //    — the reroll's classes are the programs, so a
                                            //    covered prefix exercises every program and every
                                            //    boundary the bundle ships; the tail is rebound
                                            //    to the kept prefix's last hidden-out and the
                                            //    golden/walk/staging all filter on the twin's
                                            //    bound set. A graph the law cannot truncate
                                            //    (short bundles, or a transition the prefix
                                            //    cannot cover) gates its OWN full depth — the
                                            //    identity rebind is the built-in equivalence
                                            //    test. ──
                                            #[cfg(feature = "spyre-emu")]
                                            {
                                                let prefill_gate = prefill_classes.map(
                                                    |(mq, rotary_cos, rotary_sin)| {
                                                        superdsc::PrefillGate {
                                                            mq,
                                                            rotary_cos,
                                                            rotary_sin,
                                                        }
                                                    },
                                                );
                                                // The rung under test, one fact — see
                                                // `superdsc::GateRung`.
                                                let gate_rung = superdsc::GateRung {
                                                    fp: &bfp,
                                                    cap,
                                                    active_cap,
                                                    query_rows,
                                                    shards,
                                                };
                                                // ⭐ THE SHIPPED BUNDLE'S OWN PIN WINDOW — read
                                                // from its emitted layout, so the twin's minted
                                                // programs carry the SAME `lx_window_fits` budget
                                                // the bake proved (auditor ruling 2026-10-27). A
                                                // missing entry or an unstated window here is a
                                                // defect the gate names, not a zero it charges.
                                                let lx_pin_window = superdsc::emitted_lx_pin_window_bytes::<$W>(&bfp).unwrap_or_else(|| {
                                                    panic!(
                                                        "[spyre-superdsc] {base}: bundle {bfp} has \
                                                         no emitted layout stating its LX pin window \
                                                         — the twin cannot mint programs without the \
                                                         budget the bake proved"
                                                    )
                                                });
                                                let run_gate =
                                                    |prefill: Option<&superdsc::PrefillGate>| {
                                                        // The twin's own groups, fresh from the
                                                        // truncated unroll — the same interned
                                                        // programs the bundle's first K launches
                                                        // bind, so the emulator session runs
                                                        // exactly the classes under test.
                                                        let twin = superdsc::covering_twin_groups::<$W>(
                                                            &rolled,
                                                            Some(lx_pin_window),
                                                        )
                                                        .unwrap_or_else(|e| {
                                                            panic!(
                                                                "[spyre-superdsc] {base}: \
                                                                         covering twin for {bfp} \
                                                                         failed to lower — {e}"
                                                            )
                                                        });
                                                        match twin {
                                                            Some((twin_groups, trunc)) => {
                                                                match superdsc::numeric_gate::<$W, _>(
                                                                    &twin_groups,
                                                                    &krg,
                                                                    &weight_ids,
                                                                    &gate_rung,
                                                                    prefill,
                                                                    Some(&trunc),
                                                                ) {
                                                                    Ok(()) => {}
                                                                    Err(e) => panic!(
                                                                        "[spyre-superdsc] {base}: \
                                                                         NUMERIC GATE REFUSED for \
                                                                         bundle {bfp} — {e}"
                                                                    ),
                                                                }
                                                            }
                                                            None => {
                                                                match superdsc::with_emitted_groups::<$W, _>(
                                                                    &bfp,
                                                                    |groups| {
                                                                        superdsc::numeric_gate::<$W, _>(
                                                                            groups,
                                                                            &krg,
                                                                            &weight_ids,
                                                                            &gate_rung,
                                                                            prefill,
                                                                            None,
                                                                        )
                                                                    },
                                                                ) {
                                                                    Some(Ok(())) => {}
                                                                    Some(Err(e)) => panic!(
                                                                        "[spyre-superdsc] {base}: \
                                                                         NUMERIC GATE REFUSED for \
                                                                         bundle {bfp} — {e}"
                                                                    ),
                                                                    None => eprintln!(
                                                                        "[spyre-superdsc] {base}: NUMERIC \
                                                                         GATE could not borrow {bfp}'s \
                                                                         launch groups (not Ready) — \
                                                                         UNCHECKED"
                                                                    ),
                                                                }
                                                            }
                                                        }
                                                    };
                                                match prefill_mq {
                                                    None => run_gate(None),
                                                    Some(_)
                                                        if active_cap
                                                            == superdsc::ActiveCap::NONE =>
                                                    {
                                                        run_gate(prefill_gate.as_ref())
                                                    }
                                                    Some(mq) => eprintln!(
                                                        "[spyre-superdsc] {base}: NUMERIC GATE \
                                                         SKIP prefill bundle {bfp} (mq={mq}, \
                                                         prefix-capable) — the continuation-chunk \
                                                         mask law (resident prefix ∘ [mq, mq] \
                                                         triangle) is the next rung; UNCHECKED"
                                                    ),
                                                }
                                            }
                                            eprintln!(
                                                "[spyre-superdsc] {base}: 3 bundles baked — prefix {pfp} / body {bfp} + {} class(es), {} launch(es) \
                                                 / suffix {sfp}; page_stride={} B (dxp: SMALL bundles → seconds)",
                                                rolled.bodies.len(),
                                                rolled.launches.len(),
                                                rolled.kv_page_stride,
                                            );
                                            // Stamp the DECODE body_fp or the PREFILL body_fp by phase
                                            // (the prefill reroll must NOT clobber the decode fp).
                                            if let Some(mq) = prefill_mq.filter(|_| active_cap == scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::FULL) {
                                            // A PREFIX-CAPABLE prefill rung: continuation chunks
                                            // (start>0) run on these, since they DO have resident KV to
                                            // attend — each on the smallest rung that holds it.
                                            fx.push(BakeEffect::PrefillPrefixRung { mq, fp: bfp });
                                            prefill_fp_stamped = true;
                                        } else if let Some(mq) = prefill_mq {
                                            // Ladder rung (deduped on mq: a rebake of the same width
                                            // must not double-enter). The TOP rung additionally lands in
                                            // `superdsc_prefill_fp` — the single-bundle prefill slot —
                                            // which the rung loop below arranges by baking it FIRST.
                                            fx.push(BakeEffect::PrefillRung {
                                                mq,
                                                fp: bfp.clone(),
                                            });
                                            // Rungs bake ASCENDING, so the LAST write is the widest
                                            // -- the ceiling the worker treats as its single prefill
                                            // bundle. Unconditional: non-superdsc bakes exactly one.
                                            fx.push(BakeEffect::PrefillFp(bfp));
                                            prefill_fp_stamped = true;
                                        } else {
                                            // ⛔ AN EMPTY fp IS A FAILED RUNG, NOT AN UNREGISTERED
                                            // ONE: the batch ladder's contract is that every rung
                                            // the bake was asked for SHIPS — an emit that failed
                                            // above already panicked, and an empty fp reaching
                                            // here means a bundle nobody emitted would otherwise be
                                            // silently absent from the ladder.
                                            if bfp.is_empty() {
                                                panic!(
                                                    "[spyre-superdsc] {base}: decode batch rung \
                                                     seqs={decode_rows} NOT BAKED (no body \
                                                     fingerprint) — every rung the bake was asked \
                                                     for must ship"
                                                );
                                            } else {
                                                // `cap` IS this body's swept extent: `bfp` is the
                                                // CEILING rung (the interior sweep rungs go to
                                                // `sk_bucket_rungs`), and `ActiveCap::FULL` resolves
                                                // to `cap`.
                                                fx.push(BakeEffect::DecodeRung {
                                                    rows: decode_rows,
                                                    cap,
                                                    fp: bfp.clone(),
                                                    // THIS rung's own kinds — the wiring `gk`
                                                    // derived from THIS bake's `krg`, whose
                                                    // mint ran at `decode_rows`. Not the
                                                    // single-request wiring's list: the ten
                                                    // uniform `[rows, 1]` tables stage at the
                                                    // rung's own rows.
                                                    consts: gk.load_consts.clone(),
                                                });
                                            }
                                            // The one-request bundle stays THE decode bundle for
                                            // every path that predates the ladder.
                                            //
                                            // ⛔ AND `decode_rows <= 1` DOES NOT MEAN THE GRAPH IS
                                            // ONE ROW. The CB emission (`ktir_decode_cb_*`) passes
                                            // `decode_rows = 1` on purpose — its rows are a batch
                                            // SYMBOL the DEM resolves at runtime — while the graph
                                            // it lowered is `CB_BATCH_TEMPLATE` = 96 rows wide. It
                                            // runs LAST, so it won this slot and every decode token
                                            // ran a 96-row program: MEASURED as `[96, 49155]` logits
                                            // and 195,428,640 elements read back per token, for a
                                            // ~90 K result. Correct (row 0 is the real token) and
                                            // ~96x the work. `seq_sym.is_none()` is the same
                                            // discriminator the superdsc block above uses to exclude
                                            // that symbolic tape.
                                            if decode_rows <= 1 && seq_sym.is_none() {
                                                fx.push(BakeEffect::DecodeFp(bfp));
                                            }
                                        }
                                        }
                                        // HARD BUILD ERROR (guard-every-crash-at-build-time): a failed bundle
                                        // write here left `superdsc_fp`/`superdsc_prefill_fp` unset, so the
                                        // placeholder-group synthesis at the decode-bundle call site
                                        // (`default_decode_groups.is_empty() && superdsc_fp.borrow().is_some()`)
                                        // never fires — the SengraphBundle ships with ZERO groups, and the
                                        // FIRST signal is the worker's runtime "bundle has no layer groups
                                        // (split produced 0)" at pod load, with none of this context. Panicking
                                        // here instead surfaces the REAL SuperDSC lowering failure at `cargo
                                        // build` time, where it belongs.
                                        (pre, bod, suf) => panic!(
                                            "[spyre-superdsc] {base}: 3-bundle write failed — \
                                         prefix_err={:?} body_err={:?} suffix_err={:?}",
                                            pre.err(),
                                            bod.err(),
                                            suf.err(),
                                        ),
                                    }
                                }
                                Err(e) => {
                                    // ⛔ NO RUNG MAY FAIL TO LOWER. The old "rung contract" — a
                                    // failed rung is skipped and the batch falls to a narrower one —
                                    // shipped binaries whose batch decode silently ran elsewhere
                                    // (measured: the sampler's grid defect skipped EVERY decode
                                    // batch rung and nothing said so). A rung the bake was asked
                                    // for is a build obligation: failure is a build failure, naming
                                    // the rung and the error.
                                    panic!(
                                        "[spyre-superdsc] {base}: RE-ROLL NOT LOWERABLE at \
                                         decode_rows={decode_rows} prefill_mq={prefill_mq:?} — {e}. \
                                         Every rung the bake was asked for must lower; a skip ships \
                                         a binary whose batch silently runs elsewhere"
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            panic!(
                                "[spyre-superdsc] {base}: SubtileIR invalid for reroll — {e:?} (previously \
                             swallowed by an eprintln, leaving superdsc_fp unset and shipping a ZERO-group \
                             bundle that only crashed at pod load.)"
                            )
                        }
                    }
                }
                // Sengraph lowering DELETED (dead end — SDSC-native only). The sdsc bundle rides the
                // reroll bake (`superdsc_fp`) above; this decode graph_json stays empty and is
                // overridden by the SUPERDSC_BUNDLE sentinel downstream. `group_graphs` stays empty
                // (a placeholder group is synthesized by the caller to carry the sentinel).
                let sengraph_json = String::new();
                // (the wiring `gk` was already derived above the reroll bake — the decode batch
                // rung's effect carries this bake's own load_consts, so the derivation precedes
                // the bake)
                // The census entry for this program: the fold set the wiring just recognized,
                // sorted by tid so two walks that found the same folds in a different node
                // order still compare equal.
                {
                    let mut folds = gk.weight_scale_folds.clone();
                    folds.sort_unstable_by_key(|(tid, _)| *tid);
                    fx.push(BakeEffect::FoldCensus {
                        base: base.clone(),
                        folds,
                    });
                }
                // ── THE LAUNCH FACTS, TYPED, FROM ONE WALK ─────────────────
                //
                // `gk.result_tensor`, `gk.tensor_shapes`, `gk.attn_mask` and — the
                // one that matters — `lwd.bindings()`, already the `SourceBinding`
                // ENUM, become values of spyre's OWN types (`wiring::Wiring`),
                // exactly as every target is handed statics of its runtime's types.
                // ONE walk states them: a second, stringified copy of the same
                // bindings (the old `manifest.json`) and the check that the two
                // agreed are gone with it — a fact stated once cannot disagree.
                {
                    // Tensor ids are PER PROGRAM: the prefill graph numbers its
                    // own. Keeping only one would bind decode ids into prefill
                    // launches, so both are kept and the worker selects.
                    //
                    // ⛔ AND THE DECODE SLOT TAKES THE CONCRETE, SINGLE-REQUEST GRAPH ONLY. There is
                    // one slot but several non-prefill emissions, so a plain `else` is last-write-
                    // wins — and the last one is `ktir_decode_cb_*`, lowered at `CB_BATCH_TEMPLATE`
                    // (96). The worker forwards `m_cap` ROWS and reads `m_cap` off this wiring's
                    // EmbeddedHidden shape (`spyre_load.rs:109`, `spyre_forward.rs:1377`), so that
                    // stamp made every single-token decode step compute 96 activation rows.
                    //
                    // Both guards are the discriminators this file already uses. `seq_sym.is_none()`
                    // is how the superdsc block at the top of this closure excludes "the emulator-only
                    // symbolic CB-decode tape (Some(("s0",..)))" — whose rows are a batch SYMBOL the
                    // DEM resolves at runtime, not a count anything may forward. `decode_rows ==
                    // primary_m` keeps a wider batch-ladder rung out: a rung is the same tape baked
                    // at B rows and is addressed by FINGERPRINT, not through this wiring.
                    // ⛔ AND A PREFILL WIRING SHIPS ONLY FOR A PREFILL THAT BAKED. The
                    // prefill walk's KTIR emission and wiring stamp survive their own
                    // reroll's failure (a failed reroll panics after the walk has already
                    // emitted), so `is_prefill` alone would promise a batched prefill no bundle
                    // carries — see the `prefill_fp_stamped` declaration above for the
                    // measured consequence. A rung that did not bake stamps NEITHER
                    // slot: falling through to the decode arm would ship the m=31
                    // prefill wiring as the DECODE wiring (both its guards hold for the
                    // prefill rung call — `seq_sym.is_none()` because superdsc keeps
                    // the prefill tape concrete, `decode_rows == primary_m` because the
                    // rung passes 1), overwriting the real decode wiring after the fact
                    // and poisoning every step's staging (MEASURED: V7b tiny8 — the
                    // same ten PROBE11 clobbers, now routed through the decode slot).
                    // Stamped per call, so a failed WIDER rung cannot overwrite a
                    // narrower rung's good wiring either: the last successful prefill
                    // call has the last word, which under the ascending bake order is
                    // the widest rung that actually lowered (`commit_bake` applies the
                    // calls' effects in that order however the rungs were baked).
                    let slot = if is_prefill {
                        prefill_fp_stamped.then_some(WiringSlot::Prefill)
                    } else if seq_sym.is_none() && decode_rows == primary_m {
                        Some(WiringSlot::Decode)
                    } else {
                        None
                    };
                    fx.push(BakeEffect::Registry {
                        prefill: is_prefill,
                        base: format!(
                            "{base} (prefill_mq={prefill_mq:?}, rows={decode_rows}, sweep={active_cap:?})"
                        ),
                        scales: gk.scalarmul_scales.clone(),
                    });
                    fx.push(BakeEffect::Wiring {
                        slot,
                        gk: Box::new(gk),
                        decode_position: cap.saturating_sub(1),
                    });
                }
                ((sengraph_json, group_graphs), fx)
            }
            };
        }
        let emit_bundle_deferred =
            world_bake!(scratchy_target_spyre::bundle_code::collective::TpWorld<1>);
        let emit_bundle_deferred_sharded =
            world_bake!(scratchy_target_spyre::bundle_code::collective::Build);
        // Apply one bake's effects to `world`'s state, in the order the bake recorded them. Serial,
        // and the only code that touches the expansion's shared state.
        let commit_bake = |world: &WorldBake<'_>, lwd: &BakeTape, fx: Vec<BakeEffect>| {
            for e in fx {
                match e {
                    // Deduped on mq like the prefix-free ladder: a rebake of a width must not double-enter.
                    BakeEffect::PrefillPrefixRung { mq, fp } => {
                        let mut r = world.prefill_prefix_rungs.borrow_mut();
                        if !r.iter().any(|(m, _)| *m == mq) {
                            r.push((mq, fp));
                        }
                    }
                    // Ladder rung (deduped on mq: a rebake of the same width must not double-enter).
                    BakeEffect::PrefillRung { mq, fp } => {
                        let mut r = world.prefill_rungs.borrow_mut();
                        if !r.iter().any(|(m, _)| *m == mq) {
                            r.push((mq, fp));
                        }
                    }
                    BakeEffect::PrefillFp(fp) => *world.prefill_fp.borrow_mut() = Some(fp),
                    BakeEffect::DecodeRung {
                        rows,
                        cap,
                        fp,
                        consts,
                    } => {
                        let mut r = world.decode_rungs.borrow_mut();
                        if !r.iter().any(|(n, _, _, _)| *n == rows) {
                            r.push((rows, cap, fp, consts));
                        }
                    }
                    BakeEffect::DecodeFp(fp) => *world.fp.borrow_mut() = Some(fp),
                    BakeEffect::FoldCensus { base, folds } => {
                        let mut census = world.fold_census.borrow_mut();
                        match &*census {
                            None => *census = Some((base, folds)),
                            Some((ref_label, ref_folds)) if *ref_folds == folds => {}
                            Some((ref_label, ref_folds)) => {
                                let show = |f: &[(u32, f32)]| {
                                    f.iter()
                                        .map(|(t, s)| format!("t{t}×{s}"))
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                };
                                panic!(
                                    "[spyre] {base}: its ScalarMul weight folds disagree with \
                                     {ref_label}'s — every program this model bakes runs over ONE set \
                                     of staged weight bytes, so a fold one program recognizes and \
                                     another runs as an op is a double-applied (or never-applied) \
                                     multiplier. {base} folds [{}]; {ref_label} folds [{}]. The \
                                     recognition is structural (shared weight, shared producer, \
                                     sliced read), so a disagreement means the two graphs' producer \
                                     structure differs around a folded ScalarMul — refuse rather \
                                     than serve silently-wrong logits.",
                                    show(&folds),
                                    show(ref_folds),
                                );
                            }
                        }
                    }
                    // The wiring is built HERE, on the expansion's own thread: it is a token stream,
                    // and tokens do not cross threads.
                    BakeEffect::Wiring {
                        slot,
                        gk,
                        decode_position,
                    } => {
                        let w = emit_superdsc_wiring(&gk, lwd, model, decode_position);
                        let mut s = world.wiring.borrow_mut();
                        let mut bound = registry_bound.borrow_mut();
                        match slot {
                            Some(WiringSlot::Prefill) => {
                                s.1 = Some(w);
                                bound.1 = Some(gk.scalarmul_scales.clone());
                            }
                            Some(WiringSlot::Decode) => {
                                s.0 = Some(w);
                                bound.0 = Some(gk.scalarmul_scales.clone());
                            }
                            None => {}
                        }
                    }
                    // ⭐ ONE LEDGER ENTRY per baked bundle — the parity law judges the
                    // whole ledger once, after the last bake.
                    BakeEffect::WeightSeg { base, bank0, banks } => {
                        world
                            .weight_seg_ledger
                            .borrow_mut()
                            .push((base, bank0, banks));
                    }
                    BakeEffect::Registry {
                        prefill,
                        base,
                        scales,
                    } => registry_ledger.borrow_mut().push((prefill, base, scales)),
                }
            }
        };
        let emit_bundle =
            |lwd: &BakeTape,
             base: String,
             cap: u32,
             seq_sym: Option<(&'static str, u32)>,
             prefill_mq: Option<u32>,
             active_cap: scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap,
             decode_rows: u32| {
                let (r, fx) = emit_bundle_deferred(
                    lwd,
                    base,
                    cap,
                    seq_sym,
                    prefill_mq,
                    active_cap,
                    decode_rows,
                    None,
                );
                commit_bake(&one_rank, lwd, fx);
                r
            };

        // Decode bundle (m = KTIR_M, default 1): the worker's fast per-token path.
        // The dd2-PROVEN static-SDPA single graph is emitted from THIS m=1 decode
        // tape (`seq_sym=None`) and baked as the sendnn bundle — ONE graph for both
        // prefill-chunks and decode (host-grown KV). This is the only sendnn lowering
        // (the offline_decoder CB-decode / SDPA-prefill bundle path was removed); a
        // PLAIN `cargo build -Fsendnn` (no env) produces it. This is also the TOP
        // decode cap bucket (cap == prefix_len); the narrower buckets are baked below.
        let (_decode_graph_unused, mut default_decode_groups) = emit_bundle(
            &lowered,
            format!("ktir_decode_{stem_id}"),
            prefix_len.get(),
            None, // KTIR decode is m=1; its sengraph is replaced by the CB one below
            None, // decode path (Store + PagedAttnCompute) -- no prefill width
            scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::FULL,
            primary_m,
        );

        // ⭐ THE SHARDED WORLD — the same decode forward, sharded `TP_WORLD` ways by the shared
        // analysis and baked through the same pipeline into its own world's state. Every collective
        // in it was placed by `tensor_parallel::shard` from the dataflow; the slice law each source
        // is loaded by rides out with the wiring (`tp_sources`).
        //
        // ⭐ BAKED ONLY WHEN THE BUILD NAMES IT (`tp/<W>` → this crate's `tp-<W>`, read once as
        // `TP_WORLD`): a build that names no world bakes none, and ships an empty `SUPERDSC_TP`.
        let tp_sources: Option<Vec<scratchy_subtile::tensor_parallel::SourceShard>> = if !<
            scratchy_target_spyre::bundle_code::collective::Build as scratchy_target_spyre::bundle_code::collective::World
        >::SHARDED
        {
            None
        } else {
            let (tp_lwd, sf) = scratchy_subtile::tensor_parallel::shard_lowered::<
                scratchy_target_spyre::bundle_code::collective::Build,
            _,
        >(
                &lowered,
                scratchy_target_spyre::tp_plan::SPYRE_SPLIT_GRANULE,
                &chained_row_gathers(&lowered, model, true),
            )
            .unwrap_or_else(|e| {
                panic!("[spyre-tp] {stem}: the {TP_WORLD}-rank forward is malformed — {e:?}")
            });
            let gathers = sf
                .collectives
                .iter()
                .filter(|c| c.kind == scratchy_subtile::tensor_parallel::CollectiveKind::AllGather)
                .count();
            eprintln!(
                "[spyre-tp] {stem}: tp={TP_WORLD} — {} ops, {} all-reduce(s), {gathers} all-gather(s)",
                tp_lwd.input.ops.len(),
                sf.collectives.len() - gathers,
            );
            // ⭐ THE COLLECTIVE LEDGER — every site the analysis wove, with the op whose output it
            // reads and the weights that op contracts, by their generated field names: what an
            // independent placement of the same model diffs against, site by site.
            let weights_of = |op: usize| -> Vec<String> {
                tp_lwd.input.ops[op]
                    .inputs
                    .iter()
                    .filter_map(|i| match i {
                        scratchy_subtile::lower::InputRef::Ext(s) => tp_lwd.bindings().get(*s),
                        scratchy_subtile::lower::InputRef::Op(_) => None,
                    })
                    .filter_map(|b| match b {
                        scratchy_subtile::handoff::SourceBinding::Weight { id, index } => Some(
                            scratchy_forward_compiler_macro::emit::weight_field_name(program, WeightId(*id), *index)
                                .to_string(),
                        ),
                        _ => None,
                    })
                    .collect()
            };
            for (k, c) in sf.collectives.iter().enumerate() {
                let reads = match c.reads {
                    scratchy_subtile::lower::InputRef::Op(r) => r,
                    // A gathered row's partial sum: the gather is its contraction.
                    scratchy_subtile::lower::InputRef::Ext(e) => {
                        eprintln!(
                            "[spyre-tp] {stem}: collective {k}: {:?} at op {} reading source t{e} \
                             ({:?}), a row gathered from a column-split table",
                            c.kind,
                            c.op,
                            tp_lwd.bindings().get(e),
                        );
                        continue;
                    }
                };
                // The contraction behind the read: back along the first operand through the
                // weightless ops between it and the collective (granite's residual multiplier).
                let mut at = reads;
                while weights_of(at).is_empty() {
                    match tp_lwd.input.ops[at].inputs.first() {
                        Some(scratchy_subtile::lower::InputRef::Op(p)) => at = *p,
                        _ => break,
                    }
                }
                eprintln!(
                    "[spyre-tp] {stem}: collective {k}: {:?} at op {} reading op {} ({:?}); \
                     contracted at op {} ({:?}) over [{}]",
                    c.kind,
                    c.op,
                    reads,
                    tp_lwd.input.ops[reads].op,
                    at,
                    tp_lwd.input.ops[at].op,
                    weights_of(at).join(", ")
                );
            }
            // ⭐ A VOCABULARY-SPLIT HEAD SHIPS ITS WORLD. The decode's chained head gathers the
            // drawn token's row from the lm head's own table, which the analysis split by columns
            // with the head; the gather fact (`chained_row_gathers`) made that row a partial sum the
            // analysis reduces, and each rank's chain gathers its own block (`mint_embed_chain`
            // over the split tail, the rank's `ChainBlock` in its tables and seeds).
            for g in &sf.gathered {
                eprintln!(
                    "[spyre-tp] {stem}: the decode's input row t{} is gathered from the \
                     column-split table t{} — each rank gathers its own vocabulary block",
                    g.row, g.table
                );
            }
            let (_, fx) = emit_bundle_deferred_sharded(
                &tp_lwd,
                format!("ktir_decode_tp{TP_WORLD}_{stem_id}"),
                prefix_len.get(),
                None,
                None,
                scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::FULL,
                primary_m,
                Some(&sf.sources),
            );
            commit_bake(&sharded_world, &tp_lwd, fx);
            Some(sf.sources)
        };
        // The decode's source bindings, named apart: every later rung of the sharded world checks
        // its cuts against them (`refuse_if_staged_cut_apart`), and the prefill ladder's own
        // per-width lowering is also called `lowered`.
        let decode_bindings = lowered.bindings();
        // ── DECODE BATCH LADDER (superdsc) ────────────────────────────────────────────────────────
        // A decode step runs one token of each RUNNING request, and how many that is changes step to
        // step — the scheduler decides it. A SuperDSC bundle is a static-shape graph, so the emitter
        // cannot take that number; it bakes a rung per width and the worker picks the smallest that
        // holds the live count, exactly as the prefill ladder already works.
        //
        // Each rung is the SAME tape at a wider row count — prefill's own emission, which is the
        // whole point: decode is that path at mq=B, not a second implementation. What differs is the
        // two things a batch needs and a prompt does not, and both are already gated on the row count
        // being requests: one page map per request, and one prefix-validity row per request.
        //
        // Rungs share ONE resident weights+KV — no term in either placement depends on the row count,
        // the same property that licenses the prefill ladder. A rung that fails to lower FAILS THE
        // BUILD, naming the rung and the error.
        // ⛔ CARD-ONLY, same reason as the prefill ladder below: the worker reaches a batch rung
        // through `superdsc_decode_rungs` / `BatchSlot`, which it consults under `spyre-hw` alone —
        // an emulator build sets `batched_prefill: None` (`spyre_load.rs:1713`) and never runs one.
        // Five more whole-model bakes for nothing.
        if sendnn_superdsc && cfg!(feature = "spyre-lower") {
            // RUNGS PAST 8. A decode step's cost is dominated by a FIXED penalty for leaving the
            // mq==1 fast path — the body goes 271 ops/layer at one request to 612 at two, then only
            // +16 per further request — plus one launch per request. Measured 36 ms fixed + ~2 ms per
            // request, so the fixed part is what the ladder amortizes and wider rungs are close to
            // free: 8 requests reach ~2.4x the weight stream's worth of work, 32 should reach ~4x.
            // A rung that fails to lower fails the build, so adding one cannot ship silently.
            //
            // THE LADDER COMES FROM THE POOL, not from a literal here. A page holds exactly
            // `PagedKvPool::ROWS` requests, and a rung wider than that would bake a launch whose
            // request axis walks past its kv head into the next one's keys — silently, and only for
            // the highest rows of the widest rung. `PagedKvPool` const-asserts that its widest rung
            // is `ROWS`, so the two cannot drift: changing either one alone is a `cargo build` error.
            //
            // ⭐ WIDEST FIRST. The widest rung's per-page fold is the bake's longest dxp group (2,156
            // ops at 32 requests, ~4 minutes alone), and dxp compiles groups in submission order, so
            // emitted last it started ~200 s into a granite-2b bake and WAS the bake's tail. Order is
            // free here: `decode_rung_lits` sorts the ladder by width before baking it.
            for seqs in ktir_superdsc::sdsc_abstract::PagedKvPool::BATCH_RUNGS
                .into_iter()
                .rev()
            {
                match scratchy_forward_compiler_macro::to_wavefront::lower_decode_to_wavefront(
                    fuf,
                    inferred,
                    decode_bounds,
                    &bridge,
                    scratchy_forward_compiler_macro::to_wavefront::PrefixCapacity::new(prefix_len),
                    seqs,
                ) {
                    Ok(lwd_b) => {
                        let lwd_b = bake_tape(stem, lwd_b);
                        emit_bundle(
                            &lwd_b,
                            format!("ktir_decode_{stem_id}"),
                            // `cap`: this bundle is baked for the full prefix.
                            prefix_len.get(),
                            // `seq_sym` / `prefill_mq`: decode, so neither applies.
                            None,
                            None,
                            scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::FULL,
                            seqs,
                        );
                        // ⭐ AND THE RUNG IN THE SHARDED WORLD: a sharded session serves every live
                        // request count on its own world's rung, never the one-rank ladder's.
                        if let Some(shards) = tp_sources.as_deref() {
                            let (tp_lwd, sf_b) =
                                scratchy_subtile::tensor_parallel::shard_lowered::<
                                    scratchy_target_spyre::bundle_code::collective::Build,
                                    _,
                                >(
                                    &lwd_b,
                                    scratchy_target_spyre::tp_plan::SPYRE_SPLIT_GRANULE,
                                    &chained_row_gathers(&lwd_b, model, true),
                                )
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "[spyre-tp] {stem}: the {TP_WORLD}-rank decode batch \
                                     seqs={seqs} is malformed — {e:?}"
                                    )
                                });
                            refuse_if_staged_cut_apart(
                                stem,
                                &format!("ktir_decode_tp{TP_WORLD}_{stem_id} seqs={seqs}"),
                                (lowered.bindings(), shards),
                                (lwd_b.bindings(), &sf_b.sources),
                            );
                            let (_, fx) = emit_bundle_deferred_sharded(
                                &tp_lwd,
                                format!("ktir_decode_tp{TP_WORLD}_{stem_id}"),
                                prefix_len.get(),
                                None,
                                None,
                                scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::FULL,
                                seqs,
                                Some(&sf_b.sources),
                            );
                            commit_bake(&sharded_world, &tp_lwd, fx);
                        }
                    }
                    Err(e) => panic!(
                        "[spyre-superdsc] {stem}: decode batch rung seqs={seqs} NOT LOWERED — {e}. \
                         Every rung the bake was asked for must lower; a skip ships a binary whose \
                         batch silently runs elsewhere"
                    ),
                }
            }
        }
        // SuperDSC replaces the sengraph ENTIRELY: `emit_bundle` above baked the work-divided dxp bundle
        // and set `superdsc_fp`, but the SENGRAPH decode lowering is REFUSED for granite (the lm_head
        // vocab-slice `t65` non-whole-region) so `default_decode_groups` is EMPTY. The SengraphBundle still
        // needs ONE [prefill,decode] group to carry the `SUPERDSC_BUNDLE:{fp}` sentinel (both slots are
        // overridden to it below), so synthesize a placeholder. `in_t`/`res_t` are UNUSED by the superdsc
        // worker path (it reads the generated `Wiring`); the graph_json is the sentinel. This
        // is the group STRUCTURE only — the actual compute is the proven dxp bundle, not a sengraph.
        if default_decode_groups.is_empty() && superdsc_fp.borrow().is_some() {
            default_decode_groups.push((0u32, 0u32, 0u32, String::new()));
        }

        // ⭐ THE HEAD WIDTH IS MINTED AS A TYPE at the bake's one door: whole sticks or a BUILD
        // ERROR that names the model and its head_dim (the config's, typed at load).
        let prefill_head_width = ktir_superdsc::sdsc_abstract::HeadWidth::of_whole_sticks_at_bake(
            u64::from(model.text.head_dim.get()),
            &model.name,
        );
        // The rungs THIS model bakes, by the same law the emitter forms the causal block with.
        let rung_admitted = |r: u32| {
            r <= SUB_BLOCKED_PREFILL_CEILING
                || ktir_superdsc::sdsc_abstract::NewBlockForm::of(r, prefill_head_width, false)
                    == ktir_superdsc::sdsc_abstract::NewBlockForm::Chunk
        };
        let admitted_ceiling = PREFILL_RUNGS
            .into_iter()
            .filter(|&r| rung_admitted(r))
            .max()
            .unwrap_or(SUB_BLOCKED_PREFILL_CEILING);
        // ⛔ BUILD GUARD (guard-every-crash, on-card PROVEN 2026-06-24/25 dd2): mq is
        // NOT free-valued — the DEM/DSM partitioner only tiles the static-paged
        // PagedAttnCompute batch axis (`mq`) cleanly for a SPECIFIC empirically-
        // verified set. Out-of-set mq crashes on-card during `session_create`
        // (CompileGraph), BEFORE any Predict, with one of two signatures depending
        // on the value (sweep at nblk=4, SmolLM2-135M):
        //   mq ∈ {2,4,8}     → COMPILES + runs + coherent (decode 5.5 / 6.4 / 9.1 ms)
        //   mq == 96         → COMPILES + runs + coherent (decode 58.7 ms) [baseline]
        //   mq ∈ {16,32,256} → ABORT `vector::_M_range_check (0 >= 0)` (DEM tiler:
        //                      empty partition vector indexed at 0)
        //   mq == 64         → ABORT `DtException: sumVpPair == tensorProp.shape[d]`
        //                      (dsm/sharedFuncs.cpp:2804 — partition-sum != axis len)
        //   mq == 1          → already build-refused upstream (unit-axis Reshape vs
        //                      symbolic-batch collision, graphOptimizer.cpp:5422)
        // The plain `mq <= 96` ceiling did NOT catch the mid-range {16,32,64} crashes
        // (all <= 96). This allowlist turns every out-of-set mq into a `cargo build`
        // error instead of an on-card `session_create threw` at model load. To add a
        // new mq, PROVE it compiles+runs on dd2 first, then extend MQ_DEM_SAFE.
        // Real LONGER CONTEXT comes from a LARGE paged CACHE (prefix_len/nblk —
        // proven to compile at 512/2048) + CHUNKED prefill + 1-token incremental
        // decode, NOT a larger mq. A SMALLER decode mq (≤8) is the proven roofline
        // lever: decode COMPUTE scales ~linearly in mq (≈4.1 ms floor + 0.57 ms·mq
        // at nblk=4), so mq=2 is ~10.7× faster per decode step than mq=96.
        //
        // SMALL-MQ DECODE LEVER (PROVEN ~10.7× per-token, on-card 2026-06-25): the
        // static-paged DECODE graph is emitted at the SMALL `DECODE_MQ` (incremental
        // decode has 1 new token, so the old mq=96 over-provisions ~48×; decode COMPUTE
        // scales ~linearly in mq — 286 ms → ~10 ms/token at mq=2). NUMERICS ARE
        // IDENTICAL — pure right-sizing: the real token is row 0, the rest is pad
        // (last-real-row replicated, then discarded; the shim returns row n_new-1).
        //
        // ONE session drives BOTH prefill chunks and incremental decode through this
        // ONE graph — the resident KV lives in the session's on-card allocation and
        // persists across Predicts, and it CANNOT be shared with a second graph/session
        // (on-card PROVEN 2026-06-25: a separate prefill graph at mq=96 + a host-buffer
        // KV hand-off to a mq=2 decode session produced GARBLED output — the prefill's
        // on-card cache writes are NOT mirrored to the host buffer, so the copy is
        // stale; both mq=96 and mq=2 garbled with the two-session split, while the
        // single-session incremental path is coherent). So prefill chunks ride the SAME
        // small mq (a slower one-time TTFT cost; decode is the dominant repeated cost).
        // `CB_BATCH_TEMPLATE` is kept ONLY for the DEFAULT (non-paged) mode's CB-decode
        // template and as the MQ_DEM_SAFE upper anchor.
        const CB_BATCH_TEMPLATE: u32 = 96; // default-mode CB template (not block_size 64)
        // PAGED prefill+decode mq = the CONTINUOUS-BATCHING batch axis (#45): mq is the
        // number of decode ROWS one Predict processes, so it is the max number of
        // concurrent sequences whose next-token is computed per Predict (the CB batch
        // axis). DECODE_MQ=2 is the chat-optimal single-stream width: on-card @ nblk=32,
        // mq=2 → ~9.7 ms/Predict ≈ 104 tok/s with byte-identical output (committed
        // 206ece8d; ∈ MQ_DEM_SAFE). The worker packs B≤mq requests into the mq rows (1
        // real token each) via predict_perrow with per-row block_table; B>mq sub-batches
        // (ceil(B/mq) Predicts). NUMERICS identical: each row is its own seq's real token
        // over its own scheduler blocks; padding rows replicate the last real row and are
        // discarded. The per-row CB mechanism is on-card-VERIFIED at mq=8 (C=4 → ~99 tok/s,
        // 4 concurrent prompts each correct incl Jupiter, ZERO cross-contamination).
        //
        // mq is FIXED per compiled graph, so a single static mq trades C=1 latency vs
        // batch throughput: per-Predict cost grows with BOTH mq and nblk (mq=8 @ nblk=32
        // → ~33 ms ⇒ C=1 only ~30 tok/s; mq=96 @ nblk=32 → ~58.7 ms but 96 rows ⇒ ~1635
        // tok/s aggregate at full batch ≈ 36% of the ~4500 tok/s weight-bandwidth
        // roofline). The PATH to that throughput WITHOUT regressing chat is DYNAMIC mq —
        // bake decode graphs at several mq ∈ MQ_DEM_SAFE and have the worker pick the
        // smallest that holds the live batch (C=1 → mq=2, large batch → mq=96). Until that
        // lands, change `DECODE_MQ` below for an 8-way concurrent server.
        // DEM/DSM-safe mq values, on-card verified (dd2 SENTIENT, nblk=4). MUST stay
        // sorted; every entry has been observed to CompileGraph + Predict coherently.
        const MQ_DEM_SAFE: [u32; 4] = [2, 4, 8, 96];
        // The smallest DEM-safe mq that holds `min_rows` rows: the law DECODE_MQ
        // FOLLOWS from, and the same answer the dynamic-mq path gives the live
        // batch (worker picks the smallest safe mq ≥ the live request count).
        // Reads MQ_DEM_SAFE ascending, so the table's sorted-order law is its
        // precondition (asserted below).
        const fn smallest_dem_safe_mq(min_rows: u32) -> u32 {
            let mut i = 0;
            while i < MQ_DEM_SAFE.len() {
                if MQ_DEM_SAFE[i] >= min_rows {
                    return MQ_DEM_SAFE[i];
                }
                i += 1;
            }
            panic!("MQ_DEM_SAFE holds no mq >= the minimal decode row count");
        }
        // Incremental decode carries ONE new token per sequence, and the device
        // already build-refuses mq=1 (unit-axis Reshape vs symbolic-batch
        // collision, graphOptimizer.cpp:5422 — see the guard note above), so the
        // minimal lawful decode row count is 2: 1 real token + 1 pad.
        const MINIMAL_DECODE_ROWS: u32 = 2;
        // PAGED prefill+decode mq = CB batch axis default (chat-optimal; see note).
        // DERIVED, not picked: the smallest DEM-safe mq ≥ the minimal decode row
        // count — identical for every model (min{mq ∈ {2,4,8,96} : mq ≥ 2} = 2).
        const DECODE_MQ: u32 = smallest_dem_safe_mq(MINIMAL_DECODE_ROWS);
        const fn mq_is_dem_safe(mq: u32) -> bool {
            // const-fn linear scan (arrays aren't const-iterable on stable).
            let mut i = 0;
            while i < MQ_DEM_SAFE.len() {
                if MQ_DEM_SAFE[i] == mq {
                    return true;
                }
                i += 1;
            }
            false
        }
        const _: () = assert!(
            mq_is_dem_safe(CB_BATCH_TEMPLATE),
            "CB_BATCH_TEMPLATE (PREFILL mq) is not a DEM/DSM-safe static-paged mq. Only {{2,4,8,96}} \
             are on-card PROVEN (dd2) to CompileGraph + Predict; other values abort \
             `session_create` BEFORE any Predict with vector::_M_range_check (mq in {{16,32,256}}) \
             or DtException sumVpPair==shape (mq==64). Pick a value from MQ_DEM_SAFE; to add \
             one, prove it compiles+runs on dd2 first."
        );
        const _: () = assert!(
            mq_is_dem_safe(DECODE_MQ),
            "DECODE_MQ (smallest DEM-safe mq ≥ MINIMAL_DECODE_ROWS) is not in MQ_DEM_SAFE — \
             the derivation and the table have drifted apart."
        );
        // smallest_dem_safe_mq reads MQ_DEM_SAFE ascending; pin that ordering.
        const _: () = assert!(
            {
                let mut i = 1;
                let mut sorted = true;
                while i < MQ_DEM_SAFE.len() {
                    if MQ_DEM_SAFE[i - 1] >= MQ_DEM_SAFE[i] {
                        sorted = false;
                    }
                    i += 1;
                }
                sorted
            },
            "MQ_DEM_SAFE must stay strictly ascending: smallest_dem_safe_mq scans it in order."
        );
        // Batched-prefill bundle: the worker runs the prompt through it (one m=N
        // forward per chunk), keeping the m=1 decode bundle for the generation
        // loop. Emitted BY DEFAULT so a plain spyre build prefills the prompt;
        // m=1 ⇒ skip (decode already covers single-token steps, e.g. the m=1
        // self-check gate where prefix_len defaults to its tiny value).
        //
        // PAGED: the prefill graph is emitted but UNUSED at runtime (one session uses
        // the decode graph for both phases — see DECODE_MQ note). It is still lowered
        // (to satisfy the bundle's [prefill,decode] pairing + KV-shape-agreement guards)
        // at the small DECODE_MQ so its kcache_/vcache_ shapes match decode exactly.
        // DEFAULT (host-grown) prefill uses the LX-derived width
        // (`emu_prefill_lx_width`, module level). `KTIR_PREFILL_LEN` overrides.
        let prefill_m: u32 = std::env::var("KTIR_PREFILL_LEN")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(if sendnn_paged {
                // The prefill graph is emitted-but-UNUSED in the single-graph path (the mq=1
                // decode graph drives BOTH phases). It must still be emitted at mq>1, else the
                // `.filter(|&v| v > 1)` below drops it → the bundle ships no prefill graph → the
                // paged-bundle guard panics (prefill_present=false). kcache_/vcache_ shapes are
                // mq-INDEPENDENT ([nblk,64,nkv,hd]), so this mq-2 placeholder prefill graph's KV
                // shapes still match the mq=1 decode exactly (the KV-shape-agreement guard passes).
                DECODE_MQ.max(2)
            } else if sendnn_superdsc && cfg!(feature = "spyre-lower") {
                // SuperDSC bakes the whole WIDTH LADDER and lets the WIDEST rung that actually
                // lowers become the ceiling (see the rung loop below), so this is just where the
                // ladder STARTS looking — not a value that has to be right. That is what demotes
                // `KTIR_PREFILL_LEN` from a required flag to a diagnostic override, and what stops
                // a model whose FFN blows the LX budget at the top width from failing the BUILD:
                // it simply gets a shorter ladder. Capped by the widest rung THIS model admits
                // (`rung_admitted`), and by nothing else.
                //
                // ⛔ NOT BY THE RESIDENT CACHE (`prefix_len`, 256 by default). That cap dates from the
                // unpaged cache, where a chunk's keys had to fit the baked prefix rows. A paged chunk
                // writes the POOL, one page part per page (`EmittedOp::kv_slot_offset`), so its width is
                // the pool's law (`PagedKvPool::PREFILL_CHUNK_SLOTS`, asserted equal to the ladder
                // ceiling above) and not the cache's. MEASURED: with the cap, the 512-row ladder baked
                // to 256 in silence — `PREFILL ladder = [..., 192, 256]` at load, and every TTFT cell
                // equal to the 256-row build's.
                admitted_ceiling
            } else {
                // The EMU's ONE width, LX-derived from the model's own fp8
                // gemms (`emu_prefill_lx_width`, module level): the old
                // constant `prefix_len.min(PREFILL_LX_SAFE)` refused ~40 s
                // into gemma-4-12b's first chat — rung 31's fp8 epilogue
                // needs 2,887,680 B against a 2 MiB core.
                emu_prefill_lx_width(fuf, model, prefix_len.get())
            });
        // ── PREFILL WIDTH LADDER: bake EVERY rung up to the ceiling, ascending ──
        // `prefill` ends up as the WIDEST rung that actually lowered, so a rung the model cannot fit
        // (the LX budget scales with mq x intermediate) shortens the ladder instead of failing the
        // build. Ascending order also means the widest bake is LAST, so it is the one that lands in
        // `superdsc_prefill_fp` (the single-bundle prefill slot) -- the worker treats that as the
        // ceiling and every narrower rung as a step below it.
        // Non-superdsc keeps exactly one width and still PANICS if it will not lower: those paths
        // have no ladder to fall back to, so a missing prefill graph must stay a build failure.
        // ⛔ THE LADDER IS A CARD FEATURE — BAKE ONE WIDTH OTHERWISE. Every rung is a full lowering
        // of the whole model, and the emulator worker can only ever run ONE of them:
        // `KtirBundle::prefill` is a single `Option` (`manifest.rs:167`) and the narrower rungs are
        // reached through `prefill_rungs`, which `spyre_forward.rs` reads under `spyre-hw` alone.
        // With `PREFILL_RUNGS` running to 96 that was 21 bakes to use 1 (25 now, to 256) — the generated
        // model source and rustc's single-threaded pass over it both paid for all of them.
        let prefill_rung_widths: Vec<u32> = if sendnn_superdsc && cfg!(feature = "spyre-lower") {
            let mut v: Vec<u32> = PREFILL_RUNGS
                .into_iter()
                .filter(|&r| r > 1 && r <= prefill_m && rung_admitted(r))
                .collect();
            // An explicit KTIR_PREFILL_LEN that is not itself a rung still gets baked, as the top.
            if v.last() != Some(&prefill_m) && prefill_m > 1 {
                v.push(prefill_m);
            }
            v
        } else {
            Some(prefill_m).filter(|&v| v > 1).into_iter().collect()
        };
        type PrefillPick = BakeOut;
        let mut prefill: Option<PrefillPick> = None;
        // ⭐ THE RUNGS ARE BAKED IN PARALLEL AND COMMITTED IN ORDER. Each rung is a whole-model lowering
        // that reads nothing another rung writes — `emit_bundle_deferred` records its effects instead of
        // applying them — so the 25 bakes run side by side and `commit_bake` then applies their effects
        // rung by rung, ascending, exactly as the serial loop did (the last rung that lowered is still
        // the ceiling).
        //
        // A rung that does not lower PANICS inside its own bake — every entry that reaches
        // the commit below is `Ok`, so the committed ladder is every rung asked for.
        // A rung's bake in each world: the one-rank lowering, its pick and effects, and — when the
        // sharded world baked its decode — the rung's sharded lowering and effects.
        type RungBake = (
            BakeTape,
            PrefillPick,
            Vec<BakeEffect>,
            Option<(BakeTape, Vec<BakeEffect>)>,
        );
        let per_rung: Vec<Result<RungBake, (u32, String)>> = {
            use rayon::prelude::*;
            prefill_rung_widths.par_iter().map(|&pl| -> Result<RungBake, (u32, String)> {
            let lowered = to_wavefront::lower_decode_to_wavefront(
                fuf,
                inferred,
                decode_bounds,
                &bridge,
                // Prefill and decode model the SAME resident prefix-KV cache, so
                // they MUST share ONE capacity — the KTIR emulator bakes a STATIC
                // `[prefix_len, kv]` cache + a `[1, prefix_len]` runtime length mask,
                // and the worker reads that mask width as the bundle's max context
                // (`b.capacity`). A prefill baked at a smaller capacity than the
                // decode makes the worker reject any prompt longer than that capacity
                // (`positions 0..N exceed bundle prefix capacity`). The prior default
                // used `1` — a host-grown-sengraph optimization (valid_len-1=0 drops
                // the prefix seg on silicon, where the host passes exactly `valid`
                // rows at runtime so the template size is immaterial). But the KTIR
                // cache is NOT host-grown; at prefix_len=1 the prompt can never
                // exceed 1 token. Use the resident capacity for BOTH paths (paged and
                // default), matching the decode bundle: at start=0 the runtime mask
                // keeps 0 prefix rows so the prompt still attends only itself
                // causally, and chunk k attends the k*mq rows written so far.
                scratchy_forward_compiler_macro::to_wavefront::PrefixCapacity::new(prefix_len),
                pl,
            );
            match lowered {
                Ok(lwd_p) => {
                    let lwd_p = bake_tape(stem, lwd_p);
                    // One rung's bake, stamped for each world `lwd` may be lowered for.
                    macro_rules! bake_rung_with {
                        ($bake:ident) => {
                            |lwd: &BakeTape,
                                     base: String,
                                     shards: Option<&[scratchy_subtile::tensor_parallel::SourceShard]>| $bake(
                        lwd,
                        base,
                        prefix_len.get(),
                        // prefill: rewrite the concrete prompt length `pl` → symbol
                        // s0 (offline_decoder idx-0 graph: prompt sym = Tq = Tkv).
                        // ── superdsc: keep the prefill tape CONCRETE (m=pl) so the reroll
                        //    bakes a real m=N dxp bundle (the dxp reroll is NOT symbolic; the
                        //    worker runs it at the fixed pl). Only the sengraph offline_decoder
                        //    path uses the s0 symbol. ──
                        if sendnn_superdsc {
                            None
                        } else {
                            Some(("s0", pl))
                        },
                        Some(pl), // prefill path (SDPA + Store), ladder rung m=pl
                        // ⛔ ZERO PREFIX BLOCKS, AND THAT IS ONLY RIGHT FOR THE FIRST CHUNK.
                        // `ActiveCap::NONE` is documented as valid ONLY where `start == 0`, but
                        // `manifest.rs`'s `KtirBundle::prefill` is a single `Option` and
                        // `spyre_forward.rs:1544` hands that one bundle to EVERY chunk, so chunk k
                        // attends only its own `mq` tokens and ignores the `k*mq` before it. The
                        // prefix-capable emission below (`:10176`) is reachable only through
                        // `prefill_prefix`, which the worker reads under `spyre-hw` alone.
                        //
                        // ⚠️ FLIPPING THIS TO `FULL` DOES NOT FIX IT — MEASURED. `ActiveCap` is the
                        // sweep extent of a DECODE attention, so `FULL` here lowers the chunk's
                        // attention in the decode form: its per-layer `new_k` comes back one row
                        // (`kv_dim`) while `forward_chunk` reads `n * kv_dim`
                        // (`spyre_forward.rs:1460`), and any prompt long enough to chunk PANICS
                        // (`range end index 49152 out of range for slice of length 512`, 49152 =
                        // 96*512). Multi-chunk prefill needs prefix blocks AND `mq` new rows
                        // together; that combination is not what this flag selects.
                        scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::NONE,
                        1, // a prompt chunk is one request however many positions it spans
                        shards,
                    )
                        };
                    }
                    let bake_rung = bake_rung_with!(emit_bundle_deferred);
                    let bake_rung_sharded = bake_rung_with!(emit_bundle_deferred_sharded);
                    let (r, fx) = bake_rung(&lwd_p, format!("ktir_prefill_{stem_id}"), None);
                    // ⭐ AND THE RUNG IN THE TWO-RANK WORLD, iff that world baked its decode — a
                    // world ships whole or not at all.
                    let tp2 = tp_sources.as_deref().map(|decode_cuts| {
                        let (tp_lwd, sf_p) = scratchy_subtile::tensor_parallel::shard_lowered::<
                            scratchy_target_spyre::bundle_code::collective::Build,
                        _,
                    >(
                            &lwd_p,
                            scratchy_target_spyre::tp_plan::SPYRE_SPLIT_GRANULE,
                            &[],
                        )
                        .unwrap_or_else(|e| {
                            panic!("[spyre-tp] {stem}: the {TP_WORLD}-rank prefill m={pl} is malformed — {e:?}")
                        });
                        refuse_if_staged_cut_apart(
                            stem,
                            &format!("ktir_prefill_tp{TP_WORLD}_{stem_id} m={pl}"),
                            (decode_bindings, decode_cuts),
                            (lwd_p.bindings(), &sf_p.sources),
                        );
                        let (_, fx) = bake_rung_sharded(
                            &tp_lwd,
                            format!("ktir_prefill_tp{TP_WORLD}_{stem_id}"),
                            Some(&sf_p.sources),
                        );
                        (tp_lwd, fx)
                    });
                    Ok((lwd_p, r, fx, tp2))
                }
                Err(e) if sendnn_superdsc => {
                    panic!(
                        "[spyre-superdsc] {stem}: prefill ladder rung m={pl} NOT LOWERED — {e}. \
                         Every rung the bake was asked for must lower; a skip ships a binary whose \
                         chunks silently run elsewhere"
                    )
                }
                Err(e) => panic!("[wavefront] {stem}: prefill (m={pl}) NOT LOWERED — {e}"),
            }
            }).collect()
        };
        // A rung that fails to lower panics inside the parallel bake above, so every
        // entry here is `Ok` — the window guard below still runs, over what baked.
        for rung in per_rung {
            match rung {
                Ok((lwd_p, r, fx, tp2)) => {
                    commit_bake(&one_rank, &lwd_p, fx);
                    if let Some((tp_lwd, fx)) = tp2 {
                        commit_bake(&sharded_world, &tp_lwd, fx);
                    }
                    prefill = Some(r);
                }
                Err((pl, e)) => {
                    panic!("[spyre-superdsc] {stem}: prefill ladder rung m={pl} NOT LOWERED — {e}")
                }
            }
        }
        // ⛔⛔⛔ AND THE LADDER THAT LOWERED MUST REACH THE POOL'S WINDOW, for a model that admits it — on
        // the rungs codegen actually baked, never the const ladder. A short ladder runs, correctly and
        // slowly, with nothing anywhere saying so; see `PagedKvPool::prefill_ladder_reaches_the_window`.
        if sendnn_superdsc && cfg!(feature = "spyre-lower") {
            let baked: Vec<u32> = superdsc_prefill_rungs
                .borrow()
                .iter()
                .map(|(m, _)| *m)
                .collect();
            let window = ktir_superdsc::sdsc_abstract::PagedKvPool::PREFILL_CHUNK_SLOTS as u32;
            if let Err(e) =
                ktir_superdsc::sdsc_abstract::PagedKvPool::prefill_ladder_reaches_the_window(
                    rung_admitted(window),
                    prefill_m,
                    &baked,
                    &[],
                )
            {
                panic!("[spyre-superdsc] {stem}: {e}");
            }
        }

        // ── PREFIX-CAPABLE PREFILL LADDER ── the prefill bakes with a full resident-prefix sweep. The
        // ladder rungs above are all prefix-FREE, which is correct only for a chunk with start==0; a
        // continuation chunk has resident KV to attend and must run on one of these.
        // ⛔ AND IT IS CARD-ONLY TOO: it is reached through `prefill_prefix`, which the worker reads
        // under `spyre-hw` alone. An emulator build baked this whole extra model and could never
        // launch it.
        // ⭐ AT THE PAGE AND AT THE WINDOW — `PagedKvPool::prefix_rungs`, over the widths that baked.
        // Every mid-prompt continuation chunk is a whole window (the step-back leaves it room for one),
        // so only a prompt's LAST chunk can be narrower, and a page is the narrowest a padded write can
        // be placed in without stepping back into the previous chunk's page. Their attention is the
        // whole-chunk continuation, whose program is flat in the rows (`attn_program_is_flat_in_rows`
        // refuses anything else at the door), so the widest rung costs no more to lower than the
        // narrowest.
        // ⛔ ONE WIDTH WAS THE MEASURED COST. This baked the widest rung alone, and a 52-row tail ran a
        // full 512-row chunk — granite-3.1-2b at ~2k tokens: 460 ms against 384 ms on the 256-row build.
        if sendnn_superdsc && cfg!(feature = "spyre-lower") {
            let baked: Vec<u32> = superdsc_prefill_rungs
                .borrow()
                .iter()
                .map(|(m, _)| *m)
                .collect();
            let asked = ktir_superdsc::sdsc_abstract::PagedKvPool::prefix_rungs(&baked);
            for &m in &asked {
                match scratchy_forward_compiler_macro::to_wavefront::lower_decode_to_wavefront(
                    fuf,
                    inferred,
                    decode_bounds,
                    &bridge,
                    scratchy_forward_compiler_macro::to_wavefront::PrefixCapacity::new(prefix_len),
                    m,
                ) {
                    Ok(lwd_p) => {
                        let lwd_p = bake_tape(stem, lwd_p);
                        emit_bundle(
                            &lwd_p,
                            format!("ktir_prefill_{stem_id}"),
                            prefix_len.get(),
                            None,
                            Some(m),
                            scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::FULL,
                            1, // a prompt chunk is one request
                        );
                        // ⭐ AND THE RUNG IN THE SHARDED WORLD — a world ships whole or not at all: a
                        // continuation chunk on a sharded session runs on ITS world's prefix rung.
                        if let Some(decode_cuts) = tp_sources.as_deref() {
                            let (tp_lwd, sf_p) = scratchy_subtile::tensor_parallel::shard_lowered::<
                                scratchy_target_spyre::bundle_code::collective::Build,
                            _,
                        >(
                                &lwd_p,
                                scratchy_target_spyre::tp_plan::SPYRE_SPLIT_GRANULE,
                                &[],
                            )
                            .unwrap_or_else(|e| {
                                panic!(
                                    "[spyre-tp] {stem}: the {TP_WORLD}-rank prefix prefill m={m} \
                                     is malformed — {e:?}"
                                )
                            });
                            // The rung's OWN slice laws, as the prefix-free ladder passes its own:
                            // a prompt's rows are staged whole, never the decode's gathered row.
                            refuse_if_staged_cut_apart(
                                stem,
                                &format!("ktir_prefill_tp{TP_WORLD}_{stem_id} prefix m={m}"),
                                (decode_bindings, decode_cuts),
                                (lwd_p.bindings(), &sf_p.sources),
                            );
                            let (_, fx) = emit_bundle_deferred_sharded(
                                &tp_lwd,
                                format!("ktir_prefill_tp{TP_WORLD}_{stem_id}"),
                                prefix_len.get(),
                                None,
                                Some(m),
                                scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::FULL,
                                1,
                                Some(&sf_p.sources),
                            );
                            commit_bake(&sharded_world, &tp_lwd, fx);
                        }
                    }
                    Err(e) => panic!(
                        "[spyre-superdsc] {stem}: prefix-capable prefill rung m={m} NOT LOWERED — \
                         {e}. Every rung the bake was asked for must lower"
                    ),
                }
            }
            // ⛔⛔⛔ THE PREFIX LADDER MUST TOP OUT AT THE PREFIX-FREE LADDER's TOP. A mid-prompt continuation
            // chunk runs at the widest rung's width (it is capped there and the step-back leaves it room), so
            // a prefix ladder that stops short would put a mid-prompt chunk on a bundle narrower than its rows
            // — a refusal on the card. Checked on what BAKED, not on the widths asked for.
            let prefix: Vec<u32> = superdsc_prefill_prefix_rungs
                .borrow()
                .iter()
                .map(|(m, _)| *m)
                .collect();
            if let Err(e) = ktir_superdsc::sdsc_abstract::PagedKvPool::prefix_ladder_tops_the_ladder(
                &baked, &prefix,
            ) {
                panic!("[spyre-superdsc] {stem}: {e}");
            }
            // ⛔⛔⛔ AND EVERY RUNG BELOW THE TOP THAT WAS ASKED FOR MUST HAVE BAKED. A missing one is silent:
            // its chunks run on the next wider rung. See `PagedKvPool::prefix_ladder_bakes_every_rung`.
            if let Err(e) =
                ktir_superdsc::sdsc_abstract::PagedKvPool::prefix_ladder_bakes_every_rung(
                    &asked,
                    &prefix,
                    &[],
                )
            {
                panic!("[spyre-superdsc] {stem}: {e}");
            }
        }

        // ⛔⛔⛔ ONE MODEL PACKS ONE RESIDENT WEIGHT SEGMENT — the parity ledger's verdict,
        // judged ONCE here, after the last superdsc bake of this model. Every bundle that
        // lowered recorded its packed weight segment (bank 0 + extra banks), exactly as the
        // runtime's `alias_seg` guard reads it when a session takes another's resident
        // weights; a bundle that reserves bytes a sibling does not is a card-load refusal
        // there, so it is a BUILD error here, naming both bundles. Empty ledger (nothing
        // baked a superdsc section) is trivially fine.
        if let Err(e) = scratchy_target_spyre::lower_subtile_tape_to_superdsc::weight_segment_parity(
            &weight_seg_ledger.borrow(),
        ) {
            panic!("[spyre-superdsc] {stem}: {e}");
        }
        // AND EACH SHARDED WORLD'S, APART: a rank packs its own shard of the weights, one resident
        // segment per rank, which no one-rank bundle shares.
        if let Err(e) = scratchy_target_spyre::lower_subtile_tape_to_superdsc::weight_segment_parity(
            &sharded_world.weight_seg_ledger.borrow(),
        ) {
            panic!("[spyre-superdsc] {stem} (tp={TP_WORLD}): {e}");
        }
        // ⛔⛔ AND EVERY BUNDLE OF A PHASE PLACES THE REGISTRY ITS PHASE'S WIRING BINDS — the emulator
        // worker stages one wiring's scales for every bundle of a phase. (A card launch binds its own
        // session's placed registry, `SuperDscSession::bake_facts`, so its rungs may differ: a decode
        // batch rung's carries the batched kernel's clamp.)
        if !cfg!(feature = "spyre-lower") {
            let (decode_bound, prefill_bound) = &*registry_bound.borrow();
            let wrong: Vec<String> = registry_ledger
                .borrow()
                .iter()
                .filter_map(|(prefill, base, scales)| {
                    let bound = if *prefill {
                        prefill_bound
                    } else {
                        decode_bound
                    };
                    bound
                        .as_ref()
                        .filter(|b| *b != scales)
                        .map(|b| format!("{base}: places {scales:?}, its wiring binds {b:?}"))
                })
                .collect();
            if !wrong.is_empty() {
                panic!(
                    "[spyre-superdsc] {stem}: {} bundle(s) place a scalar registry other than the \
                     one their phase's wiring binds, so their constants are read by the wrong index:\n{}",
                    wrong.len(),
                    wrong.join("\n")
                );
            }
        }

        // ── CB decode sengraph (the baked `--target sendnn` decode graph). The
        //    offline_decoder decode isengraph must carry a BATCH symbol on its K/Q
        //    data — the DEM reads batchSym from the PagedAttnStore KEY input
        //    (dem_frontend decoderSymCheck). The KTIR decode is m=1, so build a
        //    SEPARATE m=DECODE_MQ tape whose ROWS are the batch dim, then
        //    symbolicize that value → s0 (batch); the tkv symbol (s1) is emitted
        //    explicitly on block_table dim0. batch=1 at runtime via symVals[s0]=1. ──
        // The mq allowlist guard (MQ_DEM_SAFE) + CB_BATCH_TEMPLATE (prefill mq) +
        // DECODE_MQ (decode mq) are defined ABOVE (before the prefill block, so the
        // paged prefill can use CB_BATCH_TEMPLATE). The DECODE graph is emitted at the
        // SMALL DECODE_MQ (the proven ~10.7× decode lever); prefill stays at 96.
        // ⛔ BUILD GUARD (guard-every-crash, option (b) resident KV): the paged path
        // serves LONGER CONTEXT only if the resident cache (cap = prefix_len, in
        // 64-token pages) holds strictly MORE than one forward's worth of tokens
        // (`mq`). A model whose context were <= CB_BATCH_TEMPLATE would overflow the
        // cache after the first chunk (the shim's slot-bound guard would reject step 2
        // at RUNTIME) — turn that into a `cargo build` failure here (this proc-macro
        // body runs at build time). Only gated to the paged mode.
        if std::env::var("SCRATCHY_SENDNN_MODE").as_deref() == Ok("paged")
            && prefix_len.get() <= CB_BATCH_TEMPLATE
        {
            panic!(
                "[spyre-sendnn] paged mode: the model's context {prefix_len} <= mq \
                 (CB_BATCH_TEMPLATE={CB_BATCH_TEMPLATE}). The resident cache must hold MORE than one \
                 incremental forward's worth of tokens to serve longer context."
            );
        }
        // `_cb_decode_graph` = the whole-graph decode lowering (stays empty under
        // the flit-cap guard for 30 layers); the runnable artifacts are the
        // per-group `cb_decode_groups`.
        // DECODE graph mq: the SMALL DECODE_MQ for paged (the proven ~10.7× decode
        // lever — incremental decode has 1 real token, so mq=2 right-sizes the per-
        // token forward); CB_BATCH_TEMPLATE for default mode (which drives one graph
        // for both phases). NUMERICS identical: row 0 is the real token, the rest pad.
        let decode_mq = if sendnn_paged {
            DECODE_MQ
        } else {
            CB_BATCH_TEMPLATE
        };
        let (_cb_decode_graph, mut cb_decode_groups): (String, GroupGraphs) =
            match scratchy_forward_compiler_macro::to_wavefront::lower_decode_to_wavefront(
                fuf,
                inferred,
                decode_bounds,
                &bridge,
                scratchy_forward_compiler_macro::to_wavefront::PrefixCapacity::new(prefix_len),
                decode_mq,
            ) {
                Ok(lwd_cb) => {
                    let lwd_cb = bake_tape(stem, lwd_cb);
                    let (sg, groups) = emit_bundle(
                        &lwd_cb,
                        format!("ktir_decode_cb_{stem_id}"),
                        prefix_len.get(),
                        Some(("s0", decode_mq)),
                        None,
                        scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::FULL,
                        // The CB sengraph's rows are a batch SYMBOL the DEM resolves at runtime, not
                        // a baked request count.
                        1,
                    );
                    (sg, groups)
                }
                Err(e) => {
                    panic!("[wavefront] {stem}: CB decode (m={decode_mq}) NOT LOWERED — {e}")
                }
            };

        // DEFAULT-MODE OVERRIDE: replace the offline_decoder CB-decode groups with
        // the single static-SDPA default-mode graph (emitted from the m=1 decode
        // tape above). The bundle bake below stays unchanged structurally — it
        // bakes one `SengraphBundleGroup` per group; in default mode there is ONE
        // group whose `prefill` and `decode` slots both hold the SAME single graph
        // (the worker drives it for both phases; KV is host-grown, not resident, so
        // there is no prefill↔decode KV-by-name linkage to satisfy). The `>=2`
        // / KV-shape / count guards below all pass trivially (one group, identical
        // graphs). The offline_decoder env is NOT required for this graph.
        // DEFAULT mode: one group, prefill slot == decode slot (host-grown KV, single
        // static-SDPA graph drives both phases). PAGED mode: KEEP the distinct paged
        // decode groups the CB-decode `emit_bundle` produced above (Store+Compute over
        // the resident cache) — do NOT clobber them with the default clone.
        if !sendnn_paged {
            cb_decode_groups = default_decode_groups.clone();
        }

        // ⛔ GUARD (guard-every-crash-at-build-time) for deeprt.cpp:3317
        // "Insufficient sengraphs passed for decoder compilation, expected >=2":
        // offline_decoder compiles [prefill, decode] TOGETHER. Emitting a decode
        // sengraph without a prefill one would make the shim pass a single graph and
        // crash on-card. Fail the BUILD instead.
        // With the layer-group split the runnable artifacts are the per-group
        // graphs; the whole-graph `pg`/`cb_decode_graph` stay empty under the
        // flit-cap guard. The "both graphs present" check is now per-GROUP: the
        // prefill + decode group lists must be non-empty.
        // "prefill present": DEFAULT mode — the single static-SDPA graph fills both
        // slots (default_decode_groups non-empty). PAGED mode — default_decode_groups is
        // empty (the m=1 decode emits no paged sengraph); the prefill groups live in the
        // `prefill` emit_bundle result (the paged SDPA+Store graph). Without this, paged
        // would see prefill_sg_present=false and CLEAR the paged decode groups below →
        // "bundle has no layer groups (split produced 0)" at worker load.
        let prefill_sg_present = if sendnn_paged {
            prefill.as_ref().is_some_and(|(_, pgs)| !pgs.is_empty())
        } else {
            !default_decode_groups.is_empty()
        };
        // offline_decoder compiles [prefill, decode] TOGETHER (deeprt.cpp:3317 expects
        // >=2). If the decode groups exist but prefill ones do NOT, the bundle is
        // INCOMPLETE — most commonly because the single-symbol SDPA prefill was REFUSED
        // by the flit-cap guard (its 256-bucket logits output = 2.29 MB > the 1.03 MB
        // cap; the only fix is a chunked prefill, not yet landed). Rather than fail
        // `cargo build` (which would block EVERY -Fsendnn build, incl. other models),
        // mirror the flit-cap guard's LOG-NOT-PANIC pattern: drop the decode groups so
        // the baked SENGRAPH_BUNDLE is EMPTY, and let the spyre worker's startup guard
        // (spyre_worker.rs: empty prefill/decode graph_json) reject the model at load
        // with a clear message — the card never sees a single-graph (>=2 violating) job.
        if !cb_decode_groups.is_empty() && !prefill_sg_present {
            eprintln!(
                "[spyre-sendnn] {stem}: sendnn bundle has decode group graphs but NO prefill ones \
                 (the SDPA prefill was likely flit-cap-refused; chunked prefill not yet landed) — \
                 offline_decoder requires BOTH per group (deeprt.cpp:3317 expects >=2). Baking an \
                 EMPTY SENGRAPH_BUNDLE; the worker startup guard will reject `--device sendnn` for \
                 this model. (Set KTIR_PREFILL_LEN / land chunked prefill to produce a runnable bundle.)"
            );
            cb_decode_groups.clear();
        }

        // ⛔ PRIORITY-1 BUILD-TIME GUARD (guard-every-crash-at-build-time): the opt-in
        // paged path was explicitly requested (SCRATCHY_SENDNN_MODE=paged), so an EMPTY
        // or INCOMPLETE offline_decoder bundle MUST be a `cargo build` error — NOT the
        // deferred worker-load "bundle has no layer groups (split produced 0)" runtime
        // failure (the exact error the prefill_sg_present fix addressed). A runnable
        // paged bundle needs BOTH a prefill (SDPA+PagedAttnStore) and a decode
        // (PagedAttnCompute) graph per group; emitting neither/one is a lowering/codegen
        // bug. (The graceful log-not-panic above is DEFAULT-path-only, where the SDPA
        // prefill may be flit-refused without blocking other models' -Fsendnn builds.)
        if sendnn_paged && (cb_decode_groups.is_empty() || !prefill_sg_present) {
            panic!(
                "[spyre-sendnn] {stem}: SCRATCHY_SENDNN_MODE=paged produced an INCOMPLETE bundle \
                 (prefill_present={prefill_sg_present}, decode_groups={}) — the paged offline_decoder \
                 bundle needs BOTH a prefill (SDPA+PagedAttnStore) and a decode (PagedAttnCompute) \
                 graph per group. Check lower_graph_to_sengraph_paged_prefill/_decode + the \
                 emit_bundle paged routing; a worker-load failure must never be the first signal.",
                cb_decode_groups.len()
            );
        }

        // Bake the bundle into the binary as a const the per-arch
        // `ScratchyWeights::ktir_bundle()` override returns — no runtime disk read.
        // ⭐ A BUNDLE IS NAMED, NOT COPIED. Its programs were submitted to the registry above as
        // const data; this records WHICH set is this model's, and `bundle_code::bundle(fp)`
        // resolves it — the same resolver the ladder rungs and re-rolled siblings already use.
        let fp_lit = |fp: &std::cell::RefCell<Option<String>>| {
            let fp = fp.borrow().clone().unwrap_or_default();
            let l = proc_macro2::Literal::string(&fp);
            quote! {
                ::scratchy_target_spyre::manifest::KtirBundleData { fp: #l }
            }
        };
        let decode_lits: Vec<proc_macro2::TokenStream> = vec![fp_lit(&superdsc_fp)];
        let prefill_tokens = if superdsc_prefill_fp.borrow().is_some() {
            let p = fp_lit(&superdsc_prefill_fp);
            quote! { ::core::option::Option::Some(#p) }
        } else {
            quote! { ::core::option::Option::None }
        };
        *ktir_bundle_out = quote! {
            /// Embedded KTIR bundle — baked by `#[forward]` under `-Fspyre`.
            /// The fingerprint-matched `ScratchyWeights::ktir_bundle()` returns
            /// `&KTIR_BUNDLE`; the spyre worker runs it in-memory (no disk read).
            pub const KTIR_BUNDLE: ::scratchy_target_spyre::manifest::KtirBundle =
                ::scratchy_target_spyre::manifest::KtirBundle {
                    decode: &[ #(#decode_lits),* ],
                    prefill: #prefill_tokens,
                };
        };

        // ── `--target sendnn`: bake the LAYER-GROUP sub-bundles. Each group is a
        //    `[prefill, decode]` offline_decoder pair (the worker compiles each
        //    independently + threads the hidden state group→group on the host).
        //    Pair the prefill-group and decode-group by group index; carry the
        //    group IO wiring (input/result tensor ids = the residual edge). ──
        // DEFAULT mode: the "prefill" slot holds the SAME single static-SDPA graph as
        // the "decode" slot (one graph drives both phases; host-grown KV). PAGED mode:
        // pair the REAL paged prefill groups (SDPA+Store, distinct from the paged decode)
        // from the `prefill` emit_bundle so the bundle has genuinely distinct
        // [prefill,decode] graphs sharing the resident cache by name.
        let prefill_groups: &GroupGraphs = if sendnn_paged {
            prefill
                .as_ref()
                .map(|(_, pgs)| pgs)
                .unwrap_or(&default_decode_groups)
        } else {
            &default_decode_groups
        };
        // Pair groups by index (prefill ⟷ decode). They MUST agree in count (same
        // model, same group_size) — guard it.
        if !cb_decode_groups.is_empty() && prefill_groups.len() != cb_decode_groups.len() {
            panic!(
                "[spyre-sendnn] {stem}: prefill split into {} groups but decode into {} — \
                 the layer-group split must produce the same group count for both phases",
                prefill_groups.len(),
                cb_decode_groups.len()
            );
        }
        // ⛔ GUARD (guard-every-crash-at-build-time, priority-1, SILENT-FAILURE class):
        // the offline_decoder links prefill↔decode KV by NODE NAME
        // (`_KVCACHE_TENSOR_ADDRESSES`, dem_frontend.cpp:559-628 +
        // `DT_CHECK(otherIsg->hasNode(name))`). The resident `kcache_{L}`/`vcache_{L}`
        // sources are bound to the SAME on-device tensor across both graphs. If
        // prefill and decode emit DIFFERENT shapes for the same-named cache (e.g. a
        // prefill `cap` hardcoded to 256 vs a decode `cap` derived from the tape for
        // a non-256 prefix), the card silently reads/writes a mismatched KV layout —
        // NO crash, WRONG numerics. Cross-check every `{k,v}cache_*` input node's
        // shape between the paired prefill/decode group graphs; a mismatch fails the
        // BUILD. Extracts shapes from the shipped JSON (the exact on-card artifact).
        let kv_cache_shapes = |graph_json: &str| -> std::collections::BTreeMap<String, Vec<i64>> {
            let mut out = std::collections::BTreeMap::new();
            let v: serde_json::Value =
                serde_json::from_str(graph_json).unwrap_or(serde_json::Value::Null);
            if let Some(inputs) = v.get("input_nodes").and_then(|n| n.as_array()) {
                for node in inputs {
                    let name = node.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    if !(name.starts_with("kcache_") || name.starts_with("vcache_")) {
                        continue;
                    }
                    // Resident KV caches are CONCRETE rank-4; collect the integer dims
                    // (a symbolic dim would already be a separate guard violation).
                    let shape = node
                        .get("output_tensors")
                        .and_then(|t| t.as_array())
                        .and_then(|a| a.first())
                        .and_then(|t| t.get("shape"))
                        .and_then(|s| s.as_array())
                        .map(|dims| {
                            dims.iter()
                                .map(|d| d.as_i64().unwrap_or(-1))
                                .collect::<Vec<i64>>()
                        })
                        .unwrap_or_default();
                    out.insert(name.to_string(), shape);
                }
            }
            out
        };

        let mut group_tokens: Vec<proc_macro2::TokenStream> = Vec::new();
        for (gi, (g_idx, in_t, res_t, dec_json)) in cb_decode_groups.iter().enumerate() {
            // Matching prefill group (same index); its graph json + IO wiring.
            let pre_json = prefill_groups
                .get(gi)
                .map(|(_, _, _, j)| j.as_str())
                .unwrap_or("");
            let pre_in = prefill_groups
                .get(gi)
                .map(|(_, i, _, _)| *i)
                .unwrap_or(*in_t);
            let pre_res = prefill_groups
                .get(gi)
                .map(|(_, _, r, _)| *r)
                .unwrap_or(*res_t);
            // prefill + decode group IO must match (same residual edges).
            if pre_in != *in_t || pre_res != *res_t {
                panic!(
                    "[spyre-sendnn] {stem}: group {g_idx} IO mismatch prefill (in t{pre_in}, out t{pre_res}) \
                     vs decode (in t{in_t}, out t{res_t})"
                );
            }
            // ⛔ KV-cache shape agreement (priority-1, silent-numerics): every
            // same-named resident cache MUST have identical shapes in both phases.
            let pre_kv = kv_cache_shapes(pre_json);
            let dec_kv = kv_cache_shapes(dec_json);
            for (name, pshape) in &pre_kv {
                if let Some(dshape) = dec_kv.get(name)
                    && pshape != dshape
                {
                    panic!(
                        "[spyre-sendnn] {stem}: group {g_idx} KV-cache `{name}` shape MISMATCH — \
                         prefill {pshape:?} vs decode {dshape:?}. The offline_decoder links the \
                         resident KV by node name (_KVCACHE_TENSOR_ADDRESSES); disagreeing shapes \
                         silently corrupt the on-card KV layout (no crash, wrong numerics). Both \
                         arms in lower_subtile_tape_to_sengraph.rs must derive `cap` the SAME way \
                         (from b.graph.shape(inputs[1].tensor).rows)."
                    );
                }
            }
            let g_lit = *g_idx;
            let in_lit = *in_t;
            let res_lit = *res_t;
            // ── AHEAD-OF-TIME g2 cache (mirror of cuda's cudaforge `.cu` drop) ──
            // Compute a stable fingerprint of each shipped graph JSON and DUMP the
            // JSON to the sengraphforge cache keyed by it. `scratchy-builder-spyre`'s
            // build.rs (which depends on this model crate, so it runs AFTER this
            // emit — the cuda emit-then-compile ordering) picks each `<fp>.sengraph.json`
            // up, runs CompileGraph → SerializeToString on the card, and writes the
            // compiled `<fp>.g2.sen` next to it; that g2 is `include_bytes!`-baked into
            // the binary and reached at run time by this same fingerprint. The empty
            // string => no AoT key (the worker always CompileGraphs at load).
            // SuperDSC is its OWN target: it emits ONLY the work-divided dxp bundle
            // (compiled + embedded, stamped SUPERDSC_BUNDLE below) and NEVER a
            // sengraph. So under `--features superdsc` we do NOT drop a sengraph into
            // the sengraphforge cache — there is no g2 AoT bake, no CompileGraph, no
            // sengraph at all on this target.
            // ⛔ NO SENGRAPH IS DROPPED. There is no g2 ahead-of-time bake and no `CompileGraph`
            // on this target — the bundle's programs are what the device runs.
            let (pre_fp, dec_fp) = (String::new(), String::new());
            let pre_fp_lit = proc_macro2::Literal::string(&pre_fp);
            let dec_fp_lit = proc_macro2::Literal::string(&dec_fp);
            // SuperDSC ROUTING (#51 item 3): under `SCRATCHY_SENDNN_MODE=superdsc`
            // the m=1 decode walk produced a work-divided dxp bundle keyed by
            // `superdsc_fp`. Stamp BOTH the prefill and decode `graph_json` of the
            // (single) group to the sentinel `SUPERDSC_BUNDLE:{fp}` so the worker
            // resolves the baked bundle dir (`baked_superdsc_dir(fp)`) rather than
            // treating the string as a sengraph JSON. The sengraph `graph_json` is
            // overridden ONLY in superdsc mode; the default/paged sengraph path is
            // untouched (superdsc_fp stays None).
            let superdsc_sentinel = superdsc_fp
                .borrow()
                .as_ref()
                .map(|fp| format!("SUPERDSC_BUNDLE:{fp}"));
            // C4: the PREFILL slot resolves the DISTINCT m=N prefill dxp bundle (lm_head skipped)
            // when it baked; otherwise it falls back to the decode fp (worker uses decode for both
            // = sequential prefill, the prior behavior). Never a sengraph string in superdsc mode.
            let superdsc_prefill_sentinel = superdsc_prefill_fp
                .borrow()
                .as_ref()
                .map(|fp| format!("SUPERDSC_BUNDLE:{fp}"));
            let (pre_graph, dec_graph): (&str, &str) =
                match (&superdsc_prefill_sentinel, &superdsc_sentinel) {
                    (Some(pre), Some(dec)) => (pre.as_str(), dec.as_str()),
                    (None, Some(dec)) => (dec.as_str(), dec.as_str()),
                    _ => (pre_json, dec_json),
                };
            let pre_json_lit = proc_macro2::Literal::string(pre_graph);
            let dec_json_lit = proc_macro2::Literal::string(dec_graph);
            let (prefill_rung_lits, decode_rung_lits, prefill_prefix_lits) = ladder_lits(
                &superdsc_prefill_rungs.borrow(),
                &superdsc_decode_rungs.borrow(),
                &superdsc_prefill_prefix_rungs.borrow(),
            );
            group_tokens.push(quote! {
                ::scratchy_target_spyre::manifest::SengraphBundleGroup {
                    group: #g_lit,
                    input_tensor: #in_lit,
                    result_tensor: #res_lit,
                    prefill: ::scratchy_target_spyre::manifest::SengraphBundleData {
                        graph_json: #pre_json_lit,
                        g2_fingerprint: #pre_fp_lit,
                    },
                    decode: ::scratchy_target_spyre::manifest::SengraphBundleData {
                        graph_json: #dec_json_lit,
                        g2_fingerprint: #dec_fp_lit,
                    },
                    prefill_rungs: &[#(#prefill_rung_lits),*],
                    decode_rungs: &[#(#decode_rung_lits),*],
                    prefill_prefix: &[#(#prefill_prefix_lits),*],
                }
            });
        }
        // ⭐ THE COMPILED DEVICE CODE GOES INTO THE BINARY. This drains the bounded bake queue and
        // renders every bundle it compiled as an `inventory::submit!`ed value the runtime looks up by
        // fingerprint.
        ktir_bundle_out.extend(superdsc_bundle_tokens(model));
        // The typed launch wiring, spliced into the SAME per-model module that
        // holds `Weights` — so the binding of a source to a weight can be
        // generated code naming a real field, the way the GPU target's forward names
        // `wm.q_proj`, instead of a disk-key string resolved at load.
        {
            let (dec, pre) = &*wiring_tokens.borrow();
            if let Some(dec) = dec {
                let pre = match pre {
                    Some(p) => quote! { ::core::option::Option::Some(#p) },
                    None => quote! { ::core::option::Option::None },
                };
                ktir_bundle_out.extend(quote! {
                    /// The launch wiring for this model, emitted by `#[forward]`.
                    /// Replaces parsing `manifest.json` at model load.
                    pub static SUPERDSC_WIRINGS: ::scratchy_target_spyre::wiring::Wirings =
                        ::scratchy_target_spyre::wiring::Wirings {
                            decode: #dec,
                            prefill: #pre,
                        };
                });
            }
        }
        // ⭐ THE SHARDED WORLDS this model baked — one `TpForward` per world whose bake committed a
        // decode bundle and its wiring.
        {
            use scratchy_subtile::tensor_parallel::SourceShard as S;
            let tp_ranks = proc_macro2::Literal::u32_unsuffixed(TP_WORLD);
            let tp_forwards: Vec<proc_macro2::TokenStream> = match (
                tp_fp.borrow().clone(),
                tp_wiring.borrow().clone(),
                tp_sources.as_ref(),
            ) {
                (Some(fp), (Some(wiring), prefill_wiring), Some(sources)) => {
                    let fp = proc_macro2::Literal::string(&fp);
                    // The prefill bundle and its wiring ship together or not at all, as the
                    // one-rank world's do.
                    let (prefill, prefill_wiring) = match (
                        tp_prefill_fp.borrow().clone(),
                        prefill_wiring,
                    ) {
                        (Some(pfp), Some(pw)) => {
                            let pfp = proc_macro2::Literal::string(&pfp);
                            (
                                quote! { ::core::option::Option::Some(
                                    ::scratchy_target_spyre::manifest::KtirBundleData { fp: #pfp }
                                ) },
                                quote! { ::core::option::Option::Some(#pw) },
                            )
                        }
                        _ => (
                            quote! { ::core::option::Option::None },
                            quote! { ::core::option::Option::None },
                        ),
                    };
                    let (tp_prefill_rung_lits, tp_decode_rung_lits, tp_prefill_prefix_lits) =
                        ladder_lits(
                            &tp_prefill_rungs.borrow(),
                            &tp_decode_rungs.borrow(),
                            &tp_prefill_prefix_rungs.borrow(),
                        );
                    let sources = sources.iter().map(|s| match s {
                        S::Whole => quote! { ::scratchy_target_spyre::wiring::SourceShard::Whole },
                        S::Rows => quote! { ::scratchy_target_spyre::wiring::SourceShard::Rows },
                        S::Cols => quote! { ::scratchy_target_spyre::wiring::SourceShard::Cols },
                        S::Partial => {
                            quote! { ::scratchy_target_spyre::wiring::SourceShard::Partial }
                        }
                        S::ColBlocks { block } | S::RowBlocks { block } => {
                            let b = proc_macro2::Literal::u32_unsuffixed(block.get());
                            let block = quote! {
                                match ::core::num::NonZeroU32::new(#b) {
                                    ::core::option::Option::Some(n) => n,
                                    ::core::option::Option::None => panic!("an empty block"),
                                }
                            };
                            if matches!(s, S::ColBlocks { .. }) {
                                quote! {
                                    ::scratchy_target_spyre::wiring::SourceShard::ColBlocks {
                                        block: #block,
                                    }
                                }
                            } else {
                                quote! {
                                    ::scratchy_target_spyre::wiring::SourceShard::RowBlocks {
                                        block: #block,
                                    }
                                }
                            }
                        }
                        S::HeadRows { heads } => {
                            let h = proc_macro2::Literal::u32_unsuffixed(heads.get());
                            quote! {
                                ::scratchy_target_spyre::wiring::SourceShard::HeadRows {
                                    heads: match ::core::num::NonZeroU32::new(#h) {
                                        ::core::option::Option::Some(n) => n,
                                        ::core::option::Option::None => panic!("zero heads"),
                                    },
                                }
                            }
                        }
                    });
                    vec![quote! {
                        ::scratchy_target_spyre::wiring::TpForward::<
                            ::scratchy_target_spyre::bundle_code::collective::Build,
                        > {
                            decode: ::scratchy_target_spyre::manifest::KtirBundleData { fp: #fp },
                            prefill: #prefill,
                            prefill_rungs: &[#(#tp_prefill_rung_lits),*],
                            decode_rungs: &[#(#tp_decode_rung_lits),*],
                            prefill_prefix: &[#(#tp_prefill_prefix_lits),*],
                            wirings: ::scratchy_target_spyre::wiring::Wirings {
                                decode: #wiring,
                                prefill: #prefill_wiring,
                            },
                            sources: &[#(#sources),*],
                            world: ::core::marker::PhantomData,
                        }
                    }]
                }
                _ => Vec::new(),
            };
            ktir_bundle_out.extend(quote! {
                // ⛔ THE BAKE'S WORLD IS THE BINARY'S. The macro runs on the host with its own copy
                // of the world crate; the binary's `Build` must be the world it sharded for, or a
                // session would load bundles of another world. A mismatch is this build error.
                const _: () = assert!(
                    <::scratchy_target_spyre::bundle_code::collective::Build
                        as ::scratchy_target_spyre::bundle_code::collective::World>::RANKS
                        == #tp_ranks,
                    "#[forward] sharded for a different tensor-parallel world than this binary's \
                     `Build` — name the same `tp/N` for the bake and the runtime"
                );
                /// This model's sharded worlds, emitted by `#[forward]` — see
                /// `scratchy_target_spyre::wiring::TpForward`.
                pub static SUPERDSC_TP: &[::scratchy_target_spyre::wiring::TpForward<
                    ::scratchy_target_spyre::bundle_code::collective::Build,
                >] = &[#(#tp_forwards),*];
            });
        }
        ktir_bundle_out.extend(quote! {
            /// Embedded sendnn LAYER-GROUP bundle — baked by `#[forward]` under
            /// `-Fspyre`. Reached by the `sendnn`-feature spyre worker, which
            /// compiles EACH group's `[prefill, decode]` independently on the AIU
            /// and threads the hidden state group→group on the host.
            pub const SENGRAPH_BUNDLE: ::scratchy_target_spyre::manifest::SengraphBundle =
                ::scratchy_target_spyre::manifest::SengraphBundle {
                    groups: &[ #(#group_tokens),* ],
                };
        });
        // KTIR rides the embedded const + try_load.
    }
}

#[cfg(test)]
mod tests {
    use super::emu_prefill_lx_width;
    use scratchy_forward_compiler_macro::fixtures::{config_of, fuf_of, variant_of};

    /// ⛔ THE REGRESSION THIS GATE EXISTS FOR: the emu's ONE baked prefill width
    /// is DERIVED per model (`emu_prefill_lx_width`) from its own fp8 gemms —
    /// `3·m·N·2 + N·2` bytes of intra-node epilogue tiles vs 7/8 of a core's
    /// 2 MiB LX, measured on an isolated production-entry probe (flip at
    /// m=21/23 for N=15360). The old constant 31 was llama-1b's dense-FFN
    /// number: on gemma-4-12b it refused ~40 s into the first chat
    /// (`LinalgBroadcast: LX capacity exceeded … 1935360 + 952320 > 2097152`).
    /// This pins the derivation's rung table on the real configs — and, hidden
    /// in the 12b/31b rows, the lm-head exclusion: the vocab-wide gemm
    /// (N=262144 on the 12b) refuses EVERY rung, so a broken row-local tail
    /// walk panics the derivation here instead of failing an emu build.
    #[test]
    fn emu_prefill_lx_width_pins_the_gemma4_rungs() {
        for (arch, stem, preset, want) in [
            ("gemma4", "gemma-4-12b-it", "fp8-dynamic-per-channel", 19),
            ("gemma4", "gemma-4-31b-it", "fp8-dynamic-per-channel", 11),
            (
                "granite",
                "granite-3.1-2b-instruct",
                "fp8-dynamic-per-channel",
                31,
            ),
            ("llama", "llama-3.2-1b", "", 31),
        ] {
            let model = if preset.is_empty() {
                config_of(arch, stem)
            } else {
                match variant_of(arch, stem, preset) {
                    Some(model) => model,
                    // Load-or-refuse: a mixed-bit variant skips under a
                    // backend that can't lower it.
                    None => continue,
                }
            };
            let (_, fuf) = fuf_of(arch, &model);
            assert_eq!(
                emu_prefill_lx_width(&fuf, &model, 512),
                want,
                "{stem}: the emu prefill width must stay the derived rung"
            );
            // A prefix shorter than the derived rung bakes its OWN width.
            assert_eq!(
                emu_prefill_lx_width(&fuf, &model, 12),
                12,
                "{stem}: a prefix shorter than the derived rung must bake at the prefix"
            );
        }
    }
}
