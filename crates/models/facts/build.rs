// SPDX-License-Identifier: Apache-2.0
//! STAGE 1: one type per selected canonical, implementing `ModelFacts`.
//!
//! scratchy-models' build script (stage 2) bakes every model's tapes, and its lowerings are generic
//! over the model's facts (`fn lower<M: ModelFacts>`). A type a build script monomorphises over has
//! to exist when that script is COMPILED, and some facts come out of the DSL front end, not the
//! config alone, so the types are generated one build earlier: here. This script runs the same
//! front end and fill stage 2 runs (`carrier_facts`) over the same carriers
//! (`scope_compiled_carriers`, with this crate's features forwarded one-for-one from
//! scratchy-models'), and writes `$OUT_DIR/facts.rs`:
//!
//! * `pub mod <arch> { pub struct <Canonical>; impl ModelFacts for <Canonical> { … } }`
//! * `pub fn dispatch(arch, canonical, visitor)`, which calls `visitor.visit::<Canonical>()` for
//!   exactly the canonicals this build selected, so stage 2 instantiates nothing else.

use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use rayon::prelude::*;
use scratchy_forward_compiler_macro::{
    CanonicalFacts, carrier_facts, render_tokens, scope_compiled_carriers,
};
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // scratchy-models' crate dir: its `dsl/` and `configs/`.
    let arch_root = manifest.join("../arch");
    check_feature_parity(&arch_root.join("Cargo.toml"), &manifest.join("Cargo.toml"));
    let presets = scratchy_quantizations::enabled_presets()
        .into_iter()
        .collect();
    let targets = scope_compiled_carriers(&arch_root, &presets).targets;
    // Tokens are not `Send`, so each arch renders its module and its dispatch entries to text
    // inside its own task, as scratchy-forwards.rs renders each arch's emission.
    let per_arch: Vec<(String, String, usize)> = targets
        .par_iter()
        .map(|t| {
            let facts = carrier_facts(t.carrier(), &t.configs_dir)
                .unwrap_or_else(|e| panic!("model facts ({}): {e}", t.mod_name));
            let arch = Ident::new(&t.mod_name, Span::call_site());
            let (mut module, mut entries) = (String::new(), String::new());
            render_tokens(arch_module(&arch, &facts), &mut module);
            render_tokens(dispatch_entries(&arch, &facts), &mut entries);
            (module, entries, facts.len())
        })
        .collect();
    let parse = |text: &str| -> TokenStream {
        text.parse()
            .unwrap_or_else(|e| panic!("re-read rendered model facts: {e}"))
    };
    let modules = per_arch.iter().map(|(module, _, _)| parse(module));
    let entries = per_arch.iter().map(|(_, entries, _)| parse(entries));
    let count = per_arch.iter().map(|(_, _, n)| n).sum::<usize>();
    let file: TokenStream = quote! {
        #(#modules)*

        /// Run `visitor` at the facts of canonical `canonical` of arch `arch`: `None` for a
        /// canonical this build did not select. Each entry instantiates the visitor at one selected
        /// canonical's type and at no other.
        pub fn dispatch<V: ::scratchy_ir::facts::FactsVisitor>(
            arch: &str,
            canonical: &str,
            visitor: V,
        ) -> ::core::option::Option<V::Out> {
            let canonicals: [Entry<V>; #count] = [#(#entries)*];
            canonicals
                .into_iter()
                .find(|e| e.arch == arch && e.canonical == canonical)
                .map(|e| (e.visit)(visitor))
        }
    };
    let mut rendered = String::with_capacity(1 << 16);
    render_tokens(file, &mut rendered);
    let path = out_dir.join("facts.rs");
    if std::fs::read_to_string(&path).ok().as_deref() != Some(rendered.as_str()) {
        std::fs::write(&path, rendered).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }
}

/// This build script selects canonicals from its own `CARGO_FEATURE_*` env, so this crate carries
/// exactly scratchy-models' feature names, each forwarded from there (`F = [..,
/// "scratchy-model-facts/F"]`), and each enabling here exactly what it enables there among
/// features (`all` → its stems, a stem → its `arch-<name>`) and pipeline-lib features
/// (`scratchy-forward-compiler-macro/…`), so this crate selects the same canonicals built alone.
/// Any drift would make the two stages disagree about which canonicals exist, so it fails this
/// build, naming the feature.
fn check_feature_parity(models_manifest: &std::path::Path, facts_manifest: &std::path::Path) {
    const PIPELINE: &str = "scratchy-forward-compiler-macro/";
    let mirrored =
        |e: &String| e.starts_with(PIPELINE) || (!e.contains('/') && !e.starts_with("dep:"));
    let features = |path: &std::path::Path| -> toml::Table {
        println!("cargo:rerun-if-changed={}", path.display());
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let table: toml::Table = text
            .parse()
            .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        table["features"]
            .as_table()
            .cloned()
            .expect("a [features] table")
    };
    let list = |v: &toml::Value| -> Vec<String> {
        v.as_array()
            .expect("a feature is a list")
            .iter()
            .map(|x| x.as_str().expect("a feature entry is a string").to_owned())
            .collect()
    };
    let (models, facts) = (features(models_manifest), features(facts_manifest));
    for (name, enables) in &models {
        if name == "default" {
            continue;
        }
        let enables = list(enables);
        let forward = format!("scratchy-model-facts/{name}");
        assert!(
            enables.contains(&forward),
            "scratchy-models feature `{name}` must forward `{forward}`"
        );
        let expected: Vec<String> = enables.into_iter().filter(mirrored).collect();
        let here = facts
            .get(name)
            .map(list)
            .unwrap_or_else(|| panic!("scratchy-model-facts must declare feature `{name}`"));
        assert_eq!(
            here, expected,
            "scratchy-model-facts feature `{name}` enables exactly the features and pipeline-lib \
             features scratchy-models' does"
        );
    }
    for name in facts.keys() {
        assert!(
            models.contains_key(name),
            "scratchy-model-facts feature `{name}` is not a scratchy-models feature"
        );
    }
}

/// `pub mod <arch> { pub struct <Canonical>; impl ModelFacts for <Canonical> { … } … }`.
fn arch_module(arch: &Ident, facts: &[CanonicalFacts]) -> TokenStream {
    let items = facts.iter().map(|f| {
        // Two canonicals whose names map to one type are E0428 here, naming both.
        let (ty, body) = (type_name(&f.name), &f.body);
        let doc = format!("The facts of canonical `{arch}::{}`.", f.name);
        quote! {
            #[doc = #doc]
            pub struct #ty;
            impl ::scratchy_ir::facts::ModelFacts for #ty {
                #body
            }
            // Every law of every fact, checked now (see `ModelFacts::LAWS`).
            const _: () = <#ty as ::scratchy_ir::facts::ModelFacts>::LAWS;
        }
    });
    let doc = format!("The `{arch}` arch's canonicals.");
    quote! {
        #[doc = #doc]
        pub mod #arch {
            #(#items)*
        }
    }
}

/// `dispatch`'s table entries for one arch: each canonical's `(arch, name)` key and its type's
/// visit.
fn dispatch_entries(arch: &Ident, facts: &[CanonicalFacts]) -> TokenStream {
    let arch_name = arch.to_string();
    let entries = facts.iter().map(|f| {
        let (name, ty) = (&f.name, type_name(&f.name));
        quote! {
            Entry {
                arch: #arch_name,
                canonical: #name,
                visit: <V as ::scratchy_ir::facts::FactsVisitor>::visit::<#arch::#ty>,
            },
        }
    });
    quote! { #(#entries)* }
}

/// A canonical's type: its module name in UpperCamelCase (`gpt_oss_20b` → `GptOss20b`). An
/// underscore is kept only between two digits, where dropping it would merge two numbers
/// (`llama_3_2_1b` → `Llama3_2_1b`).
fn type_name(canonical: &str) -> Ident {
    let mut out = String::with_capacity(canonical.len());
    let mut upper_next = true;
    let mut chars = canonical.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '_' {
            let digits = out.ends_with(|p: char| p.is_ascii_digit())
                && chars.peek().is_some_and(char::is_ascii_digit);
            if digits {
                out.push('_');
            }
            upper_next = true;
        } else if upper_next {
            out.extend(c.to_uppercase());
            upper_next = false;
        } else {
            out.push(c);
        }
    }
    Ident::new(&out, Span::call_site())
}
