// SPDX-License-Identifier: Apache-2.0
//! The selected models, compiled for metal: the scope `scratchy-models` publishes, through the
//! shared models build driver with metal's emitter — then the two crate-root tables that span
//! every arch: the baked kernels the models name (`metal_bake.rs`), and every model that bakes a
//! tape (`metal_models.rs`).
use proc_macro2::{TokenStream, TokenTree};
use scratchy_models_build::{ModelsBuild, ScopeMetadata, write_if_changed};
use std::path::{Path, PathBuf};

/// The model modules of one arch's emit that carry a metal tape: each top-level
/// `pub mod <model> { … }` whose body names `METAL_BUCKETS` (a canonical's
/// `static`, a shim's re-export of its canonical's).
fn metal_tape_models(arch: &TokenStream) -> Vec<String> {
    fn names_metal_buckets(ts: TokenStream) -> bool {
        ts.into_iter().any(|tt| match tt {
            TokenTree::Ident(i) => i == "METAL_BUCKETS",
            TokenTree::Group(g) => names_metal_buckets(g.stream()),
            _ => false,
        })
    }
    let tts: Vec<TokenTree> = arch.clone().into_iter().collect();
    (tts.windows(4))
        .filter_map(|w| match w {
            [
                TokenTree::Ident(vis),
                TokenTree::Ident(kw),
                TokenTree::Ident(model),
                TokenTree::Group(body),
            ] if vis == "pub" && kw == "mod" && names_metal_buckets(body.stream()) => {
                Some(model.to_string())
            }
            _ => None,
        })
        .collect()
}

/// Write `$OUT_DIR/metal_models.rs`: `METAL_MODELS`, every model module this
/// build bakes a metal tape for (`<arch>::<model>`, sorted) beside its
/// `METAL_BUCKETS`. `src/lib.rs` includes it.
fn write_metal_models(out_dir: &Path, models: &[String]) {
    let rows: String = (models.iter())
        .map(|m| format!("    ({m:?}, crate::{m}::METAL_BUCKETS),\n"))
        .collect();
    let rendered = format!(
        "/// Every model module this build bakes a metal tape for (`<arch>::<model>`), beside\n\
         /// its baked buckets: read off the emitted modules by the build script, sorted.\n\
         pub static METAL_MODELS: &[(\n    \
             &str,\n    \
             &[::scratchy_target_metal::interpreter::metal::MetalBucketSpec],\n\
         )] = &[\n{rows}];\n"
    );
    write_if_changed(&out_dir.join("metal_models.rs"), &rendered);
}

/// Write `$OUT_DIR/metal_bake.rs`: the crate-root `__metal_bake` module (`src/lib.rs`) the
/// emitted models' baked kernels resolve in — every kernel the arches named, baked once. Runs
/// after every arch is emitted.
fn write_metal_bake(out_dir: &Path) {
    let mut rendered = String::with_capacity(1 << 20);
    scratchy_forward_compiler_macro::render_tokens(
        scratchy_target_metal_compiler::static_tape::bake_module(),
        &mut rendered,
    );
    write_if_changed(&out_dir.join("metal_bake.rs"), &rendered);
}

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let metadata = ScopeMetadata::from_dep_env();
    // A vision arch's tower runs on metal's vision surface, which this crate's `vision` compiles.
    assert!(
        cfg!(feature = "vision") || !metadata.features.iter().any(|f| f == "VISION"),
        "the model scope selects a vision arch, whose tower needs \
         `scratchy-target-metal-models/vision` (`scr`'s `multimodal`)"
    );
    let built = scratchy_models_build::build(&ModelsBuild {
        metadata: &metadata,
        out_dir: &out_dir,
        emitter: &scratchy_target_metal_compiler::Metal {
            turboquant: cfg!(feature = "turboquant"),
        },
        dump_dir: None,
        inspect: &metal_tape_models,
    });
    write_metal_bake(&out_dir);
    write_metal_models(&out_dir, &built.inspected);
}
