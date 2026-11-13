// SPDX-License-Identifier: Apache-2.0
//! The selected models, compiled for spyre: the scope `scratchy-models` publishes, through the
//! shared models build driver with spyre's emitter.
use scratchy_models_build::{ModelsBuild, ScopeMetadata};
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // Spyre/KTIR bundle knobs read at emit time by the bake: they bake the decode row count and
    // prefill width into the embedded KTIR bundle, so a change must re-run this build script
    // (which regenerates the bundle). Without this, cargo keeps the stale OUT_DIR codegen.
    for name in ["KTIR_M", "KTIR_PREFILL_LEN"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let metadata = ScopeMetadata::from_dep_env();
    // A `superdsc-dump` build hands the bake THIS build script's output directory — the one place
    // that knows it — so each baked bundle's dxp input lands beside the device images; any other
    // build writes no dump.
    let dump_dir = cfg!(feature = "superdsc-dump").then_some(out_dir.as_path());
    scratchy_models_build::build(&ModelsBuild {
        metadata: &metadata,
        out_dir: &out_dir,
        emitter: &scratchy_target_spyre_compiler::Spyre,
        dump_dir,
        inspect: &|_| Vec::new(),
    });
}
