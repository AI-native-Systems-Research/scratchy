// SPDX-License-Identifier: Apache-2.0
//! The selected models, compiled for spyre: the scope `scratchy-models` publishes, through the
//! shared models build driver with spyre's emitter.
use scratchy_models_build::{ModelsBuild, ScopeMetadata};
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let metadata = ScopeMetadata::from_dep_env();
    // A `superdsc-dump` build hands the bake THIS build script's output directory — the one place
    // that knows it — so each baked bundle's dxp input lands beside the device images; any other
    // build writes no dump.
    let dump_dir = cfg!(feature = "superdsc-dump").then_some(out_dir.as_path());
    scratchy_models_build::build(&ModelsBuild {
        metadata: &metadata,
        out_dir: &out_dir,
        emitter: &scratchy_target_spyre_compiler::Spyre::of_kv_features(
            cfg!(feature = "kv-fp8"),
            cfg!(feature = "kv-dense"),
        ),
        dump_dir,
        inspect: &|_| Vec::new(),
    });
}
