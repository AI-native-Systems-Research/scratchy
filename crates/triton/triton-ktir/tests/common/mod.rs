// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! Golden plumbing shared by the integration tests.
//!
//! FAIL CLOSED ON A MISSING GOLDEN. A test that SKIPS when its oracle is absent
//! prints `ok` and proves nothing -- that is the silent green, and this tree has
//! been bitten by it before (commit e35e626b9, "the golden diff fails closed when
//! the golden is missing"). So [`golden`] PANICS with the regeneration command.

use std::path::PathBuf;

/// `crates/triton/test-goldens/ktir`.
pub fn goldens_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR is .../crates/triton/triton-ktir.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join("test-goldens/ktir"))
        .expect("the crate sits one level under crates/triton")
}

/// One golden file's text, or a panic naming how to regenerate it.
pub fn golden(config: &str, stage: &str) -> String {
    let path = goldens_dir().join(config).join(stage);
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "MISSING GOLDEN {}: {e}\n\
             This test compares against the C++ toolchain's output and cannot pass \
             without it. Regenerate with:\n\
             \n    bash third_party/spyre/test/goldens/ktir/regen.sh\n",
            path.display()
        )
    })
}

/// Does this config have this stage? Used only to assert a REFUSAL is recorded, so
/// it never turns a real comparison into a skip.
#[allow(dead_code)]
pub fn has(config: &str, stage: &str) -> bool {
    goldens_dir().join(config).join(stage).exists()
}

/// The launch grid a config was generated with, read from its own `grid.txt` so the
/// test cannot drift from the golden.
#[allow(dead_code)]
pub fn grid(config: &str) -> Vec<i64> {
    golden(config, "grid.txt")
        .trim()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("grid.txt holds a comma-separated integer list"))
        .collect()
}

/// A census as a sorted `name=count` list, for a one-line assertion.
#[allow(dead_code)]
pub fn census_line(m: &triton_ktir::Module) -> String {
    m.census()
        .into_iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(" ")
}
