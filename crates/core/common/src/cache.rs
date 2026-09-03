// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! On-disk cache locations, defined once for the whole workspace.
//!
//! Both the metal aligned-weight sidecar writer
//! (`scratchy_target_metal::metal_allocator`) and the `scr model cache`
//! tooling resolve cache paths through here, so the two can never drift.

use std::path::PathBuf;

/// Base cache directory: `$XDG_CACHE_HOME` if set, else `$HOME/.cache`,
/// else `/tmp` (CI sandboxes with no `HOME`). Honors the user's
/// environment — it does not pin a literal path.
pub fn cache_base_dir() -> PathBuf {
    if let Some(x) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(x);
    }
    if let Some(h) = std::env::var_os("HOME") {
        return PathBuf::from(h).join(".cache");
    }
    PathBuf::from("/tmp")
}

/// The scratchy metal aligned-weight sidecar cache:
/// `<cache_base>/scratchy/metal-aligned-weights`. Holds realigned weight
/// blobs that metal mmaps zero-copy on relaunch.
pub fn metal_aligned_weights_dir() -> PathBuf {
    cache_base_dir()
        .join("scratchy")
        .join("metal-aligned-weights")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_dir_is_under_cache_base() {
        let base = cache_base_dir();
        let weights = metal_aligned_weights_dir();
        assert!(weights.starts_with(&base));
        assert!(weights.ends_with("scratchy/metal-aligned-weights"));
    }
}
