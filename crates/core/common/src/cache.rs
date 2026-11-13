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

/// Where every target's aligned-weight sidecar cache lives: `<cache_base>/scratchy`.
pub fn aligned_weights_root() -> PathBuf {
    cache_base_dir().join("scratchy")
}

/// A target's aligned-weight sidecar cache: `<cache_base>/scratchy/<target>-aligned-weights`.
/// Holds realigned weight blobs the target mmaps zero-copy on relaunch.
pub fn aligned_weights_dir(target: &str) -> PathBuf {
    aligned_weights_root().join(format!("{target}-aligned-weights"))
}

/// Every target's aligned-weight cache under [`aligned_weights_root`], sorted; none when the root
/// does not exist.
pub fn aligned_weights_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = (std::fs::read_dir(aligned_weights_root()).into_iter())
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && (p.file_name().and_then(|n| n.to_str()))
                    .is_some_and(|n| n.ends_with("-aligned-weights"))
        })
        .collect();
    dirs.sort();
    dirs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_dir_is_under_cache_base() {
        let base = cache_base_dir();
        let weights = aligned_weights_dir("some-target");
        assert!(weights.starts_with(&base));
        assert!(weights.ends_with("scratchy/some-target-aligned-weights"));
    }
}
