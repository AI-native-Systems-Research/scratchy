// SPDX-License-Identifier: Apache-2.0
//! The on-disk HuggingFace cache layout.
//!
//! ```text
//! <root>/models--<owner>--<name>/
//!     blobs/<etag>                     the bytes, named by the Hub's digest
//!     snapshots/<commit>/<filename>    symlink -> ../../blobs/<etag>
//!     refs/<revision>                  file containing <commit>
//! ```
//!
//! This layout is an interop contract, not an implementation detail: Python
//! `huggingface_hub`, `transformers`, `huggingface-cli` and our own `scr model ls`
//! all resolve `refs/<revision>` -> commit -> `snapshots/<commit>/<file>`.
//! Writing a plain file to `snapshots/main/<file>` — the obvious shortcut —
//! produces a cache that nothing else can read.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Root of the Hub cache, matching Python `huggingface_hub`'s resolution
/// order.
///
/// Deliberately `$HOME/.cache/huggingface/hub` rather than the platform cache
/// dir: on macOS `dirs::cache_dir()` is `~/Library/Caches`, which would put
/// our downloads somewhere no other Hub client looks.
pub fn default_root() -> PathBuf {
    if let Ok(dir) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var("HF_HOME") {
        return PathBuf::from(dir).join("hub");
    }
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("huggingface").join("hub")
}

/// `owner/name` -> `models--owner--name`.
pub fn repo_folder(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
}

/// Per-repo directory under the cache root.
pub fn repo_dir(root: &Path, repo_id: &str) -> PathBuf {
    root.join(repo_folder(repo_id))
}

pub fn blob_path(repo_dir: &Path, etag: &str) -> PathBuf {
    repo_dir.join("blobs").join(etag)
}

pub fn snapshot_path(repo_dir: &Path, commit: &str, filename: &str) -> PathBuf {
    repo_dir.join("snapshots").join(commit).join(filename)
}

pub fn ref_path(repo_dir: &Path, revision: &str) -> PathBuf {
    repo_dir.join("refs").join(revision)
}

/// Network-free lookup: `refs/<revision>` -> commit -> snapshot entry, if the
/// entry resolves to something that exists.
///
/// `Path::exists` follows symlinks, so a snapshot entry whose blob was
/// evicted correctly reports absent rather than handing back a dangling link.
pub fn cached_path(root: &Path, repo_id: &str, revision: &str, filename: &str) -> Option<PathBuf> {
    let dir = repo_dir(root, repo_id);
    let commit = std::fs::read_to_string(ref_path(&dir, revision)).ok()?;
    let path = snapshot_path(&dir, commit.trim(), filename);
    path.exists().then_some(path)
}

/// Link a snapshot entry at `<repo>/snapshots/<commit>/<filename>` to the
/// blob, and record the revision -> commit mapping.
///
/// The link target is relative so the cache stays movable. Its depth is
/// computed from `filename`, which may itself contain directories: an entry
/// at `snapshots/<commit>/a/b.json` sits three levels below the repo dir, not
/// two. Hardcoding `../../blobs` — the intuitive answer — yields a dangling
/// link for every nested file, and nested files are normal (`onnx/model.onnx`).
pub fn link_into_snapshot(
    repo_dir: &Path,
    commit: &str,
    filename: &str,
    etag: &str,
    revision: &str,
) -> Result<PathBuf> {
    let link = snapshot_path(repo_dir, commit, filename);
    let parent = link
        .parent()
        .expect("snapshot path always has a parent directory");
    std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;

    // `snapshots` + `<commit>` + one level per directory component of
    // `filename`.
    let nested = Path::new(filename).components().count() - 1;
    let mut target = PathBuf::new();
    for _ in 0..(2 + nested) {
        target.push("..");
    }
    let target = target.join("blobs").join(etag);

    // Replace any existing entry; `symlink_metadata` catches a dangling link,
    // which `exists()` would miss.
    if std::fs::symlink_metadata(&link).is_ok() {
        std::fs::remove_file(&link).map_err(|e| Error::io(&link, e))?;
    }
    symlink_or_copy(&target, &link, repo_dir)?;

    let refs = ref_path(repo_dir, revision);
    let refs_parent = refs.parent().expect("ref path always has a parent");
    std::fs::create_dir_all(refs_parent).map_err(|e| Error::io(refs_parent, e))?;
    // Write-then-rename: concurrent downloads from the same repo all write
    // this, and a truncated ref file makes the whole repo unresolvable.
    let tmp = refs.with_extension("tmp");
    std::fs::write(&tmp, commit).map_err(|e| Error::io(&tmp, e))?;
    std::fs::rename(&tmp, &refs).map_err(|e| Error::io(&refs, e))?;

    Ok(link)
}

/// Symlink where the platform supports it, copy where it does not.
///
/// Windows symlinks need developer mode or elevation; Python
/// `huggingface_hub` copies in that case and so do we, at the cost of storing
/// the bytes twice. `relative_target` is resolved against `link`'s directory
/// for the copy fallback, since a relative path means nothing to `fs::copy`.
fn symlink_or_copy(relative_target: &Path, link: &Path, repo_dir: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let _ = repo_dir;
        std::os::unix::fs::symlink(relative_target, link).map_err(|e| Error::io(link, e))
    }
    #[cfg(not(unix))]
    {
        let blob = repo_dir.join("blobs").join(
            relative_target
                .file_name()
                .expect("blob target always has a file name"),
        );
        std::fs::copy(&blob, link)
            .map(|_| ())
            .map_err(|e| Error::io(link, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_folder_matches_python_layout() {
        assert_eq!(repo_folder("Qwen/Qwen3-0.6B"), "models--Qwen--Qwen3-0.6B");
    }

    #[test]
    fn nested_filenames_get_a_deeper_link_target() {
        let dir = tempdir();
        let repo = dir.join("models--a--b");
        std::fs::create_dir_all(repo.join("blobs")).unwrap();
        std::fs::write(repo.join("blobs").join("deadbeef"), b"x").unwrap();

        // Flat and nested entries must both resolve to the same blob.
        for filename in ["config.json", "onnx/model.onnx", "a/b/c.bin"] {
            let link = link_into_snapshot(&repo, "commit0", filename, "deadbeef", "main").unwrap();
            assert!(
                link.exists(),
                "{filename}: link does not resolve — wrong `..` depth"
            );
            assert_eq!(std::fs::read(&link).unwrap(), b"x");
        }
    }

    #[test]
    fn cached_path_round_trips_through_refs() {
        let dir = tempdir();
        let repo = dir.join("models--a--b");
        std::fs::create_dir_all(repo.join("blobs")).unwrap();
        std::fs::write(repo.join("blobs").join("etag1"), b"hello").unwrap();
        link_into_snapshot(&repo, "commit0", "config.json", "etag1", "main").unwrap();

        let found = cached_path(&dir, "a/b", "main", "config.json").expect("should resolve");
        assert_eq!(std::fs::read(found).unwrap(), b"hello");
        assert!(cached_path(&dir, "a/b", "main", "absent.json").is_none());
    }

    #[test]
    fn cached_path_rejects_a_dangling_link() {
        let dir = tempdir();
        let repo = dir.join("models--a--b");
        std::fs::create_dir_all(repo.join("blobs")).unwrap();
        std::fs::write(repo.join("blobs").join("etag1"), b"hello").unwrap();
        link_into_snapshot(&repo, "commit0", "config.json", "etag1", "main").unwrap();
        std::fs::remove_file(repo.join("blobs").join("etag1")).unwrap();

        assert!(
            cached_path(&dir, "a/b", "main", "config.json").is_none(),
            "an evicted blob must read as absent, not as a usable path"
        );
    }

    /// Unique temp dir without pulling `tempfile` into this crate's tree.
    fn tempdir() -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "hfdl-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        base
    }
}
