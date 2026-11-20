// SPDX-License-Identifier: Apache-2.0
//! A build's per-model hashes held to a checked-in golden: one `<model>  <hash>` line per model,
//! in a file beside the test crate's `Cargo.toml`.
//!
//! The golden is the test's own. On a mismatch, [`assert_pinned_hashes`] writes `<golden>.new`
//! beside it and fails. That file holds this build's line for every model in its scope, and the
//! golden's line for every pinned model outside it, sorted by model. A change that means to move a
//! hash renames `.new` over the golden; nothing edits it by hand. A model the build hashes but the
//! golden does not pin fails the same way a moved one does.

use std::path::Path;

/// Holds each `(model, hash)` of `rows` to its line in `manifest_dir/golden`; on any mismatch,
/// writes `golden.new` and panics, naming every model whose hash moved.
pub fn assert_pinned_hashes(manifest_dir: &str, golden: &str, rows: &[(&str, String)]) {
    let path = Path::new(manifest_dir).join(golden);
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let pinned: Vec<(&str, &str)> = (text.lines())
        .filter_map(|l| l.split_once("  "))
        .map(|(m, h)| (m, h.trim()))
        .collect();
    let pin = |model: &str| pinned.iter().find(|(m, _)| *m == model).map(|(_, h)| *h);
    let moved: Vec<&str> = (rows.iter())
        .filter(|(model, hash)| pin(model) != Some(hash.as_str()))
        .map(|(model, _)| *model)
        .collect();
    if moved.is_empty() {
        return;
    }
    let mut lines: Vec<(&str, &str)> = (pinned.iter())
        .filter(|(m, _)| !rows.iter().any(|(model, _)| model == m))
        .copied()
        .chain(rows.iter().map(|(model, h)| (*model, h.as_str())))
        .collect();
    lines.sort_by_key(|(m, _)| *m);
    let new: String = lines.iter().map(|(m, h)| format!("{m}  {h}\n")).collect();
    std::fs::write(path.with_file_name(format!("{golden}.new")), new)
        .unwrap_or_else(|e| panic!("write {golden}.new: {e}"));
    panic!(
        "the pinned hashes of {moved:?} moved; `{golden}.new` holds this build's hashes: if the \
         move is meant, rename it over `{golden}`"
    );
}

#[cfg(test)]
mod tests {
    use super::assert_pinned_hashes;

    /// A moved hash fails and writes `.new` with this build's lines and the golden's out-of-scope
    /// ones; equal hashes pass and leave no `.new`.
    #[test]
    fn a_moved_hash_fails_and_the_new_file_keeps_out_of_scope_lines() {
        let dir = std::env::temp_dir().join(format!("golden-hash-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let golden = dir.join("g.txt");
        let new = dir.join("g.txt.new");
        std::fs::write(&golden, "a  1\nb  2\nz  9\n").unwrap();
        let _ = std::fs::remove_file(&new);
        let dir_s = dir.to_str().unwrap();

        assert_pinned_hashes(dir_s, "g.txt", &[("a", "1".into()), ("b", "2".into())]);
        assert!(!new.exists());

        let moved = std::panic::catch_unwind(|| {
            assert_pinned_hashes(dir_s, "g.txt", &[("c", "3".into()), ("b", "5".into())])
        });
        let msg = *moved.unwrap_err().downcast::<String>().unwrap();
        assert!(msg.contains(r#"["c", "b"]"#), "{msg}");
        assert_eq!(
            std::fs::read_to_string(&new).unwrap(),
            "a  1\nb  5\nc  3\nz  9\n"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
