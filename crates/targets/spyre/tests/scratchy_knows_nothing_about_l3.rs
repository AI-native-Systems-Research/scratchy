// SPDX-License-Identifier: Apache-2.0
//! 🛑🛑 A TARGET CRATE MAY NOT REACH INTO THE SCHEDULER'S INTERNALS.
//!
//! ⛔⛔ THE RULE, IN NICK'S OWN WORDS: *"the deeptools compiler should be isolated completely within
//! the `crates/compiler/deeptools` family. scratchy knows nothing about l3."* The root `CLAUDE.md`
//! says the same structurally — *"Everything common is shared. One implementation of every
//! target-neutral pass"* and *"Per-target surface = opcode lowering only."* The scheduler is
//! target-neutral by definition, so a target naming `deeptools::schedule::l3::dsc` is the boundary
//! being ABSENT rather than declared.
//!
//! ⭐⭐ THE SEAM IS `deeptools::sdsc`, and it is a module of pure `pub use`. A target names that and
//! nothing under `deeptools::schedule`, so the scheduler's internals can be rearranged — `l3::dsc`
//! merged, `dsc2` renamed, `stages` split — without a target crate changing.
//!
//! ⛔⛔ WHY THIS IS A TEST AND NOT A PARAGRAPH. `superdsc_to_l3_sdsc.rs` reached into NINE separate
//! `deeptools::schedule` paths, and every one arrived the same way: one more `use` line on a diff,
//! indistinguishable from the lines beside it, in a file whose header already said the boundary
//! mattered. Prose did not stop the ninth. This does — it names the FILE and the PATH.
//!
//! ⛔ AND IT IS SCOPED TO WHAT IT CAN HONESTLY CLAIM: it forbids the `deeptools::schedule` prefix in
//! target crates. It does NOT claim the conversion has been relocated into `deeptools` — that is
//! owed, blocked on the SuperDSC wire DTOs (`serde`, and two `scratchy_subtile` field types) — and
//! passing this test must never be read as that having happened.

use std::path::{Path, PathBuf};

/// The one forbidden prefix. ⭐ NOT `deeptools::` WHOLESALE: `arch`, `formats`, `units`, `islands`,
/// `bridges`, `model` and `sys_arch_spec` are deeptools' declared public IR/vocabulary surfaces and a
/// target is meant to name them. It is the SCHEDULER's interior that is off limits.
const FORBIDDEN: &str = "deeptools::schedule";

/// The seam that replaces it, named here so a failure says what to use instead.
const SEAM: &str = "deeptools::sdsc";

/// ⛔⛔ THE EVASION THAT WOULD MAKE [`FORBIDDEN`] COSMETIC: aliasing the crate, then reaching through
/// the alias — `use deeptools as dt;` followed by `dt::schedule::l3::dsc::SuperDsc`. The literal
/// prefix never appears and the boundary is gone. A target has no reason to rename the crate, so the
/// rename itself is what is forbidden.
const CRATE_ALIAS: &str = "use deeptools as";

/// The seam's own source, checked for the OTHER evasion — see
/// [`the_seam_re_exports_named_items_and_never_a_glob`].
const SEAM_FILE: &str = "../../compiler/deeptools/src/sdsc.rs";

/// Every `.rs` file under one directory.
fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// ⛔ THIS FILE ITSELF IS EXEMPT, AND IT HAS TO BE: [`FORBIDDEN`] is a `const` holding the very string
/// it forbids, so the rule cannot be STATED without one code line containing it. Exempting exactly one
/// file — this one, by [`file!`] — is narrower than exempting the pattern.
fn is_this_file(path: &Path) -> bool {
    Path::new(file!())
        .file_name()
        .is_some_and(|own| path.file_name() == Some(own))
}

/// A line that is CODE rather than commentary.
///
/// ⛔ COMMENTS ARE SKIPPED SO THE RULE CAN BE WRITTEN DOWN — this file's own doc comments name the
/// forbidden prefix, and so do the headers of the two files that explain the boundary. Counting those
/// would make the rule unstatable, which is the same allowance
/// `dfir_never_runtime_refuses.rs::occurrences` already makes.
fn is_code(line: &str) -> bool {
    let trimmed = line.trim_start();
    !(trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*'))
}

/// ⭐⭐ NO TARGET CRATE NAMES THE SCHEDULER'S INTERIOR — the audit, as an assertion.
///
/// ⛔ IT CARRIES THE FILE, THE LINE AND THE LINE'S TEXT, because a bare count would say the boundary
/// broke without saying where, and the fix is always *"route it through the seam"*.
#[test]
fn no_target_crate_names_the_schedulers_interior() {
    let targets = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/targets/spyre has a parent");
    let mut offenders: Vec<String> = Vec::new();
    let files = rust_files(targets);
    for file in &files {
        if is_this_file(file) {
            continue;
        }
        let Ok(source) = std::fs::read_to_string(file) else {
            continue;
        };
        for (at, line) in source.lines().enumerate() {
            if !is_code(line) {
                continue;
            }
            // ⛔ BOTH SPELLINGS OF THE SAME REACH-IN: the direct prefix, and the crate rename that
            // would hide it. A `use` alias INSIDE the seam's vocabulary (`SuperDsc as Sdsc`) is fine
            // and is not what this matches — only renaming the CRATE is.
            if line.contains(FORBIDDEN) || line.contains(CRATE_ALIAS) {
                offenders.push(format!(
                    "{}:{}: {}",
                    file.display(),
                    at + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a target crate reached into the scheduler's interior. `{FORBIDDEN}` is off limits to \
         `crates/targets/`; name `{SEAM}` instead, and if the item you want is not re-exported \
         there, ADD IT TO THE SEAM rather than reaching past it:\n  {}",
        offenders.join("\n  ")
    );

    // ⛔ THE NEGATIVE CONTROL, so a walk that found NOTHING cannot pass by finding nothing at all: a
    // typo in the path, an unreadable directory or a broken recursion would make the assertion above
    // vacuous. `superdsc_to_l3_sdsc.rs` is the file the rule exists for and it must be in the walk.
    assert!(
        files.iter().any(|file| file.ends_with("superdsc_to_l3_sdsc.rs")),
        "the walk must reach the conversion this rule was written for — it found {} file(s) under \
         {}, which means the traversal, not the tree, is what is empty",
        files.len(),
        targets.display()
    );
    // ⭐ AND THE SEAM IS ACTUALLY IN USE, which is what says the count above is zero because the
    // reach-in was ROUTED and not merely deleted along with the code that needed it.
    let conversion = files
        .iter()
        .find(|file| file.ends_with("superdsc_to_l3_sdsc.rs"))
        .expect("just asserted present");
    let source = std::fs::read_to_string(conversion).expect("the conversion is readable");
    assert!(
        source.lines().any(|line| is_code(line) && line.contains(SEAM)),
        "`{}` must name `{SEAM}` — it is the crate's SuperDSC conversion and it has to speak the \
         scheduler's vocabulary through the seam",
        conversion.display()
    );
}

/// ⭐⭐⭐ THE SEAM RE-EXPORTS **NAMED ITEMS**, NEVER A GLOB — without this,
/// [`no_target_crate_names_the_schedulers_interior`] is theatre.
///
/// ⛔⛔ THE EVASION IT CLOSES, EXACTLY. If `sdsc.rs` said `pub use crate::schedule::*;` — or even
/// `pub use crate::schedule::l3::dsc::*;` — then every internal a target ever reached for would be
/// reachable as `deeptools::sdsc::<anything>`, the prefix test would pass, and the layering would be
/// **unchanged**. The boundary is only a boundary while somebody has to WRITE DOWN each item that
/// crosses it, which is what makes adding one a visible line on a diff.
///
/// ⛔ AND A GLOB WOULD ALSO SILENTLY WIDEN OVER TIME: a new `pub` item in `l3::dsc` would appear on
/// the seam with nobody deciding it should. A named list cannot do that.
///
/// ⛔ NOT A TAUTOLOGY: the negative control asserts the file is actually READ and actually holds the
/// re-exports, so a moved or renamed seam fails here instead of passing on an empty read.
#[test]
fn the_seam_re_exports_named_items_and_never_a_glob() {
    let seam = Path::new(env!("CARGO_MANIFEST_DIR")).join(SEAM_FILE);
    let source = std::fs::read_to_string(&seam)
        .unwrap_or_else(|e| panic!("the seam must be readable at {}: {e}", seam.display()));

    let rows: Vec<&str> = source
        .lines()
        .filter(|line| is_code(line) && line.trim_start().starts_with("pub use "))
        .collect();
    // ⭐ THE CONTROL FIRST: the file really was read and really is the seam.
    assert!(
        rows.len() >= 5,
        "the seam must re-export the scheduler's vocabulary by name — found {} `pub use` row(s) in \
         {}, which means the path, not the policy, is what is empty",
        rows.len(),
        seam.display()
    );
    for row in &rows {
        assert!(
            !row.contains('*'),
            "the seam must name every item it re-exports; a glob makes the whole `schedule` tree \
             reachable as `{SEAM}::*` and leaves the layering unchanged: {}",
            row.trim()
        );
    }
    // ⛔ AND NOTHING MAY BE RE-EXPORTED FROM OUTSIDE deeptools' OWN TREE — a seam that re-exported a
    // scratchy type would make the dependency edge point backwards, which is the one direction
    // `deeptools/Cargo.toml` forbids outright (*"NO SCRATCHY DEPENDENCY, EVER"*).
    for row in &rows {
        assert!(
            row.contains("crate::") || row.contains("sys_arch_spec::"),
            "every seam row re-exports from `crate::` or the shared arch spec: {}",
            row.trim()
        );
    }
}
