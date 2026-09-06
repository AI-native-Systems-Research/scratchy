// SPDX-License-Identifier: Apache-2.0
//! THE ISA TABLES, BYTE FOR BYTE — the oracle a restructuring is allowed to run against.
//!
//! # 🛑 THIS TEST EXISTS TO BE WRITTEN *BEFORE* THE CHANGE, NOT AFTER
//!
//! ⛔⛔ `fields.rs` STATES 1,076 FIELD DEFINITIONS AND 440 OPCODE DEFINITIONS LONGHAND, AND ONLY 390
//! OF THE FIELD ROWS ARE DISTINCT. Deduplicating that is a mechanical change over ~11,500 lines, and a
//! mechanical change over a table of bit positions is exactly where a single transposed digit produces
//! a program that assembles cleanly and computes the wrong thing. No compiler catches it and no
//! reviewer reads 1,076 rows.
//!
//! ⭐⭐ SO THE PRE-CHANGE TABLE IS THE AUTHORITY FOR THE POST-CHANGE TABLE. This snapshot is generated
//! from the tables as they stand, committed, and then asserted against after every step. A dedup that
//! changes one bit fails here, and the failure names the component.
//!
//! ⛔ IT IS NOT A CLAIM THE TABLES ARE *CORRECT*. They were ported from `isa.cpp` and validated
//! against the senulator elsewhere; this only says a restructuring did not alter them. Those are
//! different claims and conflating them is how a refactor inherits a reputation it has not earned.
//!
//! ⭐ TOTAL BY CONSTRUCTION: it walks [`Comp::ALL`], so a component cannot be left out of the
//! comparison, and it covers both tables of all nine.
//!
//! # Regenerating
//!
//! Deliberately awkward — `SYS_ARCH_SPEC_BLESS=1 cargo test -p sys-arch-spec` rewrites the snapshot.
//! It must only be run when the tables are MEANT to change, and the diff must then be read.

use sys_arch_spec::fields::{Comp, TABLE_ARCH};

/// Every table of every component, rendered in a fixed order.
///
/// ⛔ ONE LINE PER ROW, so a diff points at the row that moved rather than at a reflowed block.
fn render() -> String {
    let mut out = format!("arch = {TABLE_ARCH:?}\n");
    for comp in Comp::ALL {
        let opcodes = comp.opcodes();
        let fields = comp.fields();
        out.push_str(&format!(
            "\n=== {comp:?}: {} opcodes, {} fields\n",
            opcodes.len(),
            fields.len()
        ));
        for op in opcodes {
            out.push_str(&format!("op {op:?}\n"));
        }
        for field in fields {
            out.push_str(&format!("fd {field:?}\n"));
        }
    }
    out
}

/// Which snapshot this build compares against — the arch is a feature, so it is a different table.
fn snapshot_path() -> std::path::PathBuf {
    let arch = if cfg!(feature = "arch-rcudd1a") {
        "rcudd1a"
    } else {
        "sen1p5"
    };
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/snapshots")
        .join(format!("{arch}.txt"))
}

/// 🎯 THE TABLES ARE WHAT THEY WERE.
#[test]
fn the_tables_are_unchanged() {
    let rendered = render();
    let path = snapshot_path();

    if std::env::var_os("SYS_ARCH_SPEC_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().expect("snapshot dir")).expect("create snapshot dir");
        std::fs::write(&path, &rendered).expect("write snapshot");
        // ⛔ NOT SILENT. A blessed run that printed nothing would let a real regression be blessed by
        // a caller who meant to run the test.
        println!("BLESSED {} ({} bytes)", path.display(), rendered.len());
        return;
    }

    let expected = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "no snapshot at {} ({error}) — generate it with SYS_ARCH_SPEC_BLESS=1 before changing \
             the tables, not after",
            path.display()
        )
    });

    if rendered == expected {
        return;
    }

    // ⭐ NAME THE FIRST DIVERGENCE AND ITS LINE. "The tables differ" is not a diagnosis anyone can act
    // on across 11,500 rows.
    let (mut got, mut want) = (rendered.lines(), expected.lines());
    let mut line = 0usize;
    loop {
        line += 1;
        match (got.next(), want.next()) {
            (Some(g), Some(w)) if g == w => {}
            (g, w) => panic!(
                "the ISA tables changed at line {line}\n  expected: {}\n  got:      {}",
                w.unwrap_or("<end of snapshot>"),
                g.unwrap_or("<end of tables>")
            ),
        }
    }
}
