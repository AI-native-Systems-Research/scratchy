// SPDX-License-Identifier: Apache-2.0
//! ⚠️ TEMPORARY bake-time instrumentation — per-pass counters for the scoped repro.
//! NOT FOR COMMIT: strip this module and every `crate::pc::` call site before the
//! signed landing (search `pc::` — every hit is instrumentation).
//!
//! One process-wide table of atomic rows. Every bake appends one line per bundle to
//! `/tmp/bt-passcounters.log`: per-bundle identity plus the CUMULATIVE table, so a
//! pass whose cost grows superlinearly across a bake is visible from the file alone
//! (each line's cumulative micros vs the bundle index). Rows are Relaxed atomics —
//! the bake is rayon-fanned, so every counter must be thread-safe.

use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

struct Row {
    name: &'static str,
    nanos: AtomicU64,
    calls: AtomicU64,
    items: AtomicU64,
}

const fn row(name: &'static str) -> Row {
    Row {
        name,
        nanos: AtomicU64::new(0),
        calls: AtomicU64::new(0),
        items: AtomicU64::new(0),
    }
}

static ROWS: [Row; 20] = [
    row("producer"), // lower_subtile_tape_to_ktir, whole producer walk
    row("foldscan"), // the weight-fold recognizers (full-graph scans)
    row("fp"),       // bundle_fp — fingerprint hashing
    row("audit"),    // audit_op_work
    row("inner"),    // emit_bundle_inner
    row("mint"),     // door::lower over the bundle's programs
    row("mint_attn"),
    row("mint_rope"),
    row("mint_rms"),
    row("mint_matmul"),
    row("mint_route"),
    row("mint_expert"),
    row("mint_ew"),
    row("guards"),   // the four post-lowering guard walks
    row("dump"),     // write_rendered_dxp_input + bundle_id.json (I/O only — render is its own row)
    row("launch"),   // launch_index
    row("render"),   // render_dxp_input — ONE pass, shared by the dump and the staging loop
    row("stage"),    // stage/reserve/write/submit per group
    row("cachit"),   // disk-cache hits (supply_cached true)
    row("registry"), // emitted() inserts: calls = bundles, items = ops retained
];

fn idx(name: &str) -> Option<usize> {
    ROWS.iter().position(|r| r.name == name)
}

/// Time one pass; elapsed nanos land on `name`'s row when the guard drops.
pub struct Pc {
    row: usize,
    t: Instant,
}

impl Pc {
    pub fn start(name: &str) -> Pc {
        Pc {
            row: idx(name).unwrap_or(0),
            t: Instant::now(),
        }
    }
}

impl Drop for Pc {
    fn drop(&mut self) {
        ROWS[self.row]
            .nanos
            .fetch_add(self.t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        ROWS[self.row].calls.fetch_add(1, Ordering::Relaxed);
    }
}

/// Add `n` to `name`'s item count (nodes scanned, descriptors minted, ...).
pub fn count(name: &str, n: u64) {
    if let Some(i) = idx(name) {
        ROWS[i].items.fetch_add(n, Ordering::Relaxed);
    }
}

/// The mint class for an op name — coarse buckets, enough to attribute the door's cost.
pub fn mint_class(op_name: &str) -> &'static str {
    if op_name.starts_with("attn") {
        "mint_attn"
    } else if op_name.starts_with("rope") {
        "mint_rope"
    } else if op_name.contains("rms") {
        "mint_rms"
    } else if op_name.starts_with("matmul") || op_name.starts_with("lmlast") {
        "mint_matmul"
    } else if op_name.starts_with("route") {
        "mint_route"
    } else if op_name.starts_with("expert") {
        "mint_expert"
    } else {
        "mint_ew"
    }
}

/// One line per bundle: identity plus the cumulative table (`name:µs/calls/items`).
pub fn bundle_line(fp: &str, ops: usize, dsc: usize, staged_bytes: u64) {
    static LOCK: Mutex<()> = Mutex::new(());
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut line = String::with_capacity(640);
    line.push_str(&format!(
        "PC fp={fp} ops={ops} dsc={dsc} staged={staged_bytes}"
    ));
    for r in &ROWS {
        let ns = r.nanos.load(Ordering::Relaxed);
        let calls = r.calls.load(Ordering::Relaxed);
        let items = r.items.load(Ordering::Relaxed);
        if ns == 0 && calls == 0 && items == 0 {
            continue;
        }
        line.push_str(&format!(" {}:{}", r.name, ns / 1000));
        if calls > 0 {
            line.push_str(&format!("/{calls}"));
        }
        if items > 0 {
            line.push_str(&format!("/{items}"));
        }
    }
    line.push('\n');
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/bt-passcounters.log")
    {
        let _ = f.write_all(line.as_bytes());
    }
}
