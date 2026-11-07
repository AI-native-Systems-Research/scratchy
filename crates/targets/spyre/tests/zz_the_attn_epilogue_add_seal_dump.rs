// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐⭐⭐ THE 8b ATTN-EPILOGUE ADD'S NUMERIC SEAL — one card cell for the #1112 mis-addressing
//! finding, dumped for an on-card `bundle_run_dev` run at TWO trees: MAIN 940736b557 (the shipped
//! door) and the fix line (the two-form door). Test tooling: nothing here ships, and the only
//! shipped code it calls is the elementwise door the bake calls.
//!
//! ## The claim this cell seals (pinned by the door-level probe, eightb-probe-20261107070000)
//! granite-3.1-8b fp16's attn_s8 epilogue adds (`add_o1187..1204`) pair head and tail row bands of
//! a `[1024, 128]` parent per 64-wide column half. The canonical instance is `add_o1190`
//! (rows 512, parent `t1188` `[1024, 128]`, output `t1190` `[512, 128]`), whose `c64` piece's HEAD
//! input — rows `[0, 512)` of columns `[64, 128)` — is bound by MAIN at element 32768
//! (`col_of(512, 128, 64)`: the OP's rows stride the column planes) and by the fix at element
//! 65536 (`rc_of(1024, 128, 0, 64)`: the PARENT's rows). 32768 is `tail[c0]`'s base, so MAIN's
//! `c64` piece computes `tail[c0] + tail[c64]` where the program states `head[c64] + tail[c64]` —
//! 337 moved instances across the ten adds, every one the FIRST input binding.
//!
//! ## What this dump states
//! The door call itself, at the probe-measured regions: for each 64-wide column half `c`, an
//! `Elementwise::Add` over `head = window(0, c)`, `tail = window(1, c)` and the output piece —
//! the exact regions the whole-function fold's own `window(j)`/`in_column` presentation builds
//! (`r_cover` the window's span, `c_cover` the stick's span). Both pieces share one output
//! tensor, exactly as the front end's column tiling mints them; the ops are both named
//! `add_o1190`, exactly as the door names them.
//!
//! The corners the two sides AGREE on are ASSERTED here (they are the probe's measured facts):
//! piece `c0` binds head at 0, tail at 32768, out at 0; piece `c64` binds tail at 98304 and out
//! at 32768. The `c64` HEAD corner is THE DISCRIMINATOR — recorded into `cell.json`, never
//! asserted: 32768 at main, 65536 at the fix. A tree that refuses the op records the refusal
//! verbatim instead (the cell is then NOT-RUN, which is itself a finding: main emits this op
//! today — the probe measured its bundle).
//!
//! ## The bar (checked by the runner, not here)
//! The read-back buffer holds the card image of
//! ```text
//!   out[r][c] = parent[r][c] + parent[512 + r][c]      for r in 0..512, c in 0..128
//! ```
//! — head plus tail, the windows the program states, never our opspec — with the parent filled
//! by ELEMENT-INDEX in two runs (row then column, both fp16-exact), so the first wrong element
//! is reported with its position. Expected: MAIN is wrong on exactly the `c64` lanes (it reads
//! `tail[c0]` as the head: `parent[512+r][c-64] + parent[512+r][c]`), right on every `c0` lane;
//! the fix is right on every lane.
//!
//! ## The fence protocol
//! One cell alone (the runner holds `/work/.spyre-bake.lock`), abort on the card's fence
//! signatures — the page-gather sweep's own protocol.
//!
//! ```bash
//! EPILOGUE_ADD_SEAL_ROOT=<dir> EPILOGUE_ADD_SEAL_SHA=<sha> \
//!   cargo test --release -p scratchy-target-spyre \
//!   --test zz_the_attn_epilogue_add_seal_dump -- --ignored --nocapture
//! ```
//!
//! Then on a pod, once per tree (a warm tree at the SAME sha, with
//! `target/release/examples/bundle_run_dev` built from it):
//! ```bash
//! tests/epilogue_add_seal_run.sh <dir>/<cell> <warm-tree-at-this-sha> <verdict-log>
//! ```

use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::emit::lower_ktir_to_superdsc::{Region, elementwise};
use ktir_superdsc::ktir_node::Elementwise;
use ktir_superdsc::place::{PlaceId, act_name};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;

/// The canonical instance's own tids, verbatim — the door mints `add_o{out.tid}`, so the emitted
/// ops are named `add_o1190` exactly as the bake names them.
const PARENT_TID: u32 = 1188;
const OUT_TID: u32 = 1190;
/// The parent `t1188`'s view: `[1024, 128]`.
const PARENT_ROWS: u32 = 1024;
/// The op's rows — the output `t1190`'s own `[512, 128]`.
const ROWS: u32 = 512;
const COLS: u32 = 128;
const STICK: u32 = 64;

/// Window `j` (row band `j·ROWS`), column stick `c` — the fold presentation's own `window(j)`
/// then `in_column(·, c)`: the window's span as the cover, the stick's span as the column cover.
fn band(j: u32, c: u32) -> Region {
    Region {
        tid: PARENT_TID,
        v_rows: PARENT_ROWS,
        v_cols: COLS,
        r_start: j * ROWS,
        r_len: ROWS,
        r_cover: (j * ROWS, (j + 1) * ROWS),
        c_start: c * STICK,
        c_len: STICK,
        c_cover: (c * STICK, (c + 1) * STICK),
        is_out: false,
        is_fp8: false,
        band: None,
    }
}

/// The output piece at column stick `c` — a whole-view `[512, 128]` output, windowed to its own
/// 64-wide half, exactly as the front end's column tiling states each piece.
fn out_piece(c: u32) -> Region {
    Region {
        tid: OUT_TID,
        v_rows: ROWS,
        v_cols: COLS,
        r_start: 0,
        r_len: ROWS,
        r_cover: (0, ROWS),
        c_start: c * STICK,
        c_len: STICK,
        c_cover: (c * STICK, (c + 1) * STICK),
        is_out: true,
        is_fp8: false,
        band: None,
    }
}

/// One binding as the cell's record states it — the door's own emission, read back off the
/// `EmittedOp` the door returned (never re-derived from the regions).
#[derive(serde::Serialize)]
struct Binding {
    buffer: String,
    is_input: bool,
    offset_elems: u32,
}

/// One piece's bindings, in emission order.
#[derive(serde::Serialize)]
struct Piece {
    piece: String,
    bindings: Vec<Binding>,
}

/// The cell's own manifest — everything the runner's fills and decoder need, stated once.
#[derive(serde::Serialize)]
struct Cell {
    /// The commit this dump ran at — `EPILOGUE_ADD_SEAL_SHA`, required at dump time.
    sha: String,
    parent_tid: u32,
    out_tid: u32,
    parent_rows: u32,
    rows: u32,
    cols: u32,
    stick: u32,
    /// ⭐ THE DISCRIMINATOR, recorded and never asserted: the `c64` piece's HEAD input corner in
    /// elements. 32768 at main (`col_of(op_rows, …)`), 65536 at the fix (`rc_of(parent_rows, …)`)
    /// — the probe-measured facts this cell carries onto the card.
    head_c64_corner_elems: u32,
    /// The refusal, verbatim, for a tree that will not lower the op. `None` ⇒ the bundle was
    /// written and the cell is runnable.
    missing: Option<String>,
    pieces: Vec<Piece>,
}

/// The launch map: the parent and the output as activations of one segment, sized at their whole
/// views (the handles the door declares: `rb(name, v_rows, v_cols)`).
fn layout_placing() -> BundleLayout {
    let parent_bytes = (PARENT_ROWS * COLS * 2) as u64;
    let out_bytes = (ROWS * COLS * 2) as u64;
    let mut l = BundleLayout::default();
    let mut off = 0u64;
    for (tid, size) in [(PARENT_TID, parent_bytes), (OUT_TID, out_bytes)] {
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                offset: off,
                size,
            },
        );
        off += size;
    }
    l.segment_bytes[3] = off + 4096; // + the H2D window pad
    l
}

/// The input corners of one emitted piece, sorted — read off the door's own `arg_bindings`.
fn input_corners(op: &EmittedOp, buffer: &str) -> Vec<u32> {
    let mut cs: Vec<u32> = op
        .arg_bindings
        .iter()
        .filter(|b| b.is_input && b.buffer == buffer)
        .map(|b| b.offset_elems)
        .collect();
    cs.sort_unstable();
    cs
}

/// The output corner of one emitted piece.
fn out_corner(op: &EmittedOp, buffer: &str) -> Option<u32> {
    op.arg_bindings
        .iter()
        .find(|b| !b.is_input && b.buffer == buffer)
        .map(|b| b.offset_elems)
}

/// Dump the cell. `group_0/` ([`superdsc::write_dxp_input`]), `placements.json` (the launch map,
/// the `bake.rs::write_placements` format, the output row labeled `Logits` so the runner reads it
/// back) and `cell.json` (this cell's own statement, for the runner's fills and decoder).
#[test]
#[ignore]
fn dump_attn_epilogue_add_seal() {
    let Some(root) = std::env::var_os("EPILOGUE_ADD_SEAL_ROOT") else {
        panic!("EPILOGUE_ADD_SEAL_ROOT=<dir> names where the cell is written");
    };
    let sha = std::env::var("EPILOGUE_ADD_SEAL_SHA").unwrap_or_else(|_| {
        panic!("EPILOGUE_ADD_SEAL_SHA=<sha> stamps the commit the dump runs at — a dump refuses to run unstamped")
    });
    let dir = std::path::Path::new(&root).join("epilogue-add-o1190");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let l = layout_placing();

    // ⭐ BOTH PIECES THROUGH THE SHIPPED DOOR — the same call the whole-function fold makes
    // (`elementwise(f.name, combine, &[lhs, rhs, dst], &[], sym, Some(layout))`), at the
    // probe-measured regions. A panic in the mint is a missing lowering too: recorded verbatim,
    // never unwound past (the page-gather sweep's own protocol).
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let walked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut ops: Vec<EmittedOp> = Vec::new();
        let mut sym = 0i64;
        for c in 0..(COLS / STICK) {
            ops.extend(elementwise(
                "attn_s8_epilogue_seal",
                Elementwise::Add,
                &[band(0, c), band(1, c), out_piece(c)],
                &[],
                &mut sym,
                Some(&l),
            )?);
        }
        Ok(ops)
    }));
    std::panic::set_hook(prev_hook);

    let (ops, missing): (Vec<EmittedOp>, Option<String>) = match walked {
        Ok(Ok(ops)) => (ops, None),
        Ok(Err(e)) => (Vec::new(), Some(e.message)),
        Err(payload) => (
            Vec::new(),
            Some(format!(
                "PANIC: {}",
                payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "an unprintable panic".into())
            )),
        ),
    };

    let mut head_c64 = 0u32;
    let mut pieces = Vec::new();
    if missing.is_none() {
        assert_eq!(
            ops.len(),
            (COLS / STICK) as usize,
            "the door must emit one op per column piece"
        );
        for (c, op) in ops.iter().enumerate() {
            assert_eq!(
                op.op_name, "add_o1190",
                "the door names the op for its output tensor — the bake's own name"
            );
            let parent = act_name(PARENT_TID);
            let out = act_name(OUT_TID);
            let ins = input_corners(op, &parent);
            let outs = out_corner(op, &out);
            // ⭐ THE PROBE-MEASURED FACTS BOTH SIDES AGREE ON, asserted so a tree that drifts on
            // the control corners fails at the dump rather than handing the runner a different
            // cell than the one the finding is about.
            assert_eq!(
                ins.len(),
                2,
                "the add binds its parent twice — head and tail"
            );
            assert_eq!(
                outs,
                Some(c * STICK * ROWS),
                "add_o1190: piece {c}'s output corner must be the whole-rows column corner \
                 col_of(512, 128, {}) = {}",
                c * STICK,
                c * STICK * ROWS
            );
            assert_eq!(
                ins[1],
                (PARENT_ROWS - ROWS) * STICK + c * STICK * PARENT_ROWS,
                "add_o1190: piece {c}'s TAIL input must sit at the parent's own pitch \
                 rc_of(1024, 128, 512, {}) — the corner both doors agree on",
                c * STICK
            );
            if c == 0 {
                assert_eq!(
                    ins[0], 0,
                    "add_o1190: piece 0's HEAD input sits at the parent's base on every door"
                );
            } else {
                // ⭐ THE DISCRIMINATOR — recorded, never asserted: 32768 at main, 65536 at the
                // fix. This is the one number the finding is about; the card decides it.
                head_c64 = ins[0];
            }
            pieces.push(Piece {
                piece: format!("c{}", c * STICK),
                bindings: op
                    .arg_bindings
                    .iter()
                    .map(|b| Binding {
                        buffer: b.buffer.clone(),
                        is_input: b.is_input,
                        offset_elems: b.offset_elems,
                    })
                    .collect(),
            });
        }
    }

    let cell = Cell {
        sha: sha.clone(),
        parent_tid: PARENT_TID,
        out_tid: OUT_TID,
        parent_rows: PARENT_ROWS,
        rows: ROWS,
        cols: COLS,
        stick: STICK,
        head_c64_corner_elems: head_c64,
        missing: missing.clone(),
        pieces,
    };
    let runnable = cell.missing.is_none();
    if runnable {
        superdsc::write_dxp_input(&dir, &ops, superdsc::FoldGrouping::Split)
            .unwrap_or_else(|e| panic!("{}: write_dxp_input: {e}", dir.display()));
    }
    write_placements(&dir, &l);
    std::fs::write(dir.join("cell.json"), serde_json::to_string_pretty(&cell).unwrap())
        .unwrap();
    eprintln!(
        "epilogue-add seal: head_c64_corner={head_c64} elems, {}",
        if runnable {
            "bundle written"
        } else {
            "MISSING LOWERING — see cell.json"
        }
    );
    assert!(
        runnable,
        "the door must lower the epilogue add — both trees emit this op today (the probe \
         measured both bundles); a refusal here is a finding, recorded in cell.json"
    );
}

/// The launch map, the `bake.rs::write_placements` format: segment sizes, every placement (the
/// OUTPUT row labeled `Logits`, the runner's read-back marker), an empty synth list.
fn write_placements(dir: &std::path::Path, l: &BundleLayout) {
    let synth = l.synth.borrow();
    let mut seg_bytes = l.segment_bytes;
    let intermediate = SegRole::Intermediate.segment();
    seg_bytes[intermediate] = seg_bytes[intermediate].max(synth.next);
    let mut s = String::new();
    s.push_str("{\n  \"segment_bytes\": [");
    for (i, b) in seg_bytes.iter().enumerate() {
        s.push_str(&format!("{}{}", if i == 0 { "" } else { ", " }, b));
    }
    s.push_str("],\n  \"places\": [\n");
    let rows: Vec<String> = l
        .placements
        .iter()
        .map(|(tid, p)| {
            format!(
                "    {{\"tid\": {}, \"role\": \"{}\", \"segment\": {}, \"bank\": {}, \"offset\": \
                 {}, \"size\": {}}}",
                tid,
                // The read-back tensor is the producer's own statement: the runner reads back
                // whichever place carries the `Logits` role, and this cell's output is it.
                if *tid == OUT_TID { "Logits" } else { "Activation" },
                p.segment,
                p.bank,
                p.offset,
                p.size
            )
        })
        .collect();
    s.push_str(&rows.join(",\n"));
    s.push_str("\n  ],\n  \"synth\": [\n  ],\n  \"scalarmul_scales\": [\n  ]\n}\n");
    std::fs::write(dir.join("placements.json"), s).unwrap();
}
