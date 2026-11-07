// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐⭐⭐ THE ATTN-EPILOGUE ADD READS ITS HEAD BAND AT THE PARENT'S OWN PITCH — the
//! #1112 mis-addressing finding's plain CI pin, replacing the card seal cell
//! (`zz_the_attn_epilogue_add_seal_dump.rs`, deleted with this change; the card numbers
//! stay in #1126 and the PR body as evidence). No card, no env vars, no #[ignore]:
//! this is the elementwise door the bake calls, driven at the probe-measured regions of
//! the canonical granite-3.1-8b fp16 instance `add_o1190`, and it asserts the ONE
//! binding the finding is about on the typed field (the input binding's element
//! offset), never message prose.
//!
//! ## The law
//! granite-3.1-8b fp16's attn_s8 epilogue adds (`add_o1187..1204`) pair head and tail
//! row bands of a `[1024, 128]` parent per 64-wide column half. The canonical instance
//! is `add_o1190` (rows 512, parent `t1188` `[1024, 128]`, output `t1190` `[512, 128]`),
//! whose `c64` piece's HEAD input — rows `[0, 512)` of columns `[64, 128)` — must bind
//! at element 65536: `rc_of(1024, 128, 0, 64)`, the PARENT's rows-stride pitch. The
//! pre-fix door bound it at 32768 (`col_of(512, 128, 64)`: the OP's rows striding the
//! column planes), which is `tail[c0]`'s base — so the `c64` piece computed
//! `tail[c0] + tail[c64]` where the program states `head[c64] + tail[c64]`; 337 moved
//! instances across the ten adds, every one the FIRST input binding
//! (eightb-probe-20261107070000, card-sealed in #1126: main wrong on exactly the `c64`
//! lanes, the fix right on every lane).
//!
//! The corners both sides of the fix AGREE on are asserted here as controls, exactly as
//! the seal cell asserted them at dump time — so a tree that drifts on a control corner
//! fails here rather than handing a green run for a different geometry than the one the
//! finding is about.

use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::emit::lower_ktir_to_superdsc::{Region, elementwise};
use ktir_superdsc::ktir_node::Elementwise;
use ktir_superdsc::place::{PlaceId, act_name};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};

/// The canonical instance's own tids, verbatim — the door mints `add_o{out.tid}`, so the
/// emitted ops are named `add_o1190` exactly as the bake names them.
const PARENT_TID: u32 = 1188;
const OUT_TID: u32 = 1190;
/// The parent `t1188`'s view: `[1024, 128]`.
const PARENT_ROWS: u32 = 1024;
/// The op's rows — the output `t1190`'s own `[512, 128]`.
const ROWS: u32 = 512;
const COLS: u32 = 128;
const STICK: u32 = 64;

/// Window `j` (row band `j·ROWS`), column stick `c` — the whole-function fold's own
/// `window(j)` then `in_column(·, c)` presentation: the window's span as the cover, the
/// stick's span as the column cover.
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

/// The output piece at column stick `c` — a whole-view `[512, 128]` output, windowed to
/// its own 64-wide half, exactly as the front end's column tiling states each piece.
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

/// The launch map: the parent and the output as activations of one segment, sized at
/// their whole views — the handles the door declares (`rb(name, v_rows, v_cols)`).
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
    l.segment_bytes[3] = off;
    l
}

/// The input corners of one emitted piece, sorted — read off the door's own
/// `arg_bindings`, never re-derived from the regions.
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

/// BOTH PIECES THROUGH THE SHIPPED DOOR — the same call the whole-function fold makes
/// (`elementwise(f.name, combine, &[lhs, rhs, dst], &[], sym, Some(layout))`), at the
/// probe-measured regions of `add_o1190`.
#[test]
fn the_epilogue_add_reads_its_head_band_at_the_parents_pitch() {
    let l = layout_placing();
    let mut ops: Vec<EmittedOp> = Vec::new();
    let mut sym = 0i64;
    for c in 0..(COLS / STICK) {
        ops.extend(
            elementwise(
                "attn_s8_epilogue_add_o1190",
                Elementwise::Add,
                &[band(0, c), band(1, c), out_piece(c)],
                &[],
                &mut sym,
                Some(&l),
            )
            .unwrap_or_else(|e| panic!("the door must lower the epilogue add: {}", e.message)),
        );
    }
    assert_eq!(
        ops.len(),
        (COLS / STICK) as usize,
        "the door must emit one op per column piece"
    );
    let parent = act_name(PARENT_TID);
    let out = act_name(OUT_TID);
    for (i, op) in ops.iter().enumerate() {
        let c = i as u32;
        assert_eq!(
            op.op_name, "add_o1190",
            "the door names the op for its output tensor — the bake's own name"
        );
        let ins = input_corners(op, &parent);
        let outs = out_corner(op, &out);
        // ⭐ THE PROBE-MEASURED CONTROL CORNERS BOTH SIDES OF THE FIX AGREE ON — asserted
        // so a tree that drifts on them fails here rather than pinning a different
        // geometry than the one the finding is about.
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
            // ⭐ THE FINDING'S ONE NUMBER: the `c64` piece's HEAD input must sit at the
            // PARENT's rows-stride pitch — `rc_of(1024, 128, 0, 64)` = 65536. The
            // pre-fix door bound the op's own rows stride (`col_of(512, 128, 64)` =
            // 32768), which is `tail[c0]`'s base: the piece read the tail band twice.
            assert_eq!(
                ins[0],
                c * STICK * PARENT_ROWS,
                "add_o1190: piece {c}'s HEAD input must sit at the parent's own pitch \
                 rc_of({PARENT_ROWS}, {COLS}, 0, {}) = {} — the window's rows stride the \
                 PARENT's planes, not the op's own {} rows",
                c * STICK,
                c * STICK * PARENT_ROWS,
                ROWS
            );
        }
    }
}
