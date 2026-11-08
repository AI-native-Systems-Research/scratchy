// SPDX-License-Identifier: Apache-2.0

//! ⭐ THE PAGE-GATHER CARD SWEEP AT THE ONE `index_select` GATHER — the D1 proof legs and the
//! new-territory cells of the `ladder-gather` branch, dumped for an on-card `bundle_run_dev` run.
//! Test tooling: nothing here ships, and the only shipped code it calls is the walk the bake calls.
//!
//! ## What this dump states
//! Each cell is a minimal KTIR program over four parameters — the pool, the index, and the two
//! operands of a terminal `arith.addf` — built from the op kinds the walk reads, exactly as the
//! splice builds a page read: the pool placed ROW-OUTERMOST (`allocate_row_outermost`, the paged
//! KV cache's own allocation law), an indirect tile over the pool and the index, its load stating
//! `[entries, slab]`, and a `tensor.reshape` whose `to` frame the walk's gather arm reads. The
//! terminal deliberately does NOT read the landing (a reader with extent laws of its own would
//! refuse cells the gather itself lowers); the walk lowers the reshape's gather chain whatever
//! reads it.
//!
//! The `to` frame is the cell's own statement, and the walk's OWN predicates pick the arm:
//! * `Blocks` — rank-3 `[E, rows, cols]` (slab = rows·cols): the WholePages form, one block an
//!   entry, the `[mb, y:1, x, out]` respelling. THE D1 CELLS: entry bytes 32–128 KiB — the
//!   deleted 64 KiB sub-row bound's both sides, 128 KiB INCLUDED.
//! * `Blocked` — rank-4 `[E, blocks, rows, cols]`: the multi-block landing (`batch_blocks`).
//! * `Flat` — rank-2 `[T, w]`: `w` one stick (the landing IS the frame) and `w` several sticks
//!   over several entries (the rank-2 RESTICKIFY: a row-major landing plus one whole stick
//!   permute — the value test the audit's N2 asks for, at the card executor leg).
//!
//! Entries past one op's index stick (`ENTRIES_PER_OP`, 32) are stated too: the walk chunks them.
//!
//! ## The bar (checked by the runner, not here)
//! FRAME-EXACT against a host model computed FROM THE OP'S DEFINITION, never from our opspec:
//! the read-back buffer holds the card image of `pool.index_select(0, ids).reshape(<frame>)` —
//! `index_select` and `reshape` as the reference states them — AT THE FRAME'S OWN RANK. Rank-2
//! `[T, w]`: the one stick-column law (torch-spyre `tensors_and_layouts.md` — a device layout
//! is row-major with the stick dim last, so a host `[T, w]` sits as `[w/64, T, 64]`). Rank-3/4:
//! OUR INTERMEDIATE'S OWN CONTRACT with its readers — the entry axis outermost, each
//! `[rows, cols]` block behind it carrying the rank-2 law, blocks one after another, the frame
//! `whole_function` registers as `batch_blocks` for its consumers (the batched transpose reads
//! a block at its base). NOT torch-spyre's default rank-3 layout (its
//! `tensors_and_layouts.md` example `[5,100,150]` → `[100,3,5,64]` is neither this nor the
//! folded image), and not a hardware requirement: the end-to-end proof of the contract is the
//! D3 bundle gate plus card chat on 4 models. The fills name the source element — (page, row) in two runs, col in the
//! third — so the first wrong element is reported with its position.
//!
//! ## The fence protocol
//! R>1 cells run ONE ALONE, and the runner ABORTS on a fence (a `RAS::PCI::BusFence` /
//! `PfMSIMonitor` line in any run log): the r32-c128-e2-b0 fence of the main sweep is unexplained,
//! so a fence here stops the sweep rather than logging past it.
//!
//! ## The cells (hand-stated, the audit's own list)
//! D1 WholePages: `r128` × `c{128,256,384,512}` × e2 — 32, 64, 96, 128 KiB an entry (128 KiB is
//! the deleted bound's proof cell) — plus `r256 × c{128,256}`, `r64 × c1024` (128 KiB at a third
//! geometry), and the one-entry control at 128 KiB. New territory: entries 33/64/96 past the
//! index stick; the rank-2 flats at `w` = 64; the rank-2 restickify at `w` = 128/256; and two
//! rank-4 multi-block cells.
//!
//! ```bash
//! PAGE_GATHER_SWEEP_ROOT=<dir> PAGE_GATHER_SWEEP_SHA=<sha> \
//!   cargo test --release -p scratchy-target-spyre \
//!   --test zz_page_gather_sweep -- --ignored --nocapture
//! ```
//!
//! Then on a pod (the fence protocol: one cell alone, abort on fence):
//! ```bash
//! tests/page_gather_sweep_run.sh <dir> <warm-tree-at-this-sha> <verdict-log>
//! ```

use ktir_core::affine::{AffineExpr, AffineMap};
use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;
use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::emit::whole_function::lower_function;
use ktir_superdsc::ktir_node::{BufferId, KtirNode, Program};
use ktir_superdsc::place::{PlaceId, act_name};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;

/// The stick, the one constant every frame law below is stated over.
const STICK: u64 = 64;

/// The frame a cell states — the reshape's `to`, the walk's own arm picker.
#[derive(Clone, Copy)]
enum Frame {
    /// Rank-3 `[E, rows, cols]`, slab = rows·cols — the WholePages form (one block an entry).
    Blocks { rows: u64, cols: u64 },
    /// Rank-4 `[E, blocks, rows, cols]`, slab = blocks·rows·cols — the multi-block landing.
    Blocked { blocks: u64, rows: u64, cols: u64 },
    /// Rank-2 `[T, w]`, T = entries·slab/w — the flat (`w` one stick) and restickify
    /// (`w` several sticks, several entries) forms.
    Flat { w: u64 },
}

impl Frame {
    /// The reshape's `to` dims.
    fn to(self, entries: u64, slab: u64) -> Vec<i64> {
        match self {
            Frame::Blocks { rows, cols } => vec![entries as i64, rows as i64, cols as i64],
            Frame::Blocked { blocks, rows, cols } => {
                vec![entries as i64, blocks as i64, rows as i64, cols as i64]
            }
            Frame::Flat { w } => vec![(entries * slab / w) as i64, w as i64],
        }
    }

    /// The landing REGION's declared two-D extent — `[rows, cols]` stick-major, the region the
    /// emitter mints for the value (rank-3/4 fold their entry and block axes into the rows; a
    /// rank-2 frame is its own two-D form). NOT the byte order: at rank ≥ 3 the landing's BYTES
    /// are per-block stick-major, blocks one after another — OUR INTERMEDIATE'S OWN CONTRACT
    /// with its readers, the frame `whole_function` registers as `batch_blocks` (proved
    /// end-to-end by the D3 bundle gate plus card chat on 4 models) — so the runner decodes by
    /// the frame's own rank, never by this folded extent. At rank 2 (or one block) the two
    /// coincide.
    fn rows_cols(self, entries: u64, slab: u64) -> (u64, u64) {
        match self {
            Frame::Blocks { rows, cols } => (entries * rows, cols),
            Frame::Blocked { blocks, rows, cols } => (entries * blocks * rows, cols),
            Frame::Flat { w } => (entries * slab / w, w),
        }
    }

    /// One word for the verdict line.
    fn tag(self, entries: u64) -> &'static str {
        match self {
            Frame::Blocks { .. } => "blocks1",
            Frame::Blocked { .. } => "blocks",
            // A rank-2 frame several sticks wide over several entries is the RESTICKIFY (a
            // row-major landing plus one whole stick permute); anything else rank-2 is flat.
            Frame::Flat { w } => {
                if entries > 1 && w > STICK {
                    "restick"
                } else {
                    "flat"
                }
            }
        }
    }
}

/// The sweep's own manifest row — everything the runner and its decoder need, stated once.
#[derive(serde::Serialize)]
struct Cell {
    /// The frame's `to` dims, in order (the reshape's own statement).
    frame: Vec<u32>,
    /// The frame's tag: `blocks1` (rank-3), `blocks` (rank-4), `flat` (rank-2).
    form: String,
    entries: u32,
    /// One page's slab, in elements.
    slab: u32,
    /// Pages in the pool: the index is a permutation of `0..pages`.
    pages: u32,
    /// The index tensor's words, in entry order (`idx[e]` selects entry `e`'s page).
    index: Vec<u32>,
    /// The landing's frame ON THE CARD — always the stick-major image of the frame's own
    /// `[rows, cols]`: `land_rows` rows, `land_cols` columns.
    land_rows: u32,
    land_cols: u32,
    /// The number of bytes the index tensor's placement holds (`entries` words, ≥ one stick).
    index_bytes: u64,
    /// The pool parameter's tid (the gather's value operand).
    pool_tid: u32,
    /// The index parameter's tid.
    idx_tid: u32,
    /// The terminal compute op's two operand tids (the runner stages them; their values are
    /// irrelevant — the terminal exists only so the walk sees a compute op).
    term_tids: [u32; 2],
    /// The buffer the runner reads back: the gather's landing, or — at a rank-2 restickify —
    /// the one whole stick permute's destination (the frame's own stick column).
    readback_tid: u32,
    /// The refusal, verbatim, for a cell the walk will not lower. `None` ⇒ the bundle was
    /// written and the cell is runnable.
    missing: Option<String>,
    /// The commit this dump ran at — `PAGE_GATHER_SWEEP_SHA`, required at dump time (a dump
    /// refuses to run unstamped). Every verdict line carries it, and the runner skips a cell
    /// only when sha AND group_0 md5 both match a logged verdict.
    sha: String,
}

/// Dump every cell. One directory per cell: `group_0/` ([`superdsc::write_dxp_input`]),
/// `placements.json` (the launch map, the `bake.rs::write_placements` format) and `cell.json`
/// (this sweep's own statement of the op, for the runner's fills and decoder).
#[test]
#[ignore]
fn dump_page_gather_sweep() {
    let Some(root) = std::env::var_os("PAGE_GATHER_SWEEP_ROOT") else {
        panic!("PAGE_GATHER_SWEEP_ROOT=<dir> names where the cells are written");
    };
    let sha = std::env::var("PAGE_GATHER_SWEEP_SHA")
        .unwrap_or_else(|_| panic!("PAGE_GATHER_SWEEP_SHA=<sha> stamps the commit the dump runs at — a dump refuses to run unstamped"));
    let root = std::path::Path::new(&root);
    std::fs::create_dir_all(root).unwrap();
    let mut cells = Vec::new();
    // ── D1: WholePages (rank-3, one block an entry), entry bytes 32–128 KiB (slab·2 an entry) ──
    for &(rows, cols) in &[
        (128u64, 128u64),
        (128, 256),
        (128, 384),
        (128, 512),
        (256, 128),
        (256, 256),
        (64, 1024),
    ] {
        cells.push((Frame::Blocks { rows, cols }, 2u64, rows * cols));
    }
    // The one-entry control at the proof size: one block, the landing IS the frame.
    cells.push((
        Frame::Blocks {
            rows: 128,
            cols: 512,
        },
        1,
        128 * 512,
    ));
    // ── New territory: entries past the 32-entry index stick (the chunk axis) ──
    cells.push((
        Frame::Blocks {
            rows: 128,
            cols: 128,
        },
        33,
        128 * 128,
    ));
    cells.push((
        Frame::Blocks {
            rows: 128,
            cols: 128,
        },
        64,
        128 * 128,
    ));
    cells.push((Frame::Blocks { rows: 64, cols: 64 }, 96, 64 * 64));
    // ── New territory: the rank-2 flats (`w` one stick — the landing IS the frame) ──
    cells.push((Frame::Flat { w: 64 }, 4, 8192));
    cells.push((Frame::Flat { w: 64 }, 16, 2048));
    // ── New territory: the rank-2 RESTICKIFY (several multi-stick entries — the N2 value test:
    //    a row-major landing plus ONE whole stick permute into the frame's own column) ──
    cells.push((Frame::Flat { w: 128 }, 4, 8192));
    cells.push((Frame::Flat { w: 256 }, 2, 16384));
    cells.push((Frame::Flat { w: 128 }, 8, 4096));
    // ── New territory: the rank-4 multi-block landing (`batch_blocks`) ──
    cells.push((
        Frame::Blocked {
            blocks: 2,
            rows: 64,
            cols: 128,
        },
        2,
        2 * 64 * 128,
    ));
    cells.push((
        Frame::Blocked {
            blocks: 4,
            rows: 32,
            cols: 64,
        },
        2,
        4 * 32 * 64,
    ));

    let mut missing = 0usize;
    let mut emitted = 0usize;
    for &(frame, entries, slab) in &cells {
        let cell = dump_cell(frame, entries, slab, root, &sha);
        match cell.missing {
            Some(_) => missing += 1,
            None => emitted += 1,
        }
    }
    eprintln!("page-gather sweep: {emitted} bundles, {missing} missing-lowering cells");
    assert_eq!(
        missing, 0,
        "every stated cell must lower — the walk refuses none"
    );
}

fn dump_cell(frame: Frame, entries: u64, slab: u64, root: &std::path::Path, sha: &str) -> Cell {
    let pages = entries; // the index is a permutation of the pool's own pages
    let pool_bytes = pages * slab * 2;
    // The index placement holds the entry words and, whole, one stick — the op's own index
    // granularity (`of_scratch_rows` reads a stick of int32 a run), never truncated by an
    // arbitrary choice of this harness.
    let index_bytes = (entries * 4).max(4 * 64);
    let dir = root.join(cell_name(frame, entries, slab));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // The index: a real permutation, reversed — the same indirection the card probes stated.
    let index: Vec<u32> = (0..entries).map(|e| (entries - 1 - e) as u32).collect();

    // ── the launch map: the pool (ROW-OUTERMOST, the paged cache's own allocation law), the
    // index, and the terminal compute op's two operands.
    let (pool_tid, idx_tid, a_tid, b_tid) = (201u32, 202, 203, 204);
    let mut l = BundleLayout::default();
    let mut off = 0u64;
    for (tid, size) in [
        (pool_tid, pool_bytes),
        (idx_tid, index_bytes),
        (a_tid, pool_bytes),
        (b_tid, pool_bytes),
    ] {
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
    l.allocate_row_outermost(pool_tid)
        .expect("the pool is whole sticks");
    l.segment_bytes[3] = off + 4096; // + the H2D window pad

    let k = paged_program(frame, entries, slab, pages);

    // ⛔ A BUILD-TIME PANIC IN THE EMITTER IS A MISSING LOWERING TOO — recorded verbatim, never
    // unwound past. The hook is silenced so a panicking cell does not spam the run; the payload
    // is the table's refusal text.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let walked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut sym = 0i64;
        lower_function(
            &k,
            Some(&l),
            &mut sym,
            None,
            &mut ktir_superdsc::emit::ktir_matmul_fp8::QuantizedActivations::default(),
        )
    }));
    std::panic::set_hook(prev_hook);

    let (land_rows, land_cols) = frame.rows_cols(entries, slab);
    let (ops, readback_tid, missing) = match walked {
        Ok(Ok(ops)) => match classify(&ops, frame, entries, slab) {
            Ok(tid) => (ops, tid, None),
            Err(message) => (Vec::new(), 0, Some(message)),
        },
        Ok(Err(e)) => (Vec::new(), 0, Some(e.message)),
        Err(payload) => (
            Vec::new(),
            0,
            Some(format!("PANIC: {}", panic_message(&payload))),
        ),
    };

    let cell = Cell {
        frame: frame
            .to(entries, slab)
            .into_iter()
            .map(|d| d as u32)
            .collect(),
        form: frame.tag(entries).into(),
        entries: entries as u32,
        slab: slab as u32,
        pages: pages as u32,
        index,
        land_rows: land_rows as u32,
        land_cols: land_cols as u32,
        index_bytes,
        pool_tid,
        idx_tid,
        term_tids: [a_tid, b_tid],
        readback_tid,
        missing,
        sha: sha.into(),
    };
    let runnable = cell.missing.is_none();
    write_cell(&dir, &l, runnable.then_some(&ops), &cell);
    cell
}

/// The cell's directory name: `f<frame>-e<entries>-s<slab>` — the frame's own dims in the name,
/// so a verdict line names the geometry it passed at.
fn cell_name(frame: Frame, entries: u64, slab: u64) -> String {
    let dims = frame
        .to(entries, slab)
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x");
    format!("f{dims}-e{entries}-s{slab}")
}

/// The minimal page-gather program (see the module doc): the pool `[pages, slab]` gathered by an
/// `[entries]` i32 index into a load `[E, slab]`, re-framed by a `tensor.reshape` to the cell's
/// frame, with a terminal `arith.addf` over two plain loaded parameters of the frame's shape.
fn paged_program(frame: Frame, entries: u64, slab: u64, pages: u64) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let to = frame.to(entries, slab);
    let shape = |op: Operation<'static>, dims: &[i64]| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.to_vec())))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };
    // The load and the reshape are UNSTORED intermediates, so each states its tensor type — the
    // `from`/`to` frames the walk's re-frame arm reads (`def_dims`).
    let ty = |dims: &[i64]| {
        Some(IrType::Tensor {
            dims: a.ints(dims.to_vec()),
            elem: DType::F16,
        })
    };
    let (p_pool, p_idx, p_a, p_b) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3));
    let (zero, v_pool, v_idx, ind, src, res) = (Ssa(4), Ssa(5), Ssa(6), Ssa(7), Ssa(8), Ssa(9));
    let (v_a, t_a, val_a, v_b, t_b, val_b, sum) = (
        Ssa(10),
        Ssa(11),
        Ssa(12),
        Ssa(13),
        Ssa(14),
        Ssa(15),
        Ssa(16),
    );
    // The indirect dim's subscript — `Sym(0) + Dim(0)`, the one form `gather_subscripts` builds
    // and the one `gathers_of` folds into an entry base.
    let sub0 = a.expr(AffineExpr::Add(
        a.expr(AffineExpr::Sym(0)),
        a.expr(AffineExpr::Dim(0)),
    ));
    let ops = vec![
        Operation::new(a, Some(zero), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
        // The pool: `[pages, slab]` — the row-outermost paged value.
        shape(
            Operation::new(a, Some(v_pool), OpKind::KtdpConstructMemoryView, &[p_pool]),
            &[pages as i64, slab as i64],
        ),
        // The index: `[entries]` i32.
        Operation::new(a, Some(v_idx), OpKind::KtdpConstructMemoryView, &[p_idx])
            .with_attr(
                a,
                AttrKey::Shape,
                Attr::IntList(a.ints(vec![entries as i64])),
            )
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::I32)),
        // THE INDIRECT TILE: data view first, index view second — `gather_of`'s own reading. Its
        // shape is the GATHERED tile `[E, slab]`, dim 0 indirect with the `Sym(0) + Dim(0)`
        // subscript, dim 1 a direct sub — the anchor triple a real program carries.
        shape(
            Operation::new(
                a,
                Some(ind),
                OpKind::KtdpConstructIndirectAccessTile,
                &[v_pool, v_idx],
            ),
            &[entries as i64, slab as i64],
        )
        .with_attr(
            a,
            AttrKey::DimKinds,
            Attr::StrList(a.names(vec!["indirect", "direct_sub"])),
        )
        .with_attr(a, AttrKey::DimData, Attr::IntList(a.ints(vec![0, 0])))
        .with_attr(
            a,
            AttrKey::IntermediateVars,
            Attr::Ssas(a.ssa(vec![zero, zero])),
        )
        .with_attr(
            a,
            AttrKey::DimSubs,
            Attr::AffineMapList(a.maps(vec![
                AffineMap {
                    num_dims: 2,
                    num_syms: 2,
                    exprs: std::slice::from_ref(sub0),
                },
                AffineMap {
                    num_dims: 2,
                    num_syms: 2,
                    exprs: a.exprs(vec![AffineExpr::Dim(1)]),
                },
            ])),
        ),
        // The gathered load: `[E, slab]`.
        {
            let mut op = shape(
                Operation::new(a, Some(src), OpKind::KtdpLoad, &[ind]),
                &[entries as i64, slab as i64],
            );
            op.result_type = ty(&[entries as i64, slab as i64]);
            op
        },
        // ⭐ THE RE-FRAME — the cell's own statement, the `to` the walk's gather arm reads.
        {
            let mut op = shape(
                Operation::new(a, Some(res), OpKind::TensorReshape, &[src]),
                &to,
            );
            op.result_type = ty(&to);
            op
        },
        // THE TERMINAL: `arith.addf` over two plain loaded parameters of the frame's shape.
        shape(
            Operation::new(a, Some(v_a), OpKind::KtdpConstructMemoryView, &[p_a]),
            &to,
        ),
        shape(
            Operation::new(
                a,
                Some(t_a),
                OpKind::KtdpConstructAccessTile,
                &[v_a, zero, zero],
            ),
            &to,
        ),
        {
            let mut op = shape(
                Operation::new(a, Some(val_a), OpKind::KtdpLoad, &[t_a]),
                &to,
            );
            op.result_type = ty(&to);
            op
        },
        shape(
            Operation::new(a, Some(v_b), OpKind::KtdpConstructMemoryView, &[p_b]),
            &to,
        ),
        shape(
            Operation::new(
                a,
                Some(t_b),
                OpKind::KtdpConstructAccessTile,
                &[v_b, zero, zero],
            ),
            &to,
        ),
        {
            let mut op = shape(
                Operation::new(a, Some(val_b), OpKind::KtdpLoad, &[t_b]),
                &to,
            );
            op.result_type = ty(&to);
            op
        },
        {
            let mut op = shape(
                Operation::new(a, Some(sum), OpKind::ArithAddf, &[val_a, val_b]),
                &to,
            );
            op.result_type = ty(&to);
            op
        },
    ];
    KtirNode {
        func: IRFunction {
            name: "page_gather_sweep",
            arguments: a.args(vec![
                (p_pool, IrType::Index),
                (p_idx, IrType::Index),
                (p_a, IrType::Index),
                (p_b, IrType::Index),
            ]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        // The label is inert to this door: `whole_function` special-cases only `Program::LmLast`;
        // the per-`Program` doors live elsewhere and are not called.
        program: Program::Reshape,
        bindings: vec![
            BufferId::new(201),
            BufferId::new(202),
            BufferId::new(203),
            BufferId::new(204),
        ],
        mask: None,
        node_out_tid: None,
        minted: Default::default(),
        lx_plan: Default::default(),
    }
}

/// Read the walk's chosen arm off its OWN emission — the gather ops' names and iteration spaces,
/// and the restickify's one permute — never re-derived from the predicates. A landing that does
/// not match the cell's stated frame FAILS the dump. Returns the tid the runner reads back.
fn classify(
    ops: &[EmittedOp<ktir_superdsc::collective::TpWorld<1>>],
    frame: Frame,
    entries: u64,
    slab: u64,
) -> Result<u32, String> {
    // The gather ops: `gather_o{tid}` (one op) or `gather{n}_o{tid}` (chunk n, entries past the
    // index stick). The head-major staging front's `gather{c}_{i}_o{tid}` copies are NOT this
    // arm's (an underscored chunk AND run index) and no cell here states one.
    let is_gather = |name: &str| match name.strip_prefix("gather") {
        Some(rest) => rest
            .split_once("_o")
            .is_some_and(|(mid, _)| mid.is_empty() || mid.chars().all(|c| c.is_ascii_digit())),
        None => false,
    };
    let gathers: Vec<&EmittedOp<ktir_superdsc::collective::TpWorld<1>>> =
        ops.iter().filter(|o| is_gather(&o.op_name)).collect();
    let Some(first) = gathers.first() else {
        return Err("the walk emitted no gather op".into());
    };
    let landing_tid = |name: &str| -> Result<u32, String> {
        name.rsplit("_o")
            .next()
            .and_then(|t| t.parse::<u32>().ok())
            .ok_or_else(|| format!("the op names no landing tid: {name}"))
    };
    let tid = landing_tid(&first.op_name)?;
    let dsc = first.dsc();
    let Some(dsc) = dsc.dscs_.first().and_then(|m| m.values().next()) else {
        return Err("the gather op states no descriptor".into());
    };
    let (mb, y, x, out) = (
        dsc.N_.mb_ as u64,
        dsc.N_.y_ as u64,
        dsc.N_.x_ as u64,
        dsc.N_.out_ as u64,
    );
    // The restickify's one whole stick permute — present only at a rank-2 multi-stick
    // multi-entry frame, and then the runner reads ITS destination (the frame's own column).
    let permutes: Vec<&EmittedOp<ktir_superdsc::collective::TpWorld<1>>> = ops
        .iter()
        .filter(|o| o.op_name.starts_with("permute"))
        .collect();
    match frame {
        Frame::Blocks { rows, cols } => {
            // The WholePages form: `y` is 1 (one block an entry — the `[mb, y:1, x, out]`
            // respelling), each chunk ≤ the 32-entry index stick.
            let mut seen = 0u64;
            for g in &gathers {
                let d = g.dsc();
                let d = d
                    .dscs_
                    .first()
                    .and_then(|m| m.values().next())
                    .ok_or("a chunk states no descriptor")?;
                seen += d.N_.mb_ as u64;
            }
            if y != 1 || x != rows || out != cols || seen != entries {
                return Err(format!(
                    "the walk's gather landed a frame this sweep does not state: N_ \
                     mb={mb} y={y} x={x} out={out} (blocks1 expects mb-chunks summing to \
                     {entries}, y=1, x={rows}, out={cols}; the chunks sum to {seen})"
                ));
            }
            if !permutes.is_empty() {
                return Err(format!(
                    "a blocks1 landing carries {} permute(s) — the landing IS the frame",
                    permutes.len()
                ));
            }
            Ok(tid)
        }
        Frame::Blocked { blocks, rows, cols } => {
            let mut seen = 0u64;
            for g in &gathers {
                let d = g.dsc();
                let d = d
                    .dscs_
                    .first()
                    .and_then(|m| m.values().next())
                    .ok_or("a chunk states no descriptor")?;
                seen += d.N_.mb_ as u64;
            }
            if y != blocks || x != rows || out != cols || seen != entries {
                return Err(format!(
                    "the walk's gather landed a frame this sweep does not state: N_ \
                     mb={mb} y={y} x={x} out={out} (blocks expects y={blocks}, x={rows}, \
                     out={cols}, the chunks summing to {entries}; they sum to {seen})"
                ));
            }
            if !permutes.is_empty() {
                return Err(format!(
                    "a blocks landing carries {} permute(s) — the landing is the blocks frame",
                    permutes.len()
                ));
            }
            Ok(tid)
        }
        Frame::Flat { w } => {
            if entries == 1 || w <= STICK {
                // The flat landing: `y` is 1, `x` is the entry's own rows (`slab / w`), `out`
                // the one-stick width — stick-major and row-major are the same bytes.
                let mut seen = 0u64;
                for g in &gathers {
                    let d = g.dsc();
                    let d = d
                        .dscs_
                        .first()
                        .and_then(|m| m.values().next())
                        .ok_or("a chunk states no descriptor")?;
                    seen += d.N_.mb_ as u64;
                }
                if y != 1 || x != slab / w || out != w || seen != entries {
                    return Err(format!(
                        "the walk's gather landed a frame this sweep does not state: N_ \
                         mb={mb} y={y} x={x} out={out} (flat expects y=1, x={}, out={w}, \
                         the chunks summing to {entries}; they sum to {seen})",
                        slab / w
                    ));
                }
                if !permutes.is_empty() {
                    return Err(format!(
                        "a flat landing carries {} permute(s) — one stick IS the frame",
                        permutes.len()
                    ));
                }
                Ok(tid)
            } else {
                // ⭐ THE RESTICKIFY: one-row blocks (`x` is 1, `y` is `slab / w` an entry, `out`
                // the multi-stick width) landing row-major, then ONE whole stick permute whose
                // destination is the frame's own column — the buffer the runner reads back.
                let mut seen = 0u64;
                for g in &gathers {
                    let d = g.dsc();
                    let d = d
                        .dscs_
                        .first()
                        .and_then(|m| m.values().next())
                        .ok_or("a chunk states no descriptor")?;
                    seen += d.N_.mb_ as u64;
                }
                if y != slab / w || x != 1 || out != w || seen != entries {
                    return Err(format!(
                        "the walk's gather landed a frame this sweep does not state: N_ \
                         mb={mb} y={y} x={x} out={out} (restickify expects y={}, x=1, \
                         out={w}, the chunks summing to {entries}; they sum to {seen})",
                        slab / w
                    ));
                }
                if permutes.len() != 1 {
                    return Err(format!(
                        "a rank-2 multi-stick multi-entry landing carries {} permute(s) — \
                         exactly one whole stick permute restickifies it",
                        permutes.len()
                    ));
                }
                landing_tid(&permutes[0].op_name)
            }
        }
    }
}

/// Write the bundle (`group_0/` + `placements.json`) and the manifest. `ops: None` ⇒ the cell is
/// MISSING-LOWERING: only `cell.json` is written, exactly as the bake would refuse the function.
fn write_cell(
    dir: &std::path::Path,
    l: &BundleLayout,
    ops: Option<&Vec<EmittedOp<ktir_superdsc::collective::TpWorld<1>>>>,
    cell: &Cell,
) {
    if let Some(ops) = ops.filter(|o| !o.is_empty()) {
        superdsc::write_dxp_input(dir, ops, superdsc::FoldGrouping::Split)
            .unwrap_or_else(|e| panic!("{}: write_dxp_input: {e}", dir.display()));
    }
    // The launch map, the `bake.rs::write_placements` format: the segment sizes at the synth
    // high-water (so the walk's minted landing and scratch have their bytes), every placement,
    // every synth row.
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
    // The parameters, then the walk's own synth placements (the landing and its scratch) — the
    // runtime's launcher (`bundle_run`) reads ONLY this `places` list and demands the read-back
    // tensor carry the `Logits` role, so each synth row is carried in verbatim (the walk's own
    // segment/offset/size from `l.synth`) with the READ-BACK tensor labeled `Logits` and every
    // other synth intermediate `Activation`. The placement itself is the walk's; only the
    // read-back role label is the producer's.
    let mut rows: Vec<String> = l
        .placements
        .iter()
        .map(|(tid, p)| {
            format!(
                "    {{\"tid\": {}, \"role\": \"{:?}\", \"segment\": {}, \"bank\": {}, \
                 \"offset\": {}, \"size\": {}}}",
                tid, p.role, p.segment, p.bank, p.offset, p.size
            )
        })
        .collect();
    for (name, off) in synth.map.iter() {
        let tid: u32 = name
            .trim_start_matches('t')
            .parse()
            .unwrap_or_else(|_| panic!("synth intermediate {name} is not a t<tid>"));
        rows.push(format!(
            "    {{\"tid\": {}, \"role\": \"{}\", \"segment\": {}, \"bank\": 0, \"offset\": {}, \
             \"size\": {}}}",
            tid,
            if tid == cell.readback_tid {
                "Logits"
            } else {
                "Activation"
            },
            intermediate,
            off,
            synth.sizes.get(name).copied().unwrap_or(0),
        ));
    }
    s.push_str(&rows.join(",\n"));
    s.push_str("\n  ],\n  \"synth\": [\n");
    for (i, (name, off)) in synth.map.iter().enumerate() {
        s.push_str(&format!(
            "    {{\"name\": \"{}\", \"segment\": {}, \"offset\": {}, \"size\": {}}}{}\n",
            name,
            intermediate,
            off,
            synth.sizes.get(name).copied().unwrap_or(0),
            if i + 1 == synth.map.len() { "" } else { "," },
        ));
    }
    s.push_str("  ],\n  \"scalarmul_scales\": [\n  ]\n}\n");
    std::fs::write(dir.join("placements.json"), s).unwrap();
    drop(synth);
    std::fs::write(
        dir.join("cell.json"),
        serde_json::to_string_pretty(cell).unwrap(),
    )
    .unwrap();
}

/// A panic payload as the table's refusal text.
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "an unprintable panic".into()
    }
}
