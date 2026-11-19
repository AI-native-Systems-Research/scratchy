// SPDX-License-Identifier: Apache-2.0

//! ⭐ THE FAST ROPE FORM'S CELLS — step 0's protocol (`examples/rope_half_swap_cells.rs`,
//! `350e305dcc`) for the 4-op form as it LANDS: every cell is `rope_at`'s own emission, so the
//! bundle timed is the bundle the bake ships. Test tooling: nothing here ships.
//!
//! ## The form (one per cell, `form = "fast4"`)
//! `xc = x·cos`, then ONE band multiply per stick move of the rotate — `rs_lo = x_hi · sin'_lo`,
//! `rs_hi = x_lo · sin'_hi`, written straight into the two halves of ONE `rs` through pitched views
//! of the `[mq, total]` plane (the stick-axis back gap is A1's TYPE, not a patch) — then
//! `out = xc + rs`. FOUR descriptors at every `mq`. Unlike step 0's `halfswap4` cells (head axis
//! pinned unsplit, gap patched into the render), the band multiplies take the elementwise law's
//! division, the head axis INCLUDED: the per-core start folds every coordinate of the view. So
//! `fast4` against `halfswap4` isolates the head split, and `fast4` against `main` is the A/B.
//!
//! ## The signed table
//! The rope reads `sin' = sign·sin` — the table its rotate keeps the sign in
//! (`RotateHalf::sin_table`, the one law over `rope_p_entry`). Its per-column sign is READ from
//! that law by this writer (`SinTable::sign_heads` over a unit row — `±1` is never typed here) and
//! carried in `cell.json` for the runner's fill, exactly as `halfswap4`'s cells carry theirs; the
//! host reference is unchanged, `out = x·cos + rotate_half(x)·sin` over the UNSIGNED table.
//!
//! ## Staging (the `halfswap4` contract, so the runner's fills never branch on the form)
//! `x`, `cos`, `sin` (plain), `out`, `P` (never read at these head dims) and `sin'` are placed at
//! the same tids in every cell; the rope reads `sin'` as its sin operand. The probe synthetic is
//! `rs` (`probe_kind = "rs"`: `rotate_half(x)·sin`, the form's own intermediate — it has no `rot`).
//!
//! ## The baseline
//! `main` cells come from the step-0 writer at a sha whose `rope_at` is main's (`350e305dcc`):
//! this writer's `rope_at` IS the 4-op form. The runner times both directories with one protocol.
//!
//! A CELL THAT CANNOT LOWER IS A FAILURE OF THE FORM — this writer panics naming every one.
//!
//! ```bash
//! ROPE_HALF_SWAP_ROOT=<dir> ROPE_HALF_SWAP_SHA=<sha> \
//!   cargo run --release -p scratchy-target-spyre --example rope_fast_cells
//! ```
//!
//! Requires A1 (the back-gap type, its stick-axis admission), A2 and the fast-form addendum
//! (`rope_at`'s 4-op body, `SinTable`, the per-core start's feature fold).

use ktir_superdsc::collective::TpWorld;
use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::emit::lower_ktir_to_superdsc::{RopeAt, rope_at};
use ktir_superdsc::place::{PlaceId, SynthRole, act_name};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::sdsc_abstract::{HeadWidth, Lanes, RotateHalf};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;

/// The cell's geometry, its own statement: rows (`mq`), head dim, heads. `total = heads · hd`.
#[derive(Clone, Copy)]
struct Geo {
    mq: u32,
    hd: u32,
    heads: u32,
}

impl Geo {
    fn total(self) -> u32 {
        self.heads * self.hd
    }
}

/// The form's tag in every verdict line.
const FORM: &str = "fast4";

/// The bundle's tensors — `halfswap4`'s numbering, so a runner that stages one stages the other:
/// x, cos, sin, sin', out are the `[mq, total]` plane; p is the baseline's `[hd, hd]` class.
const X_TID: u32 = 301;
const COS_TID: u32 = 302;
const SIN_TID: u32 = 303;
const OUT_TID: u32 = 304;
const P_TID: u32 = 305;
const SINS_TID: u32 = 306;

/// The sweep's own manifest row — `halfswap4`'s fields, one for one.
#[derive(serde::Serialize)]
struct Cell {
    form: String,
    mq: u32,
    hd: u32,
    heads: u32,
    total: u32,
    x_tid: u32,
    cos_tid: u32,
    sin_tid: u32,
    out_tid: u32,
    p_tid: u32,
    sin_tid_signed: u32,
    /// The per-column sign of `sin'`, READ from `RotateHalf::sin_table` (the one law over
    /// `rope_p_entry`) — the runner's fill builds `sin' = sign[col % hd]·sin` from it.
    sin_sign: Vec<i64>,
    /// The probe synthetic's name (`t{out}_rs`) — the runner reads it back directly.
    probe_synth: String,
    /// What the probe synthetic IS: `rs`, i.e. `rotate_half(x)·sin` over the UNSIGNED table.
    probe_kind: String,
    /// The rendered trip count: the launch count the card pays.
    descriptors: u32,
    /// The refusal, verbatim, for a cell the emission will not produce.
    missing: Option<String>,
    /// The commit this writer ran at — `ROPE_HALF_SWAP_SHA`, required at run time.
    sha: String,
}

fn main() {
    let Some(root) = std::env::var_os("ROPE_HALF_SWAP_ROOT") else {
        panic!("ROPE_HALF_SWAP_ROOT=<dir> names where the cells are written");
    };
    let sha = std::env::var("ROPE_HALF_SWAP_SHA").unwrap_or_else(|_| {
        panic!(
            "ROPE_HALF_SWAP_SHA=<sha> stamps the commit the writer runs at — the writer refuses \
             to run unstamped"
        )
    });
    let root = std::path::Path::new(&root);
    std::fs::create_dir_all(root).unwrap();
    // Step 0's shapes, one for one: gemma-4's sliding ropes (hd 256, Q16 and K8) and global ropes
    // (hd 512, Q16 and K1), at decode, the corrupt rungs 23/29/31 and the full chunk.
    let mut cells = Vec::new();
    for &(hd, heads) in &[(256u32, 16u32), (256, 8), (512, 16), (512, 1)] {
        for &mq in &[1u32, 23, 29, 31, 128] {
            cells.push(emit_cell(Geo { mq, hd, heads }, root, &sha));
        }
    }
    // ⛔ A CELL THAT CANNOT LOWER IS A FAILURE OF THE FORM, not a count.
    let missing: Vec<&Cell> = cells.iter().filter(|c| c.missing.is_some()).collect();
    assert!(
        missing.is_empty(),
        "the fast form must lower at every stated shape — {} cell(s) refused: {}",
        missing.len(),
        missing
            .iter()
            .map(|c| format!(
                "mq={} hd={} heads={}: {}",
                c.mq,
                c.hd,
                c.heads,
                c.missing.as_deref().unwrap_or("")
            ))
            .collect::<Vec<_>>()
            .join("; ")
    );
    eprintln!("rope fast sweep: {} bundles written", cells.len());
}

/// The cell's directory name: `f<form>-mq<mq>-hd<hd>-h<heads>`.
fn cell_name(geo: Geo) -> String {
    format!("f{FORM}-mq{}-hd{}-h{}", geo.mq, geo.hd, geo.heads)
}

/// The launch map: the six parameters in segment 3 (out labeled `Logits` by the writer — the
/// runner's read-back role), the synthetics declared by the emission.
fn build_layout(geo: Geo) -> BundleLayout {
    let total = geo.total();
    let mut l = BundleLayout::default();
    let mut off = 0u64;
    for (tid, elems) in [
        (X_TID, geo.mq * total),
        (COS_TID, geo.mq * total),
        (SIN_TID, geo.mq * total),
        (OUT_TID, geo.mq * total),
        (P_TID, geo.hd * geo.hd),
        (SINS_TID, geo.mq * total),
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
                size: elems as u64 * 2,
            },
        );
        off += elems as u64 * 2;
    }
    l.segment_bytes[3] = off + 4096; // + the H2D window pad
    l
}

/// The per-column sign of the table the rope at `hd` reads, READ from its rotate's own
/// `SinTable` — a unit row through `sign_heads`, so the law is applied, never restated.
fn sin_sign(hd: u32) -> Vec<i64> {
    let rotate = RotateHalf::of(HeadWidth::of_whole_sticks(hd), Lanes::FP16)
        .unwrap_or_else(|| panic!("hd {hd}: the sweep states factored head dims"));
    let mut row = vec![1.0f32; hd as usize];
    rotate.sin_table().sign_heads(&mut row);
    row.iter().map(|&s| s as i64).collect()
}

/// `rope_at` at the cell's const head dim — the only dispatch is the value → the const.
fn fast_ops(geo: Geo, l: &BundleLayout) -> Result<Vec<EmittedOp<TpWorld<1>>>, String> {
    let mut sib = 0i64;
    let ra = RopeAt {
        name: "rope",
        x: act_name(X_TID),
        cos: act_name(COS_TID),
        // The rope reads the table its rotate keeps the sign in: `sin'`.
        sin: act_name(SINS_TID),
        out: act_name(OUT_TID),
        t: OUT_TID,
        mq: geo.mq,
        total: geo.total(),
        sym_id_base: &mut sib,
        layout: Some(l),
    };
    match geo.hd {
        256 => rope_at::<TpWorld<1>, 256>(ra).map_err(|e| e.message),
        512 => rope_at::<TpWorld<1>, 512>(ra).map_err(|e| e.message),
        hd => panic!("the sweep states hd 256/512; this cell states {hd}"),
    }
}

fn emit_cell(geo: Geo, root: &std::path::Path, sha: &str) -> Cell {
    let dir = root.join(cell_name(geo));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let l = build_layout(geo);

    // ⛔ A BUILD-TIME PANIC IN THE EMITTER IS A REFUSAL TOO — recorded verbatim, never unwound
    // past. The hook is silenced so a panicking cell does not spam the run.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let walked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let ops = fast_ops(geo, &l)?;
        let rendered = superdsc::render_dxp_input(&ops, superdsc::FoldGrouping::Split)
            .map_err(|e| format!("render: {e}"))?;
        let descriptors = rendered
            .iter()
            .map(|g| g.files.len() as u32 - 1) // every file but bundle.mlir is one trip
            .sum();
        superdsc::write_rendered_dxp_input(&dir, &rendered).map_err(|e| format!("write: {e}"))?;
        Ok::<u32, String>(descriptors)
    }));
    std::panic::set_hook(prev_hook);

    let (descriptors, missing) = match walked {
        Ok(Ok(n)) => (n, None),
        Ok(Err(e)) => (0, Some(e)),
        Err(payload) => (0, Some(format!("PANIC: {}", panic_message(&payload)))),
    };

    let cell = Cell {
        form: FORM.into(),
        mq: geo.mq,
        hd: geo.hd,
        heads: geo.heads,
        total: geo.total(),
        x_tid: X_TID,
        cos_tid: COS_TID,
        sin_tid: SIN_TID,
        out_tid: OUT_TID,
        p_tid: P_TID,
        sin_tid_signed: SINS_TID,
        sin_sign: sin_sign(geo.hd),
        probe_synth: PlaceId::Act(OUT_TID).synth(SynthRole::Rs).to_string(),
        probe_kind: "rs".into(),
        descriptors,
        missing,
        sha: sha.into(),
    };
    write_cell(&dir, &l, &cell);
    cell
}

/// Write the launch map and the manifest — the step-0 writer's own format: the `places` list
/// carries the parameters (the read-back `out` labeled `Logits`), the synthetics ride the `synth`
/// list by NAME at the walk's own segment/offset from `l.synth`.
fn write_cell(dir: &std::path::Path, l: &BundleLayout, cell: &Cell) {
    let synth = l.synth.borrow();
    let mut seg_bytes = l.segment_bytes;
    let intermediate = SegRole::Intermediate.segment();
    seg_bytes[intermediate] = seg_bytes[intermediate].max(synth.high_water());
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
                "    {{\"tid\": {}, \"role\": \"{}\", \"segment\": {}, \"bank\": {}, \
                 \"offset\": {}, \"size\": {}}}",
                tid,
                if *tid == cell.out_tid {
                    "Logits"
                } else {
                    "Activation"
                },
                p.segment,
                p.bank,
                p.offset,
                p.size
            )
        })
        .collect();
    s.push_str(&rows.join(",\n"));
    s.push_str("\n  ],\n  \"synth\": [\n");
    for (i, (name, off, size)) in synth.iter().enumerate() {
        s.push_str(&format!(
            "    {{\"name\": \"{}\", \"segment\": {}, \"offset\": {}, \"size\": {}}}{}\n",
            name,
            intermediate,
            off,
            size,
            if i + 1 == synth.len() { "" } else { "," },
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
