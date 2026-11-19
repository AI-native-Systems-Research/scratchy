// SPDX-License-Identifier: Apache-2.0

//! ⭐ STEP 0 OF THE ROPE PLAN (`docs/superpowers/plans/2026-11-19-rope-reference-form-and-back-gap.md`):
//! the standalone card proof of the reference's rotate-half form, at one gemma layer's own shapes.
//! Test tooling: nothing here ships, and the only shipped code it calls is the emission the bake
//! calls.
//!
//! ## The three forms, one cell each
//! * **halfswap** — the 5-op form: `rot` built by TWO signed half-moves over `[mb, out, y]` views
//!   of the `[mq, total]` RowBlocked plane (each view sweeps `hd/2` columns of every head under a
//!   declared extent of `hd` — the stick-axis back gap, card-proven by this sweep), then the
//!   three whole-tensor legs `xc = x·cos`, `rs = rot·sin`, `out = xc + rs` exactly as `rope_at`
//!   emits them. The pairing and the sign are READ from `rope_p_entry` (the one statement), never
//!   restated; the gap is derived from the cell's own stated extents by ONE function.
//! * **halfswap4** — the 4-op form, NO rot tensor: `xc = x·cos` (whole-tensor), then TWO
//!   half-multiplies writing the two halves of ONE `rs` through the same gapped views —
//!   `rs_lo = x_hi · sin'_lo`, `rs_hi = x_lo · sin'_hi` — then `out = xc + rs`. `sin'` is the
//!   SIGNED table (`sign·sin`): its per-column sign is read from `rope_p_entry` by this writer
//!   (called per column; `±1` is never typed) and carried in `cell.json` for the runner's fill.
//!   The host reference is unchanged — `out = x·cos + rotate_half(x)·sin` over UNSIGNED sin.
//! * **main** — `rope_at` verbatim: the `[hd, hd]` P-matmul rotate (slab arm at `mq == 1`,
//!   per-head arm above) plus the same three legs. The baseline the new forms must match.
//!
//! ## The bridge (opus2-sanctioned 2026-11-19, harness-only; it DIES with A1's `BackGap` type)
//! The emitter states `backGapCore_: None` (`emit/mod.rs`), so the gapped views' `backGapCore_` is
//! patched into the RENDERED dxp input between `render_dxp_input` and `write_rendered_dxp_input`:
//! ONE function derives `gap = declared − whole-stick-rounded swept` (torch-spyre's production
//! law, `superdsc.py:1630-1633`, ELEMENTS) and splices
//! `"backGapCore_":{"out":{"-1":"<gap>"}},` immediately before the gapped view's allocate node's
//! `"coordinates_"`. It asserts the node carried NO `backGapCore_` before (nothing applied twice),
//! that the node occurs once, and that every patch applied exactly once across the render — a
//! silently-unpatched view is impossible. It never enters the ladder; A1's type replaces it
//! wholesale.
//!
//! ## The bar (checked by the runner, not here)
//! EXACT against a host model computed FROM THE OP DEFINITION (HF `rotate_half`, never our
//! opspec), on fp16-exact integer fills (every value and every intermediate an integer ≤ 390, so
//! the card's fp16 arithmetic and the SEN169 staging are exact and the compare is equality).
//! `rot` is read back too (the half-moves' own output), so a wrong half-move is located before
//! the legs smear it.
//!
//! A HALFSWAP CELL THAT CANNOT LOWER IS A FAILURE OF THE NEW FORM — this writer panics naming
//! every one. A main-form cell that cannot lower is a fact about main; it is printed verbatim
//! with its shape and refusal, and its directory carries the refusal in `cell.json` for the
//! report.
//!
//! ```bash
//! ROPE_HALF_SWAP_ROOT=<dir> ROPE_HALF_SWAP_SHA=<sha> \
//!   cargo run --release -p scratchy-target-spyre --example rope_half_swap_cells
//! ```
//!
//! Then on a pod:
//! ```bash
//! tests/rope_half_swap_cells_run.sh <dir> <warm-tree-at-this-sha> <log> [cell …]
//! ```

use ktir_superdsc::addr::rc_of;
use ktir_superdsc::collective::TpWorld;
use ktir_superdsc::emit::lower_ktir_to_superdsc::{rope_at, RopeAt};
use ktir_superdsc::emit::{emit_sdsc_tiled, rb, In, pw2, EmittedOp};
use ktir_superdsc::place::{PlaceId, SynthRole, act_name};
use ktir_superdsc::placement::{syn, BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::sdsc_abstract::{rope_p_entry, BlockCols, QueryRowCount, RowCount};
use ktir_superdsc::superdsc_opspec::{
    AnyTensorArg, Allocation, CoreBudget, Df, IterSym, ItDim, OpFunc, OpSpec, Role, Scale,
    SdscFoldSet, TensorArg, WorkPlan,
};
use ktir_superdsc::work::distribute_cores;
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

/// The three forms the sweep states.
#[derive(Clone, Copy, PartialEq)]
enum Form {
    /// The 5-op form: two signed half-moves over gapped views + the three legs.
    HalfSwap,
    /// The 4-op form: two half-multiplies over gapped views into one rs, no rot tensor.
    HalfSwap4,
    /// `rope_at` verbatim — the P-matmul rotate.
    Main,
}

impl Form {
    fn tag(self) -> &'static str {
        match self {
            Form::HalfSwap => "halfswap",
            Form::HalfSwap4 => "halfswap4",
            Form::Main => "main",
        }
    }
}

/// The bundle's tensors, one numbering for every cell: x, cos, sin, sin', out are the
/// `[mq, total]` plane; p is the baseline's `[hd, hd]` rope-p class. p AND sin' are placed and
/// staged in EVERY form (sin' is halfswap4's own input; the others never read it), so every
/// cell's staging is identical and the runner's fills never branch on the form.
const X_TID: u32 = 301;
const COS_TID: u32 = 302;
const SIN_TID: u32 = 303;
const OUT_TID: u32 = 304;
const P_TID: u32 = 305;
const SINS_TID: u32 = 306;

/// The sweep's own manifest row — everything the runner's fills, staging and host model need.
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
    /// The per-column sign of the signed sin' table, READ from `rope_p_entry` (one call per
    /// column; `±1` is never typed) — the entry's own type. The runner's fill builds
    /// `sin' = sign[col % hd]·sin` from this array — the writer is the one statement of the
    /// sign law.
    sin_sign: Vec<i8>,
    /// The probe synthetic's name (`t{out}_rot` or, for halfswap4, `t{out}_rs`) — the runner
    /// reads it back directly and compares it per `probe_kind`.
    probe_synth: String,
    /// What the probe synthetic IS: `rot` (rotate_half(x), the 5-op and main forms) or `rs`
    /// (rotate_half(x)·sin, halfswap4's own intermediate — it has no rot tensor).
    probe_kind: String,
    /// The rendered trip count: the launch count the card pays.
    descriptors: u32,
    /// The refusal, verbatim, for a cell the emission will not produce. `None` ⇒ the bundle was
    /// written and the cell is runnable.
    missing: Option<String>,
    /// The commit this writer ran at — `ROPE_HALF_SWAP_SHA`, required at run time (the writer
    /// refuses to run unstamped). Every verdict line carries it, and the runner skips a cell
    /// only when sha AND group_0 md5 both match a logged verdict.
    sha: String,
}

fn main() {
    let Some(root) = std::env::var_os("ROPE_HALF_SWAP_ROOT") else {
        panic!("ROPE_HALF_SWAP_ROOT=<dir> names where the cells are written");
    };
    let sha = std::env::var("ROPE_HALF_SWAP_SHA")
        .unwrap_or_else(|_| panic!("ROPE_HALF_SWAP_SHA=<sha> stamps the commit the writer runs at — the writer refuses to run unstamped"));
    let root = std::path::Path::new(&root);
    std::fs::create_dir_all(root).unwrap();
    // One gemma-4 layer's ropes, both widths: hd 256 (the sliding layers' Q16 and K8) and
    // hd 512 (the global layers' Q16 and K1), at the rungs' own row counts (19..128 sampled
    // at the corrupt ones — 23, 29, 31 — plus the decode and full-chunk ends).
    let mut cells = Vec::new();
    for form in [Form::HalfSwap, Form::HalfSwap4, Form::Main] {
        for &(hd, heads) in &[(256u32, 16u32), (256, 8), (512, 16), (512, 1)] {
            for &mq in &[1u32, 23, 29, 31, 128] {
                cells.push(emit_cell(form, Geo { mq, hd, heads }, root, &sha));
            }
        }
    }
    // ⛔ A NEW-FORM CELL THAT CANNOT LOWER IS A FAILURE OF THAT FORM, not a count.
    let new_form_missing: Vec<&Cell> = cells
        .iter()
        .filter(|c| c.form != Form::Main.tag() && c.missing.is_some())
        .collect();
    assert!(
        new_form_missing.is_empty(),
        "the new forms must lower at every stated shape — {} cell(s) refused: {}",
        new_form_missing.len(),
        new_form_missing
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
    // A main-form cell that cannot lower is a fact about main — printed verbatim for the report.
    for c in cells.iter().filter(|c| c.missing.is_some()) {
        eprintln!(
            "MAIN-FORM REFUSAL mq={} hd={} heads={}: {}",
            c.mq,
            c.hd,
            c.heads,
            c.missing.as_deref().unwrap_or("")
        );
    }
    let written = cells.iter().filter(|c| c.missing.is_none()).count();
    eprintln!(
        "rope half-swap sweep: {written} bundles written, {} main-form refusal(s)",
        cells.len() - written
    );
}

/// The cell's directory name: `f<form>-mq<mq>-hd<hd>-h<heads>` — the geometry in the name, so a
/// verdict line names the shape it passed at.
fn cell_name(form: Form, geo: Geo) -> String {
    format!(
        "f{}-mq{}-hd{}-h{}",
        form.tag(),
        geo.mq,
        geo.hd,
        geo.heads
    )
}

/// The launch map: the six parameters in segment 3 (out labeled `Logits` by the writer — the
/// runner's read-back role), the synthetics declared by whichever form emits.
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

fn emit_cell(form: Form, geo: Geo, root: &std::path::Path, sha: &str) -> Cell {
    let dir = root.join(cell_name(form, geo));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let l = build_layout(geo);

    // ⛔ A BUILD-TIME PANIC IN THE EMITTER IS A REFUSAL TOO — recorded verbatim, never unwound
    // past. The hook is silenced so a panicking cell does not spam the run.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let walked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Each new form states its gapped ops' names and HOW MANY views of each are gapped
        // (a half-move: read + write; a half-multiply: x, sin' and rs).
        let (ops, patches) = match form {
            Form::HalfSwap => {
                let (ops, names) = half_swap_ops(geo, &l)?;
                let gapped: Vec<(String, usize)> =
                    names.into_iter().map(|n| (n, 2usize)).collect();
                (ops, gap_patches(geo, &gapped))
            }
            Form::HalfSwap4 => {
                let (ops, names) = half_swap4_ops(geo, &l)?;
                let gapped: Vec<(String, usize)> =
                    names.into_iter().map(|n| (n, 3usize)).collect();
                (ops, gap_patches(geo, &gapped))
            }
            Form::Main => (main_ops(geo, &l)?, Vec::new()),
        };
        let mut rendered = superdsc::render_dxp_input(&ops, superdsc::FoldGrouping::Split)
            .map_err(|e| format!("render: {e}"))?;
        let descriptors = rendered
            .iter()
            .map(|g| g.files.len() as u32 - 1) // every file but bundle.mlir is one trip
            .sum();
        patch_back_gaps(&mut rendered, &patches);
        superdsc::write_rendered_dxp_input(&dir, &rendered)
            .map_err(|e| format!("write: {e}"))?;
        Ok::<u32, String>(descriptors)
    }));
    std::panic::set_hook(prev_hook);

    let (descriptors, missing) = match walked {
        Ok(Ok(n)) => (n, None),
        Ok(Err(e)) => (0, Some(e)),
        Err(payload) => (0, Some(format!("PANIC: {}", panic_message(&payload)))),
    };

    // The runner's probe: rot for every form but halfswap4, whose own intermediate is rs —
    // it has no rot tensor.
    let probe_role = match form {
        Form::HalfSwap4 => SynthRole::Rs,
        Form::HalfSwap | Form::Main => SynthRole::Rot,
    };

    let cell = Cell {
        form: form.tag().into(),
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
        sin_sign: sin_signs(geo.hd),
        probe_synth: PlaceId::Act(OUT_TID).synth(probe_role).to_string(),
        probe_kind: probe_role.to_string(),
        descriptors,
        missing,
        sha: sha.into(),
    };
    write_cell(&dir, &l, &cell);
    cell
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  The new form
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The half-move's source column and op func for the output range `[o_lo, o_hi)`, READ from
/// `rope_p_entry` — the ONE statement of the rotate's pairing and sign. Asserts the permutation's
/// own uniformity over the range (whole sticks swap halves at hd ∈ {256, 512}); a range where it
/// is not uniform is a harness bug, not a cell.
fn half_move_of(hd: u32, o_lo: u32, o_hi: u32) -> (i64, OpFunc) {
    let mut rel = None;
    let mut func = None;
    for o in o_lo..o_hi {
        let inn = (0..hd)
            .find(|&i| rope_p_entry(hd as usize, i as usize, o as usize) != 0)
            .unwrap_or_else(|| panic!("rope_p_entry(hd={hd}) states no source for column {o}"));
        let v = rope_p_entry(hd as usize, inn as usize, o as usize);
        let r = inn as i64 - o as i64;
        assert_eq!(
            *rel.get_or_insert(r),
            r,
            "rope_p_entry(hd={hd}) is not a uniform half-shift over [{o_lo},{o_hi})"
        );
        let f = if v < 0 { OpFunc::Neg } else { OpFunc::Identity };
        assert_eq!(
            *func.get_or_insert(f),
            f,
            "rope_p_entry(hd={hd}) changes sign inside [{o_lo},{o_hi})"
        );
    }
    (rel.unwrap(), func.unwrap())
}

/// The signed sin' table's per-column sign, READ from `rope_p_entry` — one call per column,
/// `±1` never typed, the entry's own type carried through. The ONE statement of the sign law:
/// the runner's fill builds `sin' = sign[col % hd] · sin` from this array, so the op never
/// carries a sign.
fn sin_signs(hd: u32) -> Vec<i8> {
    (0..hd)
        .map(|o| {
            let inn = (0..hd)
                .find(|&i| rope_p_entry(hd as usize, i as usize, o as usize) != 0)
                .unwrap_or_else(|| panic!("rope_p_entry(hd={hd}) states no source for column {o}"));
            rope_p_entry(hd as usize, inn as usize, o as usize)
        })
        .collect()
}

/// ONE gapped `[mb, out, y]` view of the `[mq, total]` plane — the card-proven spelling: layout
/// `["mb","out","y"]`, stick `out`, declared extent `hd`, device-element offset. Stated ONCE;
/// the half-moves and the half-multiplies consume it alike.
fn gapped_view(
    is_input: bool,
    name: &str,
    geo: Geo,
    off: u32,
    syms: [IterSym; 3],
) -> Result<AnyTensorArg, String> {
    Ok(AnyTensorArg::R3(
        TensorArg::<3>::new(
            is_input,
            name.to_string(),
            Role::Output,
            [Scale::Active; 3],
            syms,
            ["mb", "out", "y"],
            "out",
            Allocation::Hbm,
        )
        .map_err(|e| e.0)?
            .with_row_blocked(true)
            .with_device_extent("out", geo.hd)
            .map_err(|e| e.0)?
            .with_offset(off),
    ))
}

/// The half-views' shared work plan: `[mb: mq, out: hd/2 (stick), y: heads]`, `y` PINNED
/// unsplit — the per-core start fold for a gapped stick view folds only the mb and out corners
/// (`off_view`, `sdsc_abstract.rs`), so a y split's corner would be dropped and every core past
/// the first would start at head 0's window. Stated ONCE; both new forms consume it.
fn half_plan(geo: Geo) -> Result<(WorkPlan, [IterSym; 3]), String> {
    let dims = [
        ItDim {
            name: "mb",
            size: geo.mq,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
        ItDim {
            name: "out",
            size: geo.hd / 2,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        },
        ItDim {
            name: "y",
            size: geo.heads,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
    ];
    let plan = WorkPlan::divide(&dims, CoreBudget::whole(), |dims, cap| {
        let mut splits = distribute_cores(dims, cap);
        splits.remove("y");
        splits
    })?;
    let syms = plan.iter_syms(["mb", "out", "y"]);
    Ok((plan, syms))
}

/// ONE half-move: `func` over a `[mb, out, y]` view of the `[mq, total]` RowBlocked plane — the
/// read at `read_off` device elements, the write at `write_off`. The view's `out` sweeps `hd/2`
/// (the half) under a declared extent of `hd` (`.with_device_extent`), so the device walk's
/// head stride needs the stick-axis gap — THE STEP-0 BRIDGE patches it into the rendered input;
/// A1's `BackGap` type replaces the patch.
fn half_move(
    op_name: &str,
    func: OpFunc,
    geo: Geo,
    read: &str,
    read_off: u32,
    write: &str,
    write_off: u32,
    plan: &WorkPlan,
    syms: [IterSym; 3],
    sib: &mut i64,
    l: &BundleLayout,
) -> Result<EmittedOp<TpWorld<1>>, String> {
    let op = OpSpec::new(
        func,
        plan.clone(),
        vec![gapped_view(true, read, geo, read_off, syms)?],
        gapped_view(false, write, geo, write_off, syms)?,
    )
    .map_err(|e| e.0)?;
    emit_sdsc_tiled(
        op_name,
        &op,
        &SdscFoldSet::new(op.iter.cores_used()),
        sib,
        Some(l),
    )
    .map_err(|e| e.0)
}

/// ONE half-multiply: `rs = x · sin'` over the same gapped views — x read at `x_off`, sin' read
/// at `sin_off`, rs written at `rs_off` (device-element offsets). `mul` is the card's own
/// two-input elementwise op (`OpFunc::Multiply`); the SIGN rides the sin' fill, never the op.
fn half_multiply(
    op_name: &str,
    geo: Geo,
    x: &str,
    x_off: u32,
    sins: &str,
    sin_off: u32,
    rs: &str,
    rs_off: u32,
    plan: &WorkPlan,
    syms: [IterSym; 3],
    sib: &mut i64,
    l: &BundleLayout,
) -> Result<EmittedOp<TpWorld<1>>, String> {
    let op = OpSpec::new(
        OpFunc::Multiply,
        plan.clone(),
        vec![
            gapped_view(true, x, geo, x_off, syms)?,
            gapped_view(true, sins, geo, sin_off, syms)?,
        ],
        gapped_view(false, rs, geo, rs_off, syms)?,
    )
    .map_err(|e| e.0)?;
    emit_sdsc_tiled(
        op_name,
        &op,
        &SdscFoldSet::new(op.iter.cores_used()),
        sib,
        Some(l),
    )
    .map_err(|e| e.0)
}

/// The new form's ops: the two half-moves (rot's first half from x's second, negated; rot's
/// second half from x's first, verbatim) plus the three whole-tensor legs exactly as `rope_at`
/// emits them. Returns the ops and the half-moves' own names (the bridge keys on them).
fn half_swap_ops(
    geo: Geo,
    l: &BundleLayout,
) -> Result<(Vec<EmittedOp<TpWorld<1>>>, Vec<String>), String> {
    let (mq, hd, total) = (geo.mq, geo.hd, geo.total());
    let half = hd / 2;
    assert!(
        half % 64 == 0,
        "hd {hd}: the half {half} is not whole sticks — this sweep states hd 256/512"
    );
    // DECLARE BEFORE USE (`rope_at`'s own law): the synthetics at their TRUE full extent, so the
    // half-moves' partial windows price against the whole plane.
    let out_id = PlaceId::Act(OUT_TID);
    let rot = syn(Some(l), out_id.synth(SynthRole::Rot));
    let xc = syn(Some(l), out_id.synth(SynthRole::Xc));
    let rs = syn(Some(l), out_id.synth(SynthRole::Rs));
    for r in [SynthRole::Rot, SynthRole::Xc, SynthRole::Rs] {
        l.synth(out_id.synth(r), &[mq, total]);
    }
    let (x, cos, sin, out) = (
        act_name(X_TID),
        act_name(COS_TID),
        act_name(SIN_TID),
        act_name(OUT_TID),
    );
    let mut sib = 0i64;
    let mut ops = Vec::new();

    // The plan is shared by both half-moves (same work, same views' extents); `half_plan`
    // states it once for both new forms.
    let (plan, syms) = half_plan(geo)?;

    let mut half_move_names = Vec::new();
    for (o_lo, o_hi) in [(0, half), (half, hd)] {
        let (rel, func) = half_move_of(hd, o_lo, o_hi);
        // The read window starts `max(rel, 0)` columns in, the write `max(-rel, 0)` — the
        // permutation's own shift, never restated: a positive rel reads the second half and
        // writes the first, a negative rel the mirror.
        let (read_col, write_col) = if rel > 0 {
            (rel as u32, 0)
        } else {
            (0, (-rel) as u32)
        };
        let read_off = rc_of(mq, total, 0, read_col, Df::Fp16).into_raw_elems();
        let write_off = rc_of(mq, total, 0, write_col, Df::Fp16).into_raw_elems();
        let op_name = format!(
            "rope_{}_o{OUT_TID}",
            if matches!(func, OpFunc::Neg) {
                "neg"
            } else {
                "mov"
            }
        );
        ops.push(half_move(
            &op_name,
            func,
            geo,
            &x,
            read_off,
            &rot,
            write_off,
            &plan,
            syms,
            &mut sib,
            l,
        )?);
        half_move_names.push(op_name);
    }

    // The three whole-tensor legs, verbatim from `rope_at`'s own emission.
    let rows_all = RowCount::of_query_rows(QueryRowCount::of_mq(mq));
    let cols_all = BlockCols::of_feature_cols(total);
    ops.push(
        pw2(
            &format!("rope_xc_o{OUT_TID}"),
            "multiply",
            rows_all,
            cols_all,
            In::full(&rb(&x, mq, total)),
            In::full(&rb(&cos, mq, total)),
            &rb(&xc, mq, total),
            &mut sib,
            Some(l),
        )
        .map_err(|e| e.0)?,
    );
    ops.push(
        pw2(
            &format!("rope_rs_o{OUT_TID}"),
            "multiply",
            rows_all,
            cols_all,
            In::full(&rb(&rot, mq, total)),
            In::full(&rb(&sin, mq, total)),
            &rb(&rs, mq, total),
            &mut sib,
            Some(l),
        )
        .map_err(|e| e.0)?,
    );
    ops.push(
        pw2(
            &format!("rope_out_o{OUT_TID}"),
            "add",
            rows_all,
            cols_all,
            In::full(&rb(&xc, mq, total)),
            In::full(&rb(&rs, mq, total)),
            &rb(&out, mq, total),
            &mut sib,
            Some(l),
        )
        .map_err(|e| e.0)?,
    );
    Ok((ops, half_move_names))
}

/// The 4-op form's ops: NO rot tensor. `xc = x·cos` (the whole-tensor leg, verbatim from
/// `rope_at`), then the two half-multiplies writing the two halves of ONE `rs` through the same
/// gapped views — `rs[o] = x[inn(o)] · sin'[o]`, the sign riding the sin' fill, never the op —
/// then `out = xc + rs`. Returns the ops and the half-multiplies' own names (the bridge keys on
/// them; all THREE of each one's views are gapped — x, sin' and rs).
fn half_swap4_ops(
    geo: Geo,
    l: &BundleLayout,
) -> Result<(Vec<EmittedOp<TpWorld<1>>>, Vec<String>), String> {
    let (mq, hd, total) = (geo.mq, geo.hd, geo.total());
    let half = hd / 2;
    assert!(
        half % 64 == 0,
        "hd {hd}: the half {half} is not whole sticks — this sweep states hd 256/512"
    );
    // DECLARE BEFORE USE (`rope_at`'s own law): xc and rs at their TRUE full extent, so the
    // half-multiplies' partial windows price against the whole plane. rot does not exist here.
    let out_id = PlaceId::Act(OUT_TID);
    let xc = syn(Some(l), out_id.synth(SynthRole::Xc));
    let rs = syn(Some(l), out_id.synth(SynthRole::Rs));
    for r in [SynthRole::Xc, SynthRole::Rs] {
        l.synth(out_id.synth(r), &[mq, total]);
    }
    let (x, cos, sins, out) = (
        act_name(X_TID),
        act_name(COS_TID),
        act_name(SINS_TID),
        act_name(OUT_TID),
    );
    let mut sib = 0i64;
    let mut ops = Vec::new();

    // `xc = x·cos`, verbatim from `rope_at`'s own emission.
    let rows_all = RowCount::of_query_rows(QueryRowCount::of_mq(mq));
    let cols_all = BlockCols::of_feature_cols(total);
    ops.push(
        pw2(
            &format!("rope_xc_o{OUT_TID}"),
            "multiply",
            rows_all,
            cols_all,
            In::full(&rb(&x, mq, total)),
            In::full(&rb(&cos, mq, total)),
            &rb(&xc, mq, total),
            &mut sib,
            Some(l),
        )
        .map_err(|e| e.0)?,
    );

    // The two half-multiplies over the output range `[o_lo, o_hi)`. The x read and rs write
    // windows are the half-move's own (the permutation's shift, never restated); the sin' view
    // sweeps the SAME output column range rs writes — one law, `half_move_of`'s.
    let (plan, syms) = half_plan(geo)?;
    let mut half_multiply_names = Vec::new();
    for (o_lo, o_hi) in [(0, half), (half, hd)] {
        // The move's func is not consumed here — the sign rides the sin' fill (`sin_signs`).
        let (rel, _func) = half_move_of(hd, o_lo, o_hi);
        let (x_col, rs_col) = if rel > 0 {
            (rel as u32, 0)
        } else {
            (0, (-rel) as u32)
        };
        let x_off = rc_of(mq, total, 0, x_col, Df::Fp16).into_raw_elems();
        let sin_off = rc_of(mq, total, 0, o_lo, Df::Fp16).into_raw_elems();
        let rs_off = rc_of(mq, total, 0, rs_col, Df::Fp16).into_raw_elems();
        let op_name = format!(
            "rope_{}_o{OUT_TID}",
            if o_lo == 0 { "rslo" } else { "rshi" }
        );
        ops.push(half_multiply(
            &op_name,
            geo,
            &x,
            x_off,
            &sins,
            sin_off,
            &rs,
            rs_off,
            &plan,
            syms,
            &mut sib,
            l,
        )?);
        half_multiply_names.push(op_name);
    }

    // `out = xc + rs`, verbatim from `rope_at`'s own emission.
    ops.push(
        pw2(
            &format!("rope_out_o{OUT_TID}"),
            "add",
            rows_all,
            cols_all,
            In::full(&rb(&xc, mq, total)),
            In::full(&rb(&rs, mq, total)),
            &rb(&out, mq, total),
            &mut sib,
            Some(l),
        )
        .map_err(|e| e.0)?,
    );
    Ok((ops, half_multiply_names))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  The baseline
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Main's form, verbatim: `rope_at` at its const head dim (the only dispatch is the value → the
/// const, `rope_at`'s own contract).
fn main_ops(geo: Geo, l: &BundleLayout) -> Result<Vec<EmittedOp<TpWorld<1>>>, String> {
    let (mq, total) = (geo.mq, geo.total());
    let mut sib = 0i64;
    let ra = RopeAt {
        name: "rope",
        x: act_name(X_TID),
        cos: act_name(COS_TID),
        sin: act_name(SIN_TID),
        out: act_name(OUT_TID),
        t: OUT_TID,
        mq,
        total,
        p_tid: P_TID,
        sym_id_base: &mut sib,
        layout: Some(l),
        rows_are_requests: false,
    };
    match geo.hd {
        256 => rope_at::<TpWorld<1>, 256>(ra).map_err(|e| e.message),
        512 => rope_at::<TpWorld<1>, 512>(ra).map_err(|e| e.message),
        hd => panic!("the sweep states hd 256/512; this cell states {hd}"),
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  The bridge (harness-only; dies with A1)
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The gap law, stated ONCE for the bridge: the view's declared extent minus its swept extent
/// rounded UP to whole sticks (torch-spyre's production emitter, `superdsc.py:1630-1633`), in
/// ELEMENTS. No literal — the number comes from the cell's own stated extents.
fn stick_axis_gap(declared: u32, swept: u32, df: Df) -> u32 {
    let per_stick = df.elems_per_stick();
    let swept_whole = swept.div_ceil(per_stick) * per_stick;
    assert!(
        declared >= swept_whole,
        "the declared extent {declared} is below the whole-stick swept extent {swept_whole}"
    );
    declared - swept_whole
}

/// One gapped view's patch: the op that owns the view, the view's arg index (inputs then output,
/// the output LAST), and the gap in ELEMENTS.
struct GapPatch {
    op: String,
    view: usize,
    gap_elems: u32,
}

/// THE STEP-0 BRIDGE: patch `backGapCore_` into the gapped views' allocate nodes of the rendered
/// dxp input. Every half-swept view of a full-head plane carries it — a half-move's read of x
/// and write of rot (2 views), a half-multiply's x, sin' and rs (3 views). The op names come
/// from the emission itself, never restated here; the gap is derived once, from the cell's
/// stated extents.
fn gap_patches(geo: Geo, gapped_ops: &[(String, usize)]) -> Vec<GapPatch> {
    let gap = stick_axis_gap(geo.hd, geo.hd / 2, Df::Fp16);
    assert!(gap > 0, "hd {}: the stick-axis gap is zero", geo.hd);
    gapped_ops
        .iter()
        .flat_map(|(op, views)| {
            (0..*views).map(move |view| GapPatch {
                op: op.clone(),
                view,
                gap_elems: gap,
            })
        })
        .collect()
}

/// Apply the patches to every rendered group's files. An empty patch list (the baseline cells)
/// touches nothing. EVERY patch must apply exactly once across the render — a count below the
/// list's length is a bridge that silently patched nothing, and the writer refuses to write it.
fn patch_back_gaps(rendered: &mut [superdsc::GroupInput], patches: &[GapPatch]) {
    let mut applied = 0usize;
    for g in rendered {
        for (_file, contents) in &mut g.files {
            for p in patches {
                applied += usize::from(patch_one(contents, p));
            }
        }
    }
    assert_eq!(
        applied,
        patches.len(),
        "the bridge applied {applied} of {} patches — a gapped view went unpatched",
        patches.len()
    );
}

/// Patch ONE rendered file: locate the op's own allocate node for view `view` (the file's
/// top-level key is `{li}_{op}`, so the key ends `_{op}":`; the node is named
/// `allocate-Tensor{view}_hbm`), assert it carried NO `backGapCore_` before (nothing applied
/// twice) and occurs exactly once, then splice
/// `"backGapCore_":{"out":{"-1":"<gap>"}},` immediately before the node's own
/// `"coordinates_"`. Returns whether this file held the patch's node. A file that is not this
/// op's, or holds no such node, is left untouched.
fn patch_one(contents: &mut String, p: &GapPatch) -> bool {
    let node = format!("\"name_\":\"allocate-Tensor{}_hbm\"", p.view);
    let key = format!("_{}\":", p.op);
    if !contents.contains(&key) || !contents.contains(&node) {
        return false; // not this op's file, or this op has no such view
    }
    let count = contents.matches(&node).count();
    assert_eq!(
        count, 1,
        "the bridge found {count} nodes named {node} in one file — the op's own view must be one"
    );
    let node_at = contents.find(&node).unwrap();
    let coord = "\"coordinates_\":";
    let ins_at = contents[node_at..]
        .find(coord)
        .expect("the allocate node states coordinates_")
        + node_at;
    assert!(
        !contents[node_at..ins_at].contains("\"backGapCore_\""),
        "the allocate node already carries a backGapCore_ — nothing is applied twice"
    );
    let insertion =
        format!("\"backGapCore_\":{{\"out\":{{\"-1\":\"{}\"}}}},", p.gap_elems);
    contents.insert_str(ins_at, &insertion);
    true
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  The cell's files
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Write the launch map and the manifest. The `places` list carries the five parameters (the
/// read-back `out` labeled `Logits` — the runner's own read-back role, the one label that is the
/// producer's statement of intent); the synthetics ride the `synth` list by NAME (their names
/// carry a role suffix, not a bare tid), at the walk's own segment/offset from `l.synth`.
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
