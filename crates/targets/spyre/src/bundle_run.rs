// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! `scr bundle run` — LAUNCH A BUNDLE THIS BINARY DID NOT BAKE.
//!
//! Every other launch path in this crate runs a bundle the BUILD baked in: `BundleCode` is
//! `&'static` data, `superdsc_bake` produced it, and the model it belongs to picked its geometry.
//! That leaves no way to run a bundle produced OUTSIDE this repo — which is what a third-party KTIR
//! producer emits, and what `dxp_standalone` compiles.
//!
//! # WHAT A BUNDLE DIRECTORY IS
//!
//! `dxp_standalone -d <dir> -b sentient` reads `sdsc_*.json` and writes `spyreCodeDir/` beside them:
//!
//! ```text
//! <dir>/sdsc_0.json ...          the descriptors (the compile input)
//! <dir>/placements.json          each tensor's (segment, offset, size)  ← the producer's launch map
//! <dir>/spyreCodeDir/init_binary.bin   dxp's device image
//! <dir>/spyreCodeDir/spyrecode.json    the job plan: allocation, init transfer, corrections, entry
//! ```
//!
//! ⛔ `placements.json` IS NOT REDUNDANT WITH THE DESCRIPTORS. [`Executor::prepare`] allocates one
//! device region per tensor segment and keeps `tensor_allocs` POSITIONAL — entry *i* backs logical
//! segment *i* — so a launch has to know which segment each tensor lives in and at what offset. The
//! descriptors carry addresses whose segment bases stay SYMBOLIC until this launch's own correction
//! flits patch them, so that map cannot be read back out of them without re-deriving the producing
//! emitter's addressing scheme. The producer states it instead.
//!
//! # WHY THE PROGRAMS ARE EMPTY
//!
//! `LaunchGroup::programs` are `IRFunction` values and NOTHING on the card path reads them: the
//! device executes `init_binary`, which is their artifact. They are consumed only by `runner.rs`,
//! the `ktir_emulator` path behind the `runner` feature. So a card launch needs no serialised KTIR,
//! and this states `programs: &[]` rather than inventing a KTIR the directory does not contain.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

/// One `placements.json` entry.
#[derive(Debug, serde::Deserialize)]
struct PlaceRow {
    tid: u32,
    role: String,
    segment: u32,
    bank: u32,
    offset: u64,
    size: u64,
}

/// One `synth` entry: an intermediate the lowering minted, with the segment and offset it was
/// bump-allocated at. Written by `bake::write_placements` so a host can read a STAGE off the card.
#[derive(Debug, serde::Deserialize)]
struct SynthRow {
    name: String,
    segment: u32,
    offset: u64,
    size: u64,
}

#[derive(Debug, serde::Deserialize)]
struct PlacementsFile {
    segment_bytes: Vec<u64>,
    places: Vec<PlaceRow>,
    #[serde(default)]
    synth: Vec<SynthRow>,
}

/// `--input t0=path.bin`, `--output t2=path.bin`.
fn split_binding(s: &str, what: &str) -> Result<(u32, String)> {
    let (tid, path) = s
        .split_once('=')
        .ok_or_else(|| anyhow!("--{what} wants `t<id>=<path>`, got `{s}`"))?;
    let tid = tid
        .strip_prefix('t')
        .unwrap_or(tid)
        .parse::<u32>()
        .with_context(|| format!("--{what} `{s}`: `{tid}` is not a tensor id"))?;
    Ok((tid, path.to_string()))
}

/// ⛔⛔⛔ AN fp16 TENSOR'S DEVICE LAYOUT IS STICK-MAJOR, AND THE HOST HAS TO SPEAK IT.
///
/// MEASURED on card, `rmsnorm_granite` `[64, 4096]`: exactly the 64 DIAGONAL tiles — `(row r,
/// stick r)` — came back right and the other 4047 were wrong (0.0193 within_2pct, med|rel| 1.383).
/// Reading the host's row-major buffer through
///
/// ```text
///     device byte(r, c) = (c/64)·rows·128 + r·128 + (c%64)·2
/// ```
///
/// reproduces the card's output at within_2pct = **1.0000** over all 262144 elements (row-major
/// control: 0.0197). Staging `x` through that map and de-scattering the readback took the SAME
/// descriptors, unchanged, from 0.0193 to 0.9999.
///
/// ⭐ IT IS NOT OUR INVENTION, AND THAT IS WHY IT IS SAFE. It is scratchy's own production host —
/// `forward_tape::stick_scatter`, "element (r, c) lands at (c/64)*(rows*64) + r*64 + (c%64)", which
/// every prefill forward applies to its embedding and rotary tables — and production's
/// `KernelWeight` states the same map for a matmul weight as `device_size = [out/64, in, 64]`,
/// `stride_map = [64, in, 1]`. The descriptors agree independently: `rmg_o2`'s output start
/// addresses step 256 B per core at 2 rows per core (one stick per row), and the rope/matmul
/// `KERNEL` tables step `in·128` per out-stick.
///
/// ⭐ AND IT IS THE IDENTITY AT ONE ROW (their own `stick_scatter_is_the_identity_at_one_row`), which
/// is exactly why a launcher that never did it got every one-row bundle right and looked correct.
///
/// `out[device] = row_major[host]`, one table read both ways so the two directions cannot drift.
fn dev_order(rows: usize, cols: usize) -> Vec<usize> {
    let mut v = vec![0usize; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            v[(c / 64) * (rows * 64) + r * 64 + (c % 64)] = r * cols + c;
        }
    }
    v
}

/// Permute 2-byte elements between the host's row-major order and the device's stick order.
/// `to_device` stages a bind; `!to_device` de-scatters a readback.
fn restick(bytes: &[u8], rows: usize, cols: usize, to_device: bool) -> Vec<u8> {
    let mut out = vec![0u8; bytes.len()];
    for (dev, &host) in dev_order(rows, cols).iter().enumerate() {
        let (src, dst) = if to_device { (host, dev) } else { (dev, host) };
        out[dst * 2..dst * 2 + 2].copy_from_slice(&bytes[src * 2..src * 2 + 2]);
    }
    out
}

/// `stick:t<id>=<rows>` — the ROW COUNT the descriptors address tensor `<id>` with. The column count
/// follows from the placement (`size/2/rows`), so the caller states the one number it cannot derive.
fn split_stick(s: &str) -> Result<(u32, u64)> {
    let body = s.strip_prefix("stick:").unwrap_or(s);
    let (tid, rows) = body
        .split_once('=')
        .ok_or_else(|| anyhow!("wants `stick:t<id>=<rows>`, got `{s}`"))?;
    let tid = tid
        .strip_prefix('t')
        .unwrap_or(tid)
        .parse::<u32>()
        .with_context(|| format!("`{s}`: `{tid}` is not a tensor id"))?;
    let rows = rows
        .parse::<u64>()
        .with_context(|| format!("`{s}`: `{rows}` is not a row count"))?;
    if rows == 0 {
        bail!("`{s}`: zero rows");
    }
    Ok((tid, rows))
}

/// `raw:t<id>` — BIND THIS TENSOR'S BYTES VERBATIM, with NO IEEE→SEN fp16 re-encoding.
///
/// ⛔⛔⛔ AN INDEX BUFFER IS NOT A MAGNITUDE. Every staging path in [`crate::superdsc_exec`] ends in
/// `sen_convert::ieee_to_sen_bytes`, unconditionally, because until the gather landed every bound
/// tensor genuinely WAS an fp16 magnitude the card holds in SEN169 (sign 1, exp 6 bias-31, mantissa
/// 9). A gather's index table is int32: an fp16 re-encoding reads each 4-byte entry as TWO fp16
/// values and rewrites both mantissas, so row 37 does not become "roughly 37" — it becomes a
/// different, VALID, arbitrary row. The output is then a fluent embedding of the wrong tokens, with
/// no fault, no shape error, and nothing downstream that compares the two.
/// [`crate::superdsc_exec::Executor::bind_input_raw`] is the door production's own KV block table
/// already goes through; this is the spelling that reaches it from a bundle directory.
///
/// ⛔ IT IS A MARKER, NOT A BINDING. The bytes still come from this tensor's own `t<id>=<path>`
/// entry — `raw:` only says how to stage them — so it carries no `=`, and a spelling that has one is
/// refused rather than quietly read as a path nothing would ever open.
fn split_raw(s: &str) -> Result<u32> {
    let body = s
        .strip_prefix("raw:")
        .ok_or_else(|| anyhow!("wants `raw:t<id>`, got `{s}`"))?;
    if body.contains('=') {
        bail!(
            "`{s}`: `raw:` is a MARKER, not a binding — it takes no `=<path>`. The bytes come from \
             this tensor's own `t<id>=<path>` entry and `raw:` only says to stage them verbatim, so \
             pass both: `t<id>=<path> raw:t<id>`."
        );
    }
    let tid = body
        .strip_prefix('t')
        .unwrap_or(body)
        .parse::<u32>()
        .with_context(|| format!("`{s}`: `{body}` is not a tensor id"))?;
    Ok(tid)
}

/// `const:t<id>=<value>` — bind a one-element fp16 SCALAR stated on the command line.
///
/// A producer's reserved device constants (see [`RESERVED_TID_FLOOR`]) are single values: an
/// rmsnorm's `1/cols`, a `Program::ScalarMul`'s multiplier. They bind exactly like any other fp16
/// tensor, so all this spelling saves is a two-byte file — but a two-byte file is precisely where a
/// launch goes wrong in silence, because nothing about `t4294967275=/tmp/x.bin` says WHICH number is
/// in it, and a scale is the one operand whose wrongness is a clean rescaling of an otherwise
/// correct-looking answer. `const:t4294967275=12.0` puts the value in the command, where the run's
/// own log records it.
///
/// ⛔ NON-FINITE IS REFUSED, and so is any value fp16 cannot hold: a scale that overflows to `inf`
/// makes the whole output inf/NaN, and — the measured failure — a nonzero scale that UNDERFLOWS to
/// zero makes the card multiply by exactly 0, which is the same all-zero output a destroyed weight
/// gave. In both cases the value BOUND is not the value STATED, so it is named rather than rounded.
fn split_const(s: &str) -> Result<(u32, Vec<u8>)> {
    let body = s
        .strip_prefix("const:")
        .ok_or_else(|| anyhow!("wants `const:t<id>=<value>`, got `{s}`"))?;
    let (tid, value) = body
        .split_once('=')
        .ok_or_else(|| anyhow!("wants `const:t<id>=<value>`, got `{s}`"))?;
    let tid = tid
        .strip_prefix('t')
        .unwrap_or(tid)
        .parse::<u32>()
        .with_context(|| format!("`{s}`: `{tid}` is not a tensor id"))?;
    let v = value
        .parse::<f32>()
        .with_context(|| format!("`{s}`: `{value}` is not a number"))?;
    if !v.is_finite() {
        bail!("`{s}`: `{value}` is not finite, and every element it touches would become inf/NaN");
    }
    let h = half::f16::from_f32(v);
    if !h.is_finite() {
        bail!(
            "`{s}`: {v} overflows fp16 (max 65504), so the value bound would be inf and every \
             element it touches would be inf/NaN"
        );
    }
    if h.to_f32() == 0.0 && v != 0.0 {
        bail!(
            "`{s}`: {v} underflows fp16 to exactly 0, so the card would multiply by zero — the same \
             all-zero output a destroyed weight gives, from a launch that reports success"
        );
    }
    Ok((tid, h.to_le_bytes().to_vec()))
}

/// A bound tensor's BYTES, however the caller spelled them.
///
/// ⭐ ONE LIST, not a file map beside a const map: `is_weight`, the source-id count, the placement
/// size comparison and the stick-major refusal all have to treat the two spellings identically, and
/// a second map is a second thing for each of them to forget.
enum Payload {
    /// `t<id>=<path>`: the file's whole contents.
    File(String),
    /// `const:t<id>=<value>`: the fp16 encoding of a value stated on the command line.
    Inline(Vec<u8>),
}

impl Payload {
    fn bytes(&self, tid: u32) -> Result<Vec<u8>> {
        match self {
            Payload::File(path) => {
                std::fs::read(path).with_context(|| format!("--input t{tid}={path}"))
            }
            Payload::Inline(b) => Ok(b.clone()),
        }
    }

    /// How the caller stated it, so an error message can be pasted back into the command.
    fn how(&self) -> String {
        match self {
            Payload::File(p) => p.clone(),
            Payload::Inline(b) => {
                format!("const:{}", half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            }
        }
    }
}

/// Stage a `raw:t<id>` bind — the bytes into the shadow VERBATIM, through the one constructor that
/// can mint the value [`crate::superdsc_exec::Executor::bind_input_raw`] accepts.
///
/// ⛔ REACHED ONLY FROM THE POST-PREPARE SOURCE LOOP, never from weight staging: `refill_activations`
/// is the loop that consults `raw_ids` and copies instead of converting, and `stage_weights` refuses a
/// raw id outright. The weight arm refuses it earlier, where the caller's spelling is still in hand.
///
/// ⭐ THE PAYLOAD CHECK IS `place_raw`'s AND NOT A SECOND ONE HERE. Its `Option` is the whole guard:
/// a payload longer than its own placement, or one that is not a whole number of 4-byte entries, has
/// no `RawBind` value to arrive in. Restating the rule here would be a copy of it to drift from.
///
/// ⛔ SHORT IS LEGITIMATE for a raw bind where it is not for an fp16 one, so this does NOT demand the
/// exact size the arms below do — `place_raw`'s own doc records the activations that bind less than
/// their placement on purpose (a pmask's one broadcast row, the index table's one pass block).
fn bind_raw(
    exec: &mut crate::superdsc_exec::Executor,
    tid: u32,
    bytes: Vec<u8>,
    how: &str,
    want: u64,
) -> Result<()> {
    let n = bytes.len();
    let Some(rb) = exec.place_raw(ktir_superdsc::place::PlaceId::Act(tid), bytes) else {
        bail!(
            "--input t{tid}={how} raw:t{tid}: {n} B cannot be bound verbatim into a {want} B \
             placement. A raw payload must be a whole number of 4-byte entries — a partial trailing \
             entry is a truncated address — and must not exceed its own placement, because past it \
             lie the next tensor's bytes and an index table reads those as row numbers."
        );
    };
    exec.bind_input_raw(rb);
    println!("  bound t{tid} RAW: {n} B verbatim, no IEEE→SEN conversion");
    Ok(())
}

/// Tensor ids at or above this are the producer's RESERVED DEVICE CONSTANTS, not kernel parameters —
/// `ktir_superdsc::reserved_tids` places them by counting down from `u32::MAX`. They are bound as
/// weights (staged once) rather than declared as per-run sources, because the executor's source model
/// is the contiguous id range `0..n`.
const RESERVED_TID_FLOOR: u32 = u32::MAX - 1024;

pub fn run(dir: &Path, input: &[String], output: &[String]) -> Result<()> {
    // ── The producer's launch map ──
    let pf: PlacementsFile = serde_json::from_str(
        &std::fs::read_to_string(dir.join("placements.json")).with_context(|| {
            format!(
                "{}: no placements.json. A bundle directory needs the producer's launch map beside \
                 the descriptors — see this module's docs for why the descriptors cannot supply it.",
                dir.display()
            )
        })?,
    )
    .with_context(|| format!("{}/placements.json did not parse", dir.display()))?;

    // ── dxp's own artefacts, ONE GROUP PER SUBDIRECTORY ──
    //
    // ⛔⛔⛔ THE CLAIM THAT USED TO STAND HERE WAS FALSE, AND ITS EVIDENCE BELONGED TO ANOTHER BUG.
    // It read: "A LAUNCH IS ONE JOB, AND A KERNEL IS USUALLY SEVERAL … launching [six descriptors in
    // one bundle.mlir] runs only the FIRST: measured on card for `rmsnorm_granite`, where elements
    // 0..63 of every row came back correct and everything from element 64 on was wrong but non-zero
    // — one stick computed, five jobs never run."
    //
    // That partial-looking result was the STICK-MAJOR host layout (see `dev_order`): the device
    // addresses a `[64, 4096]` activation as `[cols/64][rows][64]`, so a row-major host buffer agrees
    // with it on exactly the `(row r, stick r)` DIAGONAL — one 64-element run per row, the first of
    // them at elements 0..63 — and disagrees everywhere else with plausible nonzero numbers. Nothing
    // was unlaunched.
    //
    // MEASURED, once the host spoke the device's layout, with the descriptors untouched:
    //
    // ```text
    //   rmsnorm_granite   6 descriptors: 6 groups 0.9999  |  ONE group, one dxp, one launch 0.9999
    //   rope_q32         35 descriptors: 35 groups 0.9967 |  ONE group, one dxp, one launch 0.9967
    // ```
    //
    // Both whole-bundle compiles produce a spyrecode.json with exactly ONE `ComputeOnDevice` step and
    // an `init_binary` holding every descriptor (33792 B / 348672 B), and dxp's ModuleStitcher orders
    // them by the `{idx}_{op}` key the emission already carries. So the split is not required, and
    // production's own `GroupSize::PRODUCTION = 128` descriptors per group dir says the same.
    //
    // A bundle directory holding `group_0/`, `group_1/`, ... is still read as that many groups and
    // launched in order — a producer may split for its own reasons — and a directory with
    // `spyreCodeDir/` directly is the single-group form.
    let mut group_dirs: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("{}: cannot read", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("group_"))
        })
        .collect();
    // ⛔⛔⛔ NUMERIC, NOT LEXICAL. `sort()` orders these as STRINGS, so `group_10` lands before
    // `group_2` and a bundle of ten or more groups launches in an order that violates its own
    // dataflow. MEASURED on card, rope_q32 (35 groups: 32 per-head rotations, then xc/rs/add): the
    // three pointwise groups are `group_26..28`, which lexically precede `group_29` and `group_3..9`,
    // so the add consumed `rot` before 13 of the 32 rotations had run. Those 13 heads came back as
    // `x·cos` EXACTLY (rot still zero) and the other 19 were exact — 0.7862 within_2pct, and nothing
    // in the numbers looked like an ordering bug. The index is the descriptor's own position in
    // `bundle.mlir`, which is the dataflow order, so it is parsed and compared as a NUMBER.
    group_dirs.sort_by_key(|p| {
        let n = p
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        // A name that is not `group_<digits>` sorts LAST rather than aliasing to 0: two groups
        // sharing a key would make the launch order depend on the filesystem's own iteration order.
        n.strip_prefix("group_")
            .and_then(|d| d.parse::<u64>().ok())
            .unwrap_or(u64::MAX)
    });
    if group_dirs.is_empty() {
        group_dirs.push(dir.to_path_buf());
    }
    println!("  {} launch group(s)", group_dirs.len());
    // ── The layout the executor indexes ──
    let mut segment_bytes = [0u64; crate::bundle_code::NUM_SEGMENTS];
    for (i, b) in pf.segment_bytes.iter().enumerate() {
        if i >= segment_bytes.len() {
            bail!(
                "placements.json states {} segments; this build has {}",
                pf.segment_bytes.len(),
                segment_bytes.len()
            );
        }
        segment_bytes[i] = *b;
    }
    let places: Vec<crate::bundle_code::Placement> = pf
        .places
        .iter()
        .map(|p| crate::bundle_code::Placement {
            id: ktir_superdsc::place::PlaceId::Act(p.tid),
            segment: p.segment,
            bank: p.bank,
            offset: p.offset,
            size: p.size,
            // ⛔ READ OFF THE PRODUCER'S ROLE, NOT GUESSED FROM THE SEGMENT. The read-back tensor is
            // whichever the producer marked `Logits`; a launcher that picked "the last place" or "the
            // highest segment" would read the wrong buffer on any bundle whose output is not last.
            is_logits: p.role == "Logits",
        })
        .collect();
    let out_ids: Vec<u32> = pf
        .places
        .iter()
        .filter(|p| p.role == "Logits")
        .map(|p| p.tid)
        .collect();

    let layout = crate::bundle_code::BundleLayout {
        segment_bytes,
        weight_bank_bytes: std::borrow::Cow::Owned(Vec::new()),
        places: std::borrow::Cow::Owned(places),
        kernel_weights: std::borrow::Cow::Owned(Vec::new()),
        // The producer's scalar registry. `layout::for_node` fills it for an rmsnorm epsilon or a
        // scalarmul multiplier; a bundle whose ops read none states an empty one, and an empty one is
        // what every current Triton-derived bundle states.
        scalarmul_scales: std::borrow::Cow::Owned(Vec::new()),
        // No paged KV: a single-kernel bundle has no per-request cache stride.
        kv_request_stride_bytes: 0,
    };

    println!("  parsed segment_bytes {:?}", segment_bytes);
    let mut groups: Vec<crate::bundle_code::LaunchGroup<'static>> = Vec::new();
    for gd in &group_dirs {
        let gcode = gd.join("spyreCodeDir");
        let gimg = std::fs::read(gcode.join("init_binary.bin"))
            .with_context(|| format!("{}: no init_binary.bin", gcode.display()))?;
        let gplan = std::fs::read_to_string(gcode.join("spyrecode.json"))
            .with_context(|| format!("{}: no spyrecode.json", gcode.display()))?;
        let gc =
            crate::bundle_code::correction::parse_spyrecode(&format!("{}", gd.display()), &gplan)?;
        println!(
            "    {}: init_binary {} B, job_bin_ptr 0x{:x}, {} flit(s)",
            gd.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
            gimg.len(),
            gc.job_bin_ptr,
            gc.correction.len()
        );
        groups.push(crate::bundle_code::LaunchGroup {
            kv: Default::default(),
            programs: std::borrow::Cow::Owned(Vec::new()),
            init_binary: std::borrow::Cow::Owned(gimg),
            job_bin_ptr: gc.job_bin_ptr,
            correction: std::borrow::Cow::Owned(gc.correction),
        });
    }

    // `Executor::load` takes `&'static` because a baked bundle IS static data in the binary. This one
    // came off disk, so it is leaked deliberately: the process runs one bundle and exits.
    let code: &'static crate::bundle_code::BundleCode<'static> =
        Box::leak(Box::new(crate::bundle_code::BundleCode {
            fp: std::borrow::Cow::Owned(
                dir.file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "bundle".into()),
            ),
            layout,
            groups: std::borrow::Cow::Owned(groups),
            reroll: None,
        }));

    // ── Bind, prepare, launch ──
    // The producer's launch map says WHERE each tensor lives; this says in WHAT ORDER — see
    // `dev_order`. It rides on the input list (like `synth:` on the output list) so neither caller's
    // argument plumbing has to change.
    let sticks: BTreeMap<u32, u64> = input
        .iter()
        .filter(|s| s.starts_with("stick:"))
        .map(|s| split_stick(s))
        .collect::<Result<_>>()?;
    // `raw:t<id>` — this tensor's bytes are ALREADY the device's and must not be re-encoded. See
    // `split_raw` for what an fp16 pass does to an int32 index entry.
    let raws: std::collections::BTreeSet<u32> = input
        .iter()
        .filter(|s| s.starts_with("raw:"))
        .map(|s| split_raw(s))
        .collect::<Result<_>>()?;
    let mut inputs: BTreeMap<u32, Payload> = input
        .iter()
        .filter(|s| !s.starts_with("stick:") && !s.starts_with("raw:") && !s.starts_with("const:"))
        .map(|s| split_binding(s, "input").map(|(tid, path)| (tid, Payload::File(path))))
        .collect::<Result<_>>()?;
    // `const:t<id>=<value>` — a scalar stated inline instead of in a two-byte file. It joins the SAME
    // map, so every check below applies to it unchanged.
    for spec in input.iter().filter(|s| s.starts_with("const:")) {
        let (tid, bytes) = split_const(spec)?;
        // ⛔ TWO STATEMENTS OF ONE TENSOR'S CONTENTS, which `insert` would resolve by keeping
        // whichever the caller happened to write last. A scale bound from the wrong one of two
        // spellings is a clean rescaling of a correct-looking answer, so the collision is named.
        if let Some(prev) = inputs.insert(tid, Payload::Inline(bytes)) {
            bail!(
                "`{spec}`: t{tid} is already bound by `{}` — two statements of one tensor's \
                 contents, and nothing here can know which the caller meant",
                prev.how()
            );
        }
    }
    // `synth:<name>[@rows]=<path>`. The optional `@rows` is the same statement `stick:tN=<rows>`
    // makes for a bound tensor, said for a synthetic — which has no tid to key that map by.
    let synth_out: Vec<(String, Option<u64>, String)> = output
        .iter()
        .filter_map(|o| o.strip_prefix("synth:"))
        .map(|o| {
            let (lhs, path) = o.split_once('=').ok_or_else(|| {
                anyhow!("--output wants `synth:<name>[@rows]=<path>`, got `synth:{o}`")
            })?;
            let (name, rows) = match lhs.split_once('@') {
                Some((n, r)) => (
                    n.to_string(),
                    Some(
                        r.parse::<u64>()
                            .with_context(|| format!("synth:{lhs}: `{r}` is not a row count"))?,
                    ),
                ),
                None => (lhs.to_string(), None),
            };
            Ok((name, rows, path.to_string()))
        })
        .collect::<Result<_>>()?;
    let outputs: BTreeMap<u32, String> = output
        .iter()
        .filter(|o| !o.starts_with("synth:"))
        .map(|s| split_binding(s, "output"))
        .collect::<Result<_>>()?;

    // ⛔⛔⛔ A `raw:` MARKER THAT NAMES NOTHING IS EXACTLY THE SILENT WRONG ANSWER IT EXISTS TO STOP.
    // `raw:` only changes how a tensor is STAGED, so a typo'd or stale id leaves the buffer on the
    // default path — through `ieee_to_sen_bytes`, which rewrites both halves of every int32 index
    // entry (see `split_raw`) — and the launch reports success over a table of different, valid,
    // arbitrary rows. Nothing later in this file can notice, so the marker is checked against the
    // bindings HERE, where both are in hand.
    for tid in &raws {
        match inputs.get(tid) {
            None => bail!(
                "raw:t{tid}: no `t{tid}=<path>` among the inputs. `raw:` says how to stage a bound \
                 tensor's bytes, not which bytes — so on its own it changes nothing and this \
                 tensor, if some other spelling binds it, would go through the fp16 re-encoding \
                 `raw:` exists to avoid."
            ),
            // A `const:` payload is an fp16 scalar this file just encoded; binding it verbatim would
            // hand the device IEEE bits where it reads SEN169, which is a different number.
            Some(Payload::Inline(_)) => bail!(
                "raw:t{tid} and const:t{tid}: a `const:` value is encoded here as IEEE fp16 and is \
                 staged THROUGH the SEN conversion on purpose. Binding it verbatim would give the \
                 device IEEE bits where it reads SEN169 — a different, finite, wrong number."
            ),
            Some(Payload::File(_)) => {}
        }
        // The stick-major map is defined over 2-byte fp16 elements (`dev_order`, `restick`). A raw
        // buffer is not fp16 — the only one today is an int32 index table — so permuting it would
        // shuffle half-words across entry boundaries and produce addresses no caller named.
        if sticks.contains_key(tid) {
            bail!(
                "raw:t{tid} and stick:t{tid}: the stick-major map permutes 2-byte fp16 elements, \
                 and a raw tensor is not fp16. Applying it to an int32 index table would swap \
                 half-words between entries and yield row numbers nobody asked for."
            );
        }
    }

    // ⛔⛔⛔ REFUSED, NOT ASSUMED ROW-MAJOR. A tensor spanning more than one 64-lane stick is
    // addressed STICK-MAJOR by the descriptors (see `dev_order`), and binding or reading it
    // row-major returns a fluent, plausible, WRONG answer — measured 0.0193 within_2pct on
    // rmsnorm_granite, with only the `(r, r)` diagonal tiles exact and nothing zeroed to give it
    // away. So the row count is STATED per tensor and a tensor wide enough for the question to
    // matter must state it. `stick:tN=1` is the identity, and saying it out loud is the point.
    let layout_of = |tid: u32, size: u64, what: &str| -> Result<Option<(usize, usize)>> {
        let elems = size / 2;
        match sticks.get(&tid) {
            Some(&rows) => {
                // `is_multiple_of` rather than `%`, and not only because clippy says so: it is TOTAL
                // where `%` is not. `stick:t0=0` parses, `rows > elems` does not catch it, and
                // `elems % 0` is a divide-by-zero PANIC where this refuses by name — `0` is a
                // multiple of nothing but `0`, so a non-empty placement bails with the message below.
                if rows > elems || !elems.is_multiple_of(rows) {
                    bail!(
                        "stick:t{tid}={rows}: the placement holds {elems} fp16 elements, which do \
                         not divide into {rows} rows"
                    );
                }
                let cols = elems / rows;
                if rows > 1 && !cols.is_multiple_of(64) {
                    bail!(
                        "stick:t{tid}={rows}: {cols} columns is not a whole number of 64-lane \
                         sticks, so the stick-major map would leave holes it cannot fill"
                    );
                }
                Ok((rows > 1).then_some((rows as usize, cols as usize)))
            }
            None if elems > 64 => bail!(
                "t{tid} ({what}) spans {elems} fp16 elements — more than one 64-lane stick — and no \
                 `stick:t{tid}=<rows>` states the row count the descriptors address it with. The device \
                 layout is stick-major, (c/64)·rows·128 + r·128 + (c%64)·2, so binding this row-major \
                 would return a wrong answer that looks right: MEASURED 0.0193 within_2pct on \
                 rmsnorm_granite, exactly the (r, r) diagonal tiles correct. Pass `stick:t{tid}=1` if \
                 the tensor really is one row — that map is the identity."
            ),
            None => Ok(None),
        }
    };

    let mut exec = crate::superdsc_exec::Executor::load(
        code,
        // ⛔ NOT ZERO. These three are a MODEL's geometry, and a single-kernel bundle has none — but a
        // zero pool leaves the runtime configuring a device with no KV region at all, and the launch
        // is rejected (`sync rc=-1`) with the image verified and every segment writable. Minimal
        // nonzero values: the vocab is the read-back tensor's element count, which is what the logits
        // shadow is sized from, and one page of one slot is the smallest pool that exists.
        crate::superdsc_exec::Vocab(
            pf.places
                .iter()
                .find(|p| p.role == "Logits")
                .map_or(1, |p| (p.size / 2) as usize),
        ),
        crate::superdsc_exec::KvDim(1),
        crate::superdsc_exec::PoolPages(1),
    )?;

    // ⛔ THE COUNT IS THE BOUND TENSORS, and a mismatch refuses EVERY launch rather than running a
    // program over a buffer nobody filled — `declare_sources`' own completeness check. No source is
    // resident here: a single-kernel bundle has no prefix-KV cache for the device to fill.
    // ⛔⛔⛔ `declare_sources(n)` MEANS "ids 0..n ARE CALLER-FILLED", BY ID AND NOT BY COUNT. Our
    // producer places reserved device constants at ids near u32::MAX (`RMS_INVCOLS_TID` is
    // 4294967275), so counting binds declared ids 0..4 — which claimed a t3 that does not exist and
    // said nothing about the reserved one. MEASURED: every activation segment then came back
    // `unreadable` and seg4's D2H failed rc=-1, while seg1 (the weights, staged) read fine.
    //
    // So the split is by KIND, not position: a reserved const is a WEIGHT (staged once, before
    // prepare) and the kernel's own tensors are SOURCES (bound per run, after prepare). `n` is the
    // exclusive upper bound of the source ids, so it is max(source id) + 1.
    let n_sources = inputs
        .keys()
        .filter(|t| **t < RESERVED_TID_FLOOR)
        .max()
        .map_or(0, |m| m + 1);
    exec.declare_sources(n_sources, std::iter::empty());

    for s in 0..crate::bundle_code::NUM_SEGMENTS {
        if let Some(seg) = crate::superdsc_exec::SegIdx::checked(s as i64) {
            print!("{} ", exec.seg_bytes(seg));
        }
    }
    println!("  <- shadows before stage_weights");
    // WHICH TENSORS ARE WEIGHTS IS THE LAYOUT'S CALL, NOT A TID RANGE. This used to key off
    // `tid >= RESERVED_TID_FLOOR` alone, which silently mis-sorted a REAL weight: an rmsnorm gamma is
    // a plain kernel argument (`t1`) and so took the source path, while the layout had placed it in
    // `SegRole::Weight`. A weight segment is REPLICATED to every core and an activation segment is
    // PARTITIONED across them, so a row-invariant operand bound as a source has each core reading its
    // own slice — measured on card as core k fetching `gamma[k·128 … (k+1)·128)`, with every other
    // stage of the norm exact. Reading the role off `placements.json` keeps the launcher and the
    // layout saying the same thing by construction.
    let is_weight = |tid: u32| -> bool {
        tid >= RESERVED_TID_FLOOR || pf.places.iter().any(|p| p.tid == tid && p.role == "Weight")
    };
    // The producer's reserved device constants AND the layout's weights: staged once, bound before
    // prepare.
    for (tid, payload) in inputs.iter().filter(|(t, _)| is_weight(**t)) {
        let bytes = payload.bytes(*tid)?;
        let want = pf
            .places
            .iter()
            .find(|p| p.tid == *tid)
            .ok_or_else(|| anyhow!("--input t{tid}: no such tensor in placements.json"))?
            .size;
        // ⛔⛔⛔ A RAW BIND CANNOT BE A WEIGHT, and the executor says so first: `stage_weights` refuses
        // a raw id by name, because every destination in that loop goes through `ieee_to_sen_bytes` or
        // the retile walk and neither may touch int32 entries. Raw bytes are honoured only on the
        // per-step activation path (`refill_activations`, which `run_prefix_only` calls). Refused HERE,
        // where the caller's own spelling is in hand, so the message can name the spelling rather than
        // arriving later as an executor-internal complaint about a name.
        if raws.contains(tid) {
            bail!(
                "raw:t{tid}: t{tid} is a WEIGHT here — the placement's role is `Weight`, or the id is \
                 at/above the reserved-constant floor {RESERVED_TID_FLOOR} — and a weight is staged \
                 once, before prepare, through a loop that re-encodes fp16. `stage_weights` refuses a \
                 raw id there by name. Raw bytes are only honoured on the per-step activation path, so \
                 a tensor that must be bound verbatim has to be a plain kernel argument."
            );
        }
        if bytes.len() as u64 != want {
            bail!(
                "--input t{tid}={}: {} B, but its placement is {want} B. A short buffer would \
                 leave the tail whatever the device had there, which is a silently wrong answer, and \
                 a long one would overwrite the tensor placed after it.",
                payload.how(),
                bytes.len()
            );
        }
        let bytes = match layout_of(*tid, want, "weight")? {
            Some((rows, cols)) => restick(&bytes, rows, cols, true),
            None => bytes,
        };
        exec.bind_weight(ktir_superdsc::place::PlaceId::Act(*tid), bytes);
    }

    exec.stage_weights()?;
    for s in 0..crate::bundle_code::NUM_SEGMENTS {
        if let Some(seg) = crate::superdsc_exec::SegIdx::checked(s as i64) {
            print!("{} ", exec.seg_bytes(seg));
        }
    }
    println!("  <- after stage_weights");
    exec.prepare()?;

    // What the executor actually has per segment, printed rather than assumed: a read-back goes
    // through the segment's host shadow, so a zero here explains an empty output and a nonzero one
    // moves the question to the launch.
    for s in 0..crate::bundle_code::NUM_SEGMENTS {
        if let Some(seg) = crate::superdsc_exec::SegIdx::checked(s as i64) {
            print!("{} ", exec.seg_bytes(seg));
        }
    }
    println!("  <- after prepare");

    // ⛔ A FRESHLY ALLOCATED TENSOR SEGMENT IS NOT HOST-READABLE UNTIL SOMETHING IS WRITTEN INTO IT.
    // MEASURED: after prepare, seg1 (the weights, H2D'd during staging) reads back fine while every
    // other segment reports `unreadable` and seg4's D2H fails rc=-1. Zeroing each segment this bundle
    // places both maps it and gives the kernel deterministic starting contents — which the output
    // buffer needs anyway, since a descriptor writes only the elements it computes.
    //
    // ⛔ EXCEPT A SEGMENT HOLDING A WEIGHT, WHICH STAGING ALREADY WROTE. The note above says it
    // outright — seg1 "reads back fine" after prepare because `stage_weights` H2D'd it — so zeroing it
    // here does not map an unmapped segment, it DESTROYS the staged bytes. Measured the moment a
    // gamma was first placed in `SegRole::Weight`: `zeroed seg1`, and the card then multiplied by a
    // gamma of exactly 0 (output all zeros, ramp probe resolving every element to index -1).
    let weight_segs: std::collections::BTreeSet<_> = pf
        .places
        .iter()
        .filter(|p| p.role == "Weight")
        .map(|p| p.segment)
        .collect();
    // ⭐ AND THE SEGMENTS ONLY A SYNTHETIC NAMES. A lowering bump-allocates its intermediates into
    // their own segment, and `places` never mentions it — so the loop below used to skip it, no host
    // shadow was ever sized for it, and `read_seg` came back CLAMPED TO 0 B. MEASURED on
    // swiglu_mlp_flat: seg0 holds t5/t6/t7/t7_silu (131072 B in `segment_bytes`) and every
    // `--output synth:` read failed the span check with "only 0 B are host-readable", which made the
    // per-stage comparison impossible on exactly the bundle that needed it.
    for seg_idx in pf
        .places
        .iter()
        .map(|p| p.segment)
        .chain(pf.synth.iter().map(|s| s.segment))
        .collect::<std::collections::BTreeSet<_>>()
    {
        if weight_segs.contains(&seg_idx) {
            println!("  seg{seg_idx} holds a staged weight — NOT zeroed");
            continue;
        }
        if let Some(seg) = crate::superdsc_exec::SegIdx::checked(seg_idx as i64) {
            exec.zero_seg(seg)?;
            println!("  zeroed seg{seg_idx}");
        }
    }

    // ⛔⛔⛔ AFTER `prepare`, AND THAT ORDER IS THE WHOLE CONTRACT. `bind_input`'s own doc: "a name
    // bound before prepare is recorded as a WEIGHT instead". A weight is staged once and its host
    // bytes are reclaimed — `prepare` then frees the host shadow of every segment holding no
    // activation, so binding first made all four of this kernel's tensors weights, freed segments 3
    // and 4, and left the read-back with no shadow to read. MEASURED, shadows per segment:
    //   after stage_weights   128 128 128 532608 524288 128 128
    //   after prepare (bound first)   0   0 128      0      0   0   0
    for (tid, payload) in inputs.iter().filter(|(t, _)| !is_weight(**t)) {
        let bytes = payload.bytes(*tid)?;
        let want = pf
            .places
            .iter()
            .find(|p| p.tid == *tid)
            .ok_or_else(|| anyhow!("--input t{tid}: no such tensor in placements.json"))?
            .size;
        if raws.contains(tid) {
            bind_raw(&mut exec, *tid, bytes, &payload.how(), want)?;
            continue;
        }
        if bytes.len() as u64 != want {
            bail!(
                "--input t{tid}={}: {} B, but its placement is {want} B. A short buffer would \
                 leave the tail whatever the device had there, which is a silently wrong answer, and \
                 a long one would overwrite the tensor placed after it.",
                payload.how(),
                bytes.len()
            );
        }
        let bytes = match layout_of(*tid, want, "input")? {
            Some((rows, cols)) => restick(&bytes, rows, cols, true),
            None => bytes,
        };
        exec.bind_input(ktir_superdsc::place::PlaceId::Act(*tid), bytes);
    }

    // ⛔ BEFORE `prepare`, AND NOT OPTIONAL EVEN WITH NO WEIGHTS. `stage_weights` is where the HOST
    // SHADOWS are sized (`seg_host[i] = vec![0; segment_bytes[i]]`), and a read-back goes through the
    // shadow: without this the output segment's shadow is empty, `d2h_seg_clamped` brings back 0
    // bytes, and `read_tensor` returns an EMPTY buffer from a launch that otherwise reports success.
    // Measured exactly that way before this call existed.

    // ⛔ NOT `run_full`: its own doc says "only a ROLLED bundle has those seams, so an unrolled one
    // reports `false`" — and it reports it by returning `Ok(false)` having launched NOTHING, which is
    // exactly what this launcher measured first (segments unsized, an empty read-back, exit 0).
    // `run_prefix_only` takes `ListSel::Body` for an unrolled bundle, which is what a single-kernel
    // bundle IS: one body, no prefix/suffix seams, no per-layer advance.
    exec.run_prefix_only()?;

    // ⭐ ALSO READ THE INTERMEDIATES. Which STAGE diverges is not visible in the final tensor, and the
    // producer's `placements.json` now lists every synthetic with its segment and offset — so a
    // `--output synth:<name>=<path>` reads one straight out of its segment and the host can compare it
    // against torch stage by stage.
    for (name, rows, path) in &synth_out {
        let Some(sy) = pf.synth.iter().find(|s| &s.name == name) else {
            bail!("--output synth:{name}: no such synthetic in placements.json");
        };
        let Some(seg) = crate::superdsc_exec::SegIdx::checked(sy.segment as i64) else {
            bail!("synthetic {name}: segment {} out of range", sy.segment);
        };
        let whole = exec.read_seg(seg)?;
        let a = sy.offset as usize;
        let b = (sy.offset + sy.size) as usize;
        if b > whole.len() {
            bail!(
                "synthetic {name} spans [{a}, {b}) of segment {} but only {} B are host-readable. A \
                 segment no PLACEMENT names has its host shadow reclaimed by `prepare` unless \
                 SCRATCHY_SUPERDSC_DBG is set, and a reclaimed shadow clamps this read to 0 B — set \
                 that variable to keep every shadow alive.",
                sy.segment,
                whole.len()
            );
        }
        // ⛔⛔⛔ `read_seg` IS RAW DEVICE BYTES, AND THE DEVICE fp16 IS NOT IEEE fp16. It is SEN169
        // (sign 1, exp 6 bias-31, mantissa 9) — `read_tensor` runs `sen_to_ieee_bytes` on its way out
        // and this path did not, so every synthetic came back as garbage that still LOOKED like
        // plausible fp16. MEASURED on swiglu_mlp_flat, card vs a numpy model of the same stage:
        // g=x@Wg 0.0156 within_2pct, med|rel| 7.5e-1, ratio median 1.714 — while the FINAL tensor,
        // which goes through `read_tensor`, was 0.9119 on the same launch. 1.714 is not a numeric
        // defect, it is a different exponent bias read through the wrong codec.
        let mut ieee = vec![0u8; b - a];
        crate::sen_convert::sen_to_ieee_bytes(&whole[a..b], &mut ieee);
        // And the same stick-major refusal a bound tensor gets: a synthetic wider than one 64-lane
        // stick is addressed `[cols/64][rows][64]` too, so reading it row-major is a plausible wrong
        // answer. See `dev_order`.
        let elems = sy.size / 2;
        let ieee = match rows {
            Some(r) => {
                if *r == 0 || *r > elems || elems % *r != 0 {
                    bail!("synth:{name}@{r}: {elems} fp16 elements do not divide into {r} rows");
                }
                let cols = elems / *r;
                if *r > 1 && !cols.is_multiple_of(64) {
                    bail!(
                        "synth:{name}@{r}: {cols} columns is not a whole number of 64-lane sticks"
                    );
                }
                if *r > 1 {
                    restick(&ieee, *r as usize, cols as usize, false)
                } else {
                    ieee
                }
            }
            None if elems > 64 => bail!(
                "synth:{name} spans {elems} fp16 elements — more than one 64-lane stick — and no \
                 `synth:{name}@<rows>=<path>` states the row count the descriptors address it with. \
                 Pass `@1` if it really is one row; that map is the identity."
            ),
            None => ieee,
        };
        std::fs::write(path, &ieee).with_context(|| format!("--output synth:{name}={path}"))?;
        println!("read synth {name} -> {path} ({} B)", ieee.len());
    }

    for (tid, path) in &outputs {
        let p = pf
            .places
            .iter()
            .find(|q| q.tid == *tid)
            .ok_or_else(|| anyhow!("--output t{tid}: no such tensor in placements.json"))?;
        let bytes = exec.read_tensor(
            ktir_superdsc::place::PlaceId::Act(*tid),
            (p.size / 2) as usize,
        )?;
        let bytes = match layout_of(*tid, p.size, "output")? {
            Some((rows, cols)) => restick(&bytes, rows, cols, false),
            None => bytes,
        };
        std::fs::write(path, &bytes).with_context(|| format!("--output t{tid}={path}"))?;
        println!("read t{tid} -> {path} ({} B)", bytes.len());
    }
    if outputs.is_empty() && !out_ids.is_empty() {
        println!(
            "note: the producer marks t{:?} as the output; pass --output t<id>=<path> to read it back",
            out_ids
        );
    }
    Ok(())
}
