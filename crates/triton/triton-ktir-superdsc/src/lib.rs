// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! ADAPTER: our KTIR value -> `ktir-superdsc`'s request type and per-kind entry points.
//!
//! # WHAT THIS IS AND IS NOT
//!
//! It is NOT a lowering. Every descriptor decision belongs to `ktir-superdsc`, whose output is
//! validated by byte-identity against a bake proven on silicon. This crate's whole job is to
//! present our KTIR in the shape that crate's door takes:
//!
//! ```text
//!   our Module --(triton_ktir::handoff)--> ktir_core::IRFunction
//!                                              |
//!                                              +--> KtirNode { func, program, bindings, ... }
//!                                                        |
//!                                                        +--> emit::{rmsnorm, attn_at, ...}
//! ```
//!
//! # THE FIVE FIELDS, AND WHICH ARE HARD
//!
//! `KtirNode` is `{ func, program, bindings, mask, node_out_tid }`. Four are mechanical:
//!
//! * `func` — [`triton_ktir::handoff::lower`] already builds it. Their type wants `'static`, so
//!   the arena is [`ktir_core::arena::Arena::global`].
//! * `bindings` — a `BufferId` per parameter, in order. Their crate treats these as opaque keys
//!   and never asks what a buffer is, so parameter position IS the numbering.
//! * `mask` / `node_out_tid` — `None` for everything we emit today.
//!
//! **`program` is the one that is not mechanical**, and it is the whole open question:
//! [`Program`] names what a program COMPUTES (`RmsNorm`, `Attn`, `Matmul`,
//! `Elementwise(Silu)`, …). Their producer knows it because it built the node from a tape.
//! We have to say it about a Triton kernel.
//!
//! # WHY THAT IS NOT AUTOMATICALLY A "RECOGNIZER"
//!
//! Their `Program` doc says the kind "is NOT derivable … a pointwise KTIR body is the same
//! op-DAG shape whichever function it applies". That is true of THEIR producer's KTIR and
//! false of ours: we emit `arith.addf` where an add is meant and `arith.mulf` where a multiply
//! is, so `Elementwise(Add)` is a 1:1 op mapping, not pattern matching.
//!
//! The fused kinds are the real question, and there are two honest answers — the choice is the
//! owner's, not this crate's:
//!
//! 1. **Source provenance.** Triton names these things: a `@triton.jit` helper called `rmsnorm`,
//!    a `tl.sigmoid` call the frontend sees BEFORE inlining, the kernel's own name. Carried as
//!    an attribute, the kind is a fact the PROGRAMMER stated -- not one the compiler inferred.
//! 2. **Structural matching.** Recognise the silu longhand, the rmsnorm shape, the online
//!    softmax. This is what "no kernel recognizers, no algorithm names in the compiler" rules
//!    out, and it is fragile in exactly the way that constraint anticipates.
//!
//! Until that is settled, [`drive`] takes the `Program` as an ARGUMENT. The driver states it per
//! fixture, so bring-up is not blocked on the design decision and no classifier is smuggled in
//! by accident.

use ktir_core::arena::Arena;
use ktir_superdsc::emit::{self, EmittedOp};
use ktir_superdsc::ktir_node::{BufferId, KtirNode, Program};
use ktir_superdsc::placement::BundleLayout;

pub mod bake;
// THE FOURTEEN CONFIGURATIONS. `examples/bake_py` (the dxp bake) and `triton-numeric` (the
// emulator's NUMERIC check) both need "what is `rope_q32`?" answered, and a configuration restated
// in two places is one that drifts in one of them -- see the module's own `rope_q32` note, where a
// second statement of the grid left 252 of 256 token positions unwritten and still baked.
pub mod cases;
pub mod layout;

/// What went wrong, and at which stage.
#[derive(Debug)]
pub struct Error {
    pub stage: &'static str,
    pub message: String,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.stage, self.message)
    }
}

/// Our Triton-free module as their request type.
///
/// Run [`triton_ktir::passes::to_ktir`] first — [`triton_ktir::handoff::lower`] refuses every
/// `tt.*` op by name, which is what keeps a half-converted module from reaching a device program.
pub fn node_for(m: &triton_ktir::ir::Module, program: Program) -> Result<KtirNode, Error> {
    let func = triton_ktir::passes::to_ktir_emit::lower(m, Arena::global()).map_err(|e| Error {
        stage: "handoff",
        message: e.to_string(),
    })?;
    // PARAMETER POSITION IS THE BUFFER NUMBER. Their crate documents `BufferId` as opaque —
    // it renders an operand name for it and looks it up in a `BundleLayout`, never asking what
    // the buffer holds, because extents/strides/format all come from the program's own
    // `ktdp.construct_memory_view`. So there is nothing to invent here.
    let bindings: Vec<BufferId> =
        (0..func.arguments.len()).map(|i| BufferId::new(i as u32)).collect();
    Ok(KtirNode { func, program, bindings, mask: None, node_out_tid: None })
}

/// Hand one node to the entry point its `Program` names, under a layout DERIVED FROM THE PROGRAM.
///
/// The layout comes from [`layout::for_regions`] over the node's own
/// [`regions`](emit::lower_ktir_to_superdsc::regions) — the same parameter walk the chosen body
/// reads, so the memory plan and the descriptors that address through it are built from ONE reading
/// of the program. See [`layout`] for why `None` is not an option for two of the bodies.
///
/// [`emit_node_with`] is the same call with the layout stated explicitly; it exists so the
/// no-layout arm stays MEASURABLE from the same binary, and so a caller that wants to INSPECT the
/// plan can build it itself.
pub fn emit_node(node: &KtirNode) -> Result<Vec<EmittedOp>, Error> {
    let regions = emit::lower_ktir_to_superdsc::regions(node).map_err(|e| Error {
        stage: "regions",
        message: format!("{e:?}"),
    })?;
    let l = layout::for_regions(node, &regions)?;
    emit_regions(node, &regions, Some(&l))
}

/// [`emit_node`] with the layout stated by the caller — `None` selects `ktir-superdsc`'s unit-test
/// arm (per-op `segment_base(arg_index)`, no scale registry). The DRIVER's `KTIR_NO_LAYOUT=1` path,
/// kept so "did adding a layout change this verdict?" is a question one binary can answer.
pub fn emit_node_with(
    node: &KtirNode,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let regions = emit::lower_ktir_to_superdsc::regions(node).map_err(|e| Error {
        stage: "regions",
        message: format!("{e:?}"),
    })?;
    emit_regions(node, &regions, layout)
}

fn emit_regions(
    node: &KtirNode,
    regions: &[emit::lower_ktir_to_superdsc::Region],
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let mut sym = 0i64;
    let name = node.func.name;

    let out = match node.program {
        Program::RmsNorm => {
            emit::lower_ktir_to_superdsc::rmsnorm(name, node, regions, &mut sym, layout)
        }
        Program::ScalarMul => {
            emit::lower_ktir_to_superdsc::scalarmul(name, node, regions, &mut sym, layout)
        }
        Program::Elementwise(kind) => {
            // NO BROADCAST FLAGS: this per-NODE door takes regions the caller already resolved and
            // has no program to prove a broadcast chain from. The whole-function walk is what
            // recognises `m[:, None]` and states the axes; here an empty slice says "every operand is
            // dense", which is what a `Program::Elementwise` node with resolved regions is.
            emit::lower_ktir_to_superdsc::elementwise(name, kind, regions, &[], &mut sym, layout)
        }
        Program::SiluMul => {
            emit::lower_ktir_to_superdsc::silumul(name, regions, &mut sym, layout)
        }
        Program::LmLast => emit::lower_ktir_to_superdsc::lmlast(name, node, regions, &mut sym, layout),
        Program::Transpose => {
            emit::lower_ktir_to_superdsc::transpose(name, regions, &mut sym, layout)
        }
        // `matmul` threads a per-BUNDLE fp8 activation-quantize dedup set. One node at a time
        // here, so a fresh set per call is correct; a real bundle walk threads one across nodes.
        //
        // ⛔ AND IT NEEDS THE B ORIENTATION, WHICH IS READ FROM THE NODE'S OWN `indexing_maps` — never
        // defaulted. The framing of the W REGION is `[n, k]` for the transpose-B form and `[k, n]` for
        // the plain one, the extent guards inside `matmul` read that framing, and at `k == n` neither
        // is recoverable from the extents (granite's `[4096, 4096]` output projection). This door
        // lowers ONE program per node, so it reads the node's `linalg.matmul` ops and requires them to
        // agree; the whole-function door proves it per op, which is what a multi-matmul kernel needs.
        Program::Matmul => {
            let mms: Vec<_> = node
                .func
                .operations
                .iter()
                .filter(|o| o.op_type == ktir_core::opkind::OpKind::LinalgMatmul)
                .collect();
            let mut orient = None;
            let mut disagree = false;
            for op in &mms {
                match emit::whole_function::matmul_b_orientation(&node.func, op) {
                    Err(e) => return Err(Error { stage: "b-orientation", message: e.message }),
                    Ok(b) => match orient {
                        None => orient = Some(b),
                        Some(prev) if prev != b => disagree = true,
                        Some(_) => {}
                    },
                }
            }
            let Some(b) = orient.filter(|_| !disagree) else {
                return Err(Error {
                    stage: "b-orientation",
                    message: format!(
                        "{name}: this node states `Program::Matmul` but its function holds {}                          `linalg.matmul` op(s){} — so there is no ONE proven B orientation for the                          single descriptor this door emits. The orientation decides which of the W                          region's two extents is K and cannot be recovered from the extents at                          `k == n`, so it is refused rather than assumed. The whole-function door                          (`emit_whole`) proves it per op.",
                        mms.len(),
                        if disagree { ", and they do not agree" } else { "" }
                    ),
                });
            };
            let mut quantized = std::collections::HashSet::new();
            emit::lower_ktir_to_superdsc::matmul_oriented(
                name,
                regions,
                &mut sym,
                layout,
                &mut quantized,
                b,
                // The per-`Program` door's operands are whole staged tensors, not windows of the
                // caller's parameters — the fact the spurious-pad drop discriminates on.
                emit::lower_ktir_to_superdsc::OperandOrigin::Staged,
            )
        }
        // THE MONOMORPHISATION DOOR. `rope_at::<HD>` is not reachable without the geometry, and
        // the geometry is a const parameter, so `Program::Rope` is refused here and driven from
        // `drive_rope` which takes it explicitly. Refused rather than defaulted: a wrong head
        // count is a silently wrong grouping.
        //
        // ⛔ `Program::Attn` IS REFUSED FOR A DIFFERENT REASON NOW: its pattern-matched door
        // (`drive_attn` → `attn_at::<NQH, NKVH, HD>`, scratchy's prebuilt fragment assembly
        // selected by geometry match) is REMOVED — attention is lowered from its compiled ops
        // through the whole-function door, and a per-`Program` dispatch would be a second,
        // hand-rolled path. Do not re-add one.
        Program::Attn | Program::Rope => {
            return Err(Error {
                stage: "program",
                message: format!(
                    "{:?} crosses the model-geometry const door (`rope_at::<HD>`), so it cannot \
                     be dispatched from a runtime `Program`. Call the generic entry with the \
                     fixture's own tl.constexpr values",
                    node.program
                ),
            })
        }
    };
    out.map_err(|e| Error { stage: "ktir-superdsc", message: format!("{e:?}") })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE MODEL-GEOMETRY CONST DOOR
//
//  `rope_at::<HD>` and `attn_at::<NQH, NKVH, HD>` are const-generic because the head geometry
//  parameterises the DEVICE layout — GQA grouping, slabs, head strides — so a branch on it is a
//  branch on a const the compiler can hold onto (`rope_at`'s own doc: "the collapsed RoPE form and
//  the slab RoPE form become TWO INSTANTIATIONS rather than two arms of one function"). A runtime
//  `Program` cannot reach them; something has to name the triple.
//
//  ⛔ THE VALUES ARE THE FIXTURE'S OWN `tl.constexpr`, READ, NEVER DEFAULTED. `HEAD_DIM`, `H` and
//  `GQA` are constexpr parameters of the kernels themselves (`attention_flash.py` line 339-343,
//  `rope.py` line 132-135) and are recorded per configuration in
//  `test/experiment1/index.json`'s `constexprs`, which `run_experiment1.py` fills from the
//  fixture's own `constexprs()` (line 174/223, `constexprs=ce`). A DEFAULT here would be a wrong
//  GQA grouping — every query head reading the wrong kv plane — with nothing to catch it, so the
//  driver refuses a configuration whose keys it cannot read.
//
//  ⛔ AND `GQA` IS THE GROUPING, NOT THE KV HEAD COUNT. `attention_flash.py`: "a query head reads
//  the kv plane it shares, `off_h // GQA`" and "the kv planes are nkvh-deduped, so their descriptor
//  extent is `H // GQA` planes". So NQH = H and NKVH = H / GQA — the division is the whole point,
//  and inverting it is exactly the silent mis-grouping the const door exists to make visible.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH AXIS IS OUTERMOST in a roped `[·, head_dim]` plane — the fact the view's SHAPE cannot
/// state, and therefore the one that has to be said out loud.
///
/// ⛔⛔⛔ THE TWO ORDERS PRODUCE THE SAME `regions()` SHAPE AND DIFFERENT BYTES. `rope_kv8`'s `x`
/// view is `[2048, 128]`; so is a token-major `[256, 8·128]` plane reinterpreted as `[rows·heads,
/// hd]`. Nothing in the program distinguishes them, `declare_arrangement`'s `addr_eq` admits a
/// same-total contiguous reshape, and the footprint guard sees one buffer of the right size — so a
/// mismatch here is not caught anywhere and comes out as a wrong answer.
///
/// ⭐⭐⭐ AND IT IS NOW SETTLED, BY THE PRODUCER `rope_at` WAS WRITTEN AGAINST. It is
/// [`TokenMajor`](RopeRows::TokenMajor). `KtirFunc::rope` (scratchy
/// `lower_subtile_tape_to_ktir.rs:3527`) is the whole argument in five lines:
///
/// ```text
///   let mh = rows * heads;
///   let x_view = self.view_shaped(x_t, mh, hd);          // the `[rows·heads, hd]` view
///   for ri in 0..rows {
///       let rbase = self.idx(ri * heads);                // ← THE ROW BASE OF POSITION ri
///       let xf_acc = self.tile(x_view, rbase, zero, heads, half);   // heads ROWS TALL
///       let cos1 = self.load_1d(cos_t, rows * tbl_cols, ri * tbl_cols, half);
///       let cosb = self.broadcast(cos1, vec![heads, half], 0);      // ONE row, over those heads
/// ```
///
/// Position `ri` occupies rows `ri·heads .. ri·heads + heads`, one row per head, and the SINGLE
/// cos/sin row of that position is broadcast across exactly those rows. So the row index is
/// `token·heads + head` — token outermost, head the FAST row axis. Head-outermost rows would make
/// that same tile `heads` CONSECUTIVE TOKENS of one head, every one of them rotated by position
/// `ri`'s angle, which is the silent-garbage case this enum exists to name.
///
/// ⭐ THE CONSUMER SIDE AGREES INDEPENDENTLY: scratchy's own door
/// (`ktir_superdsc_door.rs`, the rope arm) derives `heads = ins[0].r_len` — the x ACCESS TILE's row
/// extent — and then `mq = v_rows / heads`. The tile's rows ARE the head axis, inside one token, or
/// that derivation reads a token count as a head count.
///
/// ⭐ AND IT IS THE SAME BYTES AS THE PLANE ATTENTION READS. Row-major `[mq, heads·hd]` puts
/// element `(ri, h·hd + d)` at flat `ri·heads·hd + h·hd + d`, which IS row `ri·heads + h`, column `d`
/// of the tall `[mq·heads, hd]` view — the contiguous reshape `addr_eq` admits. A head-outermost
/// tall view is instead the reshape of `[heads, mq, hd]`, i.e. the TRANSPOSED plane.
///
/// ⛔ DO NOT READ `rope_at`'s COLLAPSE COMMENT AS THE OTHER ANSWER. It says "head `h` row `r` of the
/// roped `[mq, heads*hd]` tensor sits at `h*mq*hd + r*hd`, and a `[heads*mq, hd]` view indexed
/// `h*mq + r` gives exactly `(h*mq+r)*hd`" — that is the DEVICE offset of a stick-scattered
/// `[mq, total]` tensor (`dev_off: [r,c] → (c/64)·(mq·64) + r·64 + c%64`), where the column-stick
/// scattering has already made head-major contiguous ON CARD. It is a statement about the device
/// arrangement of the wide plane, not about the row order of the host view, and mistaking the two is
/// what makes a head-major fixture look plausible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeRows {
    /// Row = token, head in COLUMNS: the `[mq, heads·hd]` plane `rope_at` addresses, whose device
    /// offset for `(row r, head h)` is the `["row", "head", "feat"]` nest over `[mq, heads, hd]`.
    TokenMajor,
    /// Row = `head·n_tok + token`, i.e. head OUTERMOST — what `rope.py` builds
    /// (`x = torch.randn(h * n_tok, head_dim)`, `offs_m = off_h * N_TOK + start_m * BLOCK_M`).
    HeadMajor,
}

/// `Program::Rope` at its const head dim.
///
/// `mq`/`total` are the roped plane's TOKEN ROWS and FULL WIDTH (`heads · head_dim`) — the two
/// numbers `[rows·heads, hd]` cannot separate, since `v_rows = mq · heads` is one equation in two
/// unknowns. They come from the fixture's `N_TOK` and `H · HEAD_DIM`, and are CROSS-CHECKED against
/// the output view here so a geometry that does not belong to this program is a refusal, not an
/// address.
pub fn drive_rope(
    node: &KtirNode,
    head_dim: u32,
    mq: u32,
    total: u32,
    rows: RopeRows,
    rows_are_requests: bool,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let name = node.func.name;
    let r = emit::lower_ktir_to_superdsc::regions(node).map_err(|e| Error {
        stage: "regions",
        message: format!("{e:?}"),
    })?;
    // x, cos, sin, out — `KtirFunc::rope`'s parameters, and `RopeAt`'s four names in order.
    let (ins, out) = emit::lower_ktir_to_superdsc::split_out(name, &r, layout, 3).map_err(|e| {
        Error {
            stage: "ktir-superdsc",
            message: format!("{e:?}"),
        }
    })?;
    if head_dim == 0 || total == 0 || !total.is_multiple_of(head_dim) {
        return Err(Error {
            stage: "geometry",
            message: format!(
                "{name}: head_dim {head_dim} does not divide the roped width {total} — the fixture's \
                 own `HEAD_DIM` and `H · HEAD_DIM` must agree, and `rope_at` derives its head count \
                 as `total / hd`"
            ),
        });
    }
    let heads = total / head_dim;
    // THE CROSS-CHECK. `rope_at` addresses a `[mq, total]` plane through a `[rows·heads, hd]` view,
    // so the geometry crossed and the program's own view have to reconcile — `v_cols` IS the head
    // dim (`Region`'s doc: "`KtirFunc::rope` views `x` as `[rows·heads, hd]`, so this states the
    // HEAD DIM directly") and `v_rows` is `mq · heads`. Without this a wrong `H` would silently
    // rescale every per-(row, head) offset.
    if (out.v_rows, out.v_cols) != (mq * heads, head_dim) {
        return Err(Error {
            stage: "geometry",
            message: format!(
                "{name}: the output t{} states a `[{}, {}]` view, but head_dim {head_dim} with \
                 {mq} token row(s) and {heads} head(s) makes it `[{}, {head_dim}]` — the geometry \
                 crossed does not belong to this program",
                out.tid,
                out.v_rows,
                out.v_cols,
                mq * heads,
            ),
        });
    }
    // ⛔⛔⛔ AND THE ROW ORDER, WHICH THE SHAPES CANNOT SETTLE AND `KtirFunc::rope` DOES. Token
    // outermost — see [`RopeRows`] for the five lines of the producer that settle it and for the two
    // independent confirmations (its consumer door reads `heads` off the x tile's ROW extent; the
    // token-major tall view is the contiguous reshape of the `[mq, heads·hd]` plane attention reads).
    // `rope.py` indexes `off_h * N_TOK + start_m * BLOCK_M` over an `[H · N_TOK, HEAD_DIM]` buffer,
    // i.e. HEAD outermost, so THE KERNEL IS THE SIDE THAT IS WRONG. Same element count, same view
    // shape, different byte for every (r, h) but (0, 0) — so emitting anyway is the
    // silent-wrong-answer case, and it is refused BY NAME.
    if rows != RopeRows::TokenMajor {
        return Err(Error {
            stage: "geometry",
            message: format!(
                "{name}: this program's roped plane is HEAD-MAJOR in rows (row = head·{mq} + token, \
                 which is `rope.py`'s `offs_m = off_h * N_TOK + start_m * BLOCK_M` over an \
                 `[H · N_TOK, HEAD_DIM]` buffer) while the rope this adapter drives is TOKEN-major: \
                 `KtirFunc::rope` views the plane as `[rows·heads, hd]`, tiles it `heads` rows tall at \
                 row base `ri·heads` for position `ri`, and broadcasts that position's ONE cos/sin \
                 row across those rows — so row = token·{heads} + head. Both views are \
                 `[{}, {head_dim}]`, so no shape check, no arrangement check (`addr_eq` admits a \
                 same-total contiguous reshape) and no footprint guard distinguishes them: lowering \
                 this would place every (row, head) block at another block's address and report \
                 success. THE KERNEL IS THE SIDE THAT CHANGES — a head-major nest in `rope_at` would \
                 also have to transpose the `[mq, heads·hd]` plane its attention consumer reads.",
                out.v_rows,
            ),
        });
    }
    let mut sym = 0i64;
    let a = emit::lower_ktir_to_superdsc::RopeAt {
        name,
        x: ins[0].name(),
        cos: ins[1].name(),
        sin: ins[2].name(),
        out: out.name(),
        t: out.tid,
        mq,
        total,
        sym_id_base: &mut sym,
        layout,
        rows_are_requests,
    };
    // THE DOOR. Enumerated, so a head dim outside the set is a refusal naming it rather than a
    // monomorphisation nobody asked for; 64/128/256 are the head dims this tree's models state
    // (`rope.py`'s `HEAD_DIM = 128`, and the crate's own slab path exists for hd > 64).
    let out = match head_dim {
        64 => emit::lower_ktir_to_superdsc::rope_at::<64>(a),
        128 => emit::lower_ktir_to_superdsc::rope_at::<128>(a),
        256 => emit::lower_ktir_to_superdsc::rope_at::<256>(a),
        other => {
            return Err(Error {
                stage: "geometry",
                message: format!(
                    "{name}: head_dim {other} has no `rope_at::<HD>` instantiation here (64, 128, \
                     256). It is a CONST parameter of the device layout, so a new head dim is a new \
                     arm in this match — never a runtime value threaded past the door."
                ),
            });
        }
    };
    out.map_err(|e| Error {
        stage: "ktir-superdsc",
        message: format!("{e:?}"),
    })
}

// ⛔ `drive_attn` IS DELETED (2026-09-29). It was the pattern-match door: read `H`/`GQA`/
// `HEAD_DIM` off a `Program::Attn` kernel's constexprs, pick the matching
// `attn_at::<NQH, NKVH, HD>` instantiation, and emit scratchy's prebuilt attention fragment
// assembly — substituting a hand-rolled implementation for the given Triton code. Attention
// goes through the whole-function door (`bake_py` `KTIR_WHOLE=1`): the compiled fixture's ops,
// lowered one at a time. `attn_at` itself stays in the vendored `ktir-superdsc` crate (scratchy
// still uses it); it is simply no longer reachable from our layer. Do not re-add this door.

/// THE HANDED-OFF FUNCTION AS THEIR MATCHERS READ IT, one line per op.
///
/// An instrument, not part of the compile path. Their structural readings (`program_rmsnorm_eps`,
/// `program_score_scale`, …) walk `func.operations` / `ops_deep()` looking for exact op-kind chains,
/// so when one refuses the only way to see WHY is to read the same list it read. `Debug` on
/// `IRFunction` prints the arena's whole graph and is unusable for that.
pub fn dump(node: &KtirNode) -> String {
    use std::fmt::Write as _;
    let f = &node.func;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "@{} args={} grid={:?} top={} deep={}",
        f.name,
        f.arguments.len(),
        f.grid,
        f.operations.len(),
        f.ops_deep().len()
    );
    fn walk(ops: &[ktir_core::ir::Operation<'static>], depth: usize, s: &mut String) {
        use std::fmt::Write as _;
        for o in ops {
            let pad = "  ".repeat(depth + 1);
            let res = o.result.map(|r| format!("%{} = ", r.0)).unwrap_or_default();
            let ins: Vec<String> = o.operands.iter().map(|v| format!("%{}", v.0)).collect();
            let attrs: Vec<String> =
                o.attributes.iter().map(|(k, v)| format!("{k:?}={v:?}")).collect();
            let _ = writeln!(
                s,
                "{pad}{res}{:?}({}) {}",
                o.op_type,
                ins.join(", "),
                attrs.join(" ")
            );
            for r in o.regions {
                walk(r, depth + 1, s);
            }
        }
    }
    walk(f.operations, 0, &mut s);
    s
}

/// Lower a WHOLE-KERNEL function through `ktir-superdsc`'s whole-function door.
///
/// The per-`Program` entry points read a function's parameter list as one op's operands, which is
/// exact for scratchy's one-node-per-function producer and wrong for ours -- and for IBM's C++
/// reference producer, which puts 3 `linalg.matmul` in one function for a SwiGLU MLP and 12 for a
/// decoder layer. `emit::whole_function::lower_function` walks the ops instead, handing each body
/// only its own operands' regions. It recognises nothing: an op with no 1:1 `Program` is refused by
/// name rather than assigned a node kind on the producer's behalf.
pub fn emit_whole(
    node: &KtirNode,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let mut sym = 0i64;
    ktir_superdsc::emit::whole_function::lower_function(node, layout, &mut sym).map_err(|e| Error {
        stage: "whole-function",
        message: format!("{e:?}"),
    })
}


// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE ROPE PATH, DRIVEN FROM THE FIXTURE — the measurement `examples/bake_py` cannot make.
//
//  `bake_py` dispatches on the runtime `Program`, so a rope stops at the model-geometry const door
//  ([`emit_regions`]) and never reaches [`drive_rope`]. Everything past that door — whether the
//  layout now serves `rope_at` (its rotate matrix P is placed, so `resolve_seg_base` has a name to
//  find rather than a panic to raise) and whether the ROW ORDER is the wall that remains — is
//  therefore only reachable from a test that crosses the door itself, with the fixture's own
//  `tl.constexpr` values.
// ══════════════════════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod rope_from_the_fixture {
    use super::*;

    /// `test/fixtures/rope.py` at `rope_kv8`'s configuration (H=8, N_TOK=256, HEAD_DIM=128,
    /// BLOCK_M=64, HALF=64, grid [4, 8]) — the same values `examples/bake_py`'s `rope_kv8` case and
    /// `pure_rust_ktir.rs`'s `rope_ce` state, so the three cannot drift about what the configuration
    /// is.
    fn rope_kv8() -> ktir_superdsc::ktir_node::KtirNode {
        use triton_frontend::codegen::{ArgSpec, KernelSpec};
        use triton_frontend::semantic::Val;
        use triton_frontend::target::Target;
        let mut signature = std::collections::HashMap::new();
        for p in ["desc_x", "desc_cos", "desc_sin", "desc_o"] {
            signature.insert(p.to_string(), ArgSpec::parse("*fp16").expect("signature"));
        }
        let mut constexprs = std::collections::HashMap::new();
        for (k, v) in [
            ("H", Val::Int(8)),
            ("N_TOK", Val::Int(256)),
            ("HEAD_DIM", Val::Int(128)),
            ("BLOCK_M", Val::Int(64)),
            ("HALF", Val::Int(64)),
        ] {
            signature.insert(k.to_string(), ArgSpec::parse("constexpr").expect("constexpr"));
            constexprs.insert(k.to_string(), v);
        }
        let file = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate sits one level under crates/triton")
            .join("test-fixtures/rope.py");
        let kspec = KernelSpec {
            kernel: "rope_fwd".to_string(),
            signature,
            constexprs,
            file: file.to_string_lossy().to_string(),
        };
        let src = std::fs::read_to_string(&file).expect("read rope.py");
        let mut tt = triton_frontend::codegen::compile(&src, &kspec, Target::spyre()).expect("codegen");
        triton_frontend::opt::make_ttir(&mut tt).expect("make_ttir");
        let mut m = triton_ktir::from_ttir::convert(&tt).expect("from_ttir");
        let grid = vec![4, 8];
        triton_ktir::make_ktir(&mut m, &grid).expect("make_ktir");
        triton_ktir::passes::to_ktir::run(&mut m, &grid).expect("to_ktir");
        node_for(&m, Program::Rope).expect("node_for")
    }

    /// ⭐ THE LAYOUT SERVES A ROPE NOW. It used to refuse the program outright for naming
    /// `ROPE_P_TID`; P's extent is `[hd, hd]` at the head dim the output view states, so it is
    /// placed, and this is the test that the refusal is gone rather than merely reworded.
    #[test]
    fn the_layout_places_ropes_rotate_matrix() {
        let node = rope_kv8();
        let l = layout::for_node(&node).expect("the layout must serve a rope");
        let p = l
            .placements
            .get(&ktir_superdsc::reserved_tids::ROPE_P_TID)
            .expect("P is placed");
        // `[128, 128]` fp16 — the head dim the fixture's own `HEAD_DIM` states, doubled by the
        // element size, and NOT a number this module chose.
        assert_eq!(p.size, 128 * 128 * 2);
        // Its name is in `ids` too, which is the half `resolve_seg_base` reaches first: a placement
        // without one is invisible there and takes the build down as an undeclared synthetic.
        assert!(
            l.id_of(&ktir_superdsc::place::act_name(
                ktir_superdsc::reserved_tids::ROPE_P_TID
            ))
            .is_some(),
            "a placement without its `ids` entry is the panic this places P to avoid"
        );
    }

    /// ⛔ AND THE WALL THAT REMAINS IS THE ROW ORDER — refused, by name, for the reason
    /// [`RopeRows`] settles: `KtirFunc::rope` is TOKEN-major and `rope.py`'s plane is HEAD-major.
    ///
    /// This is the test that a HEAD-major plane cannot reach a descriptor. It is the
    /// silent-wrong-answer case (same view shape, same footprint, `addr_eq` admits the reshape), so
    /// the only thing standing between it and a bake that reports success is this refusal.
    #[test]
    fn a_head_major_plane_is_refused_by_name() {
        let node = rope_kv8();
        let l = layout::for_node(&node).expect("layout");
        // `EmittedOp` has no `Debug`, so the Ok arm says what it emitted rather than dumping it.
        let e = match drive_rope(&node, 128, 256, 1024, RopeRows::HeadMajor, false, Some(&l)) {
            Err(e) => e,
            Ok(ops) => panic!(
                "a head-major plane reached {} descriptor(s) instead of being refused",
                ops.len()
            ),
        };
        assert_eq!(e.stage, "geometry");
        assert!(e.message.contains("HEAD-MAJOR"), "{}", e.message);
        assert!(e.message.contains("TOKEN-major"), "{}", e.message);
    }

    /// ⭐⭐ THE WALL AFTER THE ROW ORDER, MEASURED AND FAILING CLOSED: an angle table bound
    /// NARROWER than the plane that reads it, caught by the footprint guard in bytes.
    ///
    /// Told the plane is token-major (the order [`RopeRows`] settles as the right one), the drive
    /// crosses the geometry door and `rope_at` reads its angle tables at the roped plane's OWN
    /// extent — `[mq·heads, hd]`, because the worker "pre-tiles the cos/sin tables to the rope's full
    /// width" (`KtirFunc::rope`'s own words) and every head's slice of a row is identical. A table
    /// bound at HALF that width is not a saving, it is an OUT-OF-BOUNDS READ, and the guard states it
    /// in bytes rather than computing it as garbage:
    ///
    /// ```text
    ///   t1: access offset 0B + 524288B exceeds its placement footprint 32768B (seg3)
    /// ```
    ///
    /// 524288 B is `mq·total·2` (the whole plane, `256·1024·2`) and 32768 B is `N_TOK·HALF·2`
    /// (`256·64·2`). The same guard was reached at 2097152 B on `rope_q32`, whose 32 heads make the
    /// plane four times as wide against the same table.
    ///
    /// ⛔⛔⛔ THE HALF-WIDTH BINDING IS CONSTRUCTED HERE AND NO LONGER BORROWED FROM THE FIXTURE, AND
    /// THAT IS THE WHOLE POINT OF THIS EDIT. This test used to get its input by calling
    /// [`layout::for_node`] on `rope_kv8()` and relying on `test/fixtures/rope.py` binding its tables
    /// at `[N_TOK, HALF]` — its then-delta 2, "a full-width table would be two identical sticks per
    /// row … halving it is deleting a duplicate". Commit `bad6820b5` made that fixture TOKEN-MAJOR
    /// with both tables at `x`'s own extent (`shape=[y_dim, HEAD_DIM]`), which is the CORRECT
    /// binding — so the fixture stopped being a witness for the defect and this test silently stopped
    /// testing anything, failing as a `should_panic` that no longer panics.
    ///
    /// A test that takes its INPUT from a file whose job is to change was always going to rot that
    /// way. So the narrow binding is stated here, in the one place that needs it: the regions are read
    /// from the program and parameters 1 and 2 — `desc_cos` and `desc_sin`, by `rope_fwd`'s own
    /// parameter order — are re-bound at the extent the fixture used to declare.
    /// [`layout::for_regions`] sizes a placement from `v_rows × v_cols`, so that is the whole of it;
    /// the NODE is untouched, so `rope_at` still reads the plane at its real extent. Narrow
    /// placement, wide reader, which is exactly the defect — and it is now reachable no matter what
    /// any fixture declares.
    ///
    /// ⭐ AND THE PREMISE IS ASSERTED BEFORE THE DRIVE, so this cannot pass on someone else's panic: a
    /// `should_panic` test whose input is silently well-formed is the failure mode being repaired
    /// here, and an assert that fires first panics with the wrong message and fails.
    ///
    /// ⛔ IT IS A `should_panic` BECAUSE THE VENDOR STILL UNWRAPS THIS GUARD'S `Err` — CHECKED, not
    /// assumed, because the `Result` form did arrive for the matmul door this session
    /// (`try_assemble_matmul_seeded`, `aba078748`) and the obvious guess is that this moved with it.
    /// It did not: rope's pointwise leg goes through `assemble_pointwise_broadcast`, whose two tails
    /// are still `.unwrap_or_else(|e| panic!(…))` (`vendor/ktir-superdsc/src/emit/mod.rs:752` and
    /// `:755`), and `vendor/PROVENANCE.md` records under "FOUND AND DELIBERATELY NOT PATCHED" that
    /// threading a `Result` out of that family reaches ~40 call sites and was left as their call. So
    /// there is no `Err` for this test to read. If that door ever returns instead, this test is the
    /// place to notice: drop the attribute and assert the `Err`.
    #[test]
    #[should_panic(expected = "exceeds its placement footprint")]
    fn a_half_width_angle_table_is_caught_in_bytes() {
        let node = rope_kv8();
        let mut r = emit::lower_ktir_to_superdsc::regions(&node).expect("regions");
        // `rope_fwd`'s parameters are `desc_x, desc_cos, desc_sin, desc_o` and `regions()` pairs them
        // BY POSITION with the bindings, so 1 and 2 are the two angle tables. `[N_TOK, HALF]` =
        // `[256, 64]` is the extent `rope.py` declared before `bad6820b5` widened it.
        assert_eq!(r.len(), 4, "four parameters, or 1 and 2 are not cos and sin");
        r[1].v_rows = 256;
        r[1].v_cols = 64;
        r[2].v_rows = 256;
        r[2].v_cols = 64;
        let l = layout::for_regions(&node, &r).expect("layout");
        // THE PREMISE, ASSERTED: the table really is bound at 32768 B, and the plane that reads it
        // really is wider. Without these the test could pass on any panic at all, which is how it
        // came to be green against a fixture that no longer produced the defect.
        let cos = l.placements.get(&1).expect("cos is placed");
        assert_eq!(
            cos.size, 32768,
            "the narrow binding must be N_TOK*HALF*2 = 32768 B, or this test is not the defect"
        );
        let x = l.placements.get(&0).expect("the roped plane is placed");
        assert!(
            x.size > cos.size,
            "the plane ({} B) must be wider than its angle table ({} B), or there is no over-read",
            x.size,
            cos.size
        );
        let _ = drive_rope(&node, 128, 256, 1024, RopeRows::TokenMajor, false, Some(&l));
    }

    /// THE GEOMETRY CROSS-CHECK, from the other side: a geometry that does not belong to this
    /// program is a refusal and not an address. `mq` and `total` cannot both be read off a
    /// `[mq·heads, hd]` view, so they are crossed — and a wrong `H` would silently rescale every
    /// per-(row, head) offset if this check were not here.
    #[test]
    fn a_geometry_that_is_not_this_programs_is_refused() {
        let node = rope_kv8();
        let l = layout::for_node(&node).expect("layout");
        // H=4 instead of 8: `total` says 4 heads, so the view would have to be `[1024, 128]` and it
        // states `[2048, 128]`.
        let e = match drive_rope(&node, 128, 256, 512, RopeRows::TokenMajor, false, Some(&l)) {
            Err(e) => e,
            Ok(ops) => panic!(
                "a 4-head geometry over an 8-head program reached {} descriptor(s)",
                ops.len()
            ),
        };
        assert_eq!(e.stage, "geometry");
        assert!(e.message.contains("does not belong to this program"), "{}", e.message);
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE 7-SEGMENT WALL, GONE — a 14-parameter function is not a 14-segment demand.
//
//  This is the test for [`layout::for_regions`]'s packing, and it does not depend on any door past
//  the layout: `decoder_layer_one_flat` used to be refused at `layout` for wanting 14 of 7 HBM
//  segments, which put the whole decoder block behind an arithmetic of ours rather than a limit of
//  the device's (`wire::SEGMENT_OFFSETS` states its 7 PER OP — per emitted descriptor's operand
//  list, three or four tensors here).
// ══════════════════════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod decoder_from_the_fixture {
    use super::*;

    /// `test/fixtures/decoder_block.py`'s `decoder_layer_fwd` at `decoder_layer_one_flat`'s
    /// configuration — `examples/bake_py`'s own `dec_ptrs_one()` order and `dec_ce(256)` values.
    fn decoder_layer_one_flat() -> ktir_superdsc::ktir_node::KtirNode {
        use triton_frontend::codegen::{ArgSpec, KernelSpec};
        use triton_frontend::semantic::Val;
        use triton_frontend::target::Target;
        let mut signature = std::collections::HashMap::new();
        for p in [
            "desc_x", "desc_o", "desc_n1", "desc_wq", "desc_wk", "desc_wv", "desc_wo", "desc_mask",
            "desc_cos", "desc_sin", "desc_n2", "desc_wg", "desc_wu", "desc_wd",
        ] {
            signature.insert(p.to_string(), ArgSpec::parse("*fp16").expect("signature"));
        }
        let mut constexprs = std::collections::HashMap::new();
        for (k, v) in [
            ("M", Val::Int(64)),
            ("D_MODEL", Val::Int(128)),
            ("D_FF", Val::Int(256)),
            ("BLOCK_N", Val::Int(256)),
            ("HALF", Val::Int(64)),
            ("EPS", Val::Float(1e-05)),
            ("INV_D", Val::Float(1.0 / 128.0)),
            ("QK_SCALE", Val::Float(0.0078125 * 1.44269504)),
            ("RM", Val::Float(0.22)),
        ] {
            signature.insert(k.to_string(), ArgSpec::parse("constexpr").expect("constexpr"));
            constexprs.insert(k.to_string(), v);
        }
        let file = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate sits one level under crates/triton")
            .join("test-fixtures/decoder_block.py");
        let kspec = KernelSpec {
            kernel: "decoder_layer_fwd".to_string(),
            signature,
            constexprs,
            file: file.to_string_lossy().to_string(),
        };
        let src = std::fs::read_to_string(&file).expect("read decoder_block.py");
        let mut tt =
            triton_frontend::codegen::compile(&src, &kspec, Target::spyre()).expect("codegen");
        triton_frontend::opt::make_ttir(&mut tt).expect("make_ttir");
        let mut m = triton_ktir::from_ttir::convert(&tt).expect("from_ttir");
        let grid = vec![1];
        triton_ktir::make_ktir(&mut m, &grid).expect("make_ktir");
        triton_ktir::passes::to_ktir::run(&mut m, &grid).expect("to_ktir");
        // The kind is stated because the signature demands one; the whole-function door never reads
        // it, and neither does the layout.
        node_for(&m, Program::Matmul).expect("node_for")
    }

    #[test]
    fn fourteen_parameters_pack_into_two_segments_without_overlapping() {
        let node = decoder_layer_one_flat();
        let l = layout::for_node(&node).expect("14 parameters must not be a refusal");
        // ONE PLACEMENT PER PARAMETER **PLUS ONE PER REGISTRY SCALE**, and the second half is new:
        // this used to be a bare `== 14` because a decoder's `scalarmul_scales` was EMPTY — `scales_for`
        // is keyed on the stated `Program` and returns nothing for `Matmul`, which is what a
        // whole-kernel bake states. The whole-function door now populates the registry from the
        // program's own shape (its epsilons and its genuine scalar multipliers, NOT the rmsnorm
        // divisors), and each of those needs a placement or `resolve_seg_base` panics on it. So the
        // invariant is stated as the two parts it has, and the PARAMETER half is checked by tid rather
        // than by count — a count alone would pass if a parameter were dropped and a scale added.
        for (i, tid) in node.bindings.iter().enumerate() {
            assert!(
                l.placements.contains_key(&tid.get()),
                "parameter {i} (t{}) has no placement",
                tid.get()
            );
        }
        assert_eq!(
            l.placements.len(),
            14 + l.scalarmul_scales.len(),
            "one placement per parameter plus one per registry scale (14 parameters, {} scale(s))",
            l.scalarmul_scales.len()
        );
        // TWO segments, whatever the parameter count: reads in `Activation`, stores in `Logits`.
        let mut segs: Vec<usize> = l.placements.values().map(|p| p.segment).collect();
        segs.sort();
        segs.dedup();
        let mut want = vec![
            ktir_superdsc::placement::SegRole::Activation.segment(),
            ktir_superdsc::placement::SegRole::Logits.segment(),
        ];
        want.sort();
        assert_eq!(segs, want, "reads pack into one segment, stores into another");
        // AND THE PROOF THAT MAKES PACKING SAFE, re-stated as a test rather than trusted of the
        // arithmetic: no two placements share a 128-B granule of one segment. `for_regions` runs
        // `overlaps_none` itself, so this asserts the same fact the other way round — from the
        // placements it produced.
        let mut spans: Vec<(usize, u64, u64, u32)> = l
            .placements
            .values()
            .map(|p| {
                (
                    p.segment,
                    p.offset,
                    p.offset + ktir_superdsc::placement::align128(p.size).max(128),
                    p.tid,
                )
            })
            .collect();
        spans.sort();
        for w in spans.windows(2) {
            let (s0, _, e0, t0) = w[0];
            let (s1, b1, _, t1) = w[1];
            assert!(
                s0 != s1 || b1 >= e0,
                "t{t1} starts at {b1}B of seg{s1} while t{t0} still occupies it through {e0}B"
            );
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE SEGMENT-TAIL FAULT — a gather's index buffer may not end flush at its segment's
//  high-water, because the leg's HBM FETCH over-reads the ids by up to one 128-B granule and a
//  segment boundary is not a place that over-read can land.
//
//  MEASURED ON CARD (`paged_score_small`, rung 4): the ids were the segment's last placement,
//  ended exactly at `segment_bytes[3]`, and the fetch of the final 128-B stick faulted
//  (`HMI=FETCH addr=0xc00004480`, `run_prefix_only sync rc=-1`). ONE GRANULE of slack past the
//  high-water (17664→17792) made the identical bundle run clean with byte-identical output, and
//  the fault is scoped, not general: the embedding configs run card-verified with their ids at
//  offset 0 (never flush — a neighbour above absorbs the over-read), and attention's flush-at-end
//  OUTPUT writes clean — it is the gather's fetch path alone that pays this.
// ══════════════════════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod paged_score_from_the_fixture {
    use super::*;

    /// `test/fixtures/paged_score.py` at `paged_score_small`'s configuration — the same values
    /// `examples/bake_py`'s own case table states, so the two cannot drift about what the
    /// configuration is.
    fn paged_score_small() -> ktir_superdsc::ktir_node::KtirNode {
        use triton_frontend::codegen::{ArgSpec, KernelSpec};
        use triton_frontend::semantic::Val;
        use triton_frontend::target::Target;
        let mut signature = std::collections::HashMap::new();
        for (p, ty) in [
            ("desc_p", "*fp16"),
            ("desc_v", "*fp16"),
            ("desc_ids", "*i32"),
            ("desc_o", "*fp16"),
        ] {
            signature.insert(p.to_string(), ArgSpec::parse(ty).expect("signature"));
        }
        let mut constexprs = std::collections::HashMap::new();
        for (k, v) in [
            ("M", Val::Int(8)),
            ("K", Val::Int(64)),
            ("V", Val::Int(128)),
            ("BLOCK_M", Val::Int(8)),
            ("HEAD_DIM", Val::Int(64)),
        ] {
            signature.insert(k.to_string(), ArgSpec::parse("constexpr").expect("constexpr"));
            constexprs.insert(k.to_string(), v);
        }
        let file = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate sits one level under crates/triton")
            .join("test-fixtures/paged_score.py");
        let kspec = KernelSpec {
            kernel: "paged_vmatmul_fwd".to_string(),
            signature,
            constexprs,
            file: file.to_string_lossy().to_string(),
        };
        let src = std::fs::read_to_string(&file).expect("read paged_score.py");
        let mut tt =
            triton_frontend::codegen::compile(&src, &kspec, Target::spyre()).expect("codegen");
        triton_frontend::opt::make_ttir(&mut tt).expect("make_ttir");
        let mut m = triton_ktir::from_ttir::convert(&tt).expect("from_ttir");
        let grid = vec![1];
        triton_ktir::make_ktir(&mut m, &grid).expect("make_ktir");
        triton_ktir::passes::to_ktir::run(&mut m, &grid).expect("to_ktir");
        node_for(&m, Program::Matmul).expect("node_for")
    }

    /// ⛔ THE FAULT'S OWN PREMISE, ASSERTED FIRST: the fixture's ids really are the segment's
    /// last placement, ending flush at the pre-fix high-water. Without this the slack test below
    /// could pass against any layout at all — which is how the card run came to fault while every
    /// value-level test stayed green.
    #[test]
    fn the_ids_end_flush_at_their_segment_high_water() {
        let node = paged_score_small();
        let l = layout::for_node(&node).expect("the layout must serve a gather");
        let ids = l.placements.get(&2).expect("the ids parameter is t2");
        // 64 i32 entries = 256 B, and `pack`'s alignment keeps 256 a whole number of granules.
        assert_eq!(ids.size, 256, "64 i32 ids, or this test is not the defect");
        // THE PREMISE: the ids' granule span IS the segment's high-water — p [8,64] at 0, table
        // [128,64] at 1024, ids at 17408, so the pre-fix `segment_bytes[3]` was 17664 exactly.
        assert_eq!(
            ids.offset + ktir_superdsc::placement::align128(ids.size).max(128),
            17664,
            "the ids' span must end at the pre-fix high-water 17664 B, or the slack test is \
             not measuring this defect"
        );
        assert_eq!(ids.segment, ktir_superdsc::placement::SegRole::Activation.segment());
    }

    /// ⭐ THE FIX: the same layout now advances the high-water ONE GRANULE past the ids' span —
    /// the measured slack that made the identical bundle run clean on card with byte-identical
    /// output. The slack is added ONLY for a flush-at-high-water ids (an ids packed anywhere
    /// else has a neighbour above it to absorb the over-read), so the twelve card-verified
    /// configs' `segment_bytes` do not move.
    #[test]
    fn the_layout_leaves_one_granule_of_slack_past_a_flush_ids() {
        let node = paged_score_small();
        let l = layout::for_node(&node).expect("the layout must serve a gather");
        let ids = l.placements.get(&2).expect("the ids parameter is t2");
        assert_eq!(
            l.segment_bytes[ids.segment],
            17664 + 128,
            "seg{} must carry one 128-B granule of slack past the ids' span ({} B), because the \
             gather leg's HBM fetch over-reads the ids by up to one granule and faults on a \
             flush segment boundary — measured on card, rung 4",
            ids.segment,
            ids.offset + ids.size
        );
    }
}
