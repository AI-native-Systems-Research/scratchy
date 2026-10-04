//! THE WHOLE RUST PATH, PYTHON SOURCE TO KTIR, against the C++ chain's own KTIR.
//!
//! Needs the `ruff` feature, because every configuration starts from PYTHON SOURCE. That
//! is the point: no stage in here is handed a file.
#![cfg(feature = "ruff")]
//!
//! ```text
//!   triton_frontend::codegen::compile   .py source -> RAW ttir       value
//!   triton_frontend::opt::make_ttir     inline/fold/cse/dce/hoist    value
//!   triton_ktir::from_ttir::convert     ttir -> KTIR                 value
//!   triton_ktir::make_ktir              the seven KTDP/KTDF passes   value
//! ```
//!
//! No arrow above serializes. The only text in this file is on the GOLDEN side:
//! `third_party/spyre/test/experiment1/ktir/<name>.ktir.mlir`, which the C++ toolchain
//! printed and `crate::text::parse` reads back so the two can be compared.
//!
//! # WHAT IS COMPARED
//!
//! `crate::text::diff`, field by field on the SSA graph: each op's kind, its attributes,
//! its result types, and the CANONICAL IDS of its operands -- so a rename is invisible and
//! a rewire is not -- in program order, recursing into regions. Plus the census, so a count
//! mismatch is reported even when the first positional difference would mask it.
//!
//! One bounded exception, which is that module's own and not this test's: a block's leading
//! `arith.constant`s are compared as a MULTISET, because MLIR's constant hoist order is
//! internal and matching it is not a claim a port can make from the outside. Value, type
//! and count are all still compared.
//!
//! # THE CONTROLS
//!
//! A green diff over a path this long could be a diff comparing nothing. So
//! `planted_*` break the converted module in each of the ways that matter and assert the
//! diff NAMES the field, and `refuses_*` assert the adapter's allow-lists refuse by name
//! rather than falling through -- the failure mode that has produced four latent defects in
//! this tree, each of them well-formed enough to look right.

use std::collections::HashMap;

use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;
use triton_frontend::{opt, ttir};

use triton_ktir::from_ttir;
use triton_ktir::ir::{Attr, AttrKey, DType, IrType, Module, Op, OpKind, Ssa};
// THE SAME LITERAL THE WRITER USES. `dot_to_linalg` sets these maps and this test recognises them;
// spelling them twice is how a reader and a writer drift apart silently.
use triton_ktir::passes::dot_to_linalg::TRANSPOSED_B_MAPS;
use triton_ktir::text::diff::Difference;
use triton_ktir::text;

fn root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .expect("crates/triton, the re-hosted ladder's root")
}

fn fixture(name: &str) -> String {
    let p = root().join("test-fixtures").join(format!("{name}.py"));
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// The C++ chain's KTIR for a configuration, as the sibling checked it in.
fn golden(name: &str) -> Option<String> {
    let p = root()
        .join("test-experiment1/ktir")
        .join(format!("{name}.ktir.mlir"));
    std::fs::read_to_string(p).ok()
}

/// One configuration: the experiment-1 name, its fixture, its kernel, its signature and
/// constexprs, the launch grid, and the target its golden was produced at.
struct Case {
    name: &'static str,
    fixture: &'static str,
    spec: KernelSpec,
    grid: Vec<i64>,
    target: Target,
    expect: Expect,
}

/// The expectation for one configuration, MACHINE-CHECKED in both directions.
///
/// `tests/fixture_status.rs` in `triton-frontend` uses the same shape for the same reason: a
/// status written in prose rots, and a configuration that starts matching must FAIL this
/// test so the status is updated deliberately rather than drifting into a silent pass.
enum Expect {
    /// Matches the C++ KTIR field by field.
    Matches,
    /// Does NOT match, for a reason that is named and whose census signature is asserted.
    /// The `&str` must appear in the rendered findings, so the gap cannot silently become a
    /// DIFFERENT gap.
    DiffersOn(&'static str),
    /// Does not match, and EVERY finding is one of [`is_transposed_dot_weight`]'s named fields.
    ///
    /// ⭐ STRICTLY NARROWER THAN [`Expect::DiffersOn`], WHICH IS WHY IT EXISTS. `DiffersOn` asserts
    /// that ONE named line is present and says nothing about the rest, so a second, unrelated
    /// regression in the same configuration would ride along unexamined. This arm requires the
    /// RESIDUE to be EMPTY: any finding the predicate does not name still fails the test, by name.
    OnlyTheTransposedDotWeight,
}

/// THE ONE PLACE THIS PATH DELIBERATELY DIVERGES FROM IBM'S C++ KTIR, recorded rather than silenced.
///
/// `dot_to_linalg` spends a `tt.trans` of a dot's WEIGHT in `linalg.matmul`'s `indexing_maps`
/// (`TRANSPOSED_B_MAPS`, commit `d0a45a91c`). The C++ spends it on the OPERAND instead. That is ONE
/// decision of ours, and `tests/golden_ktir.rs::is_recorded_divergence` already records it over four
/// fields — but THIS test's diff is stricter than that one (it also walks memory views, access-tile
/// corner operands and an op census), so the same decision surfaces here in more places. Every
/// signature below was MEASURED, and each is named separately so that a change in which ones appear
/// is itself visible.
///
/// ⛔⛔ AND THE C++ HAS **TWO** BEHAVIOURS FOR A TRANSPOSED DOT OPERAND, WHICH IS WHY THE LIST IS
/// NOT JUST THE OTHER TEST'S FOUR. Measured across the goldens:
///
///  * **the operand is a LOADED weight** (`swiglu_mlp_*`, `attention_flash_*`) — the C++ FOLDS the
///    transpose into the access tile, and in the SwiGLU goldens into the memory view as well, so no
///    `tt.trans` survives but the view/tile/load shapes are the post-transpose ones. Ours are the
///    shapes the kernel declares.
///  * **the operand is a COMPUTED value** (`decoder_*`, the RoPE'd K) — there is no access tile to
///    fold into, so the C++ LEAVES the `tt.trans` as an op. Ours is rewired past it and the trans is
///    dead, so `canonicalize` removes it.
///
/// The second case is why [`fold_dead_dot_transposes`] exists: this diff is POSITIONAL, so two ops
/// the golden has and we do not shift every later op by two and turn one decision into forty
/// `op kind differs` findings. Accepting those wholesale would be a blanket skip on the whole
/// decoder — exactly what must not happen — so the normaliser removes the CAUSE on both sides
/// instead, and what is left is diffed honestly.
///
/// ⛔⛔⛔ AND SINCE THE FIXTURES MOVED TO THE VALIDATED PAIRING, THIS PREDICATE COVERS **TWO** KINDS
/// OF THING. Stated here because they share a field signature and a reader would otherwise assume
/// one:
///
///  1. **A LOWERING DIVERGENCE** — the same kernel, and we spend the transpose somewhere else than
///     the C++ does. That is what the paragraphs above describe, and it is all this used to be.
///  2. **A FIXTURE THE GOLDEN NO LONGER DESCRIBES.** `test/fixtures/{swiglu_mlp,decoder_block}.py`
///     now present EVERY projection weight n-major with `.T` on the dot operand (swiglu delta 13,
///     decoder delta 8). The C++ goldens in `test/experiment1/ktir/` were generated from the
///     PREVIOUS fixtures and only IBM's C++ chain can regenerate them, so for the SwiGLU down
///     projection and the decoder's seven projections the golden describes a kernel whose weights
///     are presented the other way round.
///
/// MEASURED, as the finding counts moving and nothing else: `swiglu_mlp_small` 14 -> 21 (the down
/// projection's own six fields plus its `indexing_maps`), `swiglu_mlp_tiled_k` 10 -> 17,
/// `decoder_layer_one` 2 -> 43 and `decoder_two_layers` 4 -> 86. The residue stayed EMPTY throughout,
/// which is the check that the new differences really are the weight's axis order and not something
/// else the fixture edit disturbed.
///
/// ⚠ SO THIS ONE PREDICATE IS DOING A JOB IT SHOULD EVENTUALLY BE SPLIT OUT OF. Both kinds are "the
/// weight's axis order", which is why the fields coincide; they are not the same fact. The moment the
/// goldens can be regenerated from the current fixtures, (2) disappears and the counts should fall
/// back to roughly the pre-change figures — and `Expect::OnlyTheTransposedDotWeight`'s surprise arm
/// fires if any configuration becomes an exact match, so nobody has to remember to check.
///
/// ⛔ NARROW ON PURPOSE. Each clause is an AND of an op kind and a field. `op kind` and `op count`
/// are deliberately NOT in here: those are what a real regression looks like, and after the
/// normaliser above they no longer appear for this cause.
fn is_transposed_dot_weight(d: &Difference) -> bool {
    // The op is the LAST path segment, so `ktdp.load` in one config's deep nest and in another's
    // flat body read the same. `where_` is `"@f/scf.for[4]/linalg.matmul[7]"`.
    let op = d.where_.rsplit('/').next().unwrap_or("");
    let is = |kind: &str| op.starts_with(kind);

    // (1) THE MAPS THEMSELVES — we carry `indexing_maps`, the C++ carries none.
    //
    // ⛔ AND THIS CLAUSE CHECKS THE VALUES, NOT JUST THE FIELD NAME, because `attributes` is the
    // only field a WRONG map would differ on too. `(d2, d1)` where `(d1, d2)` belongs is a
    // transposed-vs-not weight — a silently wrong contraction — and a clause that accepted any
    // `linalg.matmul attributes` difference would absorb it. So the golden side must be EMPTY and
    // ours must state all three of `dot_to_linalg`'s own maps.
    let maps = is("linalg.matmul")
        && d.field == "attributes"
        && d.expected.trim() == "{}"
        && TRANSPOSED_B_MAPS.iter().all(|m| d.actual.contains(m));

    // (2) THE WEIGHT'S DECLARED BUFFER, in the SwiGLU goldens: `sizes`/`strides`/`coordinate_set`
    // and the `memref<>` type, post-transpose there and as-declared here.
    let view_ty = is("ktdp.construct_memory_view") && d.field == "result types";
    let view_at = is("ktdp.construct_memory_view") && d.field == "attributes";
    // (3) THE WINDOW OVER IT: the transposing `access_tile_order`, the swapped `access_tile_set`,
    // the post-transpose `!ktdp.access_tile<>` type, and the two corner operands in the other order.
    let tile_ty = is("ktdp.construct_access_tile") && d.field == "result types";
    let tile_at = is("ktdp.construct_access_tile") && d.field == "attributes";
    let tile_op = is("ktdp.construct_access_tile") && d.field == "operands";
    // (4) `swiglu_mlp_tiled_k` states those same two corners as two `arith.index_cast`s of the inner
    // loop's block arguments rather than as tile operands, so the swap surfaces one op earlier.
    let corner_cast = is("arith.index_cast") && d.field == "operands";
    // (5) THE VALUE READ THROUGH THAT WINDOW.
    let load_ty = is("ktdp.load") && d.field == "result types";

    maps || view_ty || view_at || tile_ty || tile_at || tile_op || corner_cast || load_ty
}

fn spec(
    fixture: &str,
    kernel: &str,
    ptrs: &[(&str, &str)],
    constexprs: &[(&str, Val)],
) -> KernelSpec {
    let mut signature = HashMap::new();
    for (k, v) in ptrs {
        signature.insert(
            (*k).to_string(),
            ArgSpec::parse(v).unwrap_or_else(|e| panic!("bad signature {k}={v}: {e}")),
        );
    }
    let mut ce = HashMap::new();
    for (k, v) in constexprs {
        signature.insert((*k).to_string(), ArgSpec::parse("constexpr").unwrap());
        ce.insert((*k).to_string(), v.clone());
    }
    KernelSpec {
        kernel: kernel.to_string(),
        signature,
        constexprs: ce,
        file: root()
            .join("third_party/spyre/test/fixtures")
            .join(format!("{fixture}.py"))
            .to_string_lossy()
            .to_string(),
    }
}

/// Every configuration the sibling checked a KTIR golden in for, with the constexprs and
/// grid taken from `test/experiment1/index.json` rather than reinvented.
///
/// `embedding_granite` at BLOCK_M=64 IS HERE NOW, and the reason it was not is worth keeping:
/// the C++ `make_ktir` used to REFUSE it at `ktdf.corelet_plan` ("the two data_bounds ranges
/// must be disjoint and contiguous and non-empty, but got [0, 0] and [0, 1]") because its i32
/// index tile is exactly ONE STICK. The `single_corelet` fix removed that refusal, a golden
/// now exists, and this is the configuration that made the fix necessary in the first place.
/// `embedding_granite_bm128_control` was the control that isolated it and stays beside it.
fn cases() -> Vec<Case> {
    let attn: &[(&str, &str)] = &[
        ("desc_q", "*fp16"),
        ("desc_k", "*fp16"),
        ("desc_v", "*fp16"),
        ("desc_o", "*fp16"),
        ("desc_mask", "*fp16"),
    ];
    let attn_ce = |stage: i128| {
        vec![
            ("Z", Val::Int(1)),
            ("H", Val::Int(4)),
            ("N_CTX", Val::Int(256)),
            ("HEAD_DIM", Val::Int(128)),
            ("BLOCK_M", Val::Int(64)),
            ("BLOCK_N", Val::Int(64)),
            ("GQA", Val::Int(2)),
            ("STAGE", Val::Int(stage)),
            ("sm_scale", Val::Float(1.0)),
        ]
    };
    let swiglu: &[(&str, &str)] = &[
        ("desc_x", "*fp16"),
        ("desc_wg", "*fp16"),
        ("desc_wu", "*fp16"),
        ("desc_wd", "*fp16"),
        ("desc_o", "*fp16"),
    ];
    let swiglu_ce = |d_model: i128, d_ff: i128, block_k: i128| {
        vec![
            ("M", Val::Int(64)),
            ("D_MODEL", Val::Int(d_model)),
            ("D_FF", Val::Int(d_ff)),
            ("BLOCK_M", Val::Int(64)),
            ("BLOCK_N", Val::Int(64)),
            ("BLOCK_K", Val::Int(block_k)),
        ]
    };
    let rope: &[(&str, &str)] = &[
        ("desc_x", "*fp16"),
        ("desc_cos", "*fp16"),
        ("desc_sin", "*fp16"),
        ("desc_o", "*fp16"),
    ];
    // NO `BLOCK_M`: `rope.py` delta 7 made a work item one whole POSITION, H rows tall, so
    // there is nothing left to block and the constexpr is gone from the kernel's signature.
    let rope_ce = |h: i128| {
        vec![
            ("H", Val::Int(h)),
            ("N_TOK", Val::Int(256)),
            ("HEAD_DIM", Val::Int(128)),
            ("HALF", Val::Int(64)),
        ]
    };
    let dec_ce = || {
        vec![
            ("M", Val::Int(64)),
            ("D_MODEL", Val::Int(128)),
            ("D_FF", Val::Int(256)),
            ("BLOCK_N", Val::Int(64)),
            ("HALF", Val::Int(64)),
            ("EPS", Val::Float(1e-05)),
            ("INV_D", Val::Float(1.0 / 128.0)),
            ("QK_SCALE", Val::Float(0.0078125 * 1.44269504)),
            ("RM", Val::Float(0.22)),
        ]
    };
    let dec_ptrs = |ptrs: &[&'static str]| -> Vec<(&'static str, &'static str)> {
        ptrs.iter().map(|p| (*p, "*fp16")).collect()
    };

    vec![
        Case {
            name: "attention_flash_noncausal",
            fixture: "attention_flash",
            spec: spec("attention_flash", "attn_fwd", attn, &attn_ce(1)),
            grid: vec![4, 4],
            // The oracle HAS the `f16 / f16 -> f32` promotion this backend turns off for
            // Spyre, and attention's epilogue is the one bare f16 divide in any fixture.
            // `tests/divergence.rs` in triton-frontend asserts the difference is exactly
            // that divide; here the golden's own target is the one to reproduce.
            target: Target::upstream_gpu(),
            // ⛔ K IS A TRANSPOSED DOT OPERAND, so this differs from the C++ golden on exactly
            // the fields [`is_transposed_dot_weight`] names — the access tile's transposing
            // `access_tile_order` and post-transpose type, the load's type, and our
            // `indexing_maps`. Four findings here, eight in the causal configuration because it
            // has two score matmuls. Anything else still fails.
            expect: Expect::OnlyTheTransposedDotWeight,
        },
        Case {
            name: "attention_flash_causal",
            fixture: "attention_flash",
            spec: spec("attention_flash", "attn_fwd", attn, &attn_ce(3)),
            grid: vec![4, 4],
            target: Target::upstream_gpu(),
            expect: Expect::OnlyTheTransposedDotWeight,
        },
        Case {
            name: "swiglu_mlp_small",
            fixture: "swiglu_mlp",
            spec: spec("swiglu_mlp", "swiglu_mlp_fwd", swiglu, &swiglu_ce(128, 256, 128)),
            grid: vec![1],
            target: Target::spyre(),
            // ⛔ THE TWO PROJECTION WEIGHTS ARE TRANSPOSED DOT OPERANDS. Same one decision as
            // attention, but the SwiGLU goldens fold the transpose into the memory VIEW as well as
            // the tile, so the recorded field list is wider here — see
            // [`is_transposed_dot_weight`]. `work_division` was ALSO missing from all three of
            // these until `plan_corelets::recover_matmul_shape` was taught to read the weight's
            // orientation off the maps; that was a real defect of ours and is fixed, not recorded.
            expect: Expect::OnlyTheTransposedDotWeight,
        },
        Case {
            name: "swiglu_mlp_granite",
            fixture: "swiglu_mlp",
            spec: spec(
                "swiglu_mlp",
                "swiglu_mlp_fwd",
                swiglu,
                &swiglu_ce(4096, 12800, 4096),
            ),
            grid: vec![1],
            target: Target::spyre(),
            expect: Expect::OnlyTheTransposedDotWeight,
        },
        Case {
            name: "swiglu_mlp_tiled_k",
            fixture: "swiglu_mlp",
            spec: spec("swiglu_mlp", "swiglu_mlp_fwd", swiglu, &swiglu_ce(128, 256, 64)),
            grid: vec![1],
            target: Target::spyre(),
            // ⛔ As `swiglu_mlp_small`, except that the K-tiled inner loop states the weight
            // tile's two corners as `arith.index_cast`s of its block arguments rather than as tile
            // operands, so the corner swap surfaces one op earlier. Same cause, named separately.
            expect: Expect::OnlyTheTransposedDotWeight,
        },
        Case {
            name: "embedding_granite_bm128_control",
            fixture: "embedding",
            spec: spec(
                "embedding",
                "embedding_fwd",
                &[("desc_ids", "*i32"), ("desc_table", "*fp16"), ("desc_o", "*fp16")],
                &[
                    ("N_TOK", Val::Int(256)),
                    ("V", Val::Int(49159)),
                    ("D_MODEL", Val::Int(4096)),
                    ("BLOCK_M", Val::Int(128)),
                    ("EMB_SCALE", Val::Float(12.0)),
                ],
            ),
            grid: vec![2],
            target: Target::spyre(),
            // WAS the one gap, and the `Expect::DiffersOn` machinery is what closed it: it
            // failed with "now MATCHES ... update `Expect`" the moment the
            // `tt.descriptor_gather` arm landed, rather than passing quietly against a stale
            // record. See `the_gather_is_lowered_like_the_cpp`.
            expect: Expect::Matches,
        },
        Case {
            name: "embedding_granite",
            fixture: "embedding",
            spec: spec(
                "embedding",
                "embedding_fwd",
                &[("desc_ids", "*i32"), ("desc_table", "*fp16"), ("desc_o", "*fp16")],
                &[
                    ("N_TOK", Val::Int(256)),
                    ("V", Val::Int(49159)),
                    ("D_MODEL", Val::Int(4096)),
                    ("BLOCK_M", Val::Int(64)),
                    ("EMB_SCALE", Val::Float(12.0)),
                ],
            ),
            grid: vec![4],
            target: Target::spyre(),
            expect: Expect::Matches,
        },
        Case {
            name: "rmsnorm_granite",
            fixture: "rmsnorm",
            spec: spec(
                "rmsnorm",
                "rmsnorm_fwd",
                &[("desc_x", "*fp16"), ("desc_w", "*fp16"), ("desc_o", "*fp16")],
                &[
                    ("M", Val::Int(64)),
                    ("D_MODEL", Val::Int(4096)),
                    ("BLOCK_M", Val::Int(64)),
                    ("EPS", Val::Float(1e-05)),
                    ("INV_D", Val::Float(1.0 / 4096.0)),
                ],
            ),
            grid: vec![1],
            target: Target::spyre(),
            expect: Expect::Matches,
        },
        // THE ROPE C++ GOLDENS DESCRIBE A SUPERSEDED KERNEL, AND THAT IS RECORDED HERE
        // RATHER THAN SKIPPED.
        //
        // `test/experiment1/ktir/rope_{q32,kv8}.ktir.mlir` were generated by the C++ chain from
        // the HEAD-MAJOR `rope.py` -- `offs_m = off_h * N_TOK + start_m * BLOCK_M`, a
        // `[BLOCK_M, HALF]` block, grid `[4, H]`. `rope.py` delta 7 made the fixture TOKEN-major
        // (row = position * H + head), because that is the order `KtirFunc::rope` -- the producer
        // the device emitter `rope_at` was written against -- tiles and broadcasts for, and
        // because a head-major tall view is the reshape of the TRANSPOSED `[mq, heads*hd]` plane
        // attention reads. The two orders share a shape, so the mismatch was a wrong ANSWER
        // rather than a build error; see `RopeRows` in `triton-ktir-superdsc/src/lib.rs`.
        //
        // So the golden is stale BY CONSTRUCTION and cannot be met without reintroducing the
        // defect. It also cannot be regenerated here: only IBM's C++ chain produces these files.
        //
        // THE NEEDLE IS THE ROW ORDER'S OWN SIGNATURE IN THE CENSUS. Head-major needed a
        // TWO-AXIS grid and delinearized it -- `distribute_work` emits one `arith.divui` and one
        // `arith.remui` to split the linear work index into (block, head), which is where
        // `off_h * N_TOK + start_m * BLOCK_M` came from. Token-major has ONE grid axis (positions),
        // so both ops are gone: MEASURED, `census/arith.divui golden: 1 ours: 0` on both
        // configurations, alongside `arith.remui` and the `arith.addi` that summed the two terms.
        //
        // The table contract (delta 8) is the second cause and shows as the cos/sin
        // `ktdp.construct_memory_view` shape, `[N_TOK, HALF]` -> `[N_TOK * H, HEAD_DIM]`. It is not
        // the needle only because the rendered diff is capped at 25 findings and the census lines
        // sort first -- not because it is less load-bearing.
        //
        // So a diff that does NOT contain this line is a diff about something other than the row
        // order, and it fails here rather than being absorbed. `Expect::DiffersOn`'s surprise arm
        // fails too if the golden ever starts matching again.
        Case {
            name: "rope_q32",
            fixture: "rope",
            spec: spec("rope", "rope_fwd", rope, &rope_ce(32)),
            grid: vec![256],
            target: Target::spyre(),
            expect: Expect::DiffersOn("census/arith.divui"),
        },
        Case {
            name: "rope_kv8",
            fixture: "rope",
            spec: spec("rope", "rope_fwd", rope, &rope_ce(8)),
            grid: vec![256],
            target: Target::spyre(),
            expect: Expect::DiffersOn("census/arith.divui"),
        },
        Case {
            name: "decoder_layer_one",
            fixture: "decoder_block",
            spec: spec(
                "decoder_block",
                "decoder_layer_fwd",
                &dec_ptrs(&[
                    "desc_x", "desc_o", "desc_n1", "desc_wq", "desc_wk", "desc_wv", "desc_wo",
                    "desc_mask", "desc_cos", "desc_sin", "desc_n2", "desc_wg", "desc_wu",
                    "desc_wd",
                ]),
                &dec_ce(),
            ),
            grid: vec![1],
            target: Target::spyre(),
            // ⛔ THE ROPE'D K IS A TRANSPOSED DOT OPERAND THAT IS **COMPUTED**, NOT LOADED, so
            // there is no access tile for the C++ to fold into and it leaves the `tt.trans`
            // standing where we rewire past it. [`fold_dead_dot_transposes`] removes that op from
            // both sides before the diff — PROVEN exact: the golden minus its `tt.trans` ops IS our
            // op list, element for element — leaving only our `indexing_maps`. Without the
            // normaliser this reported 47 findings, ~40 of them `op kind differs` on ops that are
            // identical, which is a positional shift and not a difference.
            expect: Expect::OnlyTheTransposedDotWeight,
        },
        Case {
            name: "decoder_two_layers",
            fixture: "decoder_block",
            spec: spec(
                "decoder_block",
                "decoder_two_layers_fwd",
                &dec_ptrs(&[
                    "desc_x", "desc_o", "desc_n1a", "desc_wqa", "desc_wka", "desc_wva",
                    "desc_woa", "desc_n2a", "desc_wga", "desc_wua", "desc_wda", "desc_n1b",
                    "desc_wqb", "desc_wkb", "desc_wvb", "desc_wob", "desc_n2b", "desc_wgb",
                    "desc_wub", "desc_wdb", "desc_mask", "desc_cos", "desc_sin",
                ]),
                &dec_ce(),
            ),
            grid: vec![1],
            target: Target::spyre(),
            expect: Expect::OnlyTheTransposedDotWeight,
        },
    ]
}

/// Sort each block's LEADING run of CONSTANT-LIKE ops into a canonical order, on BOTH
/// sides.
///
/// # THIS EXTENDS AN EXCEPTION `text::diff` ALREADY MAKES, FOR THE SAME REASON
///
/// That module already compares a block's leading `arith.constant`s as a MULTISET, and its
/// header says why: MLIR's constant hoist order is internal, and matching it is not a claim
/// a port can make from the outside. `triton_frontend::opt::hoist_constants` reproduces the
/// move-to-front rule and gets it right for eleven of sixteen configurations, but the greedy
/// driver interleaves hoisting with folding on a worklist, so the last permutation of three
/// or four constants is not derivable.
///
/// WHAT MADE THAT INSUFFICIENT HERE. `DecomposeDenseConstants` turns each `dense<>` splat
/// into a scalar `arith.constant` plus a `tensor.splat`, so the permutation of the leading
/// constants becomes a permutation of `tensor.splat` OPS -- which the multiset exception
/// does not cover. Every SwiGLU configuration reported it as six findings: two swapped
/// splats, plus four knock-ons, because the diff refers to a value as "result of op K" and
/// so a permutation renumbers every reference.
///
/// WHY A `tensor.splat` BELONGS IN THE EXCEPTION. It is pure, and its only operand is a
/// constant that is itself in the leading run, so two splats of different constants are
/// independent of each other and their relative position is not a fact about the program.
///
/// WHAT IS STILL COMPARED, so this narrows nothing that matters: every constant's VALUE,
/// every splat's shape, and HOW MANY of each there are -- the sort key is built from
/// exactly those, and `a_planted_constant_value_change_is_caught` is the control.
fn canonicalize_leading_constants(m: &mut Module) {
    fn key(ops: &[triton_ktir::ir::Op], o: &triton_ktir::ir::Op) -> String {
        let ty = o
            .result_type()
            .map(text::print::print_type)
            .unwrap_or_default();
        // For a splat, fold in the VALUE of the constant it reads, so two splats of the
        // same shape and different values sort deterministically and compare distinctly.
        let payload = match o.attr(&AttrKey::Value) {
            Some(a) => text::print::print_attr(a),
            None => o
                .operands
                .first()
                .and_then(|v| ops.iter().find(|d| d.results.contains(v)))
                .and_then(|d| d.attr(&AttrKey::Value))
                .map(text::print::print_attr)
                .unwrap_or_default(),
        };
        format!("{}|{ty}|{payload}", o.kind.spelling())
    }
    fn walk(ops: &mut Vec<triton_ktir::ir::Op>) {
        for o in ops.iter_mut() {
            for r in &mut o.regions {
                walk(&mut r.ops);
            }
        }
        let n = ops
            .iter()
            .take_while(|o| {
                matches!(o.kind, OpKind::ArithConstant | OpKind::TensorSplat)
                    && o.regions.is_empty()
            })
            .count();
        let snapshot = ops.clone();
        ops[..n].sort_by_key(|o| key(&snapshot, o));
    }
    walk(&mut m.ops);
}

/// FOLD A DEAD DOT TRANSPOSE — a `tt.trans` whose ONLY use is a `linalg.matmul`'s weight operand is
/// rewired past and erased. Applied to BOTH sides, exactly as [`canonicalize_leading_constants`] is.
///
/// ⛔⛔⛔ WHY A NORMALISER AND NOT A RECORDED FIELD. This diff is POSITIONAL: it compares the
/// golden's op `i` with ours. `dot_to_linalg` rewires a dot's weight past its `tt.trans` and
/// `canonicalize` then deletes the trans, so where the C++ leaves the trans standing our op list is
/// the golden's MINUS those ops — and every op after the first one is compared against the wrong
/// partner. MEASURED on the decoder goldens, and it is exact rather than approximate: deleting every
/// `tt.trans` from `decoder_layer_one`'s golden op list yields OUR list element for element (123 →
/// 121, 2 transposes), and the same for `decoder_two_layers` (218 → 214, 4 transposes, 2 per layer).
/// Un-normalised, that ONE decision was reported as 47 and 123 findings, of which ~40 and ~115 were
/// `op kind differs` on ops that are in fact identical.
///
/// Recording those as an accepted field would have meant accepting `op kind differs` — which is
/// precisely the shape of the regressions this test exists to catch, on every op in the decoder. So
/// the CAUSE is normalised away and the residue is diffed honestly; what is left is the
/// `indexing_maps` attribute, which [`is_transposed_dot_weight`] names.
///
/// ⛔ NARROW, AND SIDE-AGNOSTIC. Only a `tt.trans` feeding operand 1 of a `linalg.matmul`, whose
/// result has exactly ONE use in the whole module. A `tt.trans` anywhere else — on an activation, on
/// a stored value, read twice — is left alone and still diffs. Ours contains none of these, so this
/// is a no-op on our side today; it is applied to both because a normaliser that privileges one side
/// is a way of writing the answer down.
fn fold_dead_dot_transposes(m: &mut Module) {
    use std::collections::{HashMap, HashSet};
    // Uses of every value, over the WHOLE module including region bodies. A trans read anywhere
    // else than by the one matmul must not be folded.
    let mut uses: HashMap<Ssa, usize> = HashMap::new();
    for o in m.ops_deep() {
        for v in &o.operands {
            *uses.entry(*v).or_insert(0) += 1;
        }
    }
    // The foldable ones: trans result -> the value the trans reads.
    let mut fold: HashMap<Ssa, Ssa> = HashMap::new();
    let defs: HashMap<Ssa, (OpKind, Option<Ssa>)> = m
        .ops_deep()
        .iter()
        .flat_map(|o| {
            o.results
                .iter()
                .map(|r| (*r, (o.kind.clone(), o.operands.first().copied())))
                .collect::<Vec<_>>()
        })
        .collect();
    for o in m.ops_deep() {
        if o.kind != OpKind::LinalgMatmul || o.operands.len() < 2 {
            continue;
        }
        let w = o.operands[1];
        // `OpKind` is not `Copy` here, so the def is matched by reference.
        if let Some((k, Some(src))) = defs.get(&w) {
            if *k == OpKind::TtTrans && uses.get(&w).copied() == Some(1) {
                fold.insert(w, *src);
            }
        }
    }
    if fold.is_empty() {
        return;
    }
    let dead: HashSet<Ssa> = fold.keys().copied().collect();
    fn walk(ops: &mut Vec<Op>, fold: &std::collections::HashMap<Ssa, Ssa>, dead: &std::collections::HashSet<Ssa>) {
        for o in ops.iter_mut() {
            for v in o.operands.iter_mut() {
                if let Some(src) = fold.get(v) {
                    *v = *src;
                }
            }
            for r in o.regions.iter_mut() {
                walk(&mut r.ops, fold, dead);
            }
        }
        ops.retain(|o| !(o.kind == OpKind::TtTrans && o.results.iter().any(|r| dead.contains(r))));
    }
    walk(&mut m.ops, &fold, &dead);
}

/// What one configuration produced, so the report is data rather than prose.
struct Outcome {
    name: &'static str,
    /// `Ok` carries the converted-and-lowered module; `Err` a refusal, verbatim.
    result: std::result::Result<Module, String>,
    /// Ops in the post-`make_ttir` ttir, and ops after the adapter. MUST be equal.
    census_in: usize,
    census_out: usize,
}

/// Run the whole Rust path for one configuration. Nothing here writes or reads a file
/// except the fixture's own `.py` source.
fn run(case: &Case) -> Outcome {
    let src = fixture(case.fixture);
    let mut m = match codegen::compile(&src, &case.spec, case.target) {
        Ok(m) => m,
        Err(e) => {
            return Outcome {
                name: case.name,
                result: Err(format!("bridge one refused: {e}")),
                census_in: 0,
                census_out: 0,
            }
        }
    };
    if let Err(e) = opt::make_ttir(&mut m) {
        return Outcome {
            name: case.name,
            result: Err(format!("make_ttir refused: {e}")),
            census_in: 0,
            census_out: 0,
        };
    }
    let (_, census_in) = from_ttir::census(&m);
    let mut k = match from_ttir::convert(&m) {
        Ok(k) => k,
        Err(e) => {
            return Outcome {
                name: case.name,
                result: Err(format!("from_ttir refused: {e}")),
                census_in,
                census_out: 0,
            }
        }
    };
    let (_, census_out) = from_ttir::census_ktir(&k);
    if let Err(e) = triton_ktir::make_ktir(&mut k, &case.grid) {
        return Outcome {
            name: case.name,
            result: Err(format!("make_ktir refused: {e}")),
            census_in,
            census_out,
        };
    }
    Outcome { name: case.name, result: Ok(k), census_in, census_out }
}

/// THE HEADLINE: our KTIR against the C++ chain's, per configuration.
#[test]
fn every_configuration_matches_the_cpp_ktir_golden() {
    let mut failures: Vec<String> = Vec::new();
    let mut matched = 0usize;
    let mut differs = 0usize;
    for case in cases() {
        let out = run(&case);
        let g = match golden(case.name) {
            Some(g) => g,
            None => {
                failures.push(format!(
                    "{}: no test/experiment1/ktir/{}.ktir.mlir to diff against",
                    case.name, case.name
                ));
                continue;
            }
        };
        let ours = match &out.result {
            Ok(k) => k,
            Err(e) => {
                // A REFUSAL IS A RESULT, quoted verbatim -- but not against a
                // configuration the C++ chain produced a KTIR for.
                failures.push(format!("{}: {e}", case.name));
                continue;
            }
        };
        // THE ADAPTER NEITHER ADDS NOR REMOVES AN OPERATION. It is a change of
        // representation; a difference is the bug.
        if out.census_in != out.census_out {
            failures.push(format!(
                "{}: from_ttir took {} op(s) in and produced {} -- the adapter must be \
                 census-neutral",
                case.name, out.census_in, out.census_out
            ));
        }
        let golden_m = match text::parse::parse(&g) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("{}: the C++ golden did not parse: {e}", case.name));
                continue;
            }
        };
        let mut golden_m = golden_m;
        let mut mine = ours.clone();
        canonicalize_leading_constants(&mut golden_m);
        canonicalize_leading_constants(&mut mine);
        // Both sides, before the diff: see [`fold_dead_dot_transposes`]. A no-op on ours.
        fold_dead_dot_transposes(&mut golden_m);
        fold_dead_dot_transposes(&mut mine);
        let ours = &mine;
        let mut findings = text::diff::census_diff(&golden_m, ours);
        findings.extend(text::diff::diff(&golden_m, ours));
        let rendered: String = findings
            .iter()
            .map(|f| format!("\n    {f}"))
            .collect::<Vec<_>>()
            .join("");
        match (&case.expect, findings.is_empty()) {
            (Expect::Matches, true) => {
                matched += 1;
                eprintln!(
                    "{:<34} MATCHES  ({} op(s), census {} -> {})",
                    case.name,
                    ours.ops_deep().len(),
                    out.census_in,
                    out.census_out
                );
            }
            (Expect::Matches, false) => {
                eprintln!("{:<34} DIFFERS  ({} finding(s))", case.name, findings.len());
                failures.push(format!(
                    "{}: expected to MATCH the C++ KTIR but has {} finding(s):{}",
                    case.name,
                    findings.len(),
                    rendered
                ));
            }
            (Expect::DiffersOn(needle), false) => {
                if rendered.contains(needle) {
                    differs += 1;
                    eprintln!(
                        "{:<34} DIFFERS as recorded, on `{needle}` ({} finding(s)){}",
                        case.name,
                        findings.len(),
                        rendered
                    );
                } else {
                    failures.push(format!(
                        "{}: differs, but NOT on `{needle}` -- the recorded gap has become a \
                         DIFFERENT gap and the status is now wrong:{}",
                        case.name, rendered
                    ));
                }
            }
            (Expect::OnlyTheTransposedDotWeight, _) => {
                let residue: Vec<String> = findings
                    .iter()
                    .filter(|f| !is_transposed_dot_weight(f))
                    .map(|f| format!("{f}"))
                    .collect();
                if findings.is_empty() {
                    // SURPRISE, and the same rule as the arm below: better news than the record,
                    // and it must not pass quietly.
                    failures.push(format!(
                        "{}: now MATCHES the C++ KTIR, but is recorded as differing only on the \
                         transposed dot weight. Update `Expect` to `Matches` -- a stale status that \
                         reads worse than reality is still a wrong status.",
                        case.name
                    ));
                } else if residue.is_empty() {
                    differs += 1;
                    eprintln!(
                        "{:<34} DIFFERS only on the transposed dot weight ({} finding(s), all \
                         named)",
                        case.name,
                        findings.len()
                    );
                } else {
                    failures.push(format!(
                        "{}: differs on {} finding(s) that the transposed dot weight does NOT \
                         account for -- a SECOND divergence is hiding behind the recorded one:{}",
                        case.name,
                        residue.len(),
                        residue.iter().map(|f| format!("\n    {f}")).collect::<Vec<_>>().join("")
                    ));
                }
            }
            (Expect::DiffersOn(needle), true) => {
                // SURPRISE. Better news than the status claims, and it must not pass
                // quietly: somebody closed the gap and this file still says it is open.
                failures.push(format!(
                    "{}: now MATCHES the C++ KTIR, but is recorded as differing on \
                     `{needle}`. Update `Expect` to `Matches` -- a stale status that reads \
                     worse than reality is still a wrong status.",
                    case.name
                ));
            }
        }
    }
    eprintln!(
        "TOTAL: {matched} of {} configuration(s) match the C++ KTIR field by field; \
         {differs} differ as recorded",
        cases().len()
    );
    assert!(
        failures.is_empty(),
        "the Rust path does not reproduce the C++ KTIR for {} configuration(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        matched > 0,
        "zero configurations were compared, which is a harness failure and not a pass"
    );
}

//===----------------------------------------------------------------------===//
// The controls
//===----------------------------------------------------------------------===//

/// EVERY `linalg.matmul` WHOSE WEIGHT IS A **PRESENTED BUFFER** STATES THE TRANSPOSE-B MAPS.
///
/// ⭐ THIS IS THE INVARIANT THE FIXTURES WERE CHANGED TO ESTABLISH, AND IT NEEDS A TEST BECAUSE THE
/// FAILURE IS SILENT. An absent `indexing_maps` IS `linalg.matmul`'s default `[(d0,d2), (d2,d1),
/// (d0,d1)]`, i.e. W as `[k, n]`; nothing in `ktir-superdsc` reads the maps, so BOTH orientations
/// lower to the same descriptor and one of them computes the transposed contraction. Dropping a `.T`
/// from a fixture therefore yields a kernel that builds, bakes, and passes `dxp_standalone` while
/// computing the wrong projection. Only the KTIR says which orientation was meant, so only a test at
/// this level can hold it.
///
/// ⛔ AND **PRESENTED** IS THE LOAD-BEARING WORD, NOT A HEDGE. The maps LABEL which axis of B is `k`;
/// they do not MOVE a byte. `ktir-superdsc`'s kernel slot reads exactly one residency and it is
/// `[k, n]` (`StickLayout::kernel(k_in, n_out)`), so a transpose-B `[n, k]` operand's bytes have to be
/// placed by somebody — and for a presented weight that somebody is the HOST stage
/// (`stage_2d(&StickLayout::kernel(k, n), ..)`), once, at bake time. An IN-REGISTER COMPUTED value has
/// no host stage, no cache write, and nothing in that crate reads an access tile's `CoordinateOrder`,
/// so folding a `.T` of one into the maps DROPPED it: `decoder_block.py`'s `tl.dot(q1r, k1r.T)` and
/// `tl.dot(q2r, k2r.T)` contracted as `q1r @ k1r`, invisible to every extent guard because
/// `[M, HALF]` is 64x64 there. `dot_to_linalg` therefore folds ONLY when the transposed value is a
/// direct `tt.descriptor_load`, and this test is the two-sided statement of that rule.
///
/// ⛔ THE TWO EXCEPTIONS ARE NAMED, COUNTED, **AND DISCRIMINATED BY PROVENANCE** — a bare count
/// cannot tell them apart, and the difference between them is exactly the defect:
///
///  1. `a = tl.dot(p, v)` — a GENUINE plain-B dot. `v` is that kernel's own V-projection result, a
///     `[k, n]` tile, and `[k, n]` IS the slot's device order, so it is contracted where it lies (the
///     leg the shipped attention runs against its V cache at 41 tok/s). No `tt.trans` anywhere near
///     it. One per decoder layer.
///  2. the two SCORE matmuls — plain-B over a value a SURVIVING `tt.trans` produces. The `.T` is
///     still in the IR, `to_ktir::convert_trans` turns it into a real `linalg.transpose`, and the
///     plain `[k, n]` maps then read a buffer that is physically `[k, n]`. Two per decoder layer.
///
/// So an unmapped matmul is legitimate in exactly two shapes and the old silent drop is a THIRD:
/// unmapped, with a `.T` in the `.py` and no `tt.trans` left in the IR. That third shape is what the
/// per-operand checks below refuse, and no count alone would have seen it.
#[test]
fn every_presented_dot_weight_states_the_transpose_b_maps() {
    // (configuration, matmuls with NO maps whose B is a GENUINE plain-B value -- `p @ v`, one per
    //  decoder layer, matmuls with NO maps whose B is a surviving `tt.trans` -- the two score
    //  contractions per decoder layer, whose transposition is spent by a REAL relayout)
    let expected: &[(&str, usize, usize)] = &[
        ("swiglu_mlp_small", 0, 0),
        ("swiglu_mlp_granite", 0, 0),
        ("swiglu_mlp_tiled_k", 0, 0),
        ("decoder_layer_one", 1, 2),
        ("decoder_two_layers", 2, 4),
    ];
    let mut wrong: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for (name, want_plain, want_relayout) in expected {
        let case = cases()
            .into_iter()
            .find(|c| c.name == *name)
            .unwrap_or_else(|| panic!("{name} is in the case list"));
        let k = match run(&case).result {
            Ok(k) => k,
            Err(e) => {
                wrong.push(format!("  {name}: did not lower at all: {e}"));
                continue;
            }
        };
        // How every value in the module is defined, so operand 1's PROVENANCE can be read.
        let defs: std::collections::HashMap<Ssa, OpKind> = k
            .ops_deep()
            .iter()
            .flat_map(|o| o.results.iter().map(|r| (*r, o.kind.clone())).collect::<Vec<_>>())
            .collect();
        let mms: Vec<&Op> =
            k.ops_deep().into_iter().filter(|o| o.kind == OpKind::LinalgMatmul).collect();
        assert!(!mms.is_empty(), "{name}: no `linalg.matmul` at all, so this asserts nothing");
        let (mut mapped, mut plain, mut relayout) = (0usize, 0usize, 0usize);
        for o in &mms {
            let b_def = o.operands.get(1).and_then(|v| defs.get(v));
            let is_mapped = match o.attr(&AttrKey::IndexingMaps) {
                Some(Attr::AffineMapList(m)) => {
                    m.len() == 3 && m.iter().zip(TRANSPOSED_B_MAPS.iter()).all(|(a, b)| a == b)
                }
                _ => false,
            };
            if is_mapped {
                mapped += 1;
                // ⛔⛔⛔ THE CLAUSE THAT CATCHES THE SILENT DROP. A transpose-B B that is NOT a
                // presented buffer has nobody to place its bytes, so the descriptor computes
                // `A @ B`. If the fold ever stops discriminating, this fires here rather than on
                // a device nobody can run.
                //
                // ⭐ `ktdp.load` AND NOT `tt.descriptor_load`, BECAUSE `run` HAS ALREADY RUN
                // `convert_ttir_to_ktdp`. It is the same statement one stage later: a `ktdp.load`
                // reads an access tile of a memory view of a FUNCTION ARGUMENT — exactly the shape
                // `ktir-superdsc`'s parameter-rooted `region_for_operand` answers `Some` for, and the
                // only one the host stage places.
                if b_def != Some(&OpKind::KtdpLoad) {
                    wrong.push(format!(
                        "  {name}: a `linalg.matmul` states the TRANSPOSE-B maps over a B defined \
                         by `{:?}`, not a `ktdp.load` of a parameter. Only a presented buffer's \
                         transposition is spent by the host stage; anything else is a labelled \
                         `[n, k]` read by a `[k, n]` slot, i.e. `A @ B` computed where `A @ Bᵀ` was \
                         written, and no extent guard catches it when k == n",
                        b_def
                    ));
                }
            } else if b_def == Some(&OpKind::TtTrans) {
                relayout += 1;
            } else {
                plain += 1;
                // ⛔ AND THE OTHER SIDE OF IT: a plain-B B must not be a transposed LOAD either.
                // That shape would mean the `.T` was folded into the tile and the maps dropped,
                // which is the same wrong contraction reached from the other direction.
                assert_ne!(
                    b_def,
                    Some(&OpKind::TtTrans),
                    "{name}: unreachable — a `tt.trans` B was just counted as a relayout"
                );
            }
        }
        if plain != *want_plain || relayout != *want_relayout {
            wrong.push(format!(
                "  {name}: {mapped} of {} `linalg.matmul`(s) state the transpose-B maps; of the \
                 rest, {plain} are genuine plain-B (`p @ v`) and {relayout} read a surviving \
                 `tt.trans` -- expected {want_plain} and {want_relayout}. A `.T` that is neither \
                 folded into the maps NOR still standing as a `tt.trans` has been DROPPED, and the \
                 kernel then computes `A @ B` while the fixture wrote `A @ Bᵀ`.",
                mms.len()
            ));
        }
        checked += 1;
    }
    assert!(wrong.is_empty(), "THE WEIGHT ORIENTATION HAS MOVED:\n{}", wrong.join("\n"));
    assert_eq!(checked, expected.len(), "not every configuration was reached");
}

/// THE CONTROL FOR [`is_transposed_dot_weight`], and without it that predicate is a blanket skip
/// wearing eight clauses.
///
/// Seven configurations now pass BECAUSE their findings are all named by that predicate. If it is
/// over-broad, every one of them is a place a real regression can sit unreported — which is exactly
/// the hazard the recording was asked for. So each row below is a difference a REAL defect would
/// produce on one of the same ops, and none of them may be absorbed.
///
/// ⭐ IT TESTS THE PREDICATE DIRECTLY, on constructed [`Difference`]s, rather than by planting text
/// in a golden. `golden_ktir.rs::planted_differences_are_each_caught_by_name` plants in the text
/// because it is testing the DIFF; this is testing the FILTER, and the filter's whole input is a
/// `Difference`, so constructing them states each case exactly instead of hoping a text edit
/// produces it.
#[test]
fn the_recorded_divergence_does_not_absorb_a_real_regression() {
    let d = |where_: &str, field: &'static str, expected: &str, actual: &str| Difference {
        where_: where_.to_string(),
        field,
        expected: expected.to_string(),
        actual: actual.to_string(),
    };
    let maps_ok = format!(
        "{{indexing_maps=[affine_map<{}>, affine_map<{}>, affine_map<{}>]}}",
        TRANSPOSED_B_MAPS[0], TRANSPOSED_B_MAPS[1], TRANSPOSED_B_MAPS[2]
    );

    // (reason it must NOT be absorbed, the difference)
    let must_fail: Vec<(&str, Difference)> = vec![
        (
            "a WRONG weight map -- (d2, d1) is the UNtransposed weight, so this is the transposed \
             contraction computed the other way round: a silently wrong answer, not our recorded \
             spelling",
            d(
                "@f/linalg.matmul[7]",
                "attributes",
                "{}",
                "{indexing_maps=[affine_map<(d0, d1, d2) -> (d0, d2)>, \
                 affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}",
            ),
        ),
        (
            "the maps present on BOTH sides but different -- the golden side is not empty, so this \
             is not `we add maps where the C++ adds none`",
            d("@f/linalg.matmul[7]", "attributes", &maps_ok, "{indexing_maps=[]}"),
        ),
        (
            "an OPERAND REWIRED on the matmul -- the second matmul reading the score tile instead \
             of the probabilities, `golden_ktir.rs`'s own planted case",
            d("@f/linalg.matmul[7]", "operands", "[\"v1\", \"v2\"]", "[\"v1\", \"v3\"]"),
        ),
        (
            "the matmul REPLACED by another op",
            d("@f/linalg.matmul[7]", "op kind", "linalg.matmul", "linalg.reduce"),
        ),
        (
            "an op ADDED or REMOVED",
            d("@f/region0", "op count", "40 non-constant ops", "39 non-constant ops"),
        ),
        (
            "the matmul's RESULT TYPE changed -- f16 to f32 is what LegalizeTypes exists to stop, \
             and `result types` is a recorded field for the tile/view/load but NEVER for the matmul",
            d(
                "@f/linalg.matmul[7]",
                "result types",
                "[\"tensor<64x64xf16>\"]",
                "[\"tensor<64x64xf32>\"]",
            ),
        ),
        (
            "the CORELET PLAN's pattern -- a different work division entirely, and the gap that \
             `recover_matmul_shape` silently opened once already",
            d(
                "@f/ktdf.corelet_plan[0]",
                "attributes",
                "{pattern=\"independent_subtile\"}",
                "{pattern=\"split\"}",
            ),
        ),
        (
            "a REDUCTION AXIS changed -- axis 1 is what makes a body independent_rows; axis 0 mixes \
             rows",
            d("@f/tt.reduce[9]", "attributes", "{axis=1}", "{axis=0}"),
        ),
        (
            "a CONSTANT changed -- the qk_scale",
            d("@f/<constants>", "constant set", "1.275630e-01", "1.375630e-01"),
        ),
        (
            "a STORE's operands -- the output written from the wrong value",
            d("@f/ktdp.store[40]", "operands", "[\"v9\", \"v8\"]", "[\"v7\", \"v8\"]"),
        ),
    ];

    let mut absorbed: Vec<String> = Vec::new();
    for (why, diff) in &must_fail {
        if is_transposed_dot_weight(diff) {
            absorbed.push(format!("  ABSORBED: {why}\n    {diff}"));
        }
    }
    assert!(
        absorbed.is_empty(),
        "`is_transposed_dot_weight` is OVER-BROAD -- it accounts for {} difference(s) that a real \
         defect would produce, so every configuration recorded against it is a place a regression \
         can hide:\n{}",
        absorbed.len(),
        absorbed.join("\n")
    );

    // AND THE NEGATIVE HALF: it must still recognise the real thing, or the seven recorded
    // configurations would fail for the wrong reason and somebody would widen the predicate.
    let must_pass: Vec<Difference> = vec![
        d("@f/linalg.matmul[7]", "attributes", "{}", &maps_ok),
        d(
            "@f/scf.for[4]/ktdp.construct_memory_view[5]",
            "result types",
            "[\"memref<128x256xf16>\"]",
            "[\"memref<256x128xf16>\"]",
        ),
        d(
            "@f/ktdp.construct_access_tile[4]",
            "attributes",
            "{access_tile_order=affine_map<(d0, d1) -> (d1, d0)>}",
            "{access_tile_order=affine_map<(d0, d1) -> (d0, d1)>}",
        ),
        d(
            "@f/ktdp.construct_access_tile[4]",
            "operands",
            "[\"v13\", \"const(0:index)\", \"v25\"]",
            "[\"v13\", \"v25\", \"const(0:index)\"]",
        ),
        d("@f/ktdp.load[5]", "result types", "[\"tensor<128x64xf16>\"]", "[\"tensor<64x128xf16>\"]"),
        d("@f/scf.for[0]/arith.index_cast[4]", "operands", "[\"v22r0a0\"]", "[\"v21r0a0\"]"),
    ];
    let missed: Vec<String> = must_pass
        .iter()
        .filter(|x| !is_transposed_dot_weight(x))
        .map(|x| format!("  {x}"))
        .collect();
    assert!(
        missed.is_empty(),
        "`is_transposed_dot_weight` no longer recognises {} of the differences it was MEASURED \
         against, so the seven recorded configurations are failing for a reason this predicate \
         cannot name:\n{}",
        missed.len(),
        missed.join("\n")
    );
}

/// THE CONTROL FOR [`fold_dead_dot_transposes`] — it must fold ONLY the transpose it was written
/// for, because a normaliser that erases more than its cause is a way of making the diff agree.
///
/// Three negative cases and one positive, all built by hand so the shapes are exact.
#[test]
fn the_transpose_fold_only_folds_a_dead_dot_weight() {
    let ssa = Ssa;
    let ty = || IrType::Tensor { dims: vec![64, 64], elem: DType::F16 };
    // `tt.trans %1 -> %2`, then a consumer.
    let build = |consumer: Op| -> Module {
        let trans = Op::new(OpKind::TtTrans)
            .with_result(ssa(2), ty())
            .with_operands(vec![ssa(1)]);
        let mut m = Module::new();
        m.ops = vec![trans, consumer];
        m
    };
    let n_trans = |m: &Module| {
        m.ops_deep().iter().filter(|o| o.kind == OpKind::TtTrans).count()
    };

    // POSITIVE: the trans feeds operand 1 of a matmul and nothing else.
    let mut m = build(
        Op::new(OpKind::LinalgMatmul)
            .with_result(ssa(3), ty())
            .with_operands(vec![ssa(0), ssa(2)]),
    );
    fold_dead_dot_transposes(&mut m);
    assert_eq!(n_trans(&m), 0, "a dead dot-weight transpose must be folded");
    assert_eq!(
        m.ops[0].operands,
        vec![ssa(0), ssa(1)],
        "the matmul must be rewired to the value the transpose read"
    );

    // NEGATIVE 1: the trans feeds operand 0 — a transposed ACTIVATION, which is a different
    // contraction and which `dot_to_linalg` does not fold.
    let mut m = build(
        Op::new(OpKind::LinalgMatmul)
            .with_result(ssa(3), ty())
            .with_operands(vec![ssa(2), ssa(0)]),
    );
    fold_dead_dot_transposes(&mut m);
    assert_eq!(n_trans(&m), 1, "a transposed ACTIVATION must not be folded away");

    // NEGATIVE 2: the trans is read TWICE, so erasing it would drop a value something still needs.
    let mut m = build(
        Op::new(OpKind::LinalgMatmul)
            .with_result(ssa(3), ty())
            .with_operands(vec![ssa(2), ssa(2)]),
    );
    fold_dead_dot_transposes(&mut m);
    assert_eq!(n_trans(&m), 1, "a transpose with two uses must not be folded away");

    // NEGATIVE 3: the consumer is not a matmul at all.
    let mut m = build(
        Op::new(OpKind::ArithMulf)
            .with_result(ssa(3), ty())
            .with_operands(vec![ssa(0), ssa(2)]),
    );
    fold_dead_dot_transposes(&mut m);
    assert_eq!(n_trans(&m), 1, "a transpose feeding a pointwise op must not be folded away");
}

/// A small module the controls mutate: `rmsnorm_granite`, which has a `tt.reduce` region, a
/// generated helper the inliner had to remove, and an `arith.constant` set.
fn control() -> (Module, Module) {
    let case = cases()
        .into_iter()
        // rmsnorm_granite, because the controls need BOTH an op carrying an `axis` (its
        // `tt.reduce` and `tt.expand_dims`) and a leading `arith.constant` run. SwiGLU has
        // the constants but no `axis` at all, and a control that cannot find its victim
        // asserts nothing.
        .find(|c| c.name == "rmsnorm_granite")
        .expect("rmsnorm_granite is in the case list");
    let g = golden(case.name).expect("its golden");
    let mut golden_m = text::parse::parse(&g).expect("the golden parses");
    let mut ours = run(&case).result.expect("rmsnorm_granite lowers");
    canonicalize_leading_constants(&mut golden_m);
    canonicalize_leading_constants(&mut ours);
    (golden_m, ours)
}

/// THE NARROWNESS CONTROL for [`canonicalize_leading_constants`]: the normalization removes
/// a permutation and NOTHING else, so a changed constant VALUE inside the leading run must
/// still be caught.
#[test]
fn a_planted_constant_value_change_is_caught() {
    let (g, mut ours) = control();
    let hit = mutate_first(&mut ours, &mut |o| {
        if o.kind == OpKind::ArithConstant && o.attr(&AttrKey::Value).is_some() {
            o.set_attr(AttrKey::Value, Attr::Int(999_777));
            return true;
        }
        false
    });
    assert!(hit, "the control has an arith.constant");
    let mut mine = ours.clone();
    let mut golden_m = g.clone();
    canonicalize_leading_constants(&mut golden_m);
    canonicalize_leading_constants(&mut mine);
    let findings = text::diff::diff(&golden_m, &mine);
    assert!(
        !findings.is_empty(),
        "changing a leading constant's VALUE was not caught -- then the multiset \
         normalization is hiding content, not a permutation"
    );
}

#[test]
fn the_control_agrees_with_itself() {
    let (g, ours) = control();
    let findings = text::diff::diff(&g, &ours);
    assert!(
        findings.is_empty(),
        "the control's baseline must be clean or every planted difference proves nothing: \
         {findings:?}"
    );
}


/// Apply `f` to the FIRST op anywhere in the module (regions included) that `f` accepts,
/// returning whether one was found.
///
/// The controls need a deep walk, not a top-level one: after `DistributeWork` the body sits
/// inside the grid `scf.for`, so `rmsnorm_granite`'s `tt.reduce` -- the op carrying the
/// `axis` a control wants to break -- is nested. A control that cannot find its victim
/// asserts nothing, which is the failure this helper exists to remove.
fn mutate_first(m: &mut Module, f: &mut impl FnMut(&mut triton_ktir::ir::Op) -> bool) -> bool {
    fn walk(
        ops: &mut Vec<triton_ktir::ir::Op>,
        f: &mut impl FnMut(&mut triton_ktir::ir::Op) -> bool,
    ) -> bool {
        for o in ops.iter_mut() {
            if f(o) {
                return true;
            }
            for r in &mut o.regions {
                if walk(&mut r.ops, f) {
                    return true;
                }
            }
        }
        false
    }
    walk(&mut m.ops, f)
}

#[test]
fn a_planted_op_deletion_is_caught() {
    let (g, mut ours) = control();
    let f = ours.kernel_mut().expect("a kernel");
    let n = f.regions[0].ops.len();
    f.regions[0].ops.remove(n / 2);
    let findings = text::diff::diff(&g, &ours);
    assert!(
        !findings.is_empty(),
        "deleting an op was not caught -- the diff is not comparing the op sequence"
    );
}

#[test]
fn a_planted_attribute_change_is_caught() {
    let (g, mut ours) = control();
    let hit = mutate_first(&mut ours, &mut |o| {
        if o.attr(&AttrKey::Axis).is_some() {
            o.set_attr(AttrKey::Axis, Attr::Int(7));
            return true;
        }
        false
    });
    assert!(hit, "rmsnorm_granite has an op carrying an `axis`");
    let findings = text::diff::diff(&g, &ours);
    assert!(
        !findings.is_empty(),
        "changing an `axis` was not caught -- the diff is not comparing attributes"
    );
    assert!(
        findings.iter().any(|d| d.field.contains("attr")),
        "the finding must NAME the attribute field; got {:?}",
        findings.iter().map(|d| d.field).collect::<Vec<_>>()
    );
}

#[test]
fn a_planted_operand_rewire_is_caught() {
    let (g, mut ours) = control();
    // The operands must DIFFER, or the swap is the identity and the control proves nothing;
    // and the op must not be a leading constant-like one, whose position the normalization
    // deliberately erases.
    let hit = mutate_first(&mut ours, &mut |o| {
        if o.operands.len() == 2
            && o.operands[0] != o.operands[1]
            && !matches!(o.kind, OpKind::ArithConstant | OpKind::TensorSplat)
        {
            o.operands.swap(0, 1);
            return true;
        }
        false
    });
    assert!(hit, "an op with two distinct operands");
    let findings = text::diff::diff(&g, &ours);
    assert!(
        !findings.is_empty(),
        "swapping two distinct operands was not caught -- the diff is not comparing operand \
         linkage"
    );
}

#[test]
fn a_planted_census_change_is_caught() {
    let (g, mut ours) = control();
    let f = ours.kernel_mut().expect("a kernel");
    let extra = f.regions[0].ops[0].clone();
    f.regions[0].ops.push(extra);
    let findings = text::diff::census_diff(&g, &ours);
    assert!(
        !findings.is_empty(),
        "adding a duplicate op was not caught by the census -- a count mismatch is the \
         check that survives a positional divergence"
    );
}

//===----------------------------------------------------------------------===//
// The adapter refuses rather than falling through
//===----------------------------------------------------------------------===//

/// A minimal ttir module the refusal controls mutate, built through the front end so it is
/// the real shape rather than a hand-made one.
fn a_ttir_module() -> ttir::Module {
    let case = cases()
        .into_iter()
        .find(|c| c.name == "rmsnorm_granite")
        .expect("rmsnorm_granite");
    let src = fixture(case.fixture);
    let mut m = codegen::compile(&src, &case.spec, case.target).expect("compiles");
    opt::make_ttir(&mut m).expect("make_ttir");
    m
}

#[test]
fn an_unmodelled_op_is_refused_by_name() {
    let mut m = a_ttir_module();
    // `math.cos` has no `OpKind` variant and no SuperDSC mapping. `OpKind::from_spelling`
    // would turn it into an `Other` whose fields nobody checked, which is precisely the
    // silent-guess failure this allow-list exists to prevent.
    m.funcs[0].body.blocks[0].ops[0].name = "math.cos".to_string();
    let e = from_ttir::convert(&m).expect_err("must refuse");
    assert!(
        e.message.contains("math.cos"),
        "the refusal must NAME the op; got: {e}"
    );
}

#[test]
fn an_unmodelled_attribute_is_refused_by_name() {
    let mut m = a_ttir_module();
    m.funcs[0].body.blocks[0].ops[0]
        .attrs
        .insert("spyre.not_a_real_attr".to_string(), ttir::Attr::Unit);
    let e = from_ttir::convert(&m).expect_err("must refuse");
    assert!(
        e.message.contains("spyre.not_a_real_attr"),
        "the refusal must NAME the attribute; got: {e}"
    );
}

#[test]
fn a_second_function_is_refused_by_name() {
    let mut m = a_ttir_module();
    let dup = m.funcs[0].clone();
    m.funcs.push(dup);
    let e = from_ttir::convert(&m).expect_err("must refuse");
    assert!(
        e.message.contains("2 functions"),
        "the refusal must say how many functions it found; got: {e}"
    );
}

#[test]
fn a_multi_block_body_is_refused_rather_than_truncated() {
    let mut m = a_ttir_module();
    m.funcs[0].body.blocks.push(ttir::Block::default());
    let e = from_ttir::convert(&m).expect_err("must refuse");
    assert!(
        e.message.contains("2 blocks"),
        "the refusal must say how many blocks it found, because truncating silently would \
         delete code; got: {e}"
    );
}

/// The SSA arena must cross unchanged: `ValueId(n)` is `Ssa(n)`, and `next_ssa` starts past
/// every value bridge one made so a pass cannot mint a colliding name.
#[test]
fn the_ssa_arena_crosses_intact() {
    let m = a_ttir_module();
    let k = from_ttir::convert(&m).expect("converts");
    assert_eq!(
        k.next_ssa as usize,
        m.values.len(),
        "`next_ssa` must start past bridge one's last value, or `Module::fresh()` hands a \
         pass a name that is already in use"
    );
    assert!(
        !k.hints.is_empty(),
        "no SSA hints crossed -- the printed KTIR would be unreadable and the bundle's \
         argument names would degrade to indices"
    );
    // Every operand of every op must name a value the arena has.
    for op in k.ops_deep() {
        for v in &op.operands {
            assert!(
                (v.0 as usize) < m.values.len(),
                "`{}` reads %{} which is outside the arena bridge one built",
                op.kind.spelling(),
                v.0
            );
        }
    }
}

/// THE GATHER, LOWERED LIKE THE C++ -- the inverted form of the test that recorded it as a gap.
///
/// `ConvertTTIRToKTDP` had no `tt.descriptor_gather` arm, so the op survived to the output
/// where the C++ replaced it. It is ported now (`ConvertDescriptorGather`,
/// `ConvertTTIRToKTDP.cpp:989`, over `buildGatherSubscriptMaps` at `:581`), and
/// `embedding_granite_bm128_control` matches the C++ KTIR field by field like the other ten.
///
/// THE ASSERTIONS ARE KEPT AND INVERTED rather than deleted, because their census signature is
/// what says the arm produces the RIGHT shape and not merely some shape: no surviving
/// `tt.descriptor_gather`, and the same indirect-tile and `ktdp.load` counts as the golden.
///
/// TWO THINGS THE DIFF CAUGHT while porting it, both recorded where the code is: the region
/// carries one `index` block argument per captured variable -- emitting it EMPTY was the first
/// attempt, on the assumption that `text::parse` skips the `^bb0` line, and it does not -- and
/// the affine maps and set have to be spelled MLIR's way, `-d0 + 127` rather than `127 - d0`.
#[test]
fn the_gather_is_lowered_like_the_cpp() {
    let case = cases()
        .into_iter()
        .find(|c| c.name == "embedding_granite_bm128_control")
        .expect("the embedding control is in the case list");
    let ours = run(&case).result.expect("it lowers");
    let g = golden(case.name).expect("its golden");
    let golden_m = text::parse::parse(&g).expect("the golden parses");

    let count = |m: &Module, spelling: &str| {
        m.ops_deep().iter().filter(|o| o.kind.spelling() == spelling).count()
    };
    assert_eq!(
        count(&ours, "tt.descriptor_gather"),
        0,
        "a `tt.descriptor_gather` survived the pass -- the arm did not fire"
    );
    for spelling in ["ktdp.construct_indirect_access_tile", "ktdp.load"] {
        assert_eq!(
            count(&ours, spelling),
            count(&golden_m, spelling),
            "`{spelling}` count differs from the C++'s -- the arm fired but produced the \
             wrong shape, which is what this census signature exists to distinguish from \
             producing none"
        );
    }
    // The four attributes the indirect tile carries, by name, so a missing one is reported
    // here rather than as an opaque attribute difference.
    let tile = ours
        .ops_deep()
        .into_iter()
        .find(|o| o.kind == triton_ktir::ir::OpKind::KtdpConstructIndirectAccessTile)
        .expect("the indirect access tile");
    for key in [
        "per_dim_subscript_kinds",
        "per_dim_subscript_maps",
        "variables_space_order",
        "variables_space_set",
    ] {
        assert!(
            tile.attr(&AttrKey::Other(key.to_string())).is_some(),
            "the indirect access tile is missing `{key}`"
        );
    }
    assert_eq!(tile.regions.len(), 1, "it declares exactly one region");
    assert_eq!(
        tile.regions[0].args.len(),
        2,
        "one `index` block argument per captured variable: the single index anchor plus c_y"
    );
}
