// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ WHAT THE GATHERED DECODE BUNDLE **DECLARES**, OP BY OP, AT THREE WIDTHS — the emitter half
//! of the `job_bin_ptr + numCoresUsed_*128` launch refusal, obtained on a Mac in under a second, and now
//! the gate on the form that replaced the refused one.
//!
//! ## The card refusal this file localized
//! ```text
//! superdsc decode batch run_step (mq=8, start=97): compute launch rc=-1
//! CB status=Error locator=0x2  QGI addr=0x1c00000400 (939524104 flits)
//!     syndrome=0xc00 cases=[PrepZeroFlitCnt,PrepSwVer]
//! [segaddr] op[0] job_bin_ptr=0x1c00000000  PROG_OFFSET_BASE=0x1c00000000
//! ```
//! The offset is `mq * 128` — `mq` FLITS — measured at two widths (mq=8 → 0x400, mq=2 → 0x100), and
//! `0xc00` is syndrome bits 10|11, [`PrepZeroFlitCnt`] + [`PrepSwVer`]: the Prep unit read something at
//! that address that is neither a valid job header nor a non-zero flit count.
//!
//! ## ⭐⭐⭐⭐⭐ MEASURED ON `f8390f3bc`: THE FAULT IS IN THE **PROGRAM** SEGMENT, AND `bootstrap == 0`
//! The `[segaddr]` line above describes op[0] of whichever list launched FIRST — a PREFILL group, not the
//! op that faults — so it settled nothing about the fault. Re-measured per FAULTING op (the dump now runs
//! on every `rc != 0`, no knob), both rungs, `scr batch` on granite-3.1-2b fp8:
//! ```text
//! mq=2  FAULT op[2]  05f19f853f28683b/group_2   seg7/PROG region=128 offset=101508480 size=6016   (binary 6016 B)
//!       job_bin_ptr=0x1c00000000 bootstrap=0x0  prog_extent=6016 B = 47 flit(s)     QGI 0x1c00000100 = flit 2
//! mq=8  FAULT op[19] 6701a74bf9435e88/group_19  seg7/PROG region=128 offset=102826368 size=265728 (binary 265728 B)
//!       job_bin_ptr=0x1c00000000 bootstrap=0x0  prog_extent=265728 B = 2076 flit(s) QGI 0x1c00000400 = flit 8
//! ```
//! A QGI address is a DMVA and a DMVA's top bits ARE the segment (`SEGMENT_SIZE_BITS = 34`,
//! `PROG_SEGMENT = 7`), so `0x1c0000_0100 >> 34 == 7`: the PROGRAM segment. And no data segment can alias
//! it — at the faulting launch seg0..seg6 resolve to regions `17179869312`/`34359738496` at multi-GB
//! offsets while seg7 is region `128`. ⇒ **The "operand base resolved against the wrong segment" reading
//! is DEAD**, and with it the mask/index pitch (seg3's shift is 0 at the faulting launch), the two
//! synthetic scratch placements (seg0/4/5/6 shift 0) and the zeroed KV shift.
//!
//! ⛔ AND SO IS "ONE PAST THE PER-CORE PATCH TABLE". `bootstrap` is ZERO, so Prep started at flit 0 of the
//! group's own binary and walked `mq` flits IN before failing — flit 2 of 47 and flit 8 of 2076 are both
//! deep INSIDE the binary, not past anything. What survives: the job-header chain the Prep unit walks from
//! flit 0 is wrong at flit `mq` (it read a ZERO flit count there and a SW version that does not validate),
//! in a group whose op count is what `mq` scales.
//!
//! ## ⛔⛔⛔ THE BAKED JOB-HEADER CHAIN IS **NOT** THE DEFECT — 28 PROGRAMS PARSED, ALL WELL-FORMED
//! The header format is `deeptools/senulator/qg.h`'s `struct QGHeader` and the WRITER is
//! `deeptools/dip/dip.cpp:77-91`, which is the authority for the convention: one 128-B flit, `u8[0] =
//! SW_VER = 0xdd`, flit count in bits 21:8, terminal in bit 22, and — decisively — `myflits =
//! totalFlits_ - 1`, so a header's count EXCLUDES its own flit. Walking both faulting bundles' images
//! byte by byte under exactly that rule:
//! ```text
//! 05f19f853f28683b (mq=2)  g0 78 jobs  g1  90  g2  1  g3 2  g4 2  g5 8  g6 8  g7 51
//! 6701a74bf9435e88 (mq=8)  g0 78 jobs  g1 282  g2  1  g3..g10 2 each  g11..g18 8 each  g19 51
//! ```
//! EVERY header in all 28 programs carries `0xdd`, every declared count is exact, exactly the last job
//! of each program is terminal, and every chain ends at `1 + flits` == the file's own flit count with
//! nothing left over. ⇒ **The emission does not mis-declare a job count anywhere**, so "a header at flit
//! `mq` whose flit-count field is zero" is not something the bake wrote.
//!
//! What IS at flit `mq` is nothing at all: bytes 0..2 of flit 2 (mq=2) and flit 8 (mq=8) are `00 00 00`,
//! i.e. SW version `0x00` (≠ `0xdd`) and flit count 0 — **the `0xc00` syndrome verbatim**
//! (`prep_sw_ver` = bit 11, `prep_zero_flit_cnt` = bit 10, `app_data_sbf.hpp:360-361`). And
//! `qgi_address_flits` is `bootstrap.value() + mq` exactly (`0x38000000 + mq`, and `0x38000000` IS
//! `VirtualAddressSbf::new(7, 0).value()`), so the device ADVANCED `mq` flits from a correct bootstrap.
//! From flit 0 the baked header says the next job begins at flit 47 (mq=2) / 41 (mq=8) — never at `mq`.
//! ⇒ **Whatever told Prep the first job was `mq` flits long, it was not the header we baked.** The
//! remaining class is the DEVICE's copy of those bytes, not the bytes.
//!
//! ## ⛔ THE "44× SIZE CURVE" IS AN ARTEFACT OF COMPARING TWO DIFFERENT OPS
//! `6,016 B at mq=2 → 265,728 B at mq=8` is `group_2` measured against `group_19`, and those are not the
//! same op: `group_2` is a 1-job/47-flit program and `group_19` a 51-job/2076-flit one. `group_2` is
//! **6,016 B at BOTH rungs**, and the same-role 51-job group goes `g7` 1911 flits → `g19` 2076 flits —
//! **1.086×**, not 44×. Nothing scales super-linearly and there is no header-chain overrun to explain.
//!
//! What DOES scale is the GROUP COUNT: `4 + 2*mq` groups (8 at mq=2, 20 at mq=8), because the collapse
//! emits each per-request op as its OWN GROUP — `mq` copies of the 2-job group and `mq` of the 8-job
//! group — rather than `mq` ops inside one group. Only `group_1` grows as ops-in-a-group at all, by
//! exactly `32` jobs per request (`26 + 32*mq`: 90 at mq=2, 282 at mq=8, both under
//! `SCRATCHY_SUPERDSC_GROUP_SIZE=512`). So a request costs a PROGRAM and a LAUNCH, not a job.
//!
//! ## ⛔⛔⛔ A SEPARATE, PROVEN GAP THE BYTES EXPOSED: `ComputeOnHost`/`DataTransfer` IS UNIMPLEMENTED
//! A gather program's dxp job plan is THREE exec steps, not one. Measured over every retained
//! `spyrecode.json` on the pod — 24 of 85 carry a `ComputeOnHost`, and they are the gather probes, all
//! with the identical layout:
//! ```text
//! JobExecPlan:        ComputeOnHost  size=4224 (33 flits)  ohandle=progCorr  dsName_=ProgCorrectionFlit
//!                     DataTransfer   size=4224  dev_ptr=PROG_OFFSET_BASE + 1*128    (flits 1..33)
//!                     ComputeOnDevice           job_bin_ptr=PROG_OFFSET_BASE + 34*128
//! JobPreparationPlan: Allocate       size=133504 (1043 flits)
//!                     InitTransfer   size=125568 (981 flits)  dev_ptr=BASE + 34*128
//! ```
//! So the allocation is `[flit 0 gap][flits 1..33 correction][flits 34.. program]`, and execution starts
//! at flit 34. [`scratchy_spyre_bundle::correction`] types `DataTransfer` as
//! `JobCommand::Other` — "any step this port does not act on" — so the correction's DESTINATION is
//! discarded at parse time; `GroupCode::correction` then reaches the binary as const bytes with **no
//! runtime consumer anywhere** (`superdsc_exec.rs` never names it), and `Allocate`/`InitTransfer` are
//! never parsed at all, so the launch allocates `init_binary.len()` and H2Ds it at offset 0 regardless.
//! A correction region left ZERO is precisely a flit that reads back as SW version `0x00` and flit count
//! 0. ⛔ This is NOT the measured fault — these plans put the bootstrap at flit 34 and the faulting op
//! measured `bootstrap == 0` — but it is a real unimplemented step that fires the moment a shipped
//! group's dxp output carries a `ComputeOnHost`, and it fails with exactly this syndrome.
//!
//! ## ⛔ WHY A DIFF AND NOT A READING
//! `zz_diff_the_rung_descriptors` records what a hand-built score leg cost once: it measured
//! `MatY::of_requests` while the shipped bundle carried `MatY::of_gqa_group`, so the harness described a
//! different op than the card ran. Everything here calls `assemble_attn` itself, twice — with and
//! without the gather index — and reports only what DIFFERS. A shape present in both is, by
//! construction, a shape the card runs today.
//!
//! ## ⛔⛔⛔ THE FORM THAT FAULTED, AND HOW THIS FILE CONVICTED IT
//! The first collapsed fold carried the REQUESTS on `y`, which needs a per-batch 3-D `[y,in,out]` KERNEL.
//! Projected here, that op declared `y_=mq`, `mb_=1`, `numWkSlicesPerDim_ {y:mq, mb:1}` and
//! `numCoresUsed_ == mq` (`mb` is 1 and `out` is 64, which `work::matmul_cost_split`'s stick clause
//! forbids splitting, so `y` was the only splittable dim) — and its kernel showed **`mq` DISTINCT
//! per-core starts** where every shipped kernel shows ONE.
//!
//! So `job_bin_ptr + mq*128` was read as `job_bin_ptr + numCoresUsed_*128`: one flit per CORE, at index
//! `cores`, one past the last one the program holds (`init_binary.bin = 1920 + 128*cores`; see
//! `one-block-per-core-is-an-11x-oversized-init` — dxp's 1-op/32-core group is 1 header + 8 base + 31
//! patches). ⛔ THAT WHOLE PARAGRAPH IS REFUTED by the per-fault `[segaddr]` above — `bootstrap == 0` and
//! the fault flit is inside the binary — and `numCoresUsed_` was refuted separately on card. It is kept
//! because the *declarations* it records are still what this file gates. **RUNG 4 closed it**: at mq=4 that op is `{mb:1, y:4}` on FOUR cores, which is
//! `matmul/dims.rs`'s on-hardware-proven solo-decode split, and it faulted at `+0x200` exactly like 2 at
//! `+0x100` and 8 at `+0x400` — syndrome `0xc00`, locator `0x2`, cases `[PrepZeroFlitCnt,PrepSwVer]`
//! bit-identical, only the index moving. ⇒ The `y`-split VALUE is exonerated and the KERNEL RANK is what
//! the address is made of.
//!
//! ## ⭐⭐⭐⭐⭐ WHAT THE FIX DECLARES INSTEAD, AND IT IS THE SHIPPED OP AT ONE ROW
//! The gather is what makes the per-`y` kernel DIM unnecessary: the scratch destination is contiguous and
//! request-MINOR, so request `r`'s block is reached by a baked OFFSET
//! (`GatherScratch::kernel_row_off`) exactly as a GQA group's shared kv head is reached by `kt_off_fn`'s.
//! So the collapsed fold emits `mq` copies of the SHIPPED op, one per request, in ONE pass:
//!
//! | | shipped (`gather=false`) | gathered (`gather=true`) |
//! |---|---|---|
//! | prefix score/value op | `attn_pNsc_g{0..7}` — one per **kv head** | `attn_pNsc_g{0..7}r{0..mq-1}` — one per **(kv head, request)** |
//! | `N_` | `y=4` (the GQA group), `mb=mq` | `y=4`, **`mb=1`** |
//! | `numWkSlicesPerDim_` | `{y:4, mb:mq}` | `{y:4, mb:1}` |
//! | `numCoresUsed_` | 32 at mq=8, 8 at mq=2 | **4 at every rung** |
//! | kernel operand | `[in,out]`, **1** distinct per-core start | `[in,out]`, **1** distinct per-core start |
//! | every operand's `layoutDimOrder_`/`maxDimSizes_` | — | **IDENTICAL to the shipped op's** |
//! | fused epilogue | YES — 4 operands, actively `y`/`mb`-addressed | yes — 4 operands |
//!
//! ⭐ [`the_collapsed_op_is_the_shipped_op_at_one_row`] is the sharp version: the ONLY declared
//! quantities that differ from the shipped fold leg are the swept `mb_`, the `mb` split it forbids, and
//! the core count that follows. At `mq == 1` the two are the same op — the one running at 41 tok/s.
//!
//! ⭐ AND THE OP COUNT IS THE TRADE TO WANT, not a regression: `nkvh*mq` ops in ONE pass against `nkvh`
//! ops in each of `mq` passes — the same op count over the step, `mq`× fewer launches — and each op now
//! computes ONE row where the shipped one computes `mq` and masks `mq-1` away. Measured: 1.42 µs per op
//! against ~28 µs per fold pass.
//!
//! ⛔ AND THIS IS NOT THE PAIRING `attn.rs` FORBIDS. Its note says *"DO NOT PAIR `(gqa, request)` ONTO
//! ONE `y`"* — GARBAGE on the 8b at hd=128, 10/10 runs. That is a PACKED PAIR on a single axis
//! (`y = gqa*mq`) whose two components share one differenced step. Here `y` carries the group ALONE,
//! exactly as it ships, and the request is not on an axis at all.
//!
//! ⛔ WHAT IS STILL UNMEASURED: a LAUNCH of this form, and hd=128. Every verdict in this file is the
//! emitter's own declaration read locally. The card gate is `scr batch` against a bs=1 solo oracle per
//! row (the model's own answers, never the textbook ones — an expected-answer detector reports ~9 phantom
//! failures at bs=1).

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_attn;
use ktir_superdsc::sdsc_abstract::{AttnGeometry, PagedKvPool, attn_bundle_rows};

/// granite-3.1-2b — the model every card number is from.
const NQH: u32 = 32;
const NKVH: u32 = 8;
const HD: u32 = 64;
const CAP: u32 = PagedKvPool::PAGE_SLOTS as u32;
const ACTIVE_CAP: u32 = PagedKvPool::PAGE_SLOTS as u32;

/// One op's whole declaration, projected.
#[derive(Debug, Clone)]
struct OpPicture {
    name: String,
    /// `N_` extents the op uses, as sorted `(dim, size)`.
    iter: Vec<(String, i64)>,
    /// `numWkSlicesPerDim_`, sorted — how many slices each dim is cut into.
    split: Vec<(String, i64)>,
    /// The op-level `numCoresUsed_`.
    cores: i64,
    /// Per-operand `(layoutDimOrder_, maxDimSizes_, start-map entries, DISTINCT starts)`.
    operands: Vec<(Vec<String>, Vec<i64>, usize, usize)>,
    /// `(label, factor)` of the fold attributes on operand 0 (core / corelet / time).
    fold: Vec<(String, i64)>,
}

impl OpPicture {
    fn y(&self) -> i64 {
        self.iter
            .iter()
            .find(|(k, _)| k == "y_")
            .map_or(-1, |(_, v)| *v)
    }
    /// The op's name with the per-head / per-pass suffixes replaced by `*`, so the 32 clones of one
    /// shape collapse to a single row. Without this the report is 300 identical lines.
    fn stem(&self) -> String {
        self.name
            .split('_')
            .map(|s| {
                let numbered = |p: char| {
                    s.len() > 1 && s.starts_with(p) && s[1..].chars().all(|c| c.is_ascii_digit())
                };
                if numbered('q') {
                    "q*".to_string()
                } else if numbered('p') {
                    "p*".to_string()
                } else if numbered('g') {
                    "g*".to_string()
                } else if numbered('r') {
                    // The collapsed fold's REQUEST index — one op per (kv head, request), so without
                    // this the report is `nkvh * mq` identical lines per pass instead of one.
                    "r*".to_string()
                } else {
                    s.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("_")
    }
    /// Everything but the name — the shape identity two bundles are compared on.
    fn shape(&self) -> String {
        let j = |v: &[(String, i64)], sep: &str| {
            v.iter()
                .map(|(k, n)| format!("{k}{sep}{n}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let ops = self
            .operands
            .iter()
            .map(|(l, d, tot, dis)| format!("[{}]{d:?} st{tot}({dis})", l.join(",")))
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "N_{{{}}} wk{{{}}} cores={} fold{{{}}} | {ops}",
            j(&self.iter, "="),
            j(&self.split, "/"),
            self.cores,
            j(&self.fold, "x"),
        )
    }
}

/// Emit the whole attention body at one decode width, with or without a gather index, and project every
/// op. `gather=false` is byte-for-byte what ships.
fn emit_at(mq: u32, gather: bool) -> Vec<OpPicture> {
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let bundle_rows =
        attn_bundle_rows(geom, mq, true).unwrap_or_else(|| panic!("mq={mq} is not a baked rung"));
    let mut sym = 0i64;
    let ops = assemble_attn(
        0,
        geom,
        bundle_rows,
        CAP,
        ACTIVE_CAP,
        "t_qs",
        "t_new_k",
        "t_new_v",
        "t_kct",
        "t_vc",
        "t_pmask",
        "t_cmask",
        gather.then_some("t_kv_idx"),
        ktir_superdsc::place::PlaceId::Act(900),
        true,
        &mut sym,
        None,
    )
    .unwrap_or_else(|e| panic!("mq={mq} gather={gather}: assemble_attn refused: {}", e.0));

    ops.iter()
        .map(|e| {
            let v = serde_json::to_value(&e.op).unwrap();
            let (name, body) = v["dscs_"][0]
                .as_object()
                .and_then(|m| m.iter().next())
                .map(|(k, b)| (k.clone(), b.clone()))
                .expect("one named dsc per emitted op");
            let sorted_map = |x: &serde_json::Value| {
                let mut m: Vec<(String, i64)> = x
                    .as_object()
                    .map(|m| {
                        m.iter()
                            .filter_map(|(k, n)| n.as_i64().map(|i| (k.clone(), i)))
                            .collect()
                    })
                    .unwrap_or_default();
                m.sort();
                m
            };
            let operands = body["scheduleTree_"]
                .as_array()
                .map(|nodes| {
                    nodes
                        .iter()
                        .map(|n| {
                            let layout = n["layoutDimOrder_"]
                                .as_array()
                                .map(|a| {
                                    a.iter()
                                        .map(|d| d.as_str().unwrap_or("?").to_string())
                                        .collect()
                                })
                                .unwrap_or_default();
                            let maxd = n["maxDimSizes_"]
                                .as_array()
                                .map(|a| a.iter().map(|d| d.as_i64().unwrap_or(0)).collect())
                                .unwrap_or_default();
                            // ⛔ THE START VALUES ARE STRINGS (`{"[0, 0, 0]": "0"}`). Reading them with
                            // `as_i64` alone reports ZERO distinct starts for every operand — a broken
                            // extractor reading as a finding, paid for once already in
                            // `zz_diff_the_rung_descriptors`.
                            let m = n["startAddressCoreCorelet_"]["data_"].as_object();
                            let tot = m.map_or(0, |m| m.len());
                            let dis = m.map_or(0, |m| {
                                m.values()
                                    .filter_map(|x| {
                                        x.as_str()
                                            .and_then(|s| s.trim().parse::<i64>().ok())
                                            .or_else(|| x.as_i64())
                                    })
                                    .collect::<std::collections::BTreeSet<_>>()
                                    .len()
                            });
                            (layout, maxd, tot, dis)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let fold = body["scheduleTree_"][0]["startAddressCoreCorelet_"]["dim_prop_attr"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|d| {
                            (
                                d["label_"].as_str().unwrap_or("?").to_string(),
                                d["factor_"].as_i64().unwrap_or(0),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            OpPicture {
                name,
                iter: sorted_map(&body["N_"]),
                // ⛔ `numWkSlicesPerDim_` IS AN **OP-LEVEL** FIELD, not a `dscs_` one. Reading it off the
                // dsc body answers `{}` for every op in the bundle, which reads as "nothing is split".
                split: sorted_map(&v["numWkSlicesPerDim_"]),
                cores: v["numCoresUsed_"].as_i64().unwrap_or(-1),
                operands,
                fold,
            }
        })
        .collect()
}

/// The ops whose (stem, shape) pair appears in the gathered emission and NOT in the shipped one — what
/// the gather introduces, and the only population any of the assertions below may indict.
fn only_gathered(mq: u32) -> Vec<OpPicture> {
    let plain: std::collections::BTreeSet<(String, String)> = emit_at(mq, false)
        .iter()
        .map(|o| (o.stem(), o.shape()))
        .collect();
    emit_at(mq, true)
        .into_iter()
        .filter(|o| !plain.contains(&(o.stem(), o.shape())))
        .collect()
}

/// The report. `-- --nocapture` to read it.
#[test]
fn what_the_gather_changes() {
    for mq in [2u32, 8] {
        let mut seen: std::collections::BTreeSet<String> = Default::default();
        eprintln!("─── mq={mq}: shapes ONLY in the GATHERED emission ───");
        for o in only_gathered(mq) {
            let line = format!("{:<24} {}", o.stem(), o.shape());
            if seen.insert(line.clone()) {
                eprintln!("  + {line}");
            }
        }
        let gathered: std::collections::BTreeSet<(String, String)> = emit_at(mq, true)
            .iter()
            .map(|o| (o.stem(), o.shape()))
            .collect();
        eprintln!("─── mq={mq}: shapes ONLY in the SHIPPED emission ───");
        for o in emit_at(mq, false) {
            if gathered.contains(&(o.stem(), o.shape())) {
                continue;
            }
            let line = format!("{:<24} {}", o.stem(), o.shape());
            if seen.insert(line.clone()) {
                eprintln!("  - {line}");
            }
        }
    }
}

/// ⭐ FACT 1 — THE DECLARED `y` IS THE **GQA GROUP**, AT EVERY RUNG, AND NEVER THE RUNG WIDTH.
///
/// This is the assertion that inverts. The refused form declared `y_ == mq` (2 at rung 2, 8 at rung 8);
/// the collapsed fold declares `y_ == gqa` — the SAME batch axis the shipped bundle carries — with the
/// requests on no axis at all. A `y` that tracks the rung width again means the per-batch kernel is back,
/// and with it `numCoresUsed_ == mq` and the fault address.
///
/// Compared against the SHIPPED emission rather than against 1, because `attn_newkt*` legitimately
/// carries a width-invariant `y_=64` that has nothing to do with the batch.
#[test]
fn every_y_the_gather_introduces_is_the_gqa_group_and_never_the_rung_width() {
    const GQA: i64 = (NQH / NKVH) as i64;
    for mq in [2u32, 4, 8] {
        for o in only_gathered(mq) {
            let y = o.y();
            assert!(
                y <= 1 || y == GQA,
                "mq={mq}: {} declares y_={y}, which is neither 1 nor the GQA group ({GQA}). If it is the \
                 rung width the requests are back on `y`, which needs the per-batch 3-D kernel the card \
                 REFUSED at every rung — N_{:?}",
                o.name,
                o.iter,
            );
        }
    }
}

/// ⭐⭐⭐ FACT 2 — `numCoresUsed_` IS **OFF THE RUNG WIDTH**: it is the GQA group, 4, at every rung.
///
/// The core count is the number the fault address WAS made of — `job_bin_ptr + cores*128`, one flit per
/// core, read at index `cores`, one past the last one the program holds. The refused form's collapsed
/// onto `mq` (2, 4 and 8, all three faulting at `cores*128`); this one cannot, because `y` is the group
/// and the `mb` split is forbidden for a decode batch. **4 is also the value
/// `matmul/dims.rs`'s own doc records as on-hardware-proven** (`{mb:1, y:4}` on 4 cores, the solo-decode
/// split), so this is not a new number on the card at all.
///
/// ⛔ THE SWEEP OVER THREE RUNGS IS WHAT MAKES THIS SHARP, not a `!= mq` clause. At rung 4 the group and
/// the width are the SAME NUMBER — which is exactly why rung 4 was the card's discriminator — so no
/// single-width assertion can tell "cores is the group" from "cores is the width". Checked at 2, 4 and 8
/// together, the constant 4 can only be the group.
#[test]
fn the_gathers_core_count_is_the_gqa_group_at_every_rung_and_never_the_width() {
    const GQA: i64 = (NQH / NKVH) as i64;
    for (mq, shipped) in [(2u32, 8i64), (4, 16), (8, 32)] {
        let gathered = only_gathered(mq);
        let legs: Vec<&OpPicture> = gathered
            .iter()
            .filter(|o| o.name.contains("sc_g") || o.name.contains("ov_g"))
            .collect();
        assert!(
            !legs.is_empty(),
            "mq={mq}: the gathered emission introduced no per-request score/value leg at all — the diff \
             is broken, not the emission"
        );
        for o in &legs {
            assert_eq!(
                (o.cores, o.y()),
                (GQA, GQA),
                "mq={mq}: {} declares numCoresUsed_={} at y_={} — the collapsed fold's legs must land on \
                 the GQA group ({GQA}) at EVERY rung. A core count that tracks the width is the number \
                 `job_bin_ptr + cores*128` indexed. wk={:?}",
                o.name,
                o.cores,
                o.y(),
                o.split,
            );
        }
        // The shipped prefix score leg, for the contrast: same rung, `4 * mq` cores because it sweeps
        // `mq` rows to keep one.
        let ship = emit_at(mq, false);
        let g0 = ship
            .iter()
            .find(|o| o.name.starts_with("attn_p0sc_g0"))
            .expect("the shipped bundle has a prefix-pass-0 score leg per kv head");
        assert_eq!(
            g0.cores, shipped,
            "the shipped prefix score leg at mq={mq} should use {shipped} cores ({:?})",
            g0.split
        );
    }
}

/// ⛔⛔⛔ FACT 3 — THE FUSED EPILOGUE ON A `y`-BATCHED MATMUL **ALREADY SHIPS**, so splitting it into its
/// own op cannot be the fault.
///
/// The shipped prefix score and value legs are `y=4` batched matmuls with FOUR `scheduleTree_` operands
/// (`a`, the shared 2-D kernel, the output, and the fused epilogue source). This refutes the standing
/// "second cut" without a card run, which is the entire reason to project the emission locally.
#[test]
fn the_fused_epilogue_on_a_y_batched_matmul_already_ships() {
    for mq in [2u32, 8] {
        for stem in ["attn_p0sc_g0", "attn_p0ov_g0"] {
            let ship = emit_at(mq, false);
            let o = ship
                .iter()
                .find(|o| o.name.starts_with(stem))
                .unwrap_or_else(|| panic!("mq={mq}: the shipped bundle has no {stem}"));
            assert!(
                o.y() > 1,
                "mq={mq}: {stem} is meant to be `y`-batched, but N_={:?}",
                o.iter
            );
            assert_eq!(
                o.operands.len(),
                4,
                "mq={mq}: {stem} should carry a FUSED EPILOGUE as its 4th operand — if it does not, the \
                 refutation of the epilogue cut is void. operands={:?}",
                o.operands
                    .iter()
                    .map(|(l, _, _, _)| l.join(","))
                    .collect::<Vec<_>>(),
            );
            // ⛔ AND IT MUST BE ACTIVELY `y`-ADDRESSED, OR THE REFUTATION IS VOID. `attn.rs` documents a
            // BROADCAST pmask arm whose fused operand's address never advances per `y` or per `mb`; that
            // arm would establish nothing about the gathered form, whose epilogue advances on both. A
            // broadcast operand shows ONE distinct per-core start; this one must show as many as the
            // output does.
            let (epi, out) = (&o.operands[3], &o.operands[2]);
            assert_eq!(
                (epi.0.as_slice(), epi.3),
                (out.0.as_slice(), out.3),
                "mq={mq}: {stem}'s fused epilogue operand {:?} has {} distinct per-core start(s) against \
                 the output's {:?}/{} — a BROADCAST epilogue does not refute the epilogue cut.",
                epi.0,
                epi.3,
                out.0,
                out.3,
            );
        }
    }
}

/// ⭐⭐⭐⭐⭐ FACT 4 — **NO OP IN EITHER BUNDLE DECLARES A PER-BATCH 3-D `[y,in,out]` KERNEL, AND EVERY
/// KERNEL HAS EXACTLY ONE DISTINCT PER-CORE START.** This is the assertion the fix exists to satisfy.
///
/// The refused form's kernel was the one operand in the bundle whose per-core start count went from ONE
/// distinct address (all cores read the shared weight from one place) to `mq` — one weight start per core,
/// against a fault at index `cores` of a `cores`-long 128-byte table. The gather makes the rank
/// unnecessary: the scratch is request-minor and contiguous, so request `r`'s block is a baked OFFSET
/// (`GatherScratch::kernel_row_off`) and the kernel stays the bare shared 2-D weight the card runs.
///
/// ⛔ IT ASSERTS ON THE EMITTED DESCRIPTOR, NOT ON THE BUILDER. A builder that no longer has a 3-D door
/// is not evidence: the rank is a property of `layoutDimOrder_`, and that is what this reads.
#[test]
fn no_op_declares_a_per_batch_3d_kernel_and_every_kernel_has_one_start() {
    for mq in [2u32, 4, 8] {
        for gather in [false, true] {
            for o in emit_at(mq, gather) {
                for (l, d, _, dis) in &o.operands {
                    assert!(
                        !(l.len() == 3 && l[0] == "y" && l[1] == "in" && l[2] == "out"),
                        "mq={mq} gather={gather}: {} declares a per-batch 3-D kernel {l:?}{d:?} with \
                         {dis} distinct start(s). That form is REFUTED ON CARD — it faulted at \
                         `job_bin_ptr + numCoresUsed_*128` at rungs 2, 4 and 8 alike.",
                        o.name,
                    );
                }
            }
        }
        // ⭐ AND POSITIVELY: every collapsed-fold leg's KERNEL is `[in,out]` read from ONE address.
        // Operand order is `[a, w, o, epi?]`, so the kernel is operand 1.
        let legs: Vec<OpPicture> = only_gathered(mq)
            .into_iter()
            .filter(|o| o.name.contains("sc_g") || o.name.contains("ov_g"))
            .collect();
        assert!(
            !legs.is_empty(),
            "mq={mq}: no collapsed-fold leg in the gathered emission — the diff is broken"
        );
        for o in &legs {
            let k = &o.operands[1];
            assert_eq!(
                (k.0.join(","), k.3),
                ("in,out".to_string(), 1),
                "mq={mq}: {}'s kernel declares {:?} with {} distinct per-core start(s). The whole fix is \
                 that this operand is `[in,out]` with ONE start — core-invariant, exactly as every \
                 shipped attention kernel is — with the request supplied as a baked offset instead.",
                o.name,
                k.0,
                k.3,
            );
        }
    }
}

/// ⭐⭐⭐⭐⭐ FACT 4b — **THE COLLAPSED OP *IS* THE SHIPPED FOLD OP AT ONE ROW.** The sharpest statement of
/// the fix, and the reason it is worth a card run: every operand's `layoutDimOrder_`, every
/// `maxDimSizes_`, the contraction, the output width and the fused epilogue are IDENTICAL to the shipped
/// prefix leg's. The only declared quantities that move are the swept `mb_` (1 instead of `mq`), the
/// `mb` split that follows it, and `numCoresUsed_`.
///
/// So the shape is not new on the card — it is the `mq == 1` solo-decode leg, which runs at 41 tok/s,
/// emitted `mq` times with three different base offsets.
#[test]
fn the_collapsed_op_is_the_shipped_op_at_one_row() {
    for mq in [2u32, 4, 8] {
        let (ship, gath) = (emit_at(mq, false), emit_at(mq, true));
        for (ship_stem, gath_stem) in [
            ("attn_p0sc_g0_o", "attn_p0sc_g0_r0_o"),
            ("attn_p0ov_g0_o", "attn_p0ov_g0_r0_o"),
        ] {
            let find = |v: &[OpPicture], stem: &str| {
                v.iter()
                    .find(|o| o.name.starts_with(stem))
                    .unwrap_or_else(|| panic!("mq={mq}: no op named {stem}*"))
                    .clone()
            };
            let (s, g) = (find(&ship, ship_stem), find(&gath, gath_stem));
            // Operand DECLARATIONS: layout order and maxDimSizes, per operand. The per-core start COUNT
            // legitimately differs (it is the core count), so it is compared separately below.
            let decl = |o: &OpPicture| -> Vec<(Vec<String>, Vec<i64>)> {
                o.operands
                    .iter()
                    .map(|(l, d, _, _)| (l.clone(), d.clone()))
                    .collect()
            };
            assert_eq!(
                decl(&s),
                decl(&g),
                "mq={mq}: {gath_stem}'s operand declarations differ from the shipped {ship_stem}'s. The \
                 collapse is supposed to change only WHICH ROW an op computes, so any difference here is \
                 a new shape on the card and needs its own justification."
            );
            // `N_`: identical except `mb_`, which is the ONE row instead of the batch's `mq`.
            let iter_but_mb = |o: &OpPicture| -> Vec<(String, i64)> {
                o.iter.iter().filter(|(k, _)| k != "mb_").cloned().collect()
            };
            assert_eq!(
                iter_but_mb(&s),
                iter_but_mb(&g),
                "mq={mq}: {gath_stem} and {ship_stem} must agree on every iteration extent but `mb_`"
            );
            let mb = |o: &OpPicture| o.iter.iter().find(|(k, _)| k == "mb_").map(|(_, v)| *v);
            assert_eq!(
                (mb(&s), mb(&g)),
                (Some(i64::from(mq)), Some(1)),
                "mq={mq}: the shipped leg should sweep `mq` rows and the collapsed one exactly ONE — that \
                 difference IS the redundant arithmetic the collapse removes ({ship_stem} vs {gath_stem})"
            );
        }
    }
}

/// ⭐⭐⭐ FACT 6 — **EVERY `y` SPLIT IN EVERY ATTENTION OP, GATHERED OR NOT, IS 4 (THE GQA GROUP) OR 1.**
///
/// This was the assertion that convicted the refused form: its ops split `y` by the rung width — 2, 4 and
/// 8 — and with `mb` pinned to 1 and a one-stick `out` that split IS `numCoresUsed_`, which IS the fault's
/// flit index. Nothing that has ever launched splits `y` by anything but the group or 1.
///
/// It now covers BOTH emissions, which is the strongest form of the pin: the collapsed fold does not
/// merely avoid the refused number, it lands on the same `y` split the shipped bundle has always carried.
///
/// ⛔ AND `be3ca5355`'s CONTROL STILL DOES NOT REACH, which is why the number is asserted rather than
/// argued. That commit inferred the rank was harmless because "the `mq`-entry one-flit table is in the
/// program of BOTH kernel ranks, and the 2-D one is what ships and runs today". Its premise is the
/// ISOLATION bundle — `/work/iso-gate/superdsc_gate/sweep_bmm2d_y8/group_0/sdsc_0.json` carries
/// `numCoresUsed_:8, numWkSlicesPerDim_:{in:1,mb:1,out:1,y:8}`, measured — and the shipped 2-D form is
/// `{y:4, mb:mq}` on 32 cores, so `{mb:1, y:8}` had never run on card in either rank.
#[test]
fn every_y_split_in_every_attention_op_is_the_gqa_group_or_one() {
    const GQA: i64 = (NQH / NKVH) as i64;
    for mq in [2u32, 4, 8] {
        for gather in [false, true] {
            for o in emit_at(mq, gather) {
                let ys = o
                    .split
                    .iter()
                    .find(|(k, _)| k == "y")
                    .map_or(1, |(_, v)| *v);
                assert!(
                    ys == 1 || ys == GQA,
                    "mq={mq} gather={gather}: {} splits `y` {ys} ways. Every attention op that has ever \
                     launched splits `y` by 1 or the GQA group ({GQA}); a split at the rung width with \
                     `mb` pinned to 1 IS `numCoresUsed_`, which IS the flit index the refused form \
                     faulted at. wk={:?} cores={}",
                    o.name,
                    o.split,
                    o.cores,
                );
            }
        }
    }
}

/// ⭐ FACT 5 — THE GATHER EMITS `mq`× THE PREFIX-LEG OPS **PER PASS**, WHICH IS THE SAME OP COUNT PER
/// STEP, BECAUSE THE SHIPPED FOLD RUNS `mq` PASSES.
///
/// One op per (kv head, request) against the shipped one per kv head — and the shipped one relaunches the
/// whole group once per (page, REQUEST) while the collapsed one relaunches once per page. So the trade
/// bought is `mq`× fewer launches at an unchanged op count, plus `mq`× less arithmetic per op (`mb_=1`
/// against `mb_=mq`, see [`the_collapsed_op_is_the_shipped_op_at_one_row`]).
///
/// ⛔ IT IS NOT `nqh` OPS. That was the refused form's count (one op per QUERY head, `y` carrying the
/// requests), priced from the card at 1.42 µs/op. `nkvh*mq` is 2× that at rung 8 and 1/2× at rung 2, and
/// both are far below the ~28 µs of a fold pass it removes — but the number belongs measured, not
/// remembered.
#[test]
fn the_gathered_prefix_legs_are_one_op_per_kv_head_and_request() {
    for mq in [2u32, 4, 8] {
        let count = |gather: bool, needle: &str| {
            emit_at(mq, gather)
                .iter()
                .filter(|o| o.name.starts_with(needle))
                .count()
        };
        for leg in ["attn_p0sc_", "attn_p0ov_"] {
            let (shipped, gathered) = (count(false, leg), count(true, leg));
            assert_eq!(
                (shipped, gathered),
                (NKVH as usize, (NKVH * mq) as usize),
                "mq={mq}: prefix-pass-0 {leg} legs — shipped {shipped}, gathered {gathered}. Expected \
                 `nkvh` and `nkvh*mq`: one op per kv head per request, in ONE pass where the shipped \
                 form takes `mq`. `nqh` here would mean the refused per-query-head form is back."
            );
        }
    }
}
