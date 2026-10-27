// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ A PROMPT WINDOW WIDER THAN A PAGE IS WRITTEN IN PAGE PARTS — pinned off the op list the
//! shipped [`kv_cache_write`] returns, at both head dims the chunk form serves.
//!
//! One launch resolves ONE page per group, and a page plane is `PAGE_SLOTS` slots of one kv head, so a
//! single `[mq, hd]` copy of a 512-row chunk would run off its page into the next kv head's block. Each
//! `(kv head, slab)` copy is therefore cut into parts of `PAGE_SLOTS` rows: part `p` reads the chunk's
//! rows from `p·PAGE_SLOTS`, writes the SAME baked destination, and carries
//! its `PagePart` (minted by the cut itself, `write_parts`) so the runtime aims it at its own page. The Kᵀ re-transpose is cut
//! the same way. A window that fits one page is one part at offset 0 and emits exactly what it did.

use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::emit::lower_ktir_to_superdsc::{KvCacheWrite, kv_cache_write};
use ktir_superdsc::page_part::Slots;
use ktir_superdsc::score_scale::NewK;
use ktir_superdsc::sdsc_abstract::{PagedKvPool, QueryRowCount};

const NKVH: u32 = 8;
const PAGE: u32 = PagedKvPool::PAGE_SLOTS as u32;
const STICK: i64 = 64;

fn write(mq: u32, hd: u32, rows_are_requests: bool) -> Vec<EmittedOp> {
    let mut sym = 0i64;
    kv_cache_write(
        KvCacheWrite {
            t: ktir_superdsc::place::PlaceId::Act(7),
            mq: QueryRowCount::of_mq(mq),
            pool: PagedKvPool::new(NKVH, hd),
            rows_are_requests,
            // The door's own main-call spelling: the window starts at the cursor's row 0.
            row_start: ktir_superdsc::sdsc_abstract::WriteRowStart::at_cursor(),
            new_k: NewK::reader_stated("t_newk"),
            new_v: "t_newv",
            kc: "t_kc",
            vc: "t_vc",
            kct: "t_kct",
            ident: "t_ident",
        },
        &mut sym,
        None,
    )
    .unwrap_or_else(|e| panic!("the cache write lowers at mq={mq} hd={hd}: {}", e.message))
}

fn dsc(e: &EmittedOp) -> serde_json::Value {
    serde_json::to_value(&e.op).unwrap()["dscs_"][0][&e.op_name].clone()
}

/// The op's own base address for one operand, in elements — the minimum over its per-core start
/// addresses, so it is the emitter's answer and not a re-derivation of the placement.
fn base_elems(e: &EmittedOp, arg: usize) -> i64 {
    dsc(e)["scheduleTree_"][arg]["startAddressCoreCorelet_"]["data_"]
        .as_object()
        .expect("per-core start addresses")
        .values()
        .map(|a| a.as_str().unwrap().parse::<i64>().unwrap() / 2)
        .min()
        .expect("at least one core")
}

fn rows(e: &EmittedOp) -> u64 {
    dsc(e)["N_"]["mb_"].as_u64().expect("the copy's row extent")
}

fn cachewr(ops: &[EmittedOp]) -> Vec<&EmittedOp> {
    ops.iter()
        .filter(|e| e.op_name.starts_with("cachewr_"))
        .collect()
}

/// ⭐ 512 ROWS ARE TWO PARTS OF 256 PER `(kv head, slab, plane)`, OFFSETS 0 AND 256 — and part 1 reads
/// the chunk's rows 256.. while writing the SAME destination as part 0. At granite-2b (one slab) and
/// granite-8b (two).
#[test]
fn a_two_page_chunk_writes_two_parts_per_kv_head_and_slab() {
    for hd in [64u32, 128] {
        let slabs = hd / 64;
        let ops = write(512, hd, false);
        let wr = cachewr(&ops);
        assert_eq!(
            wr.len() as u32,
            2 * slabs * NKVH * 2,
            "hd={hd}: two parts x {slabs} slab(s) x {NKVH} kv heads x (K, V)"
        );
        for s in 0..slabs {
            for h in 0..NKVH {
                for base in ["kc", "vc"] {
                    let name = |p: u32| format!("cachewr_{base}{h}s{s}_p{p}_o7");
                    let part = |p: u32| {
                        *wr.iter()
                            .find(|e| e.op_name == name(p))
                            .unwrap_or_else(|| panic!("hd={hd}: {} is emitted", name(p)))
                    };
                    let (p0, p1) = (part(0), part(1));
                    assert_eq!(
                        (p0.kv_part.slots(), p1.kv_part.slots()),
                        (Slots::of(0), Slots::of(PAGE))
                    );
                    assert_eq!((rows(p0), rows(p1)), (256, 256), "{}", name(1));
                    assert_eq!(
                        base_elems(p1, 0) - base_elems(p0, 0),
                        i64::from(PAGE) * STICK,
                        "{}: part 1 reads the chunk's rows from {PAGE}",
                        name(1)
                    );
                    assert_eq!(
                        base_elems(p1, 2),
                        base_elems(p0, 2),
                        "{}: both parts write the same baked destination — the page differs, by the \
                         launch's shift, never the address",
                        name(1)
                    );
                    for e in [p0, p1] {
                        assert_eq!(e.kv_page_slots, PAGE);
                        assert!(e.slot_stride_bytes > 0, "{} is a slot write", e.op_name);
                    }
                }
            }
        }
    }
}

/// THE LAST PART TAKES THE REMAINDER: a 384-row chunk is a 256-row part and a 128-row part.
#[test]
fn a_part_past_the_first_page_copies_only_the_rows_left() {
    let ops = write(384, 64, false);
    for h in 0..NKVH {
        for base in ["kc", "vc"] {
            let find = |p: u32| {
                let name = format!("cachewr_{base}{h}s0_p{p}_o7");
                ops.iter()
                    .find(|e| e.op_name == name)
                    .unwrap_or_else(|| panic!("{name} is emitted"))
            };
            assert_eq!(rows(find(0)), 256);
            assert_eq!(rows(find(1)), 128);
            assert_eq!(find(1).kv_part.slots(), Slots::of(PAGE));
        }
    }
}

/// PART-MAJOR: every op of part 0 precedes every op of part 1, for the cache write and for the Kᵀ
/// re-transpose alike, so each part's copies are consecutive and fuse into one launch.
#[test]
fn the_parts_are_emitted_part_major_and_the_kt_re_transpose_follows_them() {
    for hd in [64u32, 128] {
        let ops = write(512, hd, false);
        for stem in ["cachewr_", "attn_kctpost"] {
            let offs: Vec<Slots> = ops
                .iter()
                .filter(|e| e.op_name.starts_with(stem))
                .map(|e| e.kv_part.slots())
                .collect();
            let first_p1 = offs
                .iter()
                .position(|o| *o == Slots::of(PAGE))
                .expect("a part 1");
            assert!(
                offs[..first_p1].iter().all(|o| *o == Slots::ZERO)
                    && offs[first_p1..].iter().all(|o| *o == Slots::of(PAGE)),
                "hd={hd} {stem}*: offsets {offs:?} interleave the parts"
            );
        }
        for p in 0..2u32 {
            for kvh in 0..NKVH {
                let name = format!("attn_kctpost{kvh}_p{p}_o7");
                let e = ops
                    .iter()
                    .find(|e| e.op_name == name)
                    .unwrap_or_else(|| panic!("hd={hd}: {name} is emitted"));
                assert_eq!(e.kv_part.slots(), Slots::of(p * PAGE), "{name}");
            }
        }
    }
}

/// ⛔ A WINDOW THAT FITS ONE PAGE IS UNCHANGED: one part, offset 0, and the names it always had — the
/// 256-row chunk, a decode token, and a batched decode step's per-request copies.
#[test]
fn a_window_inside_one_page_is_one_part_with_its_old_names() {
    for (mq, rows_are_requests) in [(256u32, false), (128, false), (1, false), (4, true)] {
        let ops = write(mq, 128, rows_are_requests);
        assert!(
            ops.iter().all(|e| e.kv_part.is_cursor()),
            "mq={mq}: a one-page window has no page part"
        );
        assert!(
            ops.iter().all(|e| !e.op_name.contains("_p")),
            "mq={mq}: a one-page window's names carry no part tag"
        );
    }
    let ops = write(256, 64, false);
    for h in 0..NKVH {
        let name = format!("cachewr_kc{h}s0_o7");
        let e = ops
            .iter()
            .find(|e| e.op_name == name)
            .unwrap_or_else(|| panic!("{name} is emitted"));
        assert_eq!(rows(e), 256);
    }
}
