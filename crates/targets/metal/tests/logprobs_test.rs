// SPDX-License-Identifier: Apache-2.0
//! `logprobs_topk` (`shaders/logprobs.metal`) against an f64 reference: for each entry, the
//! log-softmax of its logits row at the token emitted from that row, that token's rank, and the
//! row's 20 most likely tokens, ties to the smaller id — at Qwen3.6's vocabulary width, with the
//! entries' rows permuted and repeated, planted ties, and a row whose emitted token is its maximum.
//!
//! GPU tests — run with `--test-threads=1` (standing rule).

use half::bf16;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::logprobs::{SLOTS, THREADS, TOP, encode_raw_copies};
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::specialized_pipeline_cache::{PipelineKey, SpecializedPipelineCache};
use scratchy_target_metal::tape::ids::LogitsWidth;
use scratchy_target_metal::tape::kernel_constants::LogprobsConstants;

type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

/// Qwen3.6-35B-A3B's vocabulary.
const VOCAB: usize = 248_320;
const ROWS: usize = 6;

struct Random(u64);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 = (self.0)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() & 0xFF_FFFF) as f32 / 8_388_608.0 - 1.0
    }
}

fn buffer<T: Copy>(
    device: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>,
    data: &[T],
) -> Buffer {
    let bytes = std::mem::size_of_val(data);
    let buf = device
        .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    // SAFETY: `buf` holds at least `bytes` bytes and does not overlap `data`.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            bytes,
        )
    };
    buf
}

fn read<T: Copy>(buf: &Buffer, n: usize) -> Vec<T> {
    // SAFETY: `buf` holds `n` `T`s, written by a command buffer that is done.
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const T, n).to_vec() }
}

/// The reference scores of `row` at `token`: its log-probability and rank, then the top `TOP`
/// (id, log-probability), larger first, ties to the smaller id.
fn reference(row: &[f32], token: u32) -> ((f64, u32), Vec<(u32, f64)>) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
    let log_sum = max + sum.ln();
    let emitted = row[token as usize];
    let rank = 1 + row.iter().filter(|&&v| v > emitted).count() as u32;
    let mut order: Vec<u32> = (0..row.len() as u32).collect();
    order
        .sort_by(|&a, &b| (row[b as usize].partial_cmp(&row[a as usize]).unwrap()).then(a.cmp(&b)));
    let top = (order.iter().take(TOP))
        .map(|&i| (i, row[i as usize] as f64 - log_sum))
        .collect();
    ((emitted as f64 - log_sum, rank), top)
}

/// The kernel's scores of `entries` (logits rows of `logits`, and the tokens emitted from each row):
/// each entry's log-probabilities, token ids and ranks, its emitted token's then its top's.
fn score(
    device: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>,
    pipeline: &scratchy_target_metal::shader_cache::ComputePipelineState,
    logits: &Buffer,
    entries: &[u32],
    tokens: &[u32],
) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
    let b_rows = buffer(device, entries);
    let b_tokens = buffer(device, tokens);
    let n = entries.len() * SLOTS;
    let (b_lp, b_tok, b_rank) = (
        buffer(device, &vec![f32::NAN; n]),
        buffer(device, &vec![u32::MAX; n]),
        buffer(device, &vec![u32::MAX; n]),
    );
    let mut batch = Mtl4DispatchBatch::begin(device).expect("an MTL4 queue");
    batch.encode(
        pipeline,
        &[
            (logits, 0),
            (&b_rows, 1),
            (&b_tokens, 2),
            (&b_lp, 4),
            (&b_tok, 5),
            (&b_rank, 6),
        ],
        &[(entries.len() as u32, 3)],
        &[],
        &[],
        MTLSize {
            width: entries.len(),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
    batch.commit(true);
    (
        read::<f32>(&b_lp, n),
        read::<u32>(&b_tok, n),
        read::<u32>(&b_rank, n),
    )
}

#[test]
fn logprobs_match_the_log_softmax_rank_and_top_of_each_row() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let key = PipelineKey::new(
        "logprobs",
        "logprobs_topk_bf16",
        LogprobsConstants {
            vocab: LogitsWidth(VOCAB as u32),
        }
        .into(),
    );
    let pipeline = baked_build(&cache, &key).expect("logprobs_topk_bf16");

    let mut random = Random(0x10_9b0b5);
    let mut rows: Vec<Vec<f32>> = (0..ROWS)
        .map(|r| {
            (0..VOCAB)
                .map(|_| bf16::from_f32(random.unit() * (4.0 + 6.0 * r as f32)).to_f32())
                .collect()
        })
        .collect();
    // Ties among the leaders: three tokens of row 1 share its new maximum, so the smaller ids lead.
    let lead = rows[1].iter().copied().fold(f32::NEG_INFINITY, f32::max) + 1.0;
    for i in [91_000usize, 17, 248_319] {
        rows[1][i] = lead;
    }
    // Tokens: row 2's emitted token is its maximum; the others anywhere.
    let mut tokens: Vec<u32> = (0..ROWS).map(|_| random.next() % VOCAB as u32).collect();
    tokens[2] = (0..VOCAB as u32)
        .max_by(|&a, &b| {
            rows[2][a as usize]
                .partial_cmp(&rows[2][b as usize])
                .unwrap()
                .then(b.cmp(&a))
        })
        .unwrap();
    tokens[1] = 248_319;
    let entries: Vec<u32> = vec![3, 0, 5, 5, 1, 2];

    let logits: Vec<u16> = (rows.iter().flatten())
        .map(|&v| bf16::from_f32(v).to_bits())
        .collect();
    let b_logits = buffer(&device, &logits);
    let (lp, tok, rank) = score(&device, &pipeline, &b_logits, &entries, &tokens);
    for (e, &r) in entries.iter().enumerate() {
        let row = &rows[r as usize];
        let ((want_lp, want_rank), want_top) = reference(row, tokens[r as usize]);
        let at = e * SLOTS;
        assert_eq!(tok[at], tokens[r as usize], "entry {e}: the emitted token");
        assert_eq!(rank[at], want_rank, "entry {e}: the emitted token's rank");
        assert!(
            (lp[at] as f64 - want_lp).abs() < 1e-4,
            "entry {e}: {} vs {want_lp}",
            lp[at]
        );
        for (k, &(id, want)) in want_top.iter().enumerate() {
            let s = at + 1 + k;
            assert_eq!(tok[s], id, "entry {e} top {k}: the token");
            assert_eq!(rank[s], k as u32 + 1, "entry {e} top {k}: the rank");
            assert!(
                (lp[s] as f64 - want).abs() < 1e-4,
                "entry {e} top {k}: {} vs {want}",
                lp[s]
            );
        }
    }
    // Row 1's tied leaders: the smallest id first, and the emitted one (the largest id) third.
    let row1 = entries.iter().position(|&r| r == 1).unwrap() * SLOTS;
    assert_eq!(&tok[row1 + 1..row1 + 4], &[17, 91_000, 248_319]);
    assert_eq!(
        rank[row1], 1,
        "a tied leader's rank counts only the logits strictly above it"
    );
}

/// A grammar-masked step scores its rows' raw logits, set aside before the mask overwrites the
/// logits in place (`encode_raw_copies`), as cuda's worker (`raw_logits_for_logprobs`) and vLLM by
/// default do. Scored from the copy, a row whose disallowed tokens the mask set to -inf keeps its
/// raw log-softmax, rank and top; scored from the masked logits it does not.
#[test]
fn a_masked_row_scores_its_raw_logits() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let key = PipelineKey::new(
        "logprobs",
        "logprobs_topk_bf16",
        LogprobsConstants {
            vocab: LogitsWidth(VOCAB as u32),
        }
        .into(),
    );
    let pipeline = baked_build(&cache, &key).expect("logprobs_topk_bf16");

    const MASKED_ROWS: usize = 3;
    let mut random = Random(0x003a_5ced);
    let rows: Vec<Vec<f32>> = (0..MASKED_ROWS)
        .map(|_| {
            (0..VOCAB)
                .map(|_| bf16::from_f32(random.unit() * 8.0).to_f32())
                .collect()
        })
        .collect();
    let tokens: Vec<u32> = vec![7, 123_456, 248_000];
    // Rows 2 and 0 are scored; row 1 is not, and is never set aside.
    let entries: Vec<u32> = vec![2, 0];
    let bits = |rows: &[Vec<f32>]| -> Vec<u16> {
        (rows.iter().flatten())
            .map(|&v| bf16::from_f32(v).to_bits())
            .collect()
    };
    let b_logits = buffer(&device, &bits(&rows));
    let b_raw = buffer(&device, &vec![0u16; MASKED_ROWS * VOCAB]);
    let row_bytes = (VOCAB * 2) as u64;
    let batch = Mtl4DispatchBatch::begin(&device).expect("an MTL4 queue");
    // The copy reads and writes them: resident for the command buffer.
    let _pinned = (
        batch.residency().pin(b_logits.clone()),
        batch.residency().pin(b_raw.clone()),
    );
    encode_raw_copies(batch.encoder(), &b_logits, &b_raw, &entries, row_bytes);
    batch.commit(true);

    // The mask: every token but a thousand-token allowed range and the emitted one goes to -inf,
    // in place.
    let mut masked = rows.clone();
    for (r, row) in masked.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            if !(1000..2000).contains(&j) && j != tokens[r] as usize {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    let b_masked = buffer(&device, &bits(&masked));

    let (raw_lp, raw_tok, raw_rank) = score(&device, &pipeline, &b_raw, &entries, &tokens);
    let (masked_lp, _, _) = score(&device, &pipeline, &b_masked, &entries, &tokens);
    for (e, &r) in entries.iter().enumerate() {
        let ((want_lp, want_rank), want_top) = reference(&rows[r as usize], tokens[r as usize]);
        let at = e * SLOTS;
        assert_eq!(
            raw_tok[at], tokens[r as usize],
            "entry {e}: the emitted token"
        );
        assert_eq!(raw_rank[at], want_rank, "entry {e}: its raw rank");
        assert!(
            (raw_lp[at] as f64 - want_lp).abs() < 1e-4,
            "entry {e}: raw {} vs {want_lp}",
            raw_lp[at]
        );
        for (k, &(id, want)) in want_top.iter().enumerate() {
            assert_eq!(
                raw_tok[at + 1 + k],
                id,
                "entry {e} top {k}: the raw top's token"
            );
            assert!(
                (raw_lp[at + 1 + k] as f64 - want).abs() < 1e-4,
                "entry {e} top {k}: raw {} vs {want}",
                raw_lp[at + 1 + k]
            );
        }
        assert!(
            (masked_lp[at] as f64 - want_lp).abs() > 1e-2,
            "entry {e}: the masked logits score differently ({} vs raw {want_lp})",
            masked_lp[at]
        );
    }
}
