// SPDX-License-Identifier: Apache-2.0
// The model-independent body of the `cuda_hf_parity` tests, shared by every qwen3.5 model's
// binary (`cuda_hf_parity.rs` for qwen3.5-0.8b, `cuda_hf_parity_9b.rs` for qwen3.5-9b).
// Included with `include!` at the binary's root, not as a module, so every test keeps its name.
// The including binary supplies the model: its `CUDA_TAPES` and `CUDA_WEIGHTS`, and the
// constants `HIDDEN`, `WEIGHTS` (the weight buffers `load_weights` uploads), `UNTIED_LM_HEAD`
// (the lm_head's own tensor, `None` when it is tied to the embedding table), and the goldens'
// `MODEL` (`cuda_hf_reference.py --model`), `CHECKPOINT` and `GOLDENS` dir.
// qwen3.6-35b-a3b's `cuda_hf_parity_35b.rs` includes it too (AI-native-Systems-Research/
// scratchy#859): each binary states its P1 rule (`P1`, a `P1Rule`), and `load_checks` adds the
// 35B's MoE bundles' rows.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use half::bf16;
use scratchy_target_cuda::runtime::{
    CudaDevice, CudaWorker, DeviceBuffer, GdnPoolConfig, GdnSequence, Input, KvPoolConfig,
    KvSequence, load_weights,
};
use scratchy_target_cuda::step::gdn_bundle;
use scratchy_target_cuda::weights::{CudaWeight, open_checkpoint};

const VOCAB: usize = 248_320;
/// The smallest bucket; every prompt and greedy sequence fits it.
const M: u32 = 512;
/// Blocks per seat: 8 · 64 = 512 tokens.
const KV_POOL: KvPoolConfig = KvPoolConfig { num_blocks: 16 };
const GDN_POOL: GdnPoolConfig = GdnPoolConfig { num_slots: 2 };
const SEAT_A: (&[u32], u32) = (&[0, 1, 2, 3, 4, 5, 6, 7], 0);
const SEAT_B: (&[u32], u32) = (&[8, 9, 10, 11, 12, 13, 14, 15], 1);
/// P6's first chunk.
const CHUNK: usize = 128;

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

/// P1's rule (#859): a fixed floor on min cosine(CUDA, HF-f32), or at most `delta` below
/// HF-bf16's own min cosine on the same prompt.
enum P1Rule {
    Floor(f64),
    BelowHfBf16(f64),
}

/// Whether a cosine `cos` against HF-f32 passes `rule`, `hf_bf16_cos` HF-bf16's own cosine on the
/// same rows: P1's and P8's cosine gate.
fn p1_gate(rule: &P1Rule, cos: f64, hf_bf16_cos: f64) -> bool {
    match *rule {
        P1Rule::Floor(f) => cos >= f,
        P1Rule::BelowHfBf16(d) => cos >= hf_bf16_cos - d,
    }
}

/// Both P1 rules gate as stated, on fixed numbers (no GPU, no goldens): the fixed floor on CUDA's
/// cosine alone, the HF-bf16 rule on CUDA's cosine against HF-bf16's minus delta. Each binary
/// runs one of the two (`P1`), so this is where the other is checked.
#[test]
fn p1_rules_gate_as_stated() {
    let floor = P1Rule::Floor(0.999);
    let below = P1Rule::BelowHfBf16(0.005);
    let cases = [
        (&floor, 0.9995, 0.9, true),
        (&floor, 0.9985, 1.0, false),
        (&below, 0.980474, 0.983452, true),
        (&below, 0.977, 0.983452, false),
    ];
    let mut pass = true;
    for (rule, cos, noise, want) in cases {
        let got = p1_gate(rule, cos, noise);
        pass &= got == want;
    }
    eprintln!("P1 rules on fixed numbers {}", verdict(pass));
    assert!(pass, "a P1 rule does not gate as stated");
}

/// The H100's device memory a worker over every tape may hold after its forwards: 78.0 GiB of
/// the 79.6 GiB `cudaMemGetInfo` reports, about 1.5 GiB left for serve's KV budget (#859's
/// memory gate).
const PEAK_LIMIT: u64 = 78 << 30;

/// Bytes as GiB.
fn gib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1u32 << 30)
}

/// Little-endian byte pairs as the u16 halves `DeviceBuffer` holds.
fn le_halves(bytes: &[u8]) -> Vec<u16> {
    (bytes.as_chunks::<2>().0.iter())
        .map(|&p| u16::from_le_bytes(p))
        .collect()
}

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(GOLDENS)
}

/// The goldens' `meta.json`.
fn meta() -> &'static serde_json::Value {
    static META: OnceLock<serde_json::Value> = OnceLock::new();
    META.get_or_init(|| {
        let path = goldens_dir().join("meta.json");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "{}: {e}. Make the goldens first: python3 -I \
                 crates/models/arch/tests/cuda_hf_reference.py --model {MODEL} --checkpoint \
                 <{CHECKPOINT} dir> --prompts crates/e2e/testdata/prompts.txt --out \
                 crates/models/arch/{GOLDENS}",
                path.display()
            )
        });
        serde_json::from_str(&text).expect("meta.json parses")
    })
}

fn checkpoint() -> PathBuf {
    PathBuf::from(
        meta()["checkpoint"]
            .as_str()
            .expect("meta.json names the checkpoint"),
    )
}

/// One prompt of the goldens: its number, ids, and the f32 and bf16 logits of every row.
struct Prompt {
    p: usize,
    ids: Vec<u32>,
    f32: Vec<f32>,
    bf16: Vec<f32>,
}

fn read_f32(name: &str, rows: usize) -> Vec<f32> {
    let path = goldens_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(bytes.len(), rows * VOCAB * 4, "{name}: [L, vocab] f32");
    (bytes.as_chunks::<4>().0.iter())
        .map(|&q| f32::from_le_bytes(q))
        .collect()
}

fn prompts() -> &'static [Prompt] {
    static PROMPTS: OnceLock<Vec<Prompt>> = OnceLock::new();
    PROMPTS.get_or_init(|| {
        (meta()["prompts"]
            .as_array()
            .expect("meta.json lists prompts")
            .iter())
        .map(|e| {
            let p = e["prompt"].as_u64().expect("a prompt number") as usize;
            let ids: Vec<u32> = (e["ids"].as_array().expect("ids").iter())
                .map(|v| v.as_u64().expect("an id") as u32)
                .collect();
            assert!(
                !ids.is_empty() && ids.len() + 8 <= M as usize,
                "prompt {p}: {} ids",
                ids.len()
            );
            Prompt {
                p,
                f32: read_f32(&format!("ref_f32_{p}.bin"), ids.len()),
                bf16: read_f32(&format!("ref_bf16_{p}.bin"), ids.len()),
                ids,
            }
        })
        .collect()
    })
}

/// The device, the loaded weights, the allocation count before loading, and the loader's time.
struct Rig {
    device: CudaDevice,
    weights: Vec<DeviceBuffer>,
    before: u64,
    load_secs: f64,
}

fn rig() -> Rig {
    let device = CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"));
    let before = device.allocations();
    let t = Instant::now();
    let weights = load_weights(&device, &checkpoint(), CUDA_TAPES, CUDA_WEIGHTS)
        .unwrap_or_else(|e| panic!("load_weights: {e}"));
    Rig {
        device,
        weights,
        before,
        load_secs: t.elapsed().as_secs_f64(),
    }
}

impl Rig {
    fn worker(&self) -> CudaWorker<'_> {
        CudaWorker::new(
            &self.device,
            CUDA_TAPES,
            &self.weights,
            Some(KV_POOL),
            Some(GDN_POOL),
        )
        .unwrap_or_else(|e| panic!("CudaWorker::new: {e}"))
    }
}

fn bucket() -> usize {
    (CUDA_TAPES.iter().position(|t| t.bucket_m == M)).expect("the m=512 tape")
}

/// One forward of `ids` (the real rows) at `prefix_len` in `seat`: the logits of the real rows,
/// widened to f32.
fn forward(w: &mut CudaWorker<'_>, ids: &[u32], prefix_len: u32, seat: (&[u32], u32)) -> Vec<f32> {
    let tape = &CUDA_TAPES[bucket()];
    let q = ids.len() as u32;
    let mut padded = vec![0u32; M as usize];
    padded[..ids.len()].copy_from_slice(ids);
    let positions: Vec<u32> = (prefix_len..prefix_len + M).collect();
    let kv = KvSequence {
        block_table: seat.0,
        prefix_len,
        q_len: q,
    };
    let gdn = GdnSequence {
        slot: seat.1,
        q_len: q,
        fresh: prefix_len == 0,
    };
    let mut out = vec![0u16; tape.result.bytes as usize / 2];
    w.forward(
        bucket(),
        Input::TokenIds(&padded),
        &positions,
        Some(&kv),
        Some(&gdn),
        &mut out,
    )
    .unwrap_or_else(|e| panic!("forward: {e}"));
    (out[..ids.len() * VOCAB].iter())
        .map(|&b| bf16::from_bits(b).to_f32())
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

/// RMS and max of `a − b`.
fn error(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut sq, mut max) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = (f64::from(x) - f64::from(y)).abs();
        sq += d * d;
        max = max.max(d);
    }
    ((sq / a.len() as f64).sqrt(), max)
}

/// The indices of `row`'s `k` highest values, highest first (lowest index on a tie).
fn top_k(row: &[f32], k: usize) -> Vec<usize> {
    let mut ix: Vec<usize> = (0..row.len()).collect();
    ix.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    ix.truncate(k);
    ix
}

fn rows(v: &[f32]) -> impl Iterator<Item = &[f32]> {
    v.chunks(VOCAB)
}

/// The prompts' CUDA logits, one forward each from fresh state: computed once per test binary.
fn cuda_logits(r: &Rig) -> Vec<Vec<f32>> {
    let mut w = r.worker();
    (prompts().iter())
        .map(|p| forward(&mut w, &p.ids, 0, SEAT_A))
        .collect()
}

#[test]
fn load_checks() {
    let r = rig();
    let loaded = r.device.allocations() - r.before;
    let w = r.worker();
    let worker = r.device.allocations() - r.before - loaded;
    drop(w);
    let nonzero = (r.weights.iter()).filter(|b| b.bytes() > 0).count();
    let mut pass = loaded == WEIGHTS && worker == 3 && nonzero == WEIGHTS as usize;
    eprintln!(
        "load: allocations weights={loaded} (expect {WEIGHTS}) worker={worker} (expect 3) \
         nonzero_buffers={nonzero} load_time={:.2}s (from opening the checkpoint to the last \
         upload) {}",
        r.load_secs,
        verdict(pass)
    );
    let sizes_ok = (r.weights.iter().enumerate()).all(|(ix, b)| {
        let declared = CUDA_TAPES
            .iter()
            .map(|t| t.weight_bytes[ix])
            .max()
            .unwrap_or(0);
        b.bytes() == declared
    });
    eprintln!(
        "load: every buffer's bytes equal its weight_bytes {}",
        verdict(sizes_ok)
    );
    pass &= sizes_ok;
    // The checkpoint's own bytes, read through the same reader.
    let mut ck = open_checkpoint(&checkpoint()).unwrap_or_else(|e| panic!("{e}"));
    let Some(CudaWeight::Bf16 { tensor, .. }) = CUDA_WEIGHTS[0] else {
        panic!("weight 0 is the embedding table");
    };
    let (table, ..) = ck.take_to_cpu_bytes(tensor).expect("the embedding table");
    let device_table = r.weights[0].download().expect("download the table");
    for row in [0, VOCAB - 1] {
        let want: Vec<u16> = (table[row * HIDDEN * 2..][..HIDDEN * 2]
            .as_chunks::<2>()
            .0
            .iter())
        .map(|&p| u16::from_le_bytes(p))
        .collect();
        let ok = device_table[row * HIDDEN..][..HIDDEN] == want[..];
        eprintln!(
            "load: embed row {row} equals the checkpoint's {}",
            verdict(ok)
        );
        pass &= ok;
    }
    // An untied lm_head is its own `[vocab, hidden]` tensor: its first and last rows too.
    if let Some(lm_head) = UNTIED_LM_HEAD {
        let ix = (CUDA_WEIGHTS.iter())
            .position(|w| matches!(w, Some(CudaWeight::Bf16 { tensor, .. }) if *tensor == lm_head))
            .unwrap_or_else(|| panic!("no entry for {lm_head}"));
        let (table, ..) = ck.take_to_cpu_bytes(lm_head).expect("the lm_head");
        let device_table = r.weights[ix].download().expect("download the lm_head");
        for row in [0, VOCAB - 1] {
            let want: Vec<u16> = (table[row * HIDDEN * 2..][..HIDDEN * 2]
                .as_chunks::<2>()
                .0
                .iter())
            .map(|&p| u16::from_le_bytes(p))
            .collect();
            let ok = device_table[row * HIDDEN..][..HIDDEN] == want[..];
            eprintln!(
                "load: weight {ix} {lm_head} row {row} equals the checkpoint's {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    let (ix, bundle) = (CUDA_WEIGHTS.iter().enumerate())
        .find_map(|(ix, w)| match *w {
            Some(b @ CudaWeight::GdnBundle { .. }) => Some((ix, b)),
            _ => None,
        })
        .expect("a GDN bundle");
    let CudaWeight::GdnBundle {
        conv1d,
        a_log,
        dt_bias,
        norm,
        conv_dim,
        conv_kernel,
        nv,
        head_v,
    } = bundle
    else {
        unreachable!("matched a bundle");
    };
    let layout = gdn_bundle(conv_dim, conv_kernel, nv, head_v).expect("the bundle fits");
    let halves = r.weights[ix].download().expect("download the bundle");
    let device: Vec<f32> = (halves.as_chunks::<2>().0.iter())
        .map(|&[lo, hi]| f32::from_bits(u32::from(lo) | u32::from(hi) << 16))
        .collect();
    for (tensor, at) in [
        (conv1d, layout.conv1d),
        (a_log, layout.a_log),
        (dt_bias, layout.dt_bias),
        (norm, layout.norm),
    ] {
        let want = ck
            .tensor_to_f32(tensor)
            .unwrap_or_else(|| panic!("{tensor}"));
        let got = &device[at as usize / 4..][..want.len()];
        let ok = (got.iter().zip(&want)).all(|(g, w)| g.to_bits() == w.to_bits());
        eprintln!(
            "load: weight {ix} {tensor} ({} values at {at}) equals the checkpoint's widened to f32 {}",
            want.len(),
            verdict(ok)
        );
        pass &= ok;
    }
    // A MoE model's first and last MoE bundles (qwen3.6-35b-a3b's layers 0 and 39; none in a
    // dense model): the router whole at byte 0, experts 0 and `experts − 1`'s first `gate_up`
    // and `down` rows, and the layer's shared-expert gate, each against the checkpoint's bytes.
    let moe: Vec<(usize, CudaWeight)> = (CUDA_WEIGHTS.iter().enumerate())
        .filter_map(|(ix, w)| match *w {
            Some(b @ CudaWeight::MoeBundle { .. }) => Some((ix, b)),
            _ => None,
        })
        .collect();
    let ends = (moe.iter().enumerate()).filter(|&(n, _)| n == 0 || n + 1 == moe.len());
    for (_, &(ix, bundle)) in ends {
        let CudaWeight::MoeBundle {
            router,
            gate_up,
            down,
            experts,
            hidden,
            inter,
        } = bundle
        else {
            unreachable!("matched a MoE bundle");
        };
        let layout = scratchy_target_cuda::step::moe_bundle(experts, hidden, inter)
            .expect("the MoE bundle fits");
        let device = r.weights[ix].download().expect("download the MoE bundle");
        let (e, h, i) = (experts as usize, hidden as usize, inter as usize);
        let (want, ..) = ck.take_to_cpu_bytes(router).expect("the router");
        let at = layout.router as usize / 2;
        let ok = device[at..][..e * h] == le_halves(&want)[..];
        eprintln!(
            "load: weight {ix} {router} [{e}, {h}] at byte {} equals the checkpoint's {}",
            layout.router,
            verdict(ok)
        );
        pass &= ok;
        for (tensor, base, per_expert, row) in [
            (gate_up, layout.gate_up, 2 * i * h, h),
            (down, layout.down, h * i, i),
        ] {
            let (want, ..) = ck.take_to_cpu_bytes(tensor).expect("an expert tensor");
            for expert in [0, e - 1] {
                let from = expert * per_expert;
                let at = base as usize / 2 + from;
                let ok = device[at..][..row] == le_halves(&want[from * 2..][..row * 2])[..];
                eprintln!(
                    "load: weight {ix} {tensor} expert {expert} row 0 ({row} values) equals the \
                     checkpoint's {}",
                    verdict(ok)
                );
                pass &= ok;
            }
        }
        let gate = router.replace(".mlp.gate.weight", ".mlp.shared_expert_gate.weight");
        let gix = (CUDA_WEIGHTS.iter())
            .position(|w| matches!(w, Some(CudaWeight::Bf16 { tensor, .. }) if *tensor == gate))
            .unwrap_or_else(|| panic!("no entry for {gate}"));
        let (want, ..) = ck.take_to_cpu_bytes(&gate).expect("the shared-expert gate");
        let got = r.weights[gix].download().expect("download the shared-expert gate");
        let ok = got == le_halves(&want);
        eprintln!(
            "load: weight {gix} {gate} ({} values) equals the checkpoint's {}",
            got.len(),
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a load check failed");
}

/// `nvidia-smi`'s memory line for the memory gate's readings, or why there is none. Printed beside
/// `cudaMemGetInfo`, information only.
fn nvidia_smi() -> String {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader",
        ])
        .output()
        .map_or_else(
            |e| format!("nvidia-smi unavailable: {e}"),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        )
}

/// The memory gate (#859), run first and alone: `cudaMemGetInfo` and `nvidia-smi` after the
/// device opens, after `load_weights`, after a worker over every tape, and after one forward of
/// the largest bucket's every row and one decode. Pass: `WEIGHTS` + 3 allocations, every buffer
/// its `weight_bytes`, every bucket in the worker, and a peak (total − free after the forwards) at
/// most `PEAK_LIMIT`. The KV pool is serve's cap, one sequence's block table plus block 0
/// (`CUDA_MAX_BLOCKS_PER_SEQ` + 1); the GDN pool is the rig's. Reads only the goldens'
/// `checkpoint`.
#[test]
fn memory_fits() {
    let device = CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"));
    let reading = |when: &str| {
        let (free, total) = device.mem_info().unwrap_or_else(|e| panic!("mem_info: {e}"));
        let (free, total) = (free as u64, total as u64);
        eprintln!(
            "memory {when}: used {:.3} GiB free {:.3} GiB total {:.3} GiB (cudaMemGetInfo); \
             nvidia-smi used, total: {}",
            gib(total - free),
            gib(free),
            gib(total),
            nvidia_smi()
        );
        (free, total)
    };
    reading("after CudaDevice::open");
    let before = device.allocations();
    let t = Instant::now();
    let weights = load_weights(&device, &checkpoint(), CUDA_TAPES, CUDA_WEIGHTS)
        .unwrap_or_else(|e| panic!("load_weights: {e}"));
    let load_secs = t.elapsed().as_secs_f64();
    let loaded = device.allocations() - before;
    reading("after load_weights");
    let sizes_ok = (weights.iter().enumerate()).all(|(ix, b)| {
        b.bytes() == (CUDA_TAPES.iter().map(|t| t.weight_bytes[ix]).max()).unwrap_or(0)
    });
    let blocks = scratchy_target_cuda::step::CUDA_MAX_BLOCKS_PER_SEQ + 1;
    let mut w = CudaWorker::new(
        &device,
        CUDA_TAPES,
        &weights,
        Some(KvPoolConfig { num_blocks: blocks }),
        Some(GDN_POOL),
    )
    .unwrap_or_else(|e| panic!("CudaWorker::new: {e}"));
    let worker = device.allocations() - before - loaded;
    reading("after CudaWorker::new");
    let ms: Vec<u32> = CUDA_TAPES.iter().map(|t| t.bucket_m).collect();
    let buckets_ok = ms == [1, 512, 1024, 2048, 4096];
    let (b, tape) = (CUDA_TAPES.iter().enumerate())
        .max_by_key(|(_, t)| t.bucket_m)
        .expect("a tape");
    let rows = tape.bucket_m;
    let ids: Vec<u32> = (0..rows).map(|i| (i * 7919) % VOCAB as u32).collect();
    let positions: Vec<u32> = (0..rows).collect();
    let table: Vec<u32> = (0..blocks - 1).collect();
    let kv = KvSequence {
        block_table: &table,
        prefix_len: 0,
        q_len: rows,
    };
    let gdn = GdnSequence {
        slot: 0,
        q_len: rows,
        fresh: true,
    };
    let mut out = vec![0u16; tape.result.bytes as usize / 2];
    w.forward(
        b,
        Input::TokenIds(&ids),
        &positions,
        Some(&kv),
        Some(&gdn),
        &mut out,
    )
    .unwrap_or_else(|e| panic!("the m={rows} forward: {e}"));
    let row = decode(&mut w, 7, rows, (&table[..], 0));
    let non_finite = (out.iter().map(|&b| bf16::from_bits(b).to_f32()))
        .chain(row.iter().copied())
        .filter(|v| !v.is_finite())
        .count();
    let (free, total) = reading(&format!("after one m={rows} forward and one m=1 decode"));
    let peak = total - free;
    let ok = loaded == WEIGHTS && worker == 3 && sizes_ok && buckets_ok && peak <= PEAK_LIMIT;
    eprintln!(
        "memory: allocations weights={loaded} (expect {WEIGHTS}) worker={worker} (expect 3) \
         every buffer its weight_bytes={sizes_ok} buckets {ms:?} (expect all five) arena={} KV \
         pool={:?} ({blocks} blocks) GDN pool={:?} load_time={load_secs:.2}s non-finite \
         logits={non_finite} (information only)",
        w.arena_bytes(),
        w.kv_pool_bytes(),
        w.gdn_pool_bytes()
    );
    eprintln!(
        "memory: peak {:.3} GiB of {:.3} GiB (headroom {:.3} GiB), need <= {:.1} GiB (#859's \
         gate, about 1.5 GiB left for serve's KV budget) {}",
        gib(peak),
        gib(total),
        gib(total - peak),
        gib(PEAK_LIMIT),
        verdict(ok)
    );
    assert!(ok, "the memory gate failed: see the lines above");
}

#[test]
fn p1_to_p4_one_forward_per_prompt() {
    let r = rig();
    let cuda = cuda_logits(&r);
    let (mut p1, mut p2, mut p3, mut p4) = (true, true, true, true);
    let (mut agree, mut total) = (0usize, 0usize);
    for (p, got) in prompts().iter().zip(&cuda) {
        let n = p.ids.len();
        // P1.
        let min_cos = (rows(got).zip(rows(&p.f32)))
            .map(|(a, b)| cosine(a, b))
            .fold(f64::INFINITY, f64::min);
        let noise_cos = (rows(&p.bf16).zip(rows(&p.f32)))
            .map(|(a, b)| cosine(a, b))
            .fold(f64::INFINITY, f64::min);
        let ok1 = match P1 {
            P1Rule::Floor(f) => {
                let ok1 = p1_gate(&P1, min_cos, noise_cos);
                eprintln!(
                    "P1 prompt {}: {n} rows, min cosine(CUDA, f32) = {min_cos:.6} (HF-bf16 \
                     {noise_cos:.6}) (need >= {f}) {}",
                    p.p,
                    verdict(ok1)
                );
                if noise_cos < f {
                    eprintln!(
                        "P1 prompt {}: HF-bf16's own min cosine is below {f}: the floor is wrong, \
                         not the kernels. STOP AND REPORT",
                        p.p
                    );
                }
                ok1
            }
            P1Rule::BelowHfBf16(d) => {
                let ok1 = p1_gate(&P1, min_cos, noise_cos);
                eprintln!(
                    "P1 prompt {}: {n} rows, min cosine(CUDA, f32) = {min_cos:.6} HF-bf16 \
                     {noise_cos:.6} CUDA - HF-bf16 = {:+.6} (need >= HF-bf16 - {d}) {}",
                    p.p,
                    min_cos - noise_cos,
                    verdict(ok1)
                );
                ok1
            }
        };
        p1 &= ok1;
        // P2.
        let (rms, max) = error(got, &p.f32);
        let (noise_rms, noise_max) = error(&p.bf16, &p.f32);
        let ok2 = rms <= 2.0 * noise_rms && max <= 2.0 * noise_max;
        eprintln!(
            "P2 prompt {}: RMS(CUDA-f32)={rms:.6e} RMS(HF-bf16-f32)={noise_rms:.6e} \
             max|CUDA-f32|={max:.6e} max|HF-bf16-f32|={noise_max:.6e} (need <= 2x each) {}",
            p.p,
            verdict(ok2)
        );
        p2 &= ok2;
        // P3.
        // `ties`: the rows whose f32 margin is at most twice their noise, counted whether or not
        // CUDA agrees; `near`: those of them CUDA disagrees on.
        let (mut ties, mut near, mut wrong, mut last_ok) = (Vec::new(), 0usize, Vec::new(), true);
        for (row, ((c, f), b)) in rows(got).zip(rows(&p.f32)).zip(rows(&p.bf16)).enumerate() {
            let top = top_k(f, 2);
            let margin = f64::from(f[top[0]] - f[top[1]]);
            let noise = error(b, f).1;
            let same = top_k(c, 1)[0] == top[0];
            total += 1;
            agree += usize::from(same);
            let tie = margin <= 2.0 * noise;
            if tie {
                ties.push(row);
            }
            if !same && tie {
                near += 1;
            }
            if !same && !tie {
                wrong.push(row);
            }
            if row == n - 1 && !same {
                last_ok = tie;
                eprintln!(
                    "P3 prompt {}: last row disagrees, f32 margin {margin:.4e} vs 2x noise \
                     {:.4e}: {}",
                    p.p,
                    2.0 * noise,
                    if tie { "near-tie" } else { "NOT a near-tie" }
                );
            }
        }
        let ok3 = wrong.is_empty() && last_ok;
        eprintln!(
            "P3 prompt {}: top-1 disagreements outside near-ties={} {:?}, near-tie rows={} {:?} \
             (of them CUDA disagrees on {near}) {}",
            p.p,
            wrong.len(),
            wrong,
            ties.len(),
            ties,
            verdict(ok3)
        );
        p3 &= ok3;
        // P4.
        let last = n - 1;
        let c5: BTreeSet<usize> = top_k(&got[last * VOCAB..][..VOCAB], 5)
            .into_iter()
            .collect();
        let f5: BTreeSet<usize> = top_k(&p.f32[last * VOCAB..][..VOCAB], 5)
            .into_iter()
            .collect();
        let shared = c5.intersection(&f5).count();
        let ok4 = shared >= 4;
        eprintln!(
            "P4 prompt {}: last-row top-5 CUDA {c5:?} f32 {f5:?}, shared {shared} (need >= 4) {}",
            p.p,
            verdict(ok4)
        );
        p4 &= ok4;
    }
    // Information only, not a gate: P3 is the per-prompt rule above, no top-1 disagreement
    // outside a near-tie. The 99% floor it replaced came from 0.8b runs; qwen3.5-9b agreed on
    // 245/249 = 0.9839 rows, every disagreement a near-tie (PR #12).
    eprintln!(
        "P3 all prompts: top-1 agreement {agree}/{total} = {:.4} (information only)",
        agree as f64 / total as f64
    );
    eprintln!(
        "P1-P4 rules and their numbers' sources: P1 {}; P2 <= 2x \
         HF-bf16's RMS and max per prompt (this run's HF-bf16, #816); P3 no top-1 disagreement \
         outside a near-tie, margin <= 2x the row's max |HF-bf16 - f32| (this run's HF-bf16, the \
         user's 2026-10-23 rule); P4 >= 4 of 5 shared on the last row (#816)",
        match P1 {
            P1Rule::Floor(f) => format!("min cosine >= {f}, gated (0.8b runs, #816)"),
            P1Rule::BelowHfBf16(d) => format!(
                "min cosine >= HF-bf16's - {d} per prompt, gated (the user's 2026-10-26 rule, \
                 #859)"
            ),
        }
    );
    assert!(p1 && p2 && p3 && p4, "P1 {p1} P2 {p2} P3 {p3} P4 {p4}");
}

#[test]
fn p5_greedy_matches_hf_f32() {
    let r = rig();
    let mut w = r.worker();
    let mut near_ties = 0;
    let mut pass = true;
    for p in prompts().iter().take(5) {
        let path = goldens_dir().join(format!("greedy_{}.json", p.p));
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let golden: serde_json::Value = serde_json::from_str(&text).expect("greedy json");
        let steps = golden["steps"].as_array().expect("steps");
        let want: Vec<u32> = (steps.iter())
            .map(|s| s["token"].as_u64().expect("a token") as u32)
            .collect();
        let mut seq = p.ids.clone();
        let mut got = Vec::new();
        for _ in 0..want.len() {
            let logits = forward(&mut w, &seq, 0, SEAT_A);
            let last = &logits[(seq.len() - 1) * VOCAB..];
            let token = top_k(last, 1)[0] as u32;
            got.push(token);
            seq.push(token);
        }
        let first = got.iter().zip(&want).position(|(g, w)| g != w);
        let ok = match first {
            None => true,
            Some(step) => {
                let margin = steps[step]["margin"].as_f64().expect("a margin");
                let noise = steps[step]["bf16_max_err"].as_f64().expect("a noise");
                let tie = margin < 2.0 * noise;
                eprintln!(
                    "P5 prompt {}: first divergence at step {step}, f32 margin {margin:.4e} vs \
                     2x noise {:.4e}: {}",
                    p.p,
                    2.0 * noise,
                    if tie { "near-tie" } else { "NOT a near-tie" }
                );
                near_ties += usize::from(tie);
                tie
            }
        };
        eprintln!(
            "P5 prompt {}: CUDA {got:?} HF-f32 {want:?} {}",
            p.p,
            verdict(ok)
        );
        pass &= ok;
    }
    eprintln!(
        "P5 rule: every first divergence at a near-tie step, margin < 2x the step's bf16_max_err \
         (this run's HF-bf16, #816); near-tie prompts {near_ties} of 5, not capped (the user's \
         2026-10-26 rule, as PR #11's chat rule) {}",
        verdict(pass)
    );
    assert!(pass, "the greedy tokens differ from HF-f32's");
}

#[test]
fn p6_chunked_equals_one_shot_on_real_weights() {
    let r = rig();
    let mut w = r.worker();
    let p = &prompts()[5];
    assert!(p.ids.len() > CHUNK, "prompt 6 is {} tokens", p.ids.len());
    let one_shot = forward(&mut w, &p.ids, 0, SEAT_A);
    forward(&mut w, &p.ids[..CHUNK], 0, SEAT_B);
    let rest = forward(&mut w, &p.ids[CHUNK..], CHUNK as u32, SEAT_B);
    let tail = &one_shot[CHUNK * VOCAB..];
    let mismatched = (rest.iter().zip(tail))
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let ok = mismatched == 0;
    eprintln!(
        "P6 prompt 6 ({} tokens): {CHUNK} fresh then {} continuing against one forward, rows \
         {CHUNK}..: mismatched_bits={mismatched} {}",
        p.ids.len(),
        p.ids.len() - CHUNK,
        verdict(ok)
    );
    assert!(ok, "the chunked forward differs from the one-shot one");
}

/// Prompt 1's one-shot forward alone, for `compute-sanitizer --tool memcheck` and `--tool
/// initcheck` (`cuda_hf_parity prompt_1_one_shot --exact`): the load, one forward, finite logits.
#[test]
fn prompt_1_one_shot() {
    let r = rig();
    let mut w = r.worker();
    let p = &prompts()[0];
    let got = forward(&mut w, &p.ids, 0, SEAT_A);
    let non_finite = got.iter().filter(|v| !v.is_finite()).count();
    let ok = non_finite == 0;
    eprintln!(
        "prompt 1 one-shot: {} rows, non-finite logits={non_finite} {}",
        p.ids.len(),
        verdict(ok)
    );
    assert!(ok, "non-finite logits");
}

// ── Decode on real weights ─────────────────────────────────────────

/// One decode of `token` at position `prefix_len` in `seat` (the m=1 tape, one row): its logits,
/// widened to f32.
fn decode(w: &mut CudaWorker<'_>, token: u32, prefix_len: u32, seat: (&[u32], u32)) -> Vec<f32> {
    let b = (CUDA_TAPES.iter().position(|t| t.bucket_m == 1)).expect("the m=1 tape");
    let kv = KvSequence {
        block_table: seat.0,
        prefix_len,
        q_len: 1,
    };
    let gdn = GdnSequence {
        slot: seat.1,
        q_len: 1,
        fresh: false,
    };
    let mut out = vec![0u16; VOCAB];
    w.forward(
        b,
        Input::TokenIds(&[token]),
        &[prefix_len],
        Some(&kv),
        Some(&gdn),
        &mut out,
    )
    .unwrap_or_else(|e| panic!("decode at {prefix_len}: {e}"));
    out.iter().map(|&b| bf16::from_bits(b).to_f32()).collect()
}

/// Greedy through decode: one fresh m=512 prefill of the prompt (its last row's argmax is token
/// 1), then `steps − 1` decodes, each of the token just chosen: the tokens.
fn greedy_by_decode(w: &mut CudaWorker<'_>, ids: &[u32], steps: usize) -> Vec<u32> {
    let logits = forward(w, ids, 0, SEAT_A);
    let mut token = top_k(&logits[(ids.len() - 1) * VOCAB..], 1)[0] as u32;
    let mut got = vec![token];
    for step in 1..steps {
        let p = (ids.len() + step - 1) as u32;
        token = top_k(&decode(w, token, p, SEAT_A), 1)[0] as u32;
        got.push(token);
    }
    got
}

/// P7: prompts 1-5, one fresh m=512 prefill then 7 decodes give HF-f32's 8 greedy tokens, under
/// P5's near-tie rule (every first divergence at a step whose f32 margin is under twice that
/// step's `bf16_max_err`; the number of such prompts is not capped). Both sides printed.
#[test]
fn p7_greedy_via_decode_matches_hf_f32() {
    let r = rig();
    let mut w = r.worker();
    let mut near_ties = 0;
    let mut pass = true;
    for p in prompts().iter().take(5) {
        let path = goldens_dir().join(format!("greedy_{}.json", p.p));
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let golden: serde_json::Value = serde_json::from_str(&text).expect("greedy json");
        let steps = golden["steps"].as_array().expect("steps");
        let want: Vec<u32> = (steps.iter())
            .map(|s| s["token"].as_u64().expect("a token") as u32)
            .collect();
        let got = greedy_by_decode(&mut w, &p.ids, want.len());
        let ok = match got.iter().zip(&want).position(|(g, w)| g != w) {
            None => true,
            Some(step) => {
                let margin = steps[step]["margin"].as_f64().expect("a margin");
                let noise = steps[step]["bf16_max_err"].as_f64().expect("a noise");
                let tie = margin < 2.0 * noise;
                eprintln!(
                    "P7 prompt {}: first divergence at step {step}, f32 margin {margin:.4e} vs \
                     2x noise {:.4e}: {}",
                    p.p,
                    2.0 * noise,
                    if tie { "near-tie" } else { "NOT a near-tie" }
                );
                near_ties += usize::from(tie);
                tie
            }
        };
        eprintln!(
            "P7 prompt {}: CUDA prefill + 7 decodes {got:?} HF-f32 {want:?} {}",
            p.p,
            verdict(ok)
        );
        pass &= ok;
    }
    eprintln!(
        "P7 rule: P5's (this run's HF-bf16, #830); near-tie prompts {near_ties} of 5, not capped \
         (the user's 2026-10-26 rule, as PR #11's chat rule) {}",
        verdict(pass)
    );
    assert!(pass, "the greedy tokens via decode differ from HF-f32's");
}

/// P8: for prompts 1-5, prefill the first L − 1 tokens, then decode token L − 1. That row against
/// HF-f32's last row: cosine under the model's `P1` rule, RMS and max error ≤ 2 × HF-bf16's on that row, the same
/// top-1 unless the f32 margin is at most twice the row's max noise, top-5 overlap ≥ 4. Against
/// CUDA's own one-shot last row: cosine ≥ 0.999, the same argmax unless near-tie, max |diff|
/// printed. A one-token prompt has no prefix to prefill, and is skipped, saying so.
#[test]
fn p8_the_decode_row_matches_hf_f32() {
    let r = rig();
    let mut w = r.worker();
    let mut pass = true;
    for p in prompts().iter().take(5) {
        let n = p.ids.len();
        if n < 2 {
            eprintln!(
                "P8 prompt {}: one token, no prefix to prefill: SKIPPED",
                p.p
            );
            continue;
        }
        let one_shot = forward(&mut w, &p.ids, 0, SEAT_B);
        let one_shot = &one_shot[(n - 1) * VOCAB..];
        forward(&mut w, &p.ids[..n - 1], 0, SEAT_A);
        let row = decode(&mut w, p.ids[n - 1], (n - 1) as u32, SEAT_A);
        let (f32_row, bf16_row) = (&p.f32[(n - 1) * VOCAB..], &p.bf16[(n - 1) * VOCAB..]);
        let cos = cosine(&row, f32_row);
        let (rms, max) = error(&row, f32_row);
        let (noise_rms, noise_max) = error(bf16_row, f32_row);
        let top = top_k(f32_row, 2);
        let margin = f64::from(f32_row[top[0]] - f32_row[top[1]]);
        let tie = margin <= 2.0 * noise_max;
        let same = top_k(&row, 1)[0] == top[0];
        let c5: BTreeSet<usize> = top_k(&row, 5).into_iter().collect();
        let f5: BTreeSet<usize> = top_k(f32_row, 5).into_iter().collect();
        let shared = c5.intersection(&f5).count();
        let noise_cos = cosine(bf16_row, f32_row);
        let cos_ok = p1_gate(&P1, cos, noise_cos);
        let cos_need = match P1 {
            P1Rule::Floor(f) => format!("{f}"),
            P1Rule::BelowHfBf16(d) => format!("HF-bf16's {noise_cos:.6} - {d}"),
        };
        let ok_hf = cos_ok
            && rms <= 2.0 * noise_rms
            && max <= 2.0 * noise_max
            && (same || tie)
            && shared >= 4;
        eprintln!(
            "P8 prompt {} decode row {} vs HF-f32: cosine={cos:.6} (need >= {cos_need}) \
             RMS={rms:.6e} (HF-bf16 {noise_rms:.6e}) max={max:.6e} (HF-bf16 {noise_max:.6e}) \
             (need <= 2x each) top-1 same={same} (f32 margin {margin:.4e}, near-tie={tie}) \
             top-5 shared={shared} (need >= 4) {}",
            p.p,
            n - 1,
            verdict(ok_hf)
        );
        let cos_own = cosine(&row, one_shot);
        let (_, diff_own) = error(&row, one_shot);
        let same_own = top_k(&row, 1)[0] == top_k(one_shot, 1)[0];
        let ok_own = cos_own >= 0.999 && (same_own || tie);
        eprintln!(
            "P8 prompt {} decode row {} vs CUDA's one-shot last row: cosine={cos_own:.6} (need \
             >= 0.999) argmax same={same_own} (near-tie={tie}) max|diff|={diff_own:.6e} {}",
            p.p,
            n - 1,
            verdict(ok_own)
        );
        pass &= ok_hf && ok_own;
    }
    assert!(
        pass,
        "a decode row is out of agreement: see the lines above"
    );
}

/// Prompt 1's prefill then 7 decodes alone, for `compute-sanitizer --tool memcheck`
/// (`cuda_hf_parity prompt_1_decode --exact`): the load, the forwards, finite logits.
#[test]
fn prompt_1_decode() {
    let r = rig();
    let mut w = r.worker();
    let p = &prompts()[0];
    let logits = forward(&mut w, &p.ids, 0, SEAT_A);
    let mut token = top_k(&logits[(p.ids.len() - 1) * VOCAB..], 1)[0] as u32;
    let mut non_finite = 0;
    for step in 0..7 {
        let row = decode(&mut w, token, (p.ids.len() + step) as u32, SEAT_A);
        non_finite += row.iter().filter(|v| !v.is_finite()).count();
        token = top_k(&row, 1)[0] as u32;
    }
    let ok = non_finite == 0;
    eprintln!(
        "prompt 1 prefill + 7 decodes: non-finite decode logits={non_finite} {}",
        verdict(ok)
    );
    assert!(ok, "non-finite decode logits");
}
