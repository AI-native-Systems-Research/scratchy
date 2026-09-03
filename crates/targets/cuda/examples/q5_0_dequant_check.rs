// SPDX-License-Identifier: Apache-2.0
//! GPU dequant correctness — Q5_0.
//!
//! Bartowski's Qwen2.5 Q4_K_M GGUFs use Q5_0 for q_proj/k_proj. End-to-end
//! inference produces gibberish even with `SCRATCHY_DEQUANT_AT_LOAD=1` (which
//! routes the same dequant→cuBLAS path that Llama Q4_K_M uses on Q4_K + Q6_K
//! only). Python's `gguf.dequantize` for those exact tensors matches the HF
//! safetensors reference at quant noise (raw maxdiff ~0.05), so the on-disk
//! bytes are right. The remaining suspect is the GPU Q5_0 dequant kernel.
//!
//! Modes:
//!   - default (no args): synthesizes Q5_0 blocks with deterministic qs/qh +
//!     varied fp16 scale, runs them through `launch_dequantize_block_q5_0_f32`,
//!     and compares against a CPU implementation that mirrors `dequantize_q5_0`
//!     in `csrc/quantized.cu` bit-for-bit.
//!   - `--gguf <path> --tensor <name>`: pulls a real Q5_0 tensor from a GGUF
//!     file and runs the same comparison on its bytes.
//!
//! Float math is exact at single precision (5-bit ints × fp16 scale lifted to
//! f32), so any divergence is a kernel bug.
//!
//! Run with:
//!   cargo run -p scratchy-target-cuda --features cuda --release --example q5_0_dequant_check
//!   cargo run -p scratchy-target-cuda --features cuda --release --example q5_0_dequant_check -- \
//!       --gguf /path/to/Qwen2.5-0.5B-Q4_K_M.gguf --tensor blk.0.attn_q.weight

use scratchy_target_cuda::driver;
use scratchy_target_cuda::ggml::{GgmlDType, ggml_dequantize_f32};
// The kernel `.a` link directives come from this crate's own build.rs (the
// example links `scratchy-target-cuda`), so no separate link anchor is needed.

const Q5_0_BLOCK_BYTES: usize = 22; // fp16 (2) + qh (4) + qs (16)
const Q5_0_BLOCK_ELEMS: usize = 32;

fn pack_q5_0_block(d: half::f16, qh: u32, qs: [u8; 16]) -> [u8; Q5_0_BLOCK_BYTES] {
    let mut out = [0u8; Q5_0_BLOCK_BYTES];
    out[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
    out[2..6].copy_from_slice(&qh.to_le_bytes());
    out[6..22].copy_from_slice(&qs);
    out
}

/// CPU reference matching `dequantize_q5_0` in csrc/quantized.cu.
/// `xh_0` = bit `iqs` of qh placed at bit 4; `xh_1` = bit `iqs+16` of qh placed at bit 4.
fn cpu_dequant_q5_0(raw: &[u8]) -> Vec<f32> {
    assert_eq!(
        raw.len() % Q5_0_BLOCK_BYTES,
        0,
        "raw len must be 22 * nblocks"
    );
    let nblocks = raw.len() / Q5_0_BLOCK_BYTES;
    let mut out = vec![0.0f32; nblocks * Q5_0_BLOCK_ELEMS];
    for b in 0..nblocks {
        let off = b * Q5_0_BLOCK_BYTES;
        let d_bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
        let d = half::f16::from_bits(d_bits).to_f32();
        let qh = u32::from_le_bytes([raw[off + 2], raw[off + 3], raw[off + 4], raw[off + 5]]);
        for iqs in 0..16 {
            let xh_0 = ((qh >> iqs) << 4) & 0x10;
            let xh_1 = (qh >> (iqs + 12)) & 0x10;
            let qs_byte = raw[off + 6 + iqs];
            let lo = (qs_byte & 0x0f) as u32 | xh_0;
            let hi = (qs_byte >> 4) as u32 | xh_1;
            out[b * 32 + iqs] = (lo as f32 - 16.0) * d;
            out[b * 32 + iqs + 16] = (hi as f32 - 16.0) * d;
        }
    }
    out
}

fn init_cuda() -> cudarc::driver::sys::CUstream {
    unsafe {
        driver::init().expect("CUDA init");
        let dev = driver::device_get(0).expect("device");
        let _ctx = driver::ctx_create(dev).expect("context");
        driver::stream_create().expect("stream")
    }
}

fn run_compare(label: &str, raw_bytes: &[u8]) -> bool {
    let nblocks = raw_bytes.len() / Q5_0_BLOCK_BYTES;
    let nelems = nblocks * Q5_0_BLOCK_ELEMS;
    println!("[{label}] {nblocks} blocks ({nelems} elems)");

    unsafe {
        let stream = init_cuda();
        let gpu_src = driver::mem_alloc(raw_bytes.len()).expect("alloc src");
        driver::memcpy_htod_async(gpu_src, raw_bytes.as_ptr(), raw_bytes.len(), stream)
            .expect("htod");
        let gpu_dst = driver::mem_alloc(nelems * 4).expect("alloc dst");
        driver::stream_synchronize(stream).expect("sync upload");

        ggml_dequantize_f32(
            gpu_src,
            gpu_dst as *mut f32,
            GgmlDType::Q5_0,
            nelems,
            stream,
        );

        let mut gpu_out = vec![0.0f32; nelems];
        driver::memcpy_dtoh_async(gpu_out.as_mut_ptr() as *mut u8, gpu_dst, nelems * 4, stream)
            .expect("dtoh");
        driver::stream_synchronize(stream).expect("sync download");

        let cpu_out = cpu_dequant_q5_0(raw_bytes);

        let mut max_abs = 0.0f32;
        let mut first_mismatch: Option<(usize, f32, f32)> = None;
        let mut mismatch_count = 0;
        for i in 0..nelems {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            if diff > 1e-6 {
                mismatch_count += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((i, gpu_out[i], cpu_out[i]));
                }
            }
        }
        println!("[{label}] max_abs_diff = {:.3e}", max_abs);
        if let Some((i, g, c)) = first_mismatch {
            let block = i / Q5_0_BLOCK_ELEMS;
            let lane = i % Q5_0_BLOCK_ELEMS;
            println!(
                "[{label}] FAIL: {mismatch_count} mismatches; first @ elem {i} \
                 (block {block}, lane {lane}): gpu={g} cpu={c} diff={}",
                (g - c).abs()
            );
        }
        // Also show a few sample GPU/CPU values for sanity.
        for i in [0, 1, 16, 17, 31] {
            if i < nelems {
                println!(
                    "[{label}]   elem {i:>3}: gpu={:>10.6}  cpu={:>10.6}",
                    gpu_out[i], cpu_out[i]
                );
            }
        }

        driver::mem_free(gpu_src).expect("free src");
        driver::mem_free(gpu_dst).expect("free dst");
        driver::stream_destroy(stream).expect("destroy stream");

        first_mismatch.is_none()
    }
}

fn synthetic_blocks() -> Vec<u8> {
    // 8 blocks → 256 elements. Mix scale signs, qh patterns, and qs nibble
    // values to exercise both halves of the unpacking.
    let scales = [
        0.0125_f32, -0.0125, 0.5, -1.0, 0.001, -0.001, 0.0078125, -0.0078125,
    ];
    let qhs = [
        0x00000000_u32,
        0xFFFFFFFF,
        0xAAAAAAAA,
        0x55555555,
        0x12345678,
        0x9ABCDEF0,
        0x0000FFFF,
        0xFFFF0000,
    ];
    let mut bytes = Vec::with_capacity(scales.len() * Q5_0_BLOCK_BYTES);
    for (b, (&s, &qh)) in scales.iter().zip(qhs.iter()).enumerate() {
        let d = half::f16::from_f32(s);
        let mut qs = [0u8; 16];
        for (i, q) in qs.iter_mut().enumerate() {
            let lo = (i as u8 + b as u8) & 0xF;
            let hi = (15u8.wrapping_sub(i as u8)).wrapping_sub(b as u8) & 0xF;
            *q = (hi << 4) | lo;
        }
        bytes.extend_from_slice(&pack_q5_0_block(d, qh, qs));
    }
    bytes
}

fn parse_arg(name: &str) -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == name {
            return args.next();
        }
    }
    None
}

fn read_gguf_tensor(path: &str, tensor: &str) -> anyhow::Result<Vec<u8>> {
    use scratchy_quantizations::gguf::Content;
    use std::fs::File;
    use std::io::{BufReader, Read, Seek, SeekFrom};

    let mut reader = BufReader::new(File::open(path)?);
    let content = Content::read(&mut reader).map_err(|e| anyhow::anyhow!("GGUF parse: {e}"))?;
    let info = content
        .tensor_infos
        .get(tensor)
        .ok_or_else(|| anyhow::anyhow!("tensor not found: {tensor}"))?;
    if info.ggml_dtype.0 != 6 {
        // GGML_TYPE_Q5_0 == 6
        anyhow::bail!(
            "tensor {tensor} is dtype id {}, not Q5_0 (id 6)",
            info.ggml_dtype.0
        );
    }
    let elems = info.shape.elem_count();
    let bs = 32; // Q5_0 block size
    let ts = 22; // Q5_0 type size
    let size = (elems / bs) * ts;
    let mut buf = vec![0u8; size];
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(content.tensor_data_offset + info.offset))?;
    f.read_exact(&mut buf)?;
    println!(
        "[gguf] {tensor}: shape={:?} elems={} bytes={}",
        info.shape.dims(),
        elems,
        size
    );
    Ok(buf)
}

fn main() -> anyhow::Result<()> {
    let synth_ok = run_compare("synthetic", &synthetic_blocks());

    let gguf_path = parse_arg("--gguf");
    let tensor = parse_arg("--tensor");
    let real_ok = match (gguf_path, tensor) {
        (Some(p), Some(t)) => {
            let raw = read_gguf_tensor(&p, &t)?;
            run_compare(&format!("gguf:{t}"), &raw)
        }
        _ => true,
    };

    if !synth_ok || !real_ok {
        std::process::exit(1);
    }
    println!("OK: GPU Q5_0 dequant matches CPU reference.");
    Ok(())
}
