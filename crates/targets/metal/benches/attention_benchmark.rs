use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize};
use scratchy_target_metal::device::Device;
use scratchy_target_metal::{detect_device, mtl4_dispatch};
use std::time::Duration;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Compile `shaders/<shader_file>`'s `fn_name` kernel into a compute
/// pipeline (matches the original bench's per-shape library compile).
fn build_pipeline(device: &Device, shader_file: &str, fn_name: &str) -> Pipeline {
    let library_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .join(shader_file);
    let library_source = std::fs::read_to_string(&library_path)
        .unwrap_or_else(|_| panic!("Failed to read {shader_file}"));
    let opts = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(&library_source), Some(&opts))
        .expect("Failed to compile shader library");
    let func = library
        .newFunctionWithName(&NSString::from_str(fn_name))
        .expect("Failed to get kernel function");
    device
        .newComputePipelineStateWithFunction_error(&func)
        .expect("Failed to create pipeline state")
}

fn benchmark_single_head_attention(c: &mut Criterion) {
    let device = detect_device().expect("No Metal device found").device;

    let mut group = c.benchmark_group("single_head_attention");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for seq_len in [64, 128, 256, 512, 1024, 2048].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            seq_len,
            |b, &seq_len| {
                const HEAD_SIZE: usize = 128;
                const BLOCK_SIZE: usize = 16;
                let scale: f32 = 1.0 / (HEAD_SIZE as f32).sqrt();
                let num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

                // Create test data
                let q_data = vec![1.0f32; HEAD_SIZE];
                let k_data = vec![1.0f32; num_blocks * HEAD_SIZE * BLOCK_SIZE];
                let v_data = vec![1.0f32; num_blocks * HEAD_SIZE * BLOCK_SIZE];
                let block_table: Vec<i32> = (0..num_blocks as i32).collect();

                let q_f16: Vec<u16> = q_data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let k_f16: Vec<u16> = k_data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let v_f16: Vec<u16> = v_data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();

                let q_buffer = mtl4_dispatch::shared_slice(&device, &q_f16);
                let k_buffer = mtl4_dispatch::shared_slice(&device, &k_f16);
                let v_buffer = mtl4_dispatch::shared_slice(&device, &v_f16);
                let block_table_buffer = mtl4_dispatch::shared_slice(&device, &block_table);
                let output_buffer = mtl4_dispatch::shared_zeroed(&device, HEAD_SIZE * 2);

                #[repr(C)]
                #[derive(Clone, Copy)]
                struct PagedAttentionParams {
                    seq_len: u32,
                    head_size: u32,
                    scale: f32,
                    block_size: u32,
                    max_num_blocks: u32,
                    kv_block_stride: u32,
                    kv_head_stride: u32,
                }

                let kv_head_stride = HEAD_SIZE as u32 * BLOCK_SIZE as u32;
                let params = PagedAttentionParams {
                    seq_len: seq_len as u32,
                    head_size: HEAD_SIZE as u32,
                    scale,
                    block_size: BLOCK_SIZE as u32,
                    max_num_blocks: num_blocks as u32,
                    kv_block_stride: kv_head_stride,
                    kv_head_stride,
                };
                let params_buffer = mtl4_dispatch::shared_slice(&device, &[params]);

                let pipeline = build_pipeline(
                    &device,
                    "attention_paged.metal",
                    "attention_paged_single_head",
                );

                b.iter(|| {
                    mtl4_dispatch::dispatch_threadgroups(
                        &device,
                        &pipeline,
                        &[
                            &q_buffer,
                            &k_buffer,
                            &v_buffer,
                            &block_table_buffer,
                            &output_buffer,
                            &params_buffer,
                        ],
                        // Original: dispatch_threads(grid=256, tg=256) ⇒ one
                        // threadgroup of 256 threads.
                        MTLSize {
                            width: 1,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                    black_box(&output_buffer);
                });
            },
        );
    }
    group.finish();
}

fn benchmark_multihead_attention(c: &mut Criterion) {
    let device = detect_device().expect("No Metal device found").device;

    let mut group = c.benchmark_group("multihead_attention");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for (num_heads, seq_len) in [(4, 512), (8, 512), (16, 512), (32, 512)].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}heads_{}seq", num_heads, seq_len)),
            &(*num_heads, *seq_len),
            |b, &(num_heads, seq_len)| {
                const HEAD_SIZE: usize = 128;
                const BLOCK_SIZE: usize = 16;
                let scale: f32 = 1.0 / (HEAD_SIZE as f32).sqrt();
                let num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
                let num_kv_heads = num_heads; // No GQA for this benchmark

                let q_data = vec![1.0f32; num_heads * HEAD_SIZE];
                let k_data = vec![1.0f32; num_kv_heads * num_blocks * HEAD_SIZE * BLOCK_SIZE];
                let v_data = vec![1.0f32; num_kv_heads * num_blocks * HEAD_SIZE * BLOCK_SIZE];
                let block_table: Vec<i32> = (0..num_blocks as i32).collect();

                let q_f16: Vec<u16> = q_data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let k_f16: Vec<u16> = k_data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let v_f16: Vec<u16> = v_data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();

                let q_buffer = mtl4_dispatch::shared_slice(&device, &q_f16);
                let k_buffer = mtl4_dispatch::shared_slice(&device, &k_f16);
                let v_buffer = mtl4_dispatch::shared_slice(&device, &v_f16);
                let block_table_buffer = mtl4_dispatch::shared_slice(&device, &block_table);
                let output_buffer =
                    mtl4_dispatch::shared_zeroed(&device, num_heads * HEAD_SIZE * 2);

                #[repr(C)]
                #[derive(Clone, Copy)]
                struct MultiHeadAttentionParams {
                    seq_len: u32,
                    head_size: u32,
                    num_heads: u32,
                    num_kv_heads: u32,
                    scale: f32,
                    block_size: u32,
                    max_num_blocks: u32,
                    kv_block_stride: u32,
                    kv_head_stride: u32,
                }

                let kv_head_stride = HEAD_SIZE as u32 * BLOCK_SIZE as u32;
                let params = MultiHeadAttentionParams {
                    seq_len: seq_len as u32,
                    head_size: HEAD_SIZE as u32,
                    num_heads: num_heads as u32,
                    num_kv_heads: num_kv_heads as u32,
                    scale,
                    block_size: BLOCK_SIZE as u32,
                    max_num_blocks: num_blocks as u32,
                    kv_block_stride: kv_head_stride,
                    kv_head_stride,
                };
                let params_buffer = mtl4_dispatch::shared_slice(&device, &[params]);

                let pipeline = build_pipeline(
                    &device,
                    "attention_multihead.metal",
                    "attention_multihead_paged",
                );

                b.iter(|| {
                    mtl4_dispatch::dispatch_threadgroups(
                        &device,
                        &pipeline,
                        &[
                            &q_buffer,
                            &k_buffer,
                            &v_buffer,
                            &block_table_buffer,
                            &output_buffer,
                            &params_buffer,
                        ],
                        MTLSize {
                            width: num_heads,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                    black_box(&output_buffer);
                });
            },
        );
    }
    group.finish();
}

fn benchmark_optimized_attention(c: &mut Criterion) {
    let device = detect_device().expect("No Metal device found").device;

    let mut group = c.benchmark_group("optimized_vs_baseline");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for seq_len in [256, 512, 1024].iter() {
        // Baseline
        group.bench_with_input(
            BenchmarkId::new("baseline", seq_len),
            seq_len,
            |b, &seq_len| {
                const HEAD_SIZE: usize = 128;
                const NUM_HEADS: usize = 8;
                const NUM_KV_HEADS: usize = 2;
                const BLOCK_SIZE: usize = 16;
                let scale: f32 = 1.0 / (HEAD_SIZE as f32).sqrt();
                let num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

                let q_data = vec![1.0f32; NUM_HEADS * HEAD_SIZE];
                let k_data = vec![1.0f32; NUM_KV_HEADS * num_blocks * HEAD_SIZE * BLOCK_SIZE];
                let v_data = vec![1.0f32; NUM_KV_HEADS * num_blocks * HEAD_SIZE * BLOCK_SIZE];
                let block_table: Vec<i32> = (0..num_blocks as i32).collect();

                let q_f16: Vec<u16> = q_data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let k_f16: Vec<u16> = k_data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let v_f16: Vec<u16> = v_data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();

                let q_buffer = mtl4_dispatch::shared_slice(&device, &q_f16);
                let k_buffer = mtl4_dispatch::shared_slice(&device, &k_f16);
                let v_buffer = mtl4_dispatch::shared_slice(&device, &v_f16);
                let block_table_buffer = mtl4_dispatch::shared_slice(&device, &block_table);
                let output_buffer =
                    mtl4_dispatch::shared_zeroed(&device, NUM_HEADS * HEAD_SIZE * 2);

                #[repr(C)]
                #[derive(Clone, Copy)]
                struct MultiHeadAttentionParams {
                    seq_len: u32,
                    head_size: u32,
                    num_heads: u32,
                    num_kv_heads: u32,
                    scale: f32,
                    block_size: u32,
                    max_num_blocks: u32,
                    kv_block_stride: u32,
                    kv_head_stride: u32,
                }

                let kv_head_stride = HEAD_SIZE as u32 * BLOCK_SIZE as u32;
                let params = MultiHeadAttentionParams {
                    seq_len: seq_len as u32,
                    head_size: HEAD_SIZE as u32,
                    num_heads: NUM_HEADS as u32,
                    num_kv_heads: NUM_KV_HEADS as u32,
                    scale,
                    block_size: BLOCK_SIZE as u32,
                    max_num_blocks: num_blocks as u32,
                    kv_block_stride: kv_head_stride,
                    kv_head_stride,
                };
                let params_buffer = mtl4_dispatch::shared_slice(&device, &[params]);

                let pipeline = build_pipeline(
                    &device,
                    "attention_multihead.metal",
                    "attention_multihead_paged",
                );

                b.iter(|| {
                    mtl4_dispatch::dispatch_threadgroups(
                        &device,
                        &pipeline,
                        &[
                            &q_buffer,
                            &k_buffer,
                            &v_buffer,
                            &block_table_buffer,
                            &output_buffer,
                            &params_buffer,
                        ],
                        MTLSize {
                            width: NUM_HEADS,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                    black_box(&output_buffer);
                });
            },
        );

        // Optimized
        group.bench_with_input(
            BenchmarkId::new("optimized", seq_len),
            seq_len,
            |b, &seq_len| {
                const HEAD_SIZE: usize = 128;
                const NUM_HEADS: usize = 8;
                const NUM_KV_HEADS: usize = 2;
                const BLOCK_SIZE: usize = 16;
                let scale: f32 = 1.0 / (HEAD_SIZE as f32).sqrt();
                let num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

                let q_data = vec![1.0f32; NUM_HEADS * HEAD_SIZE];
                let k_data = vec![1.0f32; NUM_KV_HEADS * num_blocks * HEAD_SIZE * BLOCK_SIZE];
                let v_data = vec![1.0f32; NUM_KV_HEADS * num_blocks * HEAD_SIZE * BLOCK_SIZE];
                let block_table: Vec<i32> = (0..num_blocks as i32).collect();

                let q_f16: Vec<u16> = q_data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let k_f16: Vec<u16> = k_data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let v_f16: Vec<u16> = v_data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();

                let q_buffer = mtl4_dispatch::shared_slice(&device, &q_f16);
                let k_buffer = mtl4_dispatch::shared_slice(&device, &k_f16);
                let v_buffer = mtl4_dispatch::shared_slice(&device, &v_f16);
                let block_table_buffer = mtl4_dispatch::shared_slice(&device, &block_table);
                let output_buffer =
                    mtl4_dispatch::shared_zeroed(&device, NUM_HEADS * HEAD_SIZE * 2);

                #[repr(C)]
                #[derive(Clone, Copy)]
                struct MultiHeadAttentionParams {
                    seq_len: u32,
                    head_size: u32,
                    num_heads: u32,
                    num_kv_heads: u32,
                    scale: f32,
                    block_size: u32,
                    max_num_blocks: u32,
                    kv_block_stride: u32,
                    kv_head_stride: u32,
                }

                let kv_head_stride = HEAD_SIZE as u32 * BLOCK_SIZE as u32;
                let params = MultiHeadAttentionParams {
                    seq_len: seq_len as u32,
                    head_size: HEAD_SIZE as u32,
                    num_heads: NUM_HEADS as u32,
                    num_kv_heads: NUM_KV_HEADS as u32,
                    scale,
                    block_size: BLOCK_SIZE as u32,
                    max_num_blocks: num_blocks as u32,
                    kv_block_stride: kv_head_stride,
                    kv_head_stride,
                };
                let params_buffer = mtl4_dispatch::shared_slice(&device, &[params]);

                let pipeline = build_pipeline(
                    &device,
                    "attention_multihead_optimized.metal",
                    "attention_multihead_paged_optimized",
                );

                b.iter(|| {
                    mtl4_dispatch::dispatch_threadgroups(
                        &device,
                        &pipeline,
                        &[
                            &q_buffer,
                            &k_buffer,
                            &v_buffer,
                            &block_table_buffer,
                            &output_buffer,
                            &params_buffer,
                        ],
                        MTLSize {
                            width: NUM_HEADS,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                    black_box(&output_buffer);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_single_head_attention,
    benchmark_multihead_attention,
    benchmark_optimized_attention
);
criterion_main!(benches);
