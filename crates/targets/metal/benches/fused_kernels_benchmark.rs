use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize};
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch;
use std::time::Duration;

/// Compile `fn_name` out of `src` into a compute pipeline. The fused
/// kernels take their `M`/`N`/`eps` scalars as `constant uint&`/`float&`
/// buffer arguments (not function constants), so nothing is baked here —
/// the scalars ride as `shared_u32`/`shared_f32` buffers at dispatch time.
fn build_pipeline(
    device: &mtl4_dispatch::Device,
    src: &str,
    fn_name: &str,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let opts = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opts))
        .expect("compile fused-kernel shader");
    let func = library
        .newFunctionWithName(&NSString::from_str(fn_name))
        .unwrap_or_else(|| panic!("missing fn {fn_name}"));
    device
        .newComputePipelineStateWithFunction_error(&func)
        .expect("pipeline")
}

fn benchmark_fused_add_rmsnorm(c: &mut Criterion) {
    let device = detect_device().expect("No Metal device found").device;
    let src = include_str!("../shaders/fused_add_rmsnorm.metal");

    let mut group = c.benchmark_group("add_rmsnorm_fusion");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    // Test different sizes: (batch_size, hidden_dim)
    for (m, n) in [(1, 4096), (4, 4096), (16, 4096), (32, 4096), (64, 4096)].iter() {
        let size_label = format!("{}x{}", m, n);

        // Benchmark fused kernel
        group.bench_with_input(
            BenchmarkId::new("fused", &size_label),
            &(*m, *n),
            |b, &(m, n)| {
                let eps = 1e-5f32;

                // Create test data
                let input: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01).collect();
                let residual: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005).collect();
                let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();

                let input_f16: Vec<u16> = input
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let residual_f16: Vec<u16> = residual
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let weight_f16: Vec<u16> = weight
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();

                let input_buf = mtl4_dispatch::shared_slice(&device, &input_f16);
                let residual_buf = mtl4_dispatch::shared_slice(&device, &residual_f16);
                let weight_buf = mtl4_dispatch::shared_slice(&device, &weight_f16);
                let output_buf = mtl4_dispatch::shared_zeroed(&device, (m * n * 2) as usize);
                // residual_out rides at buffer(4): the consecutive MTL4
                // binding has no gaps, so the optional slot the classic path
                // left null is a real (written) buffer here.
                let residual_out_buf = mtl4_dispatch::shared_zeroed(&device, (m * n * 2) as usize);

                // use_f16 + N%4==0 → the vec4 kernel (N rides as N/4).
                let n_param = n / 4;
                let pipeline = build_pipeline(&device, src, "fused_add_rmsnorm_f16_vec4");

                let m_buf = mtl4_dispatch::shared_u32(&device, m as u32);
                let n_buf = mtl4_dispatch::shared_u32(&device, n_param as u32);
                let eps_buf = mtl4_dispatch::shared_f32(&device, eps);

                let tg_size = (n_param as usize).min(1024);
                let grid = MTLSize {
                    width: m as usize,
                    height: 1,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: tg_size,
                    height: 1,
                    depth: 1,
                };

                b.iter(|| {
                    mtl4_dispatch::dispatch_threadgroups(
                        &device,
                        &pipeline,
                        &[
                            &input_buf,
                            &residual_buf,
                            &weight_buf,
                            &output_buf,
                            &residual_out_buf,
                            &m_buf,
                            &n_buf,
                            &eps_buf,
                        ],
                        grid,
                        tg,
                    );
                    black_box(&output_buf);
                });
            },
        );

        // Benchmark separate kernels (simulated - would need actual separate implementations)
        // For now, we'll just measure the fused version to establish baseline
    }
    group.finish();
}

fn benchmark_fused_gate_up_silu_mul(c: &mut Criterion) {
    let device = detect_device().expect("No Metal device found").device;
    let src = include_str!("../shaders/fused_gate_up_silu_mul.metal");

    let mut group = c.benchmark_group("gate_up_silu_mul_fusion");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    // Test different sizes: (batch_size, intermediate_dim)
    for (m, n) in [
        (1, 11008),
        (4, 11008),
        (16, 11008),
        (32, 11008),
        (64, 11008),
    ]
    .iter()
    {
        let size_label = format!("{}x{}", m, n);

        // Benchmark fused kernel (separate inputs)
        group.bench_with_input(
            BenchmarkId::new("fused_separate", &size_label),
            &(*m, *n),
            |b, &(m, n)| {
                let gate: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01 - 2.0).collect();
                let up: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005 + 1.0).collect();

                let gate_f16: Vec<u16> = gate
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let up_f16: Vec<u16> = up
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();

                let gate_buf = mtl4_dispatch::shared_slice(&device, &gate_f16);
                let up_buf = mtl4_dispatch::shared_slice(&device, &up_f16);
                let output_buf = mtl4_dispatch::shared_zeroed(&device, (m * n * 2) as usize);

                // use_f16 + N%4==0 → the vec4 kernel (N rides as N/4).
                let n_param = n / 4;
                let pipeline = build_pipeline(&device, src, "fused_gate_up_silu_mul_f16_vec4");

                let m_buf = mtl4_dispatch::shared_u32(&device, m as u32);
                let n_buf = mtl4_dispatch::shared_u32(&device, n_param as u32);

                let tg_size = (n as usize).min(1024);
                let grid = MTLSize {
                    width: m as usize,
                    height: 1,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: tg_size,
                    height: 1,
                    depth: 1,
                };

                b.iter(|| {
                    mtl4_dispatch::dispatch_threadgroups(
                        &device,
                        &pipeline,
                        &[&gate_buf, &up_buf, &output_buf, &m_buf, &n_buf],
                        grid,
                        tg,
                    );
                    black_box(&output_buf);
                });
            },
        );

        // Benchmark fused kernel (concatenated inputs)
        group.bench_with_input(
            BenchmarkId::new("fused_concat", &size_label),
            &(*m, *n),
            |b, &(m, n)| {
                let gate: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01 - 2.0).collect();
                let up: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005 + 1.0).collect();

                // Concatenate: [gate | up] for each row
                let mut gate_up = Vec::with_capacity((m * n * 2) as usize);
                for i in 0..m as usize {
                    gate_up.extend_from_slice(&gate[i * n as usize..(i + 1) * n as usize]);
                    gate_up.extend_from_slice(&up[i * n as usize..(i + 1) * n as usize]);
                }

                let gate_up_f16: Vec<u16> = gate_up
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();

                let gate_up_buf = mtl4_dispatch::shared_slice(&device, &gate_up_f16);
                let output_buf = mtl4_dispatch::shared_zeroed(&device, (m * n * 2) as usize);

                // use_f16 + N%4==0 → the concat vec4 kernel (N rides as N/4).
                let n_param = n / 4;
                let pipeline =
                    build_pipeline(&device, src, "fused_gate_up_silu_mul_concat_f16_vec4");

                let m_buf = mtl4_dispatch::shared_u32(&device, m as u32);
                let n_buf = mtl4_dispatch::shared_u32(&device, n_param as u32);

                let tg_size = (n as usize).min(1024);
                let grid = MTLSize {
                    width: m as usize,
                    height: 1,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: tg_size,
                    height: 1,
                    depth: 1,
                };

                b.iter(|| {
                    mtl4_dispatch::dispatch_threadgroups(
                        &device,
                        &pipeline,
                        &[&gate_up_buf, &output_buf, &m_buf, &n_buf],
                        grid,
                        tg,
                    );
                    black_box(&output_buf);
                });
            },
        );
    }
    group.finish();
}

fn benchmark_vectorization_impact(c: &mut Criterion) {
    let device = detect_device().expect("No Metal device found").device;
    let src = include_str!("../shaders/fused_add_rmsnorm.metal");

    let mut group = c.benchmark_group("vectorization_impact");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    // Test with dimensions divisible by 4 (vectorizable) vs not
    for (m, n, label) in [(16, 4096, "vectorizable"), (16, 4095, "non_vectorizable")].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(*m, *n),
            |b, &(m, n)| {
                let eps = 1e-5f32;

                let input: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01).collect();
                let residual: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.005).collect();
                let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();

                let input_f16: Vec<u16> = input
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let residual_f16: Vec<u16> = residual
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let weight_f16: Vec<u16> = weight
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();

                let input_buf = mtl4_dispatch::shared_slice(&device, &input_f16);
                let residual_buf = mtl4_dispatch::shared_slice(&device, &residual_f16);
                let weight_buf = mtl4_dispatch::shared_slice(&device, &weight_f16);
                let output_buf = mtl4_dispatch::shared_zeroed(&device, (m * n * 2) as usize);
                let residual_out_buf = mtl4_dispatch::shared_zeroed(&device, (m * n * 2) as usize);

                // use_f16: vec4 kernel when N%4==0 (N rides as N/4), else the
                // scalar kernel with the full N. This is the axis under test.
                let (fn_name, n_param) = if n % 4 == 0 {
                    ("fused_add_rmsnorm_f16_vec4", n / 4)
                } else {
                    ("fused_add_rmsnorm_f16", n)
                };
                let pipeline = build_pipeline(&device, src, fn_name);

                let m_buf = mtl4_dispatch::shared_u32(&device, m as u32);
                let n_buf = mtl4_dispatch::shared_u32(&device, n_param as u32);
                let eps_buf = mtl4_dispatch::shared_f32(&device, eps);

                let tg_size = (n_param as usize).min(1024);
                let grid = MTLSize {
                    width: m as usize,
                    height: 1,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: tg_size,
                    height: 1,
                    depth: 1,
                };

                b.iter(|| {
                    mtl4_dispatch::dispatch_threadgroups(
                        &device,
                        &pipeline,
                        &[
                            &input_buf,
                            &residual_buf,
                            &weight_buf,
                            &output_buf,
                            &residual_out_buf,
                            &m_buf,
                            &n_buf,
                            &eps_buf,
                        ],
                        grid,
                        tg,
                    );
                    black_box(&output_buf);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_fused_add_rmsnorm,
    benchmark_fused_gate_up_silu_mul,
    benchmark_vectorization_impact
);
criterion_main!(benches);
