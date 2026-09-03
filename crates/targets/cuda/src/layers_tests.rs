#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    // Runtime extension traits — the `Linear::{load,forward}`,
    // `RmsNorm::load`, `Embedding::load`, `Bnb4bitLinear::forward` calls in
    // these tests resolve through the `*Ops` traits the loaders/forwards
    // relocated into (targets/cuda). `as _` since they're used for method
    // resolution only.
    use crate::layers::{Bnb4bitOps as _, EmbeddingOps as _, LinearOps as _, RmsNormOps as _};

    #[test]
    fn test_linear_dimensions() {
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[512, 4096], DType::BF16) };
        let linear = Linear::new(w, None);
        assert_eq!(linear.out_features(), 512);
        assert_eq!(linear.in_features(), 4096);
    }

    #[test]
    fn test_linear_with_bias() {
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[256, 128], DType::F16) };
        let b = unsafe { GpuTensor::new(0x2000 as *mut u8, &[256], DType::F16) };
        let linear = Linear::new(w, Some(b));
        assert!(linear.bias.is_some());
        assert_eq!(linear.out_features(), 256);
    }

    #[test]
    fn test_embedding_dimensions() {
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[32000, 4096], DType::BF16) };
        let emb = Embedding::new(w);
        assert_eq!(emb.vocab_size(), 32000);
        assert_eq!(emb.hidden_size(), 4096);
    }

    #[test]
    fn test_rms_norm_dimensions() {
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[4096], DType::BF16) };
        let norm = RmsNorm::new(w, 1e-5);
        assert_eq!(norm.hidden_size(), 4096);
        assert_eq!(norm.eps, 1e-5);
    }

    #[test]
    fn test_rms_norm_eps_values() {
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[128], DType::F16) };
        let norm = RmsNorm::new(w, 1e-6);
        assert!((norm.eps - 1e-6).abs() < 1e-10);
    }

    // GPU tests for layer loading and forward passes.
    #[cfg(feature = "cuda")]
    mod cuda_tests {
        use super::*;
        use crate::DType;
        use crate::driver;

        fn init_cuda() -> cudarc::driver::sys::CUstream {
            unsafe {
                driver::init().expect("CUDA init");
                let dev = driver::device_get(0).expect("device");
                let _ctx = driver::ctx_create(dev).expect("context");
                driver::stream_create().expect("stream")
            }
        }

        #[test]
        fn test_linear_load_from_safetensors() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            // Create a safetensors file with linear weight and bias.
            let weight_data: Vec<f32> = vec![1.0; 8]; // [2, 4]
            let bias_data: Vec<f32> = vec![0.5; 2]; // [2]
            let w_bytes: Vec<u8> = weight_data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let b_bytes: Vec<u8> = bias_data.iter().flat_map(|f| f.to_le_bytes()).collect();

            let tensors = vec![
                (
                    "proj.weight",
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        vec![2, 4],
                        &w_bytes,
                    )
                    .unwrap(),
                ),
                (
                    "proj.bias",
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        vec![2],
                        &b_bytes,
                    )
                    .unwrap(),
                ),
            ];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_single_file(&path, stream).unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };

            let linear = Linear::load(&mut gw, "proj").unwrap();
            assert_eq!(linear.out_features(), 2);
            assert_eq!(linear.in_features(), 4);
            assert!(linear.bias.is_some());

            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_linear_forward_f32() {
            let stream = init_cuda();
            unsafe {
                let mut arena = CachingAllocator::new();
                let mut cublas = CublasHandle::new(stream, &mut arena).unwrap();

                // Weight [2, 3] = [[1,0,0],[0,1,0]] (identity-ish)
                let host_w = driver::mem_alloc_host(24).unwrap();
                std::slice::from_raw_parts_mut(host_w as *mut f32, 6)
                    .copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
                let gpu_w = driver::mem_alloc(24).unwrap();
                driver::memcpy_htod_async(gpu_w, host_w, 24, stream).unwrap();

                let w = GpuTensor::new(gpu_w, &[2, 3], DType::F32);
                let linear = Linear::new(w, None);

                // Input [4, 3] = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
                let host_x = driver::mem_alloc_host(48).unwrap();
                std::slice::from_raw_parts_mut(host_x as *mut f32, 12).copy_from_slice(&[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ]);
                let gpu_x = driver::mem_alloc(48).unwrap();
                driver::memcpy_htod_async(gpu_x, host_x, 48, stream).unwrap();
                let x = GpuTensor::new(gpu_x, &[4, 3], DType::F32);
                let x_view = TensorView::from_raw(x);

                // Forward: x @ W^T = [4,3] @ [3,2] = [4,2]
                // Expected: [[1,2],[4,5],[7,8],[10,11]]
                let y = linear.forward(x_view, &mut cublas, &mut arena);
                assert_eq!(y.dim(0), 4);
                assert_eq!(y.dim(1), 2);

                let host_y = driver::mem_alloc_host(32).unwrap();
                driver::memcpy_dtoh_async(host_y, y.raw_ptr(), 32, stream).unwrap();
                driver::stream_synchronize(stream).unwrap();

                let result = std::slice::from_raw_parts(host_y as *const f32, 8);
                let expected = [1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0];
                for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (got - exp).abs() < 1e-3,
                        "linear forward mismatch at {i}: got {got}, expected {exp}"
                    );
                }

                driver::mem_free_host(host_w).unwrap();
                driver::mem_free_host(host_x).unwrap();
                driver::mem_free_host(host_y).unwrap();
                driver::mem_free(gpu_w).unwrap();
                driver::mem_free(gpu_x).unwrap();
                driver::stream_destroy(stream).unwrap();
            }
        }

        #[test]
        fn test_rms_norm_load() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let data: Vec<f32> = vec![1.0; 128];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let tensors = vec![(
                "norm.weight",
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![128], &bytes)
                    .unwrap(),
            )];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_single_file(&path, stream).unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };

            let norm = RmsNorm::load(&mut gw, "norm", 1e-5).unwrap();
            assert_eq!(norm.hidden_size(), 128);
            assert!((norm.eps - 1e-5).abs() < 1e-10);

            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_embedding_load() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let data: Vec<f32> = vec![0.0; 100 * 32]; // [100, 32]
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let tensors = vec![(
                "embed.weight",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![100, 32],
                    &bytes,
                )
                .unwrap(),
            )];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_single_file(&path, stream).unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };

            let emb = Embedding::load(&mut gw, "embed").unwrap();
            assert_eq!(emb.vocab_size(), 100);
            assert_eq!(emb.hidden_size(), 32);

            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        /// BNB 4-bit linear forward test: manually create NF4 packed data,
        /// run dequant + GEMM, compare against CPU reference.
        #[test]
        fn test_bnb4bit_linear_forward() {
            use crate::quant::NF4_CODE;

            let stream = init_cuda();
            unsafe {
                let mut alloc = crate::alloc::CachingAllocator::new();
                let mut cublas = crate::cublas::CublasHandle::new(stream, &mut alloc).unwrap();

                // Weight shape: [out=8, in=8] → 64 elements → 32 packed bytes.
                // Use blocksize=64 so all elements are in one block.
                let out_features = 8usize;
                let in_features = 8usize;
                let num_elements = out_features * in_features; // 64
                let num_packed = num_elements / 2; // 32
                let blocksize = 64usize;

                // Create packed bytes: each byte encodes 2 nibbles.
                // Use nibble values that give a recognizable pattern.
                // We'll use: row 0 = all nibble 15 (code[15]=1.0), row 1 = all nibble 0 (code[0]=-1.0), etc.
                // Each row is 8 elements = 4 packed bytes.
                let nibble_values: Vec<u8> = vec![
                    15, 15, 15, 15, 15, 15, 15, 15, // row 0: all 1.0
                    0, 0, 0, 0, 0, 0, 0, 0, // row 1: all -1.0
                    7, 7, 7, 7, 7, 7, 7, 7, // row 2: all 0.0
                    8, 8, 8, 8, 8, 8, 8, 8, // row 3: all 0.0796
                    15, 0, 15, 0, 15, 0, 15, 0, // row 4: alternating 1.0, -1.0
                    10, 10, 10, 10, 10, 10, 10, 10, // row 5: all 0.2461
                    5, 5, 5, 5, 5, 5, 5, 5, // row 6: all -0.1848
                    12, 12, 12, 12, 12, 12, 12, 12, // row 7: all 0.4407
                ];
                // Pack: BNB convention — hi nibble = first element, lo nibble = second element.
                // So packed[i] = second_nibble | (first_nibble << 4).
                let mut packed_bytes = vec![0u8; num_packed];
                for i in 0..num_packed {
                    let first = nibble_values[2 * i];
                    let second = nibble_values[2 * i + 1];
                    packed_bytes[i] = second | (first << 4);
                }

                // absmax: 1 block covering 64 elements, scale = 2.0
                let absmax_val = 2.0f32;
                let absmax_f32 = [absmax_val];

                // Compute expected dequantized weight on CPU.
                let mut expected_weight = vec![0.0f32; num_elements];
                for i in 0..num_elements {
                    expected_weight[i] = NF4_CODE[nibble_values[i] as usize] * absmax_val;
                }

                // Compute expected output: x @ W^T where x = [1, 8] all ones.
                // output[j] = sum_k(x[k] * W[j][k]) = sum_k(W[j][k])
                let mut expected_output = vec![0.0f32; out_features];
                for j in 0..out_features {
                    for k in 0..in_features {
                        expected_output[j] += expected_weight[j * in_features + k];
                    }
                }

                // Upload packed bytes to GPU.
                let packed_ptr = driver::mem_alloc(num_packed).unwrap();
                driver::memcpy_htod_async(packed_ptr, packed_bytes.as_ptr(), num_packed, stream)
                    .unwrap();
                let packed_gpu = GpuTensor::new(packed_ptr, &[num_packed], DType::U8);

                // Upload absmax to GPU.
                let absmax_ptr = driver::mem_alloc(4).unwrap();
                driver::memcpy_htod_async(absmax_ptr, absmax_f32.as_ptr() as *const u8, 4, stream)
                    .unwrap();
                let absmax_gpu = GpuTensor::new(absmax_ptr, &[1], DType::F32);

                // Upload NF4 code table to GPU.
                let code_ptr = driver::mem_alloc(64).unwrap();
                driver::memcpy_htod_async(code_ptr, NF4_CODE.as_ptr() as *const u8, 64, stream)
                    .unwrap();
                let code_gpu = GpuTensor::new(code_ptr, &[16], DType::F32);

                // Allocate dequant scratch buffer.
                let scratch_bytes = num_elements * 2; // BF16
                let scratch_ptr = driver::mem_alloc(scratch_bytes).unwrap();
                let dequant_scratch = GpuTensor::new(scratch_ptr, &[num_elements], DType::BF16);

                // Create input: [1, 8] all ones in BF16.
                let ones_bf16: Vec<u16> = vec![0x3F80; in_features]; // BF16 for 1.0
                let x_ptr = driver::mem_alloc(in_features * 2).unwrap();
                driver::memcpy_htod_async(
                    x_ptr,
                    ones_bf16.as_ptr() as *const u8,
                    in_features * 2,
                    stream,
                )
                .unwrap();
                let x = GpuTensor::new(x_ptr, &[1, in_features], DType::BF16);
                let x_view = TensorView::from_raw(x);

                // Build Bnb4bitLinear.
                let layer = Bnb4bitLinear {
                    packed_weight: packed_gpu,
                    absmax: absmax_gpu,
                    code: code_gpu,
                    dequant_scratch,
                    out_features,
                    in_features,
                    blocksize,
                    bias: None,
                };

                // Forward.
                let output = layer.forward(x_view, &mut cublas, &mut alloc, stream);
                let out_t = output.as_gpu_tensor();
                assert_eq!(out_t.dim(0), 1);
                assert_eq!(out_t.dim(1), out_features);

                // Read back output.
                let out_bytes = out_features * 2;
                let host_out = driver::mem_alloc_host(out_bytes).unwrap();
                driver::memcpy_dtoh_async(host_out, out_t.raw_ptr(), out_bytes, stream).unwrap();
                driver::stream_synchronize(stream).unwrap();

                let out_bf16 = std::slice::from_raw_parts(host_out as *const u16, out_features);
                let actual: Vec<f32> = out_bf16
                    .iter()
                    .map(|&bits| half::bf16::from_bits(bits).to_f32())
                    .collect();

                println!("Expected: {:?}", expected_output);
                println!("Actual:   {:?}", actual);

                for j in 0..out_features {
                    let diff = (actual[j] - expected_output[j]).abs();
                    assert!(
                        diff < 0.1,
                        "output[{j}]: expected {}, got {}, diff {}",
                        expected_output[j],
                        actual[j],
                        diff
                    );
                }

                // Also read back the dequantized weight to verify dequant independently.
                let dequant_bytes = num_elements * 2;
                let host_dequant = driver::mem_alloc_host(dequant_bytes).unwrap();
                driver::memcpy_dtoh_async(
                    host_dequant,
                    dequant_scratch.raw_ptr(),
                    dequant_bytes,
                    stream,
                )
                .unwrap();
                driver::stream_synchronize(stream).unwrap();

                let dequant_bf16 =
                    std::slice::from_raw_parts(host_dequant as *const u16, num_elements);
                let dequant_f32: Vec<f32> = dequant_bf16
                    .iter()
                    .map(|&bits| half::bf16::from_bits(bits).to_f32())
                    .collect();

                println!("Dequant[0..16]: {:?}", &dequant_f32[0..16]);
                println!("Expected[0..16]: {:?}", &expected_weight[0..16]);

                for i in 0..num_elements {
                    let diff = (dequant_f32[i] - expected_weight[i]).abs();
                    assert!(
                        diff < 0.01,
                        "dequant[{i}]: expected {}, got {}, diff {}",
                        expected_weight[i],
                        dequant_f32[i],
                        diff
                    );
                }

                driver::mem_free_host(host_out).unwrap();
                driver::mem_free_host(host_dequant).unwrap();
                driver::stream_destroy(stream).unwrap();
            }
        }

        /// Test fused QKV: concatenated packed bytes + absmax from 3 shards
        /// should produce same result as 3 separate matmuls.
        #[test]
        fn test_bnb4bit_fused_qkv_vs_separate() {
            use crate::quant::NF4_CODE;

            let stream = init_cuda();
            unsafe {
                let mut alloc = crate::alloc::CachingAllocator::new();
                let mut cublas = crate::cublas::CublasHandle::new(stream, &mut alloc).unwrap();

                // Simulate: q=[128, 64], k=[64, 64], v=[64, 64], blocksize=64
                let in_features = 64usize;
                let q_out = 128usize;
                let kv_out = 64usize;
                let total_out = q_out + 2 * kv_out; // 256
                let blocksize = 64usize;

                // Create distinct nibble patterns for each shard.
                // q: all nibble 15 (=1.0), k: all nibble 0 (=-1.0), v: all nibble 8 (=0.0796)
                let q_elements = q_out * in_features; // 8192
                let kv_elements = kv_out * in_features; // 4096

                let q_nibbles: Vec<u8> = vec![15; q_elements];
                let k_nibbles: Vec<u8> = vec![0; kv_elements];
                let v_nibbles: Vec<u8> = vec![8; kv_elements];

                // Pack each shard (BNB convention: hi nibble = first, lo nibble = second).
                fn pack_nibbles(nibbles: &[u8]) -> Vec<u8> {
                    nibbles.chunks(2).map(|c| c[1] | (c[0] << 4)).collect()
                }
                let q_packed = pack_nibbles(&q_nibbles);
                let k_packed = pack_nibbles(&k_nibbles);
                let v_packed = pack_nibbles(&v_nibbles);

                // Concat packed bytes.
                let mut fused_packed: Vec<u8> = Vec::new();
                fused_packed.extend_from_slice(&q_packed);
                fused_packed.extend_from_slice(&k_packed);
                fused_packed.extend_from_slice(&v_packed);

                // Each shard has its own absmax blocks.
                let q_blocks = q_elements / blocksize; // 128
                let kv_blocks = kv_elements / blocksize; // 64
                let q_absmax: Vec<f32> = vec![3.0; q_blocks];
                let k_absmax: Vec<f32> = vec![5.0; kv_blocks];
                let v_absmax: Vec<f32> = vec![7.0; kv_blocks];

                let mut fused_absmax: Vec<f32> = Vec::new();
                fused_absmax.extend_from_slice(&q_absmax);
                fused_absmax.extend_from_slice(&k_absmax);
                fused_absmax.extend_from_slice(&v_absmax);

                // Compute expected per-shard outputs on CPU.
                // x = [1, 64] all ones. Output = sum of each row.
                let q_val = NF4_CODE[15] * 3.0; // 1.0 * 3.0 = 3.0
                let k_val = NF4_CODE[0] * 5.0; // -1.0 * 5.0 = -5.0
                let v_val = NF4_CODE[8] * 7.0; // 0.0796 * 7.0 = 0.5572

                // Each output element = sum of in_features values of the same row.
                let q_out_val = q_val * in_features as f32; // 3.0 * 64 = 192.0
                let k_out_val = k_val * in_features as f32; // -5.0 * 64 = -320.0
                let v_out_val = v_val * in_features as f32; // ~35.67

                // Upload fused data to GPU.
                let packed_ptr = driver::mem_alloc(fused_packed.len()).unwrap();
                driver::memcpy_htod_async(
                    packed_ptr,
                    fused_packed.as_ptr(),
                    fused_packed.len(),
                    stream,
                )
                .unwrap();
                let packed_gpu = GpuTensor::new(packed_ptr, &[fused_packed.len()], DType::U8);

                let absmax_ptr = driver::mem_alloc(fused_absmax.len() * 4).unwrap();
                driver::memcpy_htod_async(
                    absmax_ptr,
                    fused_absmax.as_ptr() as *const u8,
                    fused_absmax.len() * 4,
                    stream,
                )
                .unwrap();
                let absmax_gpu = GpuTensor::new(absmax_ptr, &[fused_absmax.len()], DType::F32);

                let code_ptr = driver::mem_alloc(64).unwrap();
                driver::memcpy_htod_async(code_ptr, NF4_CODE.as_ptr() as *const u8, 64, stream)
                    .unwrap();
                let code_gpu = GpuTensor::new(code_ptr, &[16], DType::F32);

                let total_elements = total_out * in_features;
                let scratch_ptr = driver::mem_alloc(total_elements * 2).unwrap();
                let dequant_scratch = GpuTensor::new(scratch_ptr, &[total_elements], DType::BF16);

                // Input: [1, 64] all ones BF16.
                let ones_bf16: Vec<u16> = vec![0x3F80; in_features];
                let x_ptr = driver::mem_alloc(in_features * 2).unwrap();
                driver::memcpy_htod_async(
                    x_ptr,
                    ones_bf16.as_ptr() as *const u8,
                    in_features * 2,
                    stream,
                )
                .unwrap();
                let x = GpuTensor::new(x_ptr, &[1, in_features], DType::BF16);
                let x_view = TensorView::from_raw(x);

                let layer = Bnb4bitLinear {
                    packed_weight: packed_gpu,
                    absmax: absmax_gpu,
                    code: code_gpu,
                    dequant_scratch,
                    out_features: total_out,
                    in_features,
                    blocksize,
                    bias: None,
                };

                let output = layer.forward(x_view, &mut cublas, &mut alloc, stream);
                let out_t = output.as_gpu_tensor();
                assert_eq!(out_t.dim(0), 1);
                assert_eq!(out_t.dim(1), total_out);

                // Read back.
                let out_bytes = total_out * 2;
                let host_out = driver::mem_alloc_host(out_bytes).unwrap();
                driver::memcpy_dtoh_async(host_out, out_t.raw_ptr(), out_bytes, stream).unwrap();
                driver::stream_synchronize(stream).unwrap();

                let out_bf16 = std::slice::from_raw_parts(host_out as *const u16, total_out);
                let actual: Vec<f32> = out_bf16
                    .iter()
                    .map(|&b| half::bf16::from_bits(b).to_f32())
                    .collect();

                println!(
                    "Fused QKV output first 5 (Q region, expect ~{q_out_val}): {:?}",
                    &actual[0..5]
                );
                println!(
                    "Fused QKV output at Q/K boundary [{q_out}..{}] (K region, expect ~{k_out_val}): {:?}",
                    q_out + 5,
                    &actual[q_out..q_out + 5]
                );
                let v_start = q_out + kv_out;
                println!(
                    "Fused QKV output at K/V boundary [{v_start}..{}] (V region, expect ~{v_out_val}): {:?}",
                    v_start + 5,
                    &actual[v_start..v_start + 5]
                );

                // Check Q region.
                for (i, &val) in actual[..q_out].iter().enumerate() {
                    let diff = (val - q_out_val).abs();
                    assert!(
                        diff < 1.0,
                        "Q[{i}]: expected {q_out_val}, got {val}, diff {diff}",
                    );
                }
                // Check K region.
                for i in 0..kv_out {
                    let diff = (actual[q_out + i] - k_out_val).abs();
                    assert!(
                        diff < 1.0,
                        "K[{i}]: expected {k_out_val}, got {}, diff {diff}",
                        actual[q_out + i]
                    );
                }
                // Check V region.
                for i in 0..kv_out {
                    let diff = (actual[v_start + i] - v_out_val).abs();
                    assert!(
                        diff < 1.0,
                        "V[{i}]: expected {v_out_val}, got {}, diff {diff}",
                        actual[v_start + i]
                    );
                }

                driver::mem_free_host(host_out).unwrap();
                driver::stream_destroy(stream).unwrap();
            }
        }

        /// Test nibble order: BNB convention is hi nibble = first element, lo nibble = second.
        /// Uses distinct nibble values in each position to detect any swap.
        #[test]
        fn test_bnb4bit_nibble_order() {
            use crate::quant::NF4_CODE;

            let stream = init_cuda();
            unsafe {
                // 4 elements → 2 packed bytes, blocksize=4 (1 block), absmax=1.0.
                // Element layout: [A, B, C, D]
                // BNB packs: byte[0] = B_nibble | (A_nibble << 4)
                //            byte[1] = D_nibble | (C_nibble << 4)
                let nibble_a: u8 = 15; // code[15] =  1.0
                let nibble_b: u8 = 0; // code[0]  = -1.0
                let nibble_c: u8 = 8; // code[8]  =  0.0796
                let nibble_d: u8 = 10; // code[10] =  0.2461

                let packed: [u8; 2] = [
                    nibble_b | (nibble_a << 4), // byte 0: hi=A, lo=B
                    nibble_d | (nibble_c << 4), // byte 1: hi=C, lo=D
                ];
                let absmax: [f32; 1] = [1.0]; // single block covers all 4 elements

                let expected: [f32; 4] = [
                    NF4_CODE[nibble_a as usize], // 1.0
                    NF4_CODE[nibble_b as usize], // -1.0
                    NF4_CODE[nibble_c as usize], // 0.0796
                    NF4_CODE[nibble_d as usize], // 0.2461
                ];

                // Upload to GPU.
                let packed_ptr = driver::mem_alloc(2).unwrap();
                driver::memcpy_htod_async(packed_ptr, packed.as_ptr(), 2, stream).unwrap();
                let packed_gpu = GpuTensor::new(packed_ptr, &[2], DType::U8);

                let absmax_ptr = driver::mem_alloc(4).unwrap();
                driver::memcpy_htod_async(absmax_ptr, absmax.as_ptr() as *const u8, 4, stream)
                    .unwrap();
                let absmax_gpu = GpuTensor::new(absmax_ptr, &[1], DType::F32);

                let code_ptr = driver::mem_alloc(64).unwrap();
                driver::memcpy_htod_async(code_ptr, NF4_CODE.as_ptr() as *const u8, 64, stream)
                    .unwrap();
                let code_gpu = GpuTensor::new(code_ptr, &[16], DType::F32);

                let out_ptr = driver::mem_alloc(4 * 2).unwrap(); // 4 BF16 elements
                let out_gpu = GpuTensor::new(out_ptr, &[2, 2], DType::BF16);

                // Run dequant kernel.
                crate::kernels::dequantize_bnb4bit(
                    packed_gpu, absmax_gpu, code_gpu, out_gpu, 4, stream,
                );

                // Read back.
                let host = driver::mem_alloc_host(8).unwrap();
                driver::memcpy_dtoh_async(host, out_ptr, 8, stream).unwrap();
                driver::stream_synchronize(stream).unwrap();

                let out_bf16 = std::slice::from_raw_parts(host as *const u16, 4);
                let actual: Vec<f32> = out_bf16
                    .iter()
                    .map(|&b| half::bf16::from_bits(b).to_f32())
                    .collect();

                println!("Nibble order test: expected {:?}", expected);
                println!("Nibble order test: actual   {:?}", actual);

                for i in 0..4 {
                    let diff = (actual[i] - expected[i]).abs();
                    assert!(
                        diff < 0.01,
                        "element[{i}]: expected {}, got {}, diff {} — nibble order is wrong!",
                        expected[i],
                        actual[i],
                        diff
                    );
                }

                driver::mem_free_host(host).unwrap();
                driver::stream_destroy(stream).unwrap();
            }
        }

        /// Test double-quant absmax dequantization matches Python reference.
        /// Uses values from unsloth/Qwen3-0.6B-bnb-4bit layer 0 q_proj.
        #[test]
        fn test_double_quant_absmax_dequant() {
            use crate::weights::dequantize_double_quant_absmax;

            // From Python: absmax_u8[:5] = [61, 58, 54, 53, 54]
            // nested_quant_map = 256-element 8-bit dequant table from state2.code
            // nested_absmax = per-superblock scales from state2.absmax
            // nested_blocksize = 256, offset = 0.07990148663520813
            //
            // Python result (before offset): [-0.02827, -0.03709, -0.04886, -0.05180, -0.04886]
            // Python result (after offset):  [ 0.05163,  0.04281,  0.03104,  0.02810,  0.03104]

            // Simple synthetic test: 4 elements, nested_blocksize=2, offset=0.5
            let absmax_u8 = [3u8, 7, 1, 5];
            let nested_quant_map: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
            let nested_absmax = [2.0f32, 3.0]; // 2 superblocks of size 2
            let nested_blocksize = 2;
            let offset = 0.5f32;

            let result = dequantize_double_quant_absmax(
                &absmax_u8,
                &nested_quant_map,
                &nested_absmax,
                nested_blocksize,
                offset,
            );

            // Manual: result[0] = quant_map[3] * absmax[0/2] + 0.5 = 0.03 * 2.0 + 0.5 = 0.56
            //         result[1] = quant_map[7] * absmax[1/2] + 0.5 = 0.07 * 2.0 + 0.5 = 0.64
            //         result[2] = quant_map[1] * absmax[2/2] + 0.5 = 0.01 * 3.0 + 0.5 = 0.53
            //         result[3] = quant_map[5] * absmax[3/2] + 0.5 = 0.05 * 3.0 + 0.5 = 0.65
            let expected = [0.56f32, 0.64, 0.53, 0.65];

            assert_eq!(result.len(), 4);
            for i in 0..4 {
                let diff = (result[i] - expected[i]).abs();
                assert!(
                    diff < 1e-5,
                    "absmax[{i}]: expected {}, got {}, diff {}",
                    expected[i],
                    result[i],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_fp8_linear_dimensions() {
        // Test Fp8Linear struct construction with dummy tensors.
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[512, 4096], DType::Fp8E4m3) };
        let s = unsafe { GpuTensor::new(0x2000 as *mut u8, &[1], DType::F32) };
        let fp8 = Fp8Linear {
            weight: w,
            weight_scale: s,
            input_scale: None,
            bias: None,
            output_dtype: DType::BF16,
        };
        assert_eq!(fp8.out_features(), 512);
        assert_eq!(fp8.in_features(), 4096);
        assert!(fp8.input_scale.is_none());
        assert!(fp8.bias.is_none());
    }

    #[test]
    fn test_fp8_linear_with_static_scale() {
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[256, 128], DType::Fp8E4m3) };
        let ws = unsafe { GpuTensor::new(0x2000 as *mut u8, &[1], DType::F32) };
        let is = unsafe { GpuTensor::new(0x3000 as *mut u8, &[1], DType::F32) };
        let fp8 = Fp8Linear {
            weight: w,
            weight_scale: ws,
            input_scale: Some(is),
            bias: None,
            output_dtype: DType::BF16,
        };
        assert!(fp8.input_scale.is_some());
        assert_eq!(fp8.out_features(), 256);
    }

    #[test]
    fn test_linear_layer_fp8_variant() {
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[512, 4096], DType::Fp8E4m3) };
        let s = unsafe { GpuTensor::new(0x2000 as *mut u8, &[1], DType::F32) };
        let fp8 = Fp8Linear {
            weight: w,
            weight_scale: s,
            input_scale: None,
            bias: None,
            output_dtype: DType::BF16,
        };
        let layer = LinearLayer::Fp8(Box::new(fp8));
        assert_eq!(layer.out_features(), 512);
        assert_eq!(layer.in_features(), 4096);
    }

    #[test]
    fn test_fp8_block_linear_dimensions() {
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[512, 4096], DType::Fp8E4m3) };
        // block_size = [128, 128] → scale shape [4, 32]
        let s = unsafe { GpuTensor::new(0x2000 as *mut u8, &[4, 32], DType::F32) };
        let block = Fp8BlockLinear {
            weight: w,
            weight_scale_inv: s,
            block_size: [128, 128],
            bias: None,
            output_dtype: DType::BF16,
        };
        assert_eq!(block.out_features(), 512);
        assert_eq!(block.in_features(), 4096);
        assert_eq!(block.block_size, [128, 128]);
    }

    #[test]
    fn test_linear_layer_fp8_block_variant() {
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[256, 1024], DType::Fp8E4m3) };
        let s = unsafe { GpuTensor::new(0x2000 as *mut u8, &[2, 8], DType::F32) };
        let block = Fp8BlockLinear {
            weight: w,
            weight_scale_inv: s,
            block_size: [128, 128],
            bias: None,
            output_dtype: DType::BF16,
        };
        let layer = LinearLayer::Fp8Block(Box::new(block));
        assert_eq!(layer.out_features(), 256);
        assert_eq!(layer.in_features(), 1024);
    }

    #[test]
    fn test_fp8_block_linear_from_impl() {
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[64, 128], DType::Fp8E4m3) };
        let s = unsafe { GpuTensor::new(0x2000 as *mut u8, &[1, 1], DType::F32) };
        let block = Fp8BlockLinear {
            weight: w,
            weight_scale_inv: s,
            block_size: [64, 128],
            bias: None,
            output_dtype: DType::BF16,
        };
        let layer: LinearLayer = block.into();
        assert!(matches!(layer, LinearLayer::Fp8Block(_)));
        assert_eq!(layer.out_features(), 64);
        assert_eq!(layer.in_features(), 128);
    }
}

// ---------------------------------------------------------------------------
// CUDA tests for FP8 block-quantized linear
// ---------------------------------------------------------------------------
#[cfg(test)]
#[cfg(feature = "cuda")]
mod fp8_block_tests {
    use super::*;
    use crate::DType;
    use crate::driver;
    // `Fp8BlockLinear::forward` relocated into the `Fp8BlockOps` trait.
    use crate::layers::Fp8BlockOps as _;

    /// Convert f32 to FP8 E4M3 (1 sign + 4 exponent + 3 mantissa, bias=7).
    /// Only handles normal/subnormal positive values in the representable range.
    fn f32_to_fp8e4m3(val: f32) -> u8 {
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let frac = bits & 0x7F_FFFF;

        if exp == 0 && frac == 0 {
            return (sign << 7) as u8; // ±0
        }

        // Re-bias: FP32 bias=127, FP8E4M3 bias=7
        let new_exp = exp - 127 + 7;
        if new_exp <= 0 {
            // subnormal in fp8
            let shift = 1 - new_exp;
            let mantissa = (frac | 0x80_0000) >> (20 + shift as u32);
            return ((sign << 7) | mantissa) as u8;
        }
        if new_exp >= 15 {
            // max normal value (no inf/nan in e4m3fn): 0_1111_110 = 0x7E
            return ((sign << 7) | 0x7E) as u8;
        }
        let mantissa = frac >> 20; // top 3 bits of f32 mantissa
        ((sign << 7) | ((new_exp as u32) << 3) | mantissa) as u8
    }

    fn init_cuda() -> cudarc::driver::sys::CUstream {
        unsafe {
            driver::init().expect("CUDA init");
            let dev = driver::device_get(0).expect("device");
            let _ctx = driver::ctx_create(dev).expect("context");
            driver::stream_create().expect("stream")
        }
    }

    /// Test Fp8BlockLinear forward with a known weight/scale pattern.
    ///
    /// Weight [4, 4] FP8 with block_size=[2, 2], scale [2, 2].
    /// Verify dequant + GEMM produces correct output.
    #[test]
    fn test_fp8_block_linear_forward() {
        let stream = init_cuda();
        unsafe {
            let mut alloc = CachingAllocator::new();
            let mut cublas = CublasHandle::new(stream, &mut alloc).unwrap();

            let n = 4usize;
            let k = 4usize;
            let block_n = 2usize;
            let block_k = 2usize;

            // FP8 E4M3 weight: encode small integer values that are exact in FP8.
            // E4M3 can represent integers 0-8 exactly.
            // Weight matrix [4, 4] = [[1,2,1,2], [3,4,3,4], [1,2,1,2], [3,4,3,4]]
            let weight_vals: Vec<f32> = vec![
                1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0,
            ];
            let weight_fp8: Vec<u8> = weight_vals.iter().map(|&v| f32_to_fp8e4m3(v)).collect();

            // Scale [2, 2] — all 1.0 so dequant(w) = w * scale = w.
            let scale_vals: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

            // Upload weight.
            let gpu_w = driver::mem_alloc(n * k).unwrap();
            driver::memcpy_htod_async(gpu_w, weight_fp8.as_ptr() as *const u8, n * k, stream)
                .unwrap();
            let w = GpuTensor::new(gpu_w, &[n, k], DType::Fp8E4m3);

            // Upload scale.
            let gpu_s = driver::mem_alloc(4 * 4).unwrap(); // 4 f32s
            driver::memcpy_htod_async(gpu_s, scale_vals.as_ptr() as *const u8, 4 * 4, stream)
                .unwrap();
            let s = GpuTensor::new(gpu_s, &[2, 2], DType::F32);

            let layer = Fp8BlockLinear {
                weight: w,
                weight_scale_inv: s,
                block_size: [block_n, block_k],
                bias: None,
                output_dtype: DType::BF16,
            };

            // Input [1, 4] BF16 = [1, 1, 1, 1]
            let input_f32: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
            let input_bf16: Vec<u16> = input_f32
                .iter()
                .map(|&v| half::bf16::from_f32(v).to_bits())
                .collect();
            let gpu_x = driver::mem_alloc(k * 2).unwrap();
            driver::memcpy_htod_async(gpu_x, input_bf16.as_ptr() as *const u8, k * 2, stream)
                .unwrap();
            let x = GpuTensor::new(gpu_x, &[1, k], DType::BF16);
            let x_view = TensorView::from_raw(x);

            // Forward: y = x @ W^T, x=[1,4] all-ones, W=[4,4]
            // y[j] = sum_k(W[j][k]) = row sum
            // Row sums: [6, 14, 6, 14]
            let out = layer.forward(x_view, &mut cublas, &mut alloc, stream);
            assert_eq!(out.as_gpu_tensor().dim(0), 1);
            assert_eq!(out.as_gpu_tensor().dim(1), n);

            let host_out = driver::mem_alloc_host(n * 2).unwrap();
            driver::memcpy_dtoh_async(host_out, out.as_gpu_tensor().raw_ptr(), n * 2, stream)
                .unwrap();
            driver::stream_synchronize(stream).unwrap();

            let out_bf16 = std::slice::from_raw_parts(host_out as *const u16, n);
            let actual: Vec<f32> = out_bf16
                .iter()
                .map(|&bits| half::bf16::from_bits(bits).to_f32())
                .collect();
            let expected = [6.0, 14.0, 6.0, 14.0];
            for (i, (&got, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 0.5,
                    "fp8_block forward[{i}]: expected {exp}, got {got}"
                );
            }

            driver::mem_free_host(host_out).unwrap();
            driver::stream_destroy(stream).unwrap();
        }
    }

    /// Regression test: Fp8BlockLinear::forward must not leak
    /// the dequantized weight buffer. Running forward twice should NOT
    /// increase active_bytes (the caching allocator recycles the buffer).
    #[test]
    fn test_fp8_block_linear_no_leak() {
        let stream = init_cuda();
        unsafe {
            let mut alloc = CachingAllocator::new();
            let mut cublas = CublasHandle::new(stream, &mut alloc).unwrap();

            let n = 128usize;
            let k = 128usize;

            // Create trivial FP8 weight and scale.
            let weight_fp8: Vec<u8> = vec![0u8; n * k]; // all zeros
            let scale_vals: Vec<f32> = vec![1.0; 1]; // single block

            let gpu_w = driver::mem_alloc(n * k).unwrap();
            driver::memcpy_htod_async(gpu_w, weight_fp8.as_ptr() as *const u8, n * k, stream)
                .unwrap();
            let w = GpuTensor::new(gpu_w, &[n, k], DType::Fp8E4m3);

            let gpu_s = driver::mem_alloc(4).unwrap();
            driver::memcpy_htod_async(gpu_s, scale_vals.as_ptr() as *const u8, 4, stream).unwrap();
            let s = GpuTensor::new(gpu_s, &[1, 1], DType::F32);

            let layer = Fp8BlockLinear {
                weight: w,
                weight_scale_inv: s,
                block_size: [n, k],
                bias: None,
                output_dtype: DType::BF16,
            };

            // Input [1, k] BF16
            let input_bf16: Vec<u16> = vec![0u16; k];
            let gpu_x = driver::mem_alloc(k * 2).unwrap();
            driver::memcpy_htod_async(gpu_x, input_bf16.as_ptr() as *const u8, k * 2, stream)
                .unwrap();

            // First forward — establishes the allocator's pool.
            let x1 = TensorView::from_raw(GpuTensor::new(gpu_x, &[1, k], DType::BF16));
            let out1 = layer.forward(x1, &mut cublas, &mut alloc, stream);
            drop(out1);
            driver::stream_synchronize(stream).unwrap();
            let bytes_after_first = alloc.active_bytes();

            // Second forward — should reuse pools, NOT grow active_bytes.
            let x2 = TensorView::from_raw(GpuTensor::new(gpu_x, &[1, k], DType::BF16));
            let out2 = layer.forward(x2, &mut cublas, &mut alloc, stream);
            drop(out2);
            driver::stream_synchronize(stream).unwrap();
            let bytes_after_second = alloc.active_bytes();

            assert_eq!(
                bytes_after_first, bytes_after_second,
                "FP8 block forward leaked: {} bytes after first, {} after second",
                bytes_after_first, bytes_after_second
            );

            driver::stream_destroy(stream).unwrap();
        }
    }

    /// Test load_fp8_block_linear from safetensors.
    #[test]
    fn test_load_fp8_block_linear_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        // Create FP8 weight [8, 16] and scale [1, 2] (block_size = [8, 8]).
        let n = 8usize;
        let k = 16usize;
        let weight_data: Vec<u8> = vec![0u8; n * k]; // FP8 zeros
        let scale_data: Vec<f32> = vec![1.0, 1.0]; // [1, 2] scale
        let scale_bytes: Vec<u8> = scale_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let tensors = vec![
            (
                "proj.weight",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F8_E4M3,
                    vec![n, k],
                    &weight_data,
                )
                .unwrap(),
            ),
            (
                "proj.weight_scale_inv",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![1, 2],
                    &scale_bytes,
                )
                .unwrap(),
            ),
        ];
        safetensors::serialize_to_file(tensors, None, &path).unwrap();

        let stream = init_cuda();
        let mut gw = crate::weights::GpuWeights::from_single_file(&path, stream).unwrap();
        unsafe { driver::stream_synchronize(stream).unwrap() };

        let layer = crate::weights::load_fp8_block_linear(&mut gw, "proj", DType::BF16).unwrap();

        assert_eq!(layer.out_features(), n);
        assert_eq!(layer.in_features(), k);
        assert_eq!(layer.block_size, [8, 8]); // n/scale_rows=8/1=8, k/scale_cols=16/2=8
        assert!(layer.bias.is_none());

        unsafe { driver::stream_destroy(stream).unwrap() };
    }

    /// Test load_fused_fp8_block_linear from safetensors (2 shards).
    #[test]
    fn test_load_fused_fp8_block_linear_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        // Two shards: gate [8, 16] and up [8, 16], block_size=[8, 8].
        let n = 8usize;
        let k = 16usize;
        let weight_data: Vec<u8> = vec![0u8; n * k];
        let scale_data: Vec<f32> = vec![1.0, 1.0]; // [1, 2]
        let scale_bytes: Vec<u8> = scale_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let tensors = vec![
            (
                "gate.weight",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F8_E4M3,
                    vec![n, k],
                    &weight_data,
                )
                .unwrap(),
            ),
            (
                "gate.weight_scale_inv",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![1, 2],
                    &scale_bytes,
                )
                .unwrap(),
            ),
            (
                "up.weight",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F8_E4M3,
                    vec![n, k],
                    &weight_data,
                )
                .unwrap(),
            ),
            (
                "up.weight_scale_inv",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![1, 2],
                    &scale_bytes,
                )
                .unwrap(),
            ),
        ];
        safetensors::serialize_to_file(tensors, None, &path).unwrap();

        let stream = init_cuda();
        let mut gw = crate::weights::GpuWeights::from_single_file(&path, stream).unwrap();
        unsafe { driver::stream_synchronize(stream).unwrap() };

        let layer = crate::weights::load_fused_fp8_block_linear(
            &mut gw,
            &["gate".to_string(), "up".to_string()],
            DType::BF16,
            stream,
        )
        .unwrap();

        // Fused: [8+8, 16] = [16, 16]
        assert_eq!(layer.out_features(), 2 * n);
        assert_eq!(layer.in_features(), k);
        assert_eq!(layer.block_size, [8, 8]);
        // Scale should be [2, 2] (2 shards × 1 scale row each, 2 cols)
        assert_eq!(layer.weight_scale_inv.dim(0), 2);
        assert_eq!(layer.weight_scale_inv.dim(1), 2);

        unsafe { driver::stream_destroy(stream).unwrap() };
    }
}
