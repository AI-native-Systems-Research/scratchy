// SPDX-License-Identifier: Apache-2.0
// Compiles all CUDA kernels for vllm-rs into static .a files via cudaforge.
// Outputs land in the shared cudaforge cache (~/.cache/cudaforge/scratchy-serving-cuda/)
// and are linked by scratchy-serving-cuda's build.rs.
//
// Keeping kernel compilation in this separate crate allows Docker to cache
// the (slow) kernel build as a distinct layer that only invalidates when
// .cu source files change, independent of Rust source changes.

// Reuse the same config module the lib crate exports so that symbol
// names emitted downstream match what this build.rs renders.
#[cfg(feature = "cuda")]
#[path = "src/flashinfer_config.rs"]
mod flashinfer_config;

fn main() {
    #[cfg(feature = "cuda")]
    cuda_build();

    #[cfg(not(feature = "cuda"))]
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn cuda_build() {
    // Check-only escape hatch (CI). The nvcc kernel compile — cutlass,
    // FlashAttention-3, and the flashinfer instantiations — takes tens of
    // minutes and several GB of RAM. `cargo check`/`clippy` never link the
    // resulting .a, so the Rust surface still type-checks without it. Do NOT
    // set this for anything that actually links a binary (test/run/release):
    // the link step would fail on undefined kernel symbols.
    println!("cargo:rerun-if-env-changed=SCRATCHY_SKIP_CUDA_KERNELS");
    if std::env::var_os("SCRATCHY_SKIP_CUDA_KERNELS").is_some() {
        println!(
            "cargo:warning=SCRATCHY_SKIP_CUDA_KERNELS set — skipping nvcc kernel \
             compilation (check/clippy only; do not link binaries from this build)"
        );
        return;
    }

    let mut rerun_files: Vec<String> = vec!["build.rs".to_string()];

    // Use the same fixed cache dir as scratchy-serving-cuda so the .a files are found
    // when scratchy-serving-cuda emits its rustc-link-search directive.
    let cache_dir = dirs::cache_dir()
        .expect("no cache directory found")
        .join("cudaforge")
        .join("scratchy-serving-cuda");
    std::fs::create_dir_all(&cache_dir).expect("Failed to create cudaforge cache dir");
    let cache_str = cache_dir.to_string_lossy().to_string();

    // 1. vllm fused kernels
    let vllm_sources = [
        "../csrc/layernorm_kernels.cu",
        "../csrc/activation_kernels.cu",
        "../csrc/pos_encoding_kernels.cu",
        "../csrc/cache_kernels.cu",
        "../csrc/qk_norm_rope_kernels.cu",
        "../csrc/moe_topk_kernels.cu",
        "../csrc/grouped_topk_noaux.cu",
        "../csrc/moe_align_kernels.cu",
        "../csrc/moe_align_block_size_kernels.cu",
        "../csrc/fused_moe_gemm_kernels.cu",
        "../csrc/moe_ops_kernels.cu",
        "../csrc/sampling_kernels.cu",
        "../csrc/gptq_dequant_kernels.cu",
        "../csrc/awq_dequant_kernels.cu",
        "../csrc/embedding_kernels.cu",
        "../csrc/bnb_dequant_kernels.cu",
        "../csrc/gdn_gating_kernels.cu",
        "../csrc/gdn_conv1d_kernels.cu",
        "../csrc/gdn_recurrent_kernels.cu",
        "../csrc/gdn_split_kernels.cu",
        "../csrc/gate_split_kernels.cu",
        "../csrc/mla_kernels.cu",
        "../csrc/dequant_gather_pages.cu",
        "../csrc/fp8_scale_kernels.cu",
        "../csrc/fp8_quant_kernels.cu",
        "../csrc/fp8_block_dequant_kernels.cu",
        "../csrc/affine_dequant_kernels.cu",
        "../csrc/sdpa_naive_kernels.cu",
        "../csrc/sdpa_flash_decode_kernels.cu",
        "../csrc/fp8_post_scale_kernels.cu",
        "../csrc/gather_last_dim_kernel.cu",
        "../csrc/precision_cast_kernels.cu",
    ];
    let vllm_watch = [
        "../csrc/moeTopKFuncs.cuh",
        "../csrc/vec_utils.cuh",
        "../csrc/fp8_utils.cuh",
    ];

    rerun_files.extend(vllm_sources.iter().map(|s| s.to_string()));
    rerun_files.extend(vllm_watch.iter().map(|s| s.to_string()));

    cudaforge::KernelBuilder::new()
        .out_dir(&cache_dir)
        .source_files(vllm_sources.iter().map(|s| s.to_string()))
        .watch(vllm_watch.iter().map(|s| s.to_string()))
        .include_path("../csrc")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-extended-lambda")
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .build_lib(format!("{}/libvllm_kernels.a", cache_str))
        .expect("Failed to build vllm_kernels");

    // 2. GGML quantized kernels
    cudaforge::KernelBuilder::new()
        .out_dir(&cache_dir)
        .source_files(vec!["../csrc/quantized.cu".to_string()])
        .arg("-O3")
        .arg("--use_fast_math")
        .build_lib(format!("{}/libggml_kernels.a", cache_str))
        .expect("Failed to build ggml_kernels");

    // 3. Marlin W4A16 fused GEMM kernels
    let marlin_sources = [
        "../csrc/marlin/marlin_gemm.cu",
        "../csrc/marlin/gptq_marlin_repack.cu",
        "../csrc/marlin/awq_marlin_repack.cu",
        "../csrc/marlin/sm80_kernel_float16_u4_float16.cu",
        "../csrc/marlin/sm80_kernel_bfloat16_u4_bfloat16.cu",
        "../csrc/marlin/sm80_kernel_float16_u4b8_float16.cu",
        "../csrc/marlin/sm80_kernel_bfloat16_u4b8_bfloat16.cu",
    ];
    let marlin_watch = [
        "../csrc/marlin/marlin.cuh",
        "../csrc/marlin/kernel.h",
        "../csrc/marlin/kernel_selector.h",
        "../csrc/marlin/marlin_template.h",
        "../csrc/marlin/marlin_mma.h",
        "../csrc/marlin/dequant.h",
        "../csrc/marlin/marlin_dtypes.cuh",
        "../csrc/core/scalar_type.hpp",
    ];

    rerun_files.extend(marlin_sources.iter().map(|s| s.to_string()));
    rerun_files.extend(marlin_watch.iter().map(|s| s.to_string()));

    cudaforge::KernelBuilder::new()
        .out_dir(&cache_dir)
        .source_files(marlin_sources.iter().map(|s| s.to_string()))
        .watch(marlin_watch.iter().map(|s| s.to_string()))
        .include_path("../csrc/marlin")
        .include_path("../csrc")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("-std=c++17")
        .arg("--expt-relaxed-constexpr")
        .build_lib(format!("{}/libmarlin_kernels.a", cache_str))
        .expect("Failed to build marlin_kernels");

    // 3b. Marlin MoE kernels
    let marlin_moe_sources = [
        "../csrc/marlin_moe/marlin_moe_gemm.cu",
        "../csrc/marlin_moe/sm80_kernel_float16_u4_float16.cu",
        "../csrc/marlin_moe/sm80_kernel_bfloat16_u4_bfloat16.cu",
        "../csrc/marlin_moe/sm80_kernel_float16_u4b8_float16.cu",
        "../csrc/marlin_moe/sm80_kernel_bfloat16_u4b8_bfloat16.cu",
    ];
    let marlin_moe_watch = [
        "../csrc/marlin_moe/kernel.h",
        "../csrc/marlin_moe/kernel_selector.h",
        "../csrc/marlin_moe/marlin_template.h",
    ];

    rerun_files.extend(marlin_moe_sources.iter().map(|s| s.to_string()));
    rerun_files.extend(marlin_moe_watch.iter().map(|s| s.to_string()));

    cudaforge::KernelBuilder::new()
        .out_dir(&cache_dir)
        .source_files(marlin_moe_sources.iter().map(|s| s.to_string()))
        .watch(marlin_moe_watch.iter().map(|s| s.to_string()))
        .include_path("../csrc/marlin_moe")
        .include_path("../csrc/marlin")
        .include_path("../csrc")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("-std=c++17")
        .arg("--expt-relaxed-constexpr")
        .build_lib(format!("{}/libmarlin_moe_kernels.a", cache_str))
        .expect("Failed to build marlin_moe_kernels");

    // 4. CUTLASS scaled_mm FP8 GEMM kernels
    build_cutlass_scaled_mm(&cache_str, &mut rerun_files);

    // 4a. CUTLASS grouped MoE GEMM (gemma4-moe expert GEMM): Sm90 wgmma+TMA
    //     ptr-array warp-specialized + Sm80 mma.sync grouped (runs on sm89/Ada).
    //     Built before flash attention so a memory-constrained host that OOMs
    //     compiling the FA kernels still produces this lib (it's a hard link
    //     dependency of scratchy-target-cuda).
    build_cutlass_moe_grouped(&cache_str, &mut rerun_files);

    // 5. FlashAttention-2 paged kernels
    build_flash_attention(&cache_str, &mut rerun_files);

    // 5a. FlashAttention-3 paged decode (Hopper-only). Skipped on
    //     pre-sm_90 build hosts since the kernels are sm_90a-only and
    //     nvcc would refuse to emit them on older toolchains.
    let arch = detect_cuda_arch();
    if arch.parse::<u32>().unwrap_or(0) >= 90 {
        build_flash_attention_3(&cache_str, &mut rerun_files);
    } else {
        println!(
            "cargo:warning=FA3 build skipped (CUDA_ARCH={} < 90, Hopper-only)",
            arch
        );
    }

    // 5b. FlashInfer per-tuple paged-attention shims. Rendered at build
    //     time from `templates/` into `$OUT_DIR/flashinfer_inst/` and
    //     compiled into `libflashinfer_attn.a`.
    build_flashinfer_attention(&cache_str, &mut rerun_files);

    // 6. CUTLASS standalone GEMM launchers (128×128 + 64×64 for solver dispatch)
    build_cutlass_standalone_gemm(&cache_str, &mut rerun_files);

    // 6b. CUTLASS fused Gate GEMM + SiLU + Mul epilogue (EVT) kernel.
    build_cutlass_gemm_silu_mul(&cache_str, &mut rerun_files);

    // 6c. CUTLASS fused GEMM + bias broadcast epilogue (EVT) kernel.
    build_cutlass_gemm_bias(&cache_str, &mut rerun_files);

    for f in &rerun_files {
        let path = std::path::Path::new(f);
        if let Ok(canonical) = path.canonicalize() {
            println!("cargo:rerun-if-changed={}", canonical.display());
        } else {
            println!("cargo:rerun-if-changed={}", f);
        }
    }
}

#[cfg(feature = "cuda")]
fn build_cutlass_scaled_mm(cache_dir: &str, rerun_files: &mut Vec<String>) {
    const CUTLASS_COMMIT: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";

    // C2X SM89 (Ada) path + C3X SM90 (Hopper) path. The SM90 .cu always defines
    // its extern "C" symbols (so the Rust FFI links everywhere), but its real
    // CUTLASS-3.x body is gated behind ENABLE_SCALED_MM_SM90, set only on sm_90+
    // build hosts. On those hosts cudaforge auto-detects sm_90a, which the
    // Hopper FP8 fast-accum schedules require.
    let scaled_mm_sources = vec![
        "../csrc/cutlass_scaled_mm/scaled_mm_c2x_sm89.cu".to_string(),
        "../csrc/cutlass_scaled_mm/scaled_mm_c3x_sm90.cu".to_string(),
    ];
    let scaled_mm_watch = [
        "../csrc/cutlass_scaled_mm/common.hpp",
        "../csrc/cutlass_scaled_mm/math.hpp",
        "../csrc/cutlass_scaled_mm/scaled_mm_c2x.cuh",
        "../csrc/cutlass_scaled_mm/scaled_mm_c2x_sm89_fp8_dispatch.cuh",
        "../csrc/cutlass_scaled_mm/scaled_mm_epilogues_c2x.hpp",
        "../csrc/cutlass_scaled_mm/broadcast_load_epilogue_c2x.hpp",
        "../csrc/cutlass_scaled_mm/scaled_mm_c3x_sm90_fp8_dispatch.cuh",
        "../csrc/cutlass_scaled_mm/scaled_mm_epilogues_c3x.hpp",
        "../csrc/cutlass_scaled_mm/broadcast_load_epilogue_c3x.hpp",
    ];

    rerun_files.extend(scaled_mm_sources.iter().cloned());
    rerun_files.extend(scaled_mm_watch.iter().map(|s| s.to_string()));

    let mut builder = cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(scaled_mm_sources)
        .watch(scaled_mm_watch.iter().map(|s| s.to_string()))
        .include_path("../csrc/cutlass_scaled_mm")
        .with_cutlass(Some(CUTLASS_COMMIT))
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("-Xcompiler")
        .arg("-fPIC");
    if detect_cuda_arch().parse::<u32>().unwrap_or(0) >= 90 {
        builder = builder.arg("-DENABLE_SCALED_MM_SM90=1");
    }
    builder
        .build_lib(format!("{}/libcutlass_scaled_mm.a", cache_dir))
        .expect("Failed to build cutlass_scaled_mm");
}

#[cfg(feature = "cuda")]
fn build_flash_attention(cache_dir: &str, rerun_files: &mut Vec<String>) {
    let fa_src = std::path::Path::new("../../../../third_party/vllm-flash-attn/src");
    let shim_dir = std::path::Path::new("../../../../third_party/flash-attn-shim");

    const CUTLASS_COMMIT: &str = "62750a2b75c802660e4894434dc55e839f322277";

    let hdims = ["32", "64", "96", "128", "192", "256"];
    let mut kernel_files: Vec<String> = Vec::new();
    for hdim in &hdims {
        for suffix in &[
            "fp16_sm80",
            "bf16_sm80",
            "fp16_causal_sm80",
            "bf16_causal_sm80",
        ] {
            kernel_files.push(
                fa_src
                    .join(format!("flash_fwd_hdim{}_{}.cu", hdim, suffix))
                    .to_string_lossy()
                    .into_owned(),
            );
            kernel_files.push(
                fa_src
                    .join(format!("flash_fwd_split_hdim{}_{}.cu", hdim, suffix))
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    kernel_files.push(shim_dir.join("ffi_shim.cu").to_string_lossy().into_owned());

    let watch_files: Vec<String> = vec![
        fa_src.join("flash_fwd_kernel.h"),
        fa_src.join("flash.h"),
        fa_src.join("flash_fwd_launch_template.h"),
        fa_src.join("static_switch.h"),
        fa_src.join("utils.h"),
        fa_src.join("kernel_traits.h"),
        fa_src.join("softmax.h"),
        fa_src.join("mask.h"),
        fa_src.join("rotary.h"),
        fa_src.join("block_info.h"),
        fa_src.join("dropout.h"),
        fa_src.join("namespace_config.h"),
        fa_src.join("philox_unpack.cuh"),
        shim_dir.join("ffi_shim.cu"),
        shim_dir
            .join("compat")
            .join("ATen")
            .join("cuda")
            .join("CUDAGeneratorImpl.h"),
        shim_dir
            .join("compat")
            .join("ATen")
            .join("cuda")
            .join("detail")
            .join("UnpackRaw.cuh"),
        shim_dir
            .join("compat")
            .join("c10")
            .join("cuda")
            .join("CUDAException.h"),
    ]
    .into_iter()
    .map(|p| p.to_string_lossy().into_owned())
    .collect();

    rerun_files.extend(kernel_files.iter().cloned());
    rerun_files.extend(watch_files.iter().cloned());

    let compat_include = shim_dir.join("compat");

    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(kernel_files)
        .watch(watch_files)
        .include_path(compat_include.to_string_lossy().as_ref())
        .include_path(fa_src.to_string_lossy().as_ref())
        .with_cutlass(Some(CUTLASS_COMMIT))
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .build_lib(format!("{}/libvllm_flash_attn.a", cache_dir))
        .expect("Failed to build flash attention");
}

#[cfg(feature = "cuda")]
fn build_flash_attention_3(cache_dir: &str, rerun_files: &mut Vec<String>) {
    // FA3 is the Hopper-native FlashAttention. We compile the paged-KV
    // forward path for bf16, hdim128, sm_90a only — the slice we need for
    // Qwen-family decode. Other dtypes/hdims are excluded via the
    // FLASHATTENTION_DISABLE_* defines below; adding a new shape is just
    // dropping the matching instantiation file from
    // third_party/vllm-flash-attn-3/hopper/instantiations/ into
    // `kernel_files` and clearing the corresponding DISABLE flag.
    let fa3_src = std::path::Path::new("../../../../third_party/vllm-flash-attn-3/hopper");
    let shim_dir = std::path::Path::new("../../../../third_party/flash-attn-3-shim");

    const CUTLASS_COMMIT: &str = "62750a2b75c802660e4894434dc55e839f322277";

    let kernel_files: Vec<String> = [
        // forward instantiations (hdim128 bf16 paged, both Split=false and Split=true)
        fa3_src.join("instantiations/flash_fwd_hdim128_bf16_paged_sm90.cu"),
        fa3_src.join("instantiations/flash_fwd_hdim128_bf16_paged_split_sm90.cu"),
        // shared support .cu files
        fa3_src.join("flash_fwd_combine.cu"),
        fa3_src.join("flash_prepare_scheduler.cu"),
        // raw-ptr ABI shim
        shim_dir.join("ffi_shim.cu"),
    ]
    .into_iter()
    .map(|p| p.to_string_lossy().into_owned())
    .collect();

    // Watch every header so nvcc reruns when the vendored sources change.
    let mut watch_files: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(fa3_src) {
        for entry in entries.flatten() {
            let p = entry.path();
            if let Some(ext) = p.extension().and_then(|e| e.to_str())
                && (ext == "h" || ext == "hpp")
            {
                watch_files.push(p.to_string_lossy().into_owned());
            }
        }
    }
    watch_files.push(
        shim_dir
            .join("compat/c10/util/Exception.h")
            .to_string_lossy()
            .into_owned(),
    );

    rerun_files.extend(kernel_files.iter().cloned());
    rerun_files.extend(watch_files.iter().cloned());

    let compat_include = shim_dir.join("compat");

    // DISABLE flags trim the .o size: we only want hdim128 bf16 forward,
    // paged, no softcap. Without these, every instantiation file would
    // try to expand all hdims/dtypes via PAGEDKV_SWITCH and friends.
    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(kernel_files)
        .watch(watch_files)
        .include_path(compat_include.to_string_lossy().as_ref())
        .include_path(fa3_src.to_string_lossy().as_ref())
        .with_cutlass(Some(CUTLASS_COMMIT))
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-arch=sm_90a")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-DFLASHATTENTION_DISABLE_BACKWARD")
        .arg("-DFLASHATTENTION_DISABLE_DROPOUT")
        .arg("-DFLASHATTENTION_DISABLE_SOFTCAP")
        .arg("-DFLASHATTENTION_DISABLE_FP16")
        .arg("-DFLASHATTENTION_DISABLE_FP8")
        .arg("-DFLASHATTENTION_DISABLE_HDIM32")
        .arg("-DFLASHATTENTION_DISABLE_HDIM64")
        .arg("-DFLASHATTENTION_DISABLE_HDIM96")
        .arg("-DFLASHATTENTION_DISABLE_HDIM192")
        .arg("-DFLASHATTENTION_DISABLE_HDIM256")
        .arg("-DFLASHATTENTION_DISABLE_HDIMDIFF192")
        .arg("-DFLASHATTENTION_DISABLE_HDIMDIFF64")
        .arg("-DFLASHATTENTION_DISABLE_APPENDKV")
        .arg("-DFLASHATTENTION_DISABLE_LOCAL")
        .build_lib(format!("{}/libvllm_flash_attn_3.a", cache_dir))
        .expect("Failed to build flash attention 3");
}

#[cfg(feature = "cuda")]
fn build_flashinfer_attention(cache_dir: &str, rerun_files: &mut Vec<String>) {
    use flashinfer_config::FLASHINFER_CONFIG_SET;
    use minijinja::{Environment, context};

    // Upstream FlashInfer commit the shim + forked planner were written
    // against. Bumping this requires re-reading upstream's
    //   csrc/batch_attention_customize_config.jinja
    //   include/flashinfer/attention/scheduler.cuh  (fork source)
    // and reconciling the template text under `templates/`.
    const FLASHINFER_COMMIT: &str = "08ab45d67705b301ee66e63c6999c934c72dd41c";
    const CONFIG_TEMPLATE: &str = include_str!("templates/batch_attention_config.inc.j2");
    const SHIM_TEMPLATE: &str = include_str!("templates/flashinfer_shim.cu.j2");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let inst_dir = std::path::Path::new(&out_dir).join("flashinfer_inst");
    std::fs::create_dir_all(&inst_dir).expect("create flashinfer_inst dir");

    let mut env = Environment::new();
    env.add_template("cfg", CONFIG_TEMPLATE)
        .expect("add config template");
    env.add_template("shim", SHIM_TEMPLATE)
        .expect("add shim template");

    let mut cu_files: Vec<String> = Vec::new();
    for cfg in FLASHINFER_CONFIG_SET {
        let suffix = cfg.sym_suffix();
        let cfg_filename = format!("config_{}.inc", suffix);
        let cu_filename = format!("flashinfer_shim_{}.cu", suffix);

        let cfg_body = env
            .get_template("cfg")
            .unwrap()
            .render(context! {
                dtype_cpp => cfg.dtype.cpp_ty(),
                head_dim => cfg.head_dim,
                flashinfer_commit => FLASHINFER_COMMIT,
            })
            .expect("render config .inc");
        std::fs::write(inst_dir.join(&cfg_filename), cfg_body).expect("write config .inc");

        let shim_body = env
            .get_template("shim")
            .unwrap()
            .render(context! {
                sym_suffix => suffix,
                use_logits_soft_cap => cfg.use_logits_soft_cap,
                config_inc_filename => cfg_filename,
                flashinfer_commit => FLASHINFER_COMMIT,
            })
            .expect("render shim .cu");
        let cu_path = inst_dir.join(&cu_filename);
        std::fs::write(&cu_path, shim_body).expect("write shim .cu");
        cu_files.push(cu_path.to_string_lossy().into_owned());
    }

    // Template sources — re-render on edit. The rendered files under
    // $OUT_DIR are NOT added; they're regenerated every build and the
    // cudaforge per-object cache avoids recompilation when their content
    // is byte-identical.
    rerun_files.push("src/flashinfer_config.rs".to_string());
    rerun_files.push("templates/batch_attention_config.inc.j2".to_string());
    rerun_files.push("templates/flashinfer_shim.cu.j2".to_string());

    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(cu_files)
        .include_path(inst_dir.to_string_lossy().as_ref())
        .with_git_dependency(
            "flashinfer",
            "https://github.com/flashinfer-ai/flashinfer.git",
            FLASHINFER_COMMIT,
            vec!["include"],
            /*extra_paths=*/ vec![],
            /*recurse_submodules=*/ false,
        )
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .build_lib(format!("{}/libflashinfer_attn.a", cache_dir))
        .expect("Failed to build flashinfer_attn");
}

#[cfg(feature = "cuda")]
fn build_cutlass_standalone_gemm(cache_dir: &str, rerun_files: &mut Vec<String>) {
    // Same CUTLASS commit as scaled_mm — the standalone GEMM only uses
    // the CUTLASS 2.x device::Gemm interface, compatible with any recent commit.
    const CUTLASS_COMMIT: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";

    let sources = vec!["../csrc/cutlass_standalone_gemm.cu".to_string()];
    rerun_files.extend(sources.iter().cloned());

    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(sources)
        .with_cutlass(Some(CUTLASS_COMMIT))
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-relaxed-constexpr")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .build_lib(format!("{}/libcutlass_standalone_gemm.a", cache_dir))
        .expect("Failed to build cutlass_standalone_gemm");
}

#[cfg(feature = "cuda")]
fn build_cutlass_moe_grouped(cache_dir: &str, rerun_files: &mut Vec<String>) {
    // Grouped MoE GEMM: Sm90 (3.x ptr-array warp-specialized cooperative, wgmma+
    // TMA) + Sm80 (2.x DefaultGemmGrouped, mma.sync — runs on sm89/Ada). Same
    // CUTLASS commit as scaled_mm/standalone (4.2.1) — has the grouped ptr-array
    // API (group_array_problem_shape.hpp, KernelPtrArrayTmaWarpSpecialized*).
    // Both arch launchers always compile; the lib is single-arch-built so only
    // the matching arch's device code is emitted (Sm90 body gated on CUDA >= 12).
    const CUTLASS_COMMIT: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";

    let sources = vec!["../csrc/cutlass_moe_grouped.cu".to_string()];
    rerun_files.extend(sources.iter().cloned());

    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(sources)
        .with_cutlass(Some(CUTLASS_COMMIT))
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .build_lib(format!("{}/libcutlass_moe_grouped.a", cache_dir))
        .expect("Failed to build cutlass_moe_grouped");
}

#[cfg(feature = "cuda")]
fn build_cutlass_gemm_silu_mul(cache_dir: &str, rerun_files: &mut Vec<String>) {
    // Shares the scaled_mm CUTLASS commit (uses cutlass_2x_gemm + EVT
    // infrastructure from scaled_mm_c2x.cuh).
    const CUTLASS_COMMIT: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";

    let sources = vec!["../csrc/cutlass_gemm_silu_mul.cu".to_string()];
    let watch = [
        "../csrc/cutlass_silu_mul_epilogue.hpp",
        "../csrc/cutlass_scaled_mm/scaled_mm_c2x.cuh",
        "../csrc/cutlass_scaled_mm/common.hpp",
        "../csrc/cutlass_scaled_mm/math.hpp",
        "../csrc/cutlass_scaled_mm/scaled_mm_epilogues_c2x.hpp",
        "../csrc/cutlass_scaled_mm/broadcast_load_epilogue_c2x.hpp",
    ];

    rerun_files.extend(sources.iter().cloned());
    rerun_files.extend(watch.iter().map(|s| s.to_string()));

    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(sources)
        .watch(watch.iter().map(|s| s.to_string()))
        .include_path("../csrc")
        .include_path("../csrc/cutlass_scaled_mm")
        .with_cutlass(Some(CUTLASS_COMMIT))
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .build_lib(format!("{}/libcutlass_gemm_silu_mul.a", cache_dir))
        .expect("Failed to build cutlass_gemm_silu_mul");
}

#[cfg(feature = "cuda")]
fn build_cutlass_gemm_bias(cache_dir: &str, rerun_files: &mut Vec<String>) {
    const CUTLASS_COMMIT: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";

    let sources = vec!["../csrc/cutlass_gemm_bias.cu".to_string()];
    let watch = [
        "../csrc/cutlass_gemm_bias_epilogue.hpp",
        "../csrc/cutlass_scaled_mm/scaled_mm_c2x.cuh",
        "../csrc/cutlass_scaled_mm/common.hpp",
        "../csrc/cutlass_scaled_mm/math.hpp",
        "../csrc/cutlass_scaled_mm/scaled_mm_epilogues_c2x.hpp",
        "../csrc/cutlass_scaled_mm/broadcast_load_epilogue_c2x.hpp",
    ];

    rerun_files.extend(sources.iter().cloned());
    rerun_files.extend(watch.iter().map(|s| s.to_string()));

    cudaforge::KernelBuilder::new()
        .out_dir(cache_dir)
        .source_files(sources)
        .watch(watch.iter().map(|s| s.to_string()))
        .include_path("../csrc")
        .include_path("../csrc/cutlass_scaled_mm")
        .with_cutlass(Some(CUTLASS_COMMIT))
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .build_lib(format!("{}/libcutlass_gemm_bias.a", cache_dir))
        .expect("Failed to build cutlass_gemm_bias");
}

#[cfg(feature = "cuda")]
fn detect_cuda_arch() -> String {
    // Try CUDA_ARCH env var first, then probe via nvidia-smi.
    if let Ok(arch) = std::env::var("CUDA_ARCH") {
        return arch;
    }
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let cap = String::from_utf8_lossy(&out.stdout);
            let cap = cap.trim().lines().next().unwrap_or("8.9");
            cap.replace('.', "")
        }
        _ => "89".to_string(), // Default to sm89 (Ada Lovelace)
    }
}
