// SPDX-License-Identifier: Apache-2.0
//! Ahead-of-time Metal shader compilation + kernel-instantiation codegen.
//!
//! Every `.metal` file under `shaders/` is compiled to a per-library
//! `.metallib` in `OUT_DIR` so the runtime can `new_library_with_data`
//! a precompiled blob instead of paying the MSL→AIR frontend cost on
//! every process start. Shaders are independent (each maps to one
//! `library_name` key in `SpecializedPipelineCache`), so we keep one
//! `.metallib` per source file rather than one bundle.
//!
//! Function-constant specialization still happens at runtime through
//! `library.get_function(name, Some(constants))`; the AoT step replaces
//! only the source-compile + link stages (steps 1-2 of the Metal
//! pipeline). Step 4 (AIR → GPU machine code) remains lazy per
//! pipeline state.
//!
//! Codegen: a few kernels (currently `attention_steel_paged`) are
//! template-instantiated for one combo per (dtype, geometry-knob).
//! The instantiation list is the single source of truth in this file
//! and is mirrored to two generated files in `OUT_DIR`:
//!
//!   - `attention_steel_paged_instantiations.h` — included by the
//!     matching `.metal` source so `xcrun metal -I OUT_DIR` picks it
//!     up at MSL-compile time.
//!   - `steel_paged_kernels_generated.rs` — `include!`-ed by
//!     `kernel_identity.rs` (via the re-export below) so the
//!     dispatcher's symbol-lookup table comes from the same list.
//!
//! Adding a head-dim is a one-line edit to `STEEL_PAGED_HEAD_DIMS`
//! below.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

/// HEAD_DIMs (BD template arg of `attention_paged<...>` in
/// `mlx_steel_attn/steel_attention_paged_kernel.h`) instantiated in
/// `attention_steel_paged.metal`. Must cover every `head_dim`
/// reachable through the runtime gate in
/// `scratchy-forward-compiler/.../lowering.rs::AttentionPrefillPaged`. Sorted
/// by typical model frequency so the generated symbol table stays
/// readable.
/// `(head_dim, BK)` pairs. BK is the K-seq tile height; it must be a whole
/// multiple of the paged block size (16). BK > 16 makes a K-tile span multiple
/// pages (`PagedBlockLoaderT::kPagesPerTile`), which roughly halves the
/// per-tile barrier + online-softmax-rescale overhead per doubling → faster
/// long-context prefill on the existing simdgroup MMA (helps pre-M5 too).
/// head_dim 128 uses BK=32; others stay at 16 until validated.
const STEEL_PAGED_HEAD_DIMS: &[(u32, u32)] = &[(64, 16), (96, 16), (128, 32), (256, 16)];

/// Activation dtypes the steel kernel is instantiated for. Tag is the
/// Rust/symbol-side spelling; type is the MSL spelling used in the
/// `INST_STEEL_PAGED` macro expansion.
const STEEL_PAGED_DTYPES: &[(&str, &str)] = &[("f16", "half"), ("bf16", "bfloat")];

/// Oldest macOS (major, minor) the metal backend runs on. The MPP shaders are
/// built for this deployment target (see `reaches_mpp`), and a metallib built
/// for a newer OS refuses to load, so the backend can't start below it.
const MIN_MACOS: (u32, u32) = (26, 2);

/// Fail the build, with a clear message, when the macOS SDK is older than
/// `MIN_MACOS`, rather than producing a binary that can't load its shaders.
fn require_min_macos_sdk() {
    let out = Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-version"])
        .output()
        .unwrap_or_else(|e| panic!("spawn `xcrun --show-sdk-version` failed: {e}"));
    let version = String::from_utf8_lossy(&out.stdout);
    let mut parts = version.trim().split('.').map(|p| {
        p.parse::<u32>()
            .unwrap_or_else(|e| panic!("unparseable macOS SDK version {version:?}: {e}"))
    });
    let sdk = (parts.next().unwrap_or(0), parts.next().unwrap_or(0));
    let (major, minor) = MIN_MACOS;
    if sdk < MIN_MACOS {
        panic!(
            "the metal backend needs macOS {major}.{minor} or newer and its Xcode SDK; \
             found macOS SDK {}",
            version.trim()
        );
    }
}

/// Whether `file` reaches `<MetalPerformancePrimitives/...>` through its
/// `#include "..."` graph, i.e. uses the MPP `matmul2d` cooperative-tensor
/// intrinsics (today: everything that includes `metal_nax.h`).
///
/// On SDK 26.5 / metalfe-32023.883, the offline `xcrun metal` frontend
/// miscompiles MPP `matmul2d` (each call reduces only half its K → the
/// `affine_qmm_t_nax_*` symbols come out ~95% wrong, worst_abs ~10.45)
/// **unless** the deployment target is `>=26.2`: below 26.2 the SDK 26.5
/// MPP headers use a broken destination-tensor shim; 26.2+ selects the
/// correct indexed-operand intrinsics. This is MLX issue #3586, fixed by
/// MLX PR #3622 (the fix is exactly `-mmacosx-version-min=26.2`).
///
/// So these shaders are compiled with the extra
/// `-fno-fast-math -mmacosx-version-min=26.2 -std=metal4.0` flags (math
/// mode Safe, what mlx's `MTLCompileOptions` use). Every other shader
/// keeps the plain `-O3` flags. Deriving the set from the includes, not a
/// hand-kept list, means a new MPP shader can't silently miss the flags.
fn reaches_mpp(file: &Path, include_dirs: &[&Path], seen: &mut HashSet<PathBuf>) -> bool {
    if !seen.insert(file.to_path_buf()) {
        return false;
    }
    let src = std::fs::read_to_string(file)
        .unwrap_or_else(|e| panic!("read {} failed: {e}", file.display()));
    src.lines()
        .filter_map(|l| l.trim_start().strip_prefix("#include"))
        .any(|inc| {
            let inc = inc.trim_start();
            if inc.starts_with("<MetalPerformancePrimitives/") {
                return true;
            }
            let Some(name) = inc.strip_prefix('"').and_then(|s| s.split('"').next()) else {
                return false;
            };
            std::iter::once(file.parent().unwrap())
                .chain(include_dirs.iter().copied())
                .map(|dir| dir.join(name))
                .find(|p| p.is_file())
                .is_some_and(|p| reaches_mpp(&p, include_dirs, seen))
        })
}

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=tests");

    // 🛑 METAL 4 ONLY — FOR ALL TIME. Fail the build if classic-MTL3 dispatch
    // reappears anywhere in this backend. Runs on EVERY host (before the
    // non-macOS early-return below) so Linux/CI builds enforce it too.
    guard_no_classic_mtl3(&manifest_dir);

    // Codegen runs on every host so the Rust side compiles on Linux/CUDA
    // pods even though no .metallib is produced there.
    write_steel_paged_instantiations_h(&out_dir);
    write_steel_paged_kernels_rs(&out_dir);

    // Non-macOS targets skip the toolchain entirely — `scratchy-target-metal`
    // itself is `cfg(target_os = "macos")` at the call sites that load
    // the metallibs, so emitting nothing here lets Linux/cuda builds
    // compile this crate without needing `xcrun`.
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() != "macos" {
        return;
    }

    require_min_macos_sdk();

    let shader_dir = manifest_dir.join("shaders");

    let mut entries: Vec<_> = std::fs::read_dir(&shader_dir)
        .unwrap_or_else(|e| panic!("read shaders/ failed: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("metal"))
        .collect();
    entries.sort();

    // Each shader is an independent compile, and this step gates every crate
    // downstream, so run up to cargo's `-j` of them at once. A panic in any
    // worker still fails the build: the scope re-raises it on exit.
    let jobs: usize = std::env::var("NUM_JOBS").unwrap().parse().unwrap();
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..jobs.min(entries.len()) {
            s.spawn(|| {
                while let Some(shader) = entries.get(next.fetch_add(1, Ordering::Relaxed)) {
                    compile_shader(shader, &out_dir, &shader_dir);
                }
            });
        }
    });
}

fn compile_shader(shader: &Path, out_dir: &Path, shader_dir: &Path) {
    let stem = shader.file_stem().unwrap().to_str().unwrap();
    let air = out_dir.join(format!("{stem}.air"));
    let metallib = out_dir.join(format!("{stem}.metallib"));

    // MSL → AIR. `-O3` and `-frecord-sources=flat` so debug
    // captures retain source mapping; matches what MLX ships.
    // `-I OUT_DIR` so codegen-emitted headers (e.g.
    // `attention_steel_paged_instantiations.h`) resolve.
    //
    // MPP/NAX shaders additionally need
    // `-fno-fast-math -mmacosx-version-min=26.2 -std=metal4.0` to
    // dodge the SDK-26.5 `matmul2d` miscompile (see `reaches_mpp`).
    // Without `-mmacosx-version-min=26.2` the embedded
    // `affine_qmm_t_nax_*` metallib is ~95% wrong.
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metal", "-O3", "-frecord-sources=flat"]);
    // Only compile the sampler's telemetry-spill params/entropy when the
    // `sampler-telemetry` feature is on, so a plain engine kernel is
    // byte-identical to before (see #ifdef in sampling.metal).
    if std::env::var_os("CARGO_FEATURE_SAMPLER_TELEMETRY").is_some() {
        cmd.arg("-DSCRATCHY_SAMPLER_TELEMETRY");
    }
    if reaches_mpp(shader, &[out_dir, shader_dir], &mut HashSet::new()) {
        let (major, minor) = MIN_MACOS;
        cmd.arg("-fno-fast-math")
            .arg(format!("-mmacosx-version-min={major}.{minor}"))
            .arg("-std=metal4.0");
    }
    let status = cmd
        .arg("-I")
        .arg(out_dir)
        // `-I shaders` so headers in subdirs (mlx_steel_attn/) can
        // include top-level shader headers like `metal_nax.h`.
        .arg("-I")
        .arg(shader_dir)
        .arg("-c")
        .arg(shader)
        .arg("-o")
        .arg(&air)
        .status()
        .unwrap_or_else(|e| panic!("spawn `xcrun metal` failed: {e}"));
    if !status.success() {
        panic!("`xcrun metal` failed for {}", shader.display());
    }

    // AIR → metallib.
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .arg(&air)
        .arg("-o")
        .arg(&metallib)
        .status()
        .unwrap_or_else(|e| panic!("spawn `xcrun metallib` failed: {e}"));
    if !status.success() {
        panic!("`xcrun metallib` failed for {}", shader.display());
    }
}

/// Emit the `INST_STEEL_PAGED(tag, type, bd)` lines that
/// `attention_steel_paged.metal` includes. The `INST_STEEL_PAGED`
/// macro is defined in the .metal file itself; this header carries
/// only the per-combo expansions.
fn write_steel_paged_instantiations_h(out_dir: &std::path::Path) {
    let mut s = String::from(
        "// SPDX-License-Identifier: Apache-2.0\n\
         // Auto-generated by `scratchy-target-metal/build.rs`.\n\
         // Edit `STEEL_PAGED_HEAD_DIMS` in build.rs, not here.\n\n",
    );
    for &(bd, bk) in STEEL_PAGED_HEAD_DIMS {
        for &(tag, ty) in STEEL_PAGED_DTYPES {
            s.push_str(&format!("INST_STEEL_PAGED({tag}, {ty}, {bk}, {bd})\n"));
        }
    }
    std::fs::write(out_dir.join("attention_steel_paged_instantiations.h"), s)
        .expect("write attention_steel_paged_instantiations.h");
}

/// Emit the Rust-side mirror: a slice of head-dims (for the runtime
/// dispatch gate) and a `(dtype_tag, head_dim) -> Option<&'static str>`
/// symbol lookup. `kernel_identity.rs` re-exports both.
fn write_steel_paged_kernels_rs(out_dir: &std::path::Path) {
    let mut s = String::from(
        "// SPDX-License-Identifier: Apache-2.0\n\
         // Auto-generated by `scratchy-target-metal/build.rs`.\n\
         // Edit `STEEL_PAGED_HEAD_DIMS` in build.rs, not here.\n\n",
    );
    s.push_str(
        "/// HEAD_DIMs for which `attention_steel_paged.metal` has a\n\
         /// kernel instantiation (BD template arg). Single source of\n\
         /// truth shared with the Metal compile via the generated\n\
         /// `attention_steel_paged_instantiations.h`.\n",
    );
    s.push_str("pub const STEEL_PAGED_HEAD_DIMS: &[u32] = &[");
    for (i, &(bd, _bk)) in STEEL_PAGED_HEAD_DIMS.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&bd.to_string());
    }
    s.push_str("];\n\n");

    s.push_str(
        "/// MSL symbol for `attention_steel_paged_<dtype>_bq32_bk<bk>_bd<head_dim>_wm4_wn1_bs16`.\n\
         /// Returns `None` when the combo isn't instantiated; caller\n\
         /// must fall back to the per-token SDPA path.\n\
         pub fn steel_paged_symbol(dtype_tag: &str, head_dim: u32) -> Option<&'static str> {\n\
         \x20   match (dtype_tag, head_dim) {\n",
    );
    for &(bd, bk) in STEEL_PAGED_HEAD_DIMS {
        for &(tag, _ty) in STEEL_PAGED_DTYPES {
            let sym = format!("attention_steel_paged_{tag}_bq32_bk{bk}_bd{bd}_wm4_wn1_bs16");
            s.push_str(&format!("        ({tag:?}, {bd}) => Some({sym:?}),\n"));
        }
    }
    s.push_str("        _ => None,\n    }\n}\n");

    std::fs::write(out_dir.join("steel_paged_kernels_generated.rs"), s)
        .expect("write steel_paged_kernels_generated.rs");
}

/// 🛑 **Metal 4 ONLY — FOR ALL TIME.** Fail the build if classic-MTL3 dispatch
/// reappears in the Metal backend.
///
/// The MTL4 path creates command buffers via `device.newCommandBuffer()` +
/// `MTL4CommandBuffer::beginCommandBufferWithAllocator` and NEVER calls
/// `MTLCommandQueue::commandBuffer()`, so the exact substring `.commandBuffer()`
/// is a precise, false-positive-free marker of classic MTL3. Any dispatch must
/// go through `crate::mtl4_dispatch` (tests / cost-sweep) or
/// `interpreter::metal::run_bucket_mtl4` (production). Do NOT reintroduce classic
/// command buffers — there is no exception (turboquant, the last holdout, was
/// ported 2026-06-29). This is the build-time enforcement of the METAL-4-only
/// rule; clippy's `disallowed-methods` (clippy.toml) is the semantic/editor twin.
fn guard_no_classic_mtl3(manifest_dir: &std::path::Path) {
    const BANNED: &str = ".commandBuffer()";
    let mut offenders = Vec::new();
    for root in ["src", "tests", "cost-sweep/src"] {
        scan_for_banned(&manifest_dir.join(root), BANNED, &mut offenders);
    }
    if !offenders.is_empty() {
        let sites = offenders.join("\n  ");
        panic!(
            "\n\n🛑🛑 BANNED: classic-MTL3 `{BANNED}` found in the Metal backend.\n\n\
             The metal backend is METAL 4 ONLY, for all time. MTL4 builds command\n\
             buffers with `device.newCommandBuffer()` + `beginCommandBufferWithAllocator`,\n\
             NEVER `queue.commandBuffer()`. Dispatch through `crate::mtl4_dispatch`\n\
             (tests/cost-sweep) or `run_bucket_mtl4` (production).\n\n\
             Offending site(s):\n  {sites}\n\n"
        );
    }
}

/// Recursively flag every non-comment line under `dir` whose code contains
/// `banned`. Only the text BEFORE a `//` on each line is inspected, so doc /
/// line comments mentioning the pattern in prose don't trip it.
fn scan_for_banned(dir: &std::path::Path, banned: &str, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return; // dir may not exist (e.g. no cost-sweep/) — nothing to scan
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_for_banned(&path, banned, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            let Ok(src) = std::fs::read_to_string(&path) else {
                continue;
            };
            for (i, line) in src.lines().enumerate() {
                let code = line.split("//").next().unwrap_or("");
                if code.contains(banned) {
                    out.push(format!("{}:{}", path.display(), i + 1));
                }
            }
        }
    }
}
