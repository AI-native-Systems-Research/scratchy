// SPDX-License-Identifier: Apache-2.0
//! Ahead-of-time CUDA kernel compilation, only under the `nvcc` feature.
//!
//! Every top-level `.cu` file under `kernels/` (not recursive, so the vendored
//! `kernels/thunderkittens/` tree is never compiled as a kernel) is compiled with nvcc for the
//! H100 (`sm_90a`) into one object, and the objects are archived into
//! `libscratchy_cuda_kernels.a` in `OUT_DIR`, which this crate links. Each kernel reaches Rust as
//! one `extern "C"` launcher (`src/kernels.rs`); its GPU code is the fatbin nvcc embeds in the
//! object, which the static CUDA runtime registers at program start.
//!
//! Without the `nvcc` feature this script spawns nothing and links nothing, so the crate builds on
//! any host with no CUDA toolchain. That is the build CI and the Mac use.
//!
//! No environment variable decides anything: the script reads only Cargo's own build-script
//! variables and `PATH`, which it searches for `nvcc` the way metal's `build.rs` runs `xcrun` from
//! `PATH`. The target arch is the `const` below, because the H100 is the only target.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The one compile target: Hopper with its architecture-specific instructions (WGMMA, TMA,
/// `setmaxnreg`). No `compute_90a` PTX is embedded, so the driver cannot JIT for another GPU.
const CUDA_ARCH: &str = "sm_90a";

/// Oldest CUDA toolkit (major, minor) the vendored ThunderKittens builds with.
const MIN_CUDA: (u32, u32) = (12, 8);

/// The static library the kernels are archived into.
const KERNEL_LIB: &str = "scratchy_cuda_kernels";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=PATH");

    // The toolchain-free tier: no nvcc, no GPU, nothing linked.
    if std::env::var_os("CARGO_FEATURE_NVCC").is_none() {
        return;
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = manifest_dir.join("kernels");

    let nvcc = find_nvcc();
    require_min_cuda(&nvcc);
    // `<root>/bin/nvcc`: the toolkit root is the executable's grandparent.
    let cuda_root = nvcc
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| panic!("nvcc at {} has no <root>/bin/ parent", nvcc.display()));

    let mut sources: Vec<_> = std::fs::read_dir(&kernel_dir)
        .unwrap_or_else(|e| panic!("read {} failed: {e}", kernel_dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("cu"))
        .collect();
    sources.sort();

    let objects: Vec<PathBuf> = sources
        .iter()
        .map(|src| compile_kernel(&nvcc, src, &kernel_dir, &out_dir))
        .collect();

    // Archive with nvcc itself, so no separate `ar` is needed.
    let lib = out_dir.join(format!("lib{KERNEL_LIB}.a"));
    let _ = std::fs::remove_file(&lib);
    let status = Command::new(&nvcc)
        .arg("--lib")
        .arg("-o")
        .arg(&lib)
        .args(&objects)
        .status()
        .unwrap_or_else(|e| panic!("spawn `{} --lib` failed: {e}", nvcc.display()));
    if !status.success() {
        panic!("`nvcc --lib` failed to archive {}", lib.display());
    }

    let lib64 = cuda_root.join("lib64");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={KERNEL_LIB}");
    println!("cargo:rustc-link-search=native={}", lib64.display());
    println!("cargo:rustc-link-lib=static=cudart_static");
    // The toolkit's stub `libcuda.so` satisfies the linker; at run time the real driver library is
    // found through `LD_LIBRARY_PATH` (`/usr/local/nvidia/lib64` on GKE).
    println!(
        "cargo:rustc-link-search=native={}",
        lib64.join("stubs").display()
    );
    println!("cargo:rustc-link-lib=dylib=cuda");
    // ThunderKittens' host code uses the C++ standard library; `cudart_static` needs rt/pthread/dl.
    for lib in ["stdc++", "rt", "pthread", "dl"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}

/// The first executable file named `nvcc` on `PATH`.
fn find_nvcc() -> PathBuf {
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .map(|dir| dir.join("nvcc"))
        .find(|candidate| is_executable(candidate))
        .unwrap_or_else(|| {
            panic!(
                "the `nvcc` feature of scratchy-target-cuda compiles the CUDA kernels and needs \
                 nvcc from CUDA {}.{} or newer on PATH, but no nvcc is on PATH. Install the CUDA \
                 toolkit {}.{}+ and put its bin/ on PATH, or build without the `nvcc` feature \
                 (the toolchain-free build CI uses).",
                MIN_CUDA.0, MIN_CUDA.1, MIN_CUDA.0, MIN_CUDA.1
            )
        })
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.metadata()
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

/// Fail the build, with a clear message, when `nvcc --version` reports a toolkit older than
/// `MIN_CUDA`, or output this script cannot read.
fn require_min_cuda(nvcc: &Path) {
    let out = Command::new(nvcc)
        .arg("--version")
        .output()
        .unwrap_or_else(|e| panic!("spawn `{} --version` failed: {e}", nvcc.display()));
    let text = String::from_utf8_lossy(&out.stdout);
    let version = parse_release(&text).unwrap_or_else(|| {
        panic!(
            "cannot read the CUDA release from `{} --version`; expected a line like \
             \"Cuda compilation tools, release 12.9, V12.9.86\", got: {text:?}",
            nvcc.display()
        )
    });
    if version < MIN_CUDA {
        panic!(
            "nvcc at {} is CUDA {}.{}; the vendored ThunderKittens (67845f5f) needs CUDA {}.{} or \
             newer. Upgrade the toolkit or build without the `nvcc` feature.",
            nvcc.display(),
            version.0,
            version.1,
            MIN_CUDA.0,
            MIN_CUDA.1
        );
    }
}

/// `(X, Y)` from the `release X.Y,` of nvcc's `--version` output.
fn parse_release(text: &str) -> Option<(u32, u32)> {
    let rest = text.split("release ").nth(1)?;
    let release = rest.split([',', ' ', '\n']).next()?;
    let (major, minor) = release.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// `nvcc -c kernels/<stem>.cu -o $OUT_DIR/<stem>.o` with the flags of ThunderKittens'
/// `kernels/common.mk` (`ARCH := SM90`, `CONFIG := standalone`), minus its link-only and
/// verbose-report flags, plus `-fPIC` for a library. nvcc's stderr is inherited.
fn compile_kernel(nvcc: &Path, src: &Path, kernel_dir: &Path, out_dir: &Path) -> PathBuf {
    let stem = src.file_stem().unwrap().to_str().unwrap();
    let obj = out_dir.join(format!("{stem}.o"));
    let tk = kernel_dir.join("thunderkittens");
    let status = Command::new(nvcc)
        .args(["-std=c++20", "-O3", "--use_fast_math"])
        .args(["--expt-extended-lambda", "--expt-relaxed-constexpr"])
        .arg("-forward-unknown-to-host-compiler")
        .args(["-Xcompiler=-Wno-psabi", "-Xcompiler=-fno-strict-aliasing"])
        .arg("-Xcompiler=-fPIC")
        .args(["-DNDEBUG", "-lineinfo", "-ftemplate-backtrace-limit=0"])
        .arg("-DKITTENS_SM90")
        .arg("-gencode")
        .arg(format!(
            "arch=compute_{a},code={CUDA_ARCH}",
            a = CUDA_ARCH.trim_start_matches("sm_")
        ))
        .arg("-I")
        .arg(tk.join("include"))
        .arg("-I")
        .arg(tk.join("prototype"))
        .arg("-I")
        .arg(kernel_dir)
        .arg("-c")
        .arg(src)
        .arg("-o")
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| panic!("spawn `{}` failed: {e}", nvcc.display()));
    if !status.success() {
        panic!("nvcc failed to compile {}", src.display());
    }
    obj
}
