// SPDX-License-Identifier: Apache-2.0
//! AOT-compile MSL source to a `.metallib` blob via `xcrun metal -c`
//! and `xcrun metallib`. Shared by `scratchy-forward-compiler-macro` (proc-macro
//! expansion baking) and `scratchy-cost-sweep-metal` (benchmark of
//! the synthesized kernels) — and THE BAKE: every kernel of a shader that includes `baked.h`
//! compiled once per constant set a command carries, its constants compiled in ([`bake`]).
//!
//! Same flow as `scratchy-target-metal/build.rs`.
//!
//! Panics on `xcrun` failure — synth compile errors are build errors
//! that need to surface, not runtime soft-fail.

use std::collections::HashMap;
use std::process::Command;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;

use crate::interpreter::metal::SpecializedPipelines;
use crate::interpreter::metal::pipelines::gemm_pipeline;
use crate::specialized_pipeline_cache::{
    ComputePipelineState, ConstantType, ConstantValue, Device, PipelineKey,
    SpecializedPipelineCache,
};
use crate::stream::MetalStreamError;
use crate::tape::lowered::{
    BakedKernel, KernelId, LoweredCommand, LoweredMetalTape, MetalDtype, baked,
};

/// A library whose kernels compile only in the bake ([`bake`]).
pub struct BakedLibrary {
    pub name: &'static str,
    /// Whether it reaches MPP: it compiles with [`MPP_FLAGS`], as `build.rs` compiles such shaders.
    pub mpp: bool,
}

include!(concat!(env!("OUT_DIR"), "/baked_shaders_generated.rs"));

/// Per-call sequence so concurrent compiles of the SAME synth symbol get
/// distinct temp dirs. Many arches/models share synth kernels, and the
/// consolidated `scratchy-models` build script runs this from a rayon fan-out,
/// so `std::process::id()` is identical across the racing threads — without a
/// unique suffix they clobber each other's `.metal`/`.air` files (and one
/// thread's cleanup deletes another's dir mid-compile).
static SYNTH_SEQ: AtomicU64 = AtomicU64::new(0);

/// Compile `source` (written as `<symbol>.metal`, beside `sources` — the files it includes, by
/// relative path) to a metallib.
pub fn aot_compile_metallib(symbol: &str, source: &str, sources: &[(&str, &str)]) -> Vec<u8> {
    compile_metallib(symbol, source, sources, &[])
}

/// [`aot_compile_metallib`] with `flags` added to the MSL compile.
fn compile_metallib(
    symbol: &str,
    source: &str,
    sources: &[(&str, &str)],
    flags: &[&str],
) -> Vec<u8> {
    // Skip on non-macOS hosts (no `xcrun`). The synth metallibs are
    // only ever consumed by the metal backend.
    let host_os =
        std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| std::env::consts::OS.to_string());
    if host_os != "macos" {
        return Vec::new();
    }

    let tmp_dir = std::env::temp_dir().join(format!(
        "scratchy-synth-{}-{}-{}",
        symbol,
        std::process::id(),
        SYNTH_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&tmp_dir)
        .unwrap_or_else(|e| panic!("synth: create tmp dir {tmp_dir:?}: {e}"));

    let metal_path = tmp_dir.join(format!("{symbol}.metal"));
    // Only the final artifact needs an absolute path (it is read back
    // below). The compiler steps run with `tmp_dir` as their working
    // directory and take BARE filenames, so no build-specific path can
    // reach the emitted bytes — see the flag comment below.
    let metallib_path = tmp_dir.join(format!("{symbol}.metallib"));

    let main = format!("{symbol}.metal");
    for (path, text) in sources.iter().copied().chain([(main.as_str(), source)]) {
        let path = tmp_dir.join(path);
        std::fs::create_dir_all(path.parent().unwrap())
            .and_then(|()| std::fs::write(&path, text))
            .unwrap_or_else(|e| panic!("synth: write {path:?}: {e}"));
    }

    // Run the compiler INSIDE `tmp_dir` and hand it BARE filenames.
    // `-frecord-sources` bakes the source path it was given into the
    // blob's DEPF section, so an absolute path here puts this build's
    // PID and sequence number inside the emitted `&[u8]` — two
    // identical builds then differ by five lines and metal emission is
    // not byte-reproducible, which voids any bit-equality gate over the
    // generated code. Relative names record as `<symbol>.metal`.
    // The directory stays PID+seq unique: that is what keeps concurrent
    // compiles of the same symbol from clobbering each other, and it no
    // longer leaks into the output.
    let status = Command::new("xcrun")
        .current_dir(&tmp_dir)
        .args([
            "-sdk", "macosx", "metal", "-O3",
            // NO `-frecord-sources`. It attaches a source archive whose
            // SARC section records the compiler's WORKING DIRECTORY —
            // this build's PID-and-sequence temp dir — directly inside
            // the emitted `&[u8]`. Two identical builds then produce
            // different generated code, which voids any bit-equality
            // gate over metal emission. Dropping it
            // also shrinks each synth blob ~4x (13,976 -> 3,479 bytes
            // measured on a trivial kernel).
            //
            "-c",
        ])
        .args(flags)
        .arg(format!("{symbol}.metal"))
        .arg("-o")
        .arg(format!("{symbol}.air"))
        .status()
        .unwrap_or_else(|e| panic!("synth: spawn xcrun metal: {e}"));
    if !status.success() {
        panic!("synth: `xcrun metal` failed for `{symbol}` (source at {metal_path:?})");
    }

    let status = Command::new("xcrun")
        .current_dir(&tmp_dir)
        .args(["-sdk", "macosx", "metallib"])
        .arg(format!("{symbol}.air"))
        .arg("-o")
        .arg(format!("{symbol}.metallib"))
        .status()
        .unwrap_or_else(|e| panic!("synth: spawn xcrun metallib: {e}"));
    if !status.success() {
        panic!("synth: `xcrun metallib` failed for `{symbol}`");
    }

    let bytes = std::fs::read(&metallib_path)
        .unwrap_or_else(|e| panic!("synth: read {metallib_path:?}: {e}"));

    let _ = std::fs::remove_dir_all(&tmp_dir);
    bytes
}

/// `library`, when its kernels compile only in the bake ([`bake`]): their commands' constants
/// reach them compiled in.
pub fn baked_library(library: &str) -> Option<&'static BakedLibrary> {
    BAKED_LIBRARIES.iter().find(|l| l.name == library)
}

/// The key [`bake`] compiles `cmd`'s kernel under at activation `dtype`, when its library is baked.
pub fn bake_key(cmd: &LoweredCommand, dtype: MetalDtype) -> Option<PipelineKey> {
    let key = match (cmd.kernel, cmd.gemm_dims) {
        (KernelId::Gemm, Some(dims)) => gemm_pipeline(dtype, dims).ok()?.0,
        _ => PipelineKey::new(cmd.library, cmd.function, cmd.constants.to_vec()),
    };
    baked_library(key.library_name).map(|_| key)
}

/// What one [`bake`] compiled.
pub struct Bake {
    /// One metallib per key, in order.
    pub metallibs: Vec<Arc<[u8]>>,
    /// The keys this call compiled; an earlier call in this process had the rest.
    pub compiled: usize,
}

/// Compile the kernel each of `keys` names — `function` of a baked library, with the key's
/// constants as `constexpr`s (`shaders/baked.h`) — once per distinct key in this process, the
/// misses in parallel.
pub fn bake(keys: &[PipelineKey]) -> Bake {
    type Cell = Arc<OnceLock<Arc<[u8]>>>;
    static BAKED: OnceLock<Mutex<HashMap<PipelineKey, Cell>>> = OnceLock::new();
    let cells: Vec<Cell> = {
        let mut baked = BAKED.get_or_init(Mutex::default).lock().unwrap();
        keys.iter()
            .map(|k| baked.entry(k.clone()).or_default().clone())
            .collect()
    };
    let compiled = AtomicUsize::new(0);
    let metallibs = keys
        .par_iter()
        .zip(&cells)
        .map(|(key, cell)| {
            let bytes = cell.get_or_init(|| {
                compiled.fetch_add(1, Ordering::Relaxed);
                compile_baked(key).into()
            });
            bytes.clone()
        })
        .collect();
    Bake {
        metallibs,
        compiled: compiled.into_inner(),
    }
}

/// `keys`' kernels, baked, as the statics a baked tape holds
/// ([`ClassedTape::kernels`](crate::tape::lowered::ClassedTape::kernels)) — for kernels lowered or
/// dispatched outside the macro (kernel tests).
pub fn baked_kernels(keys: &[PipelineKey]) -> Vec<BakedKernel> {
    let bake = bake(keys);
    let kernels = keys.iter().zip(bake.metallibs);
    kernels
        .map(|(key, metallib)| BakedKernel {
            library: key.library_name,
            function: key.kernel_name,
            constants: baked(key.constants.clone()),
            metallib: baked(metallib.to_vec()),
        })
        .collect()
}

/// The standard shaders and every baked kernel `tapes` name at `dtype`, as the pool registers a
/// model's — for tapes lowered outside the macro (tests).
pub fn tape_pipelines(
    device: &Device,
    tapes: &[LoweredMetalTape],
    dtype: MetalDtype,
) -> Result<SpecializedPipelines, MetalStreamError> {
    let commands = tapes.iter().flat_map(|t| t.commands.iter());
    let keys: Vec<_> = commands
        .filter_map(|c| bake_key(&c.command, dtype))
        .collect();
    let cache = SpecializedPipelineCache::new(device.clone(), &[])?;
    cache.register_baked(&baked_kernels(&keys))?;
    Ok(SpecializedPipelines::new(Arc::new(cache)))
}

/// `key`'s pipeline from `cache`, its kernel baked first — for kernel tests that build several
/// pipelines through one cache.
pub fn baked_build(
    cache: &SpecializedPipelineCache,
    key: &PipelineKey,
) -> Result<ComputePipelineState, MetalStreamError> {
    if baked_library(key.library_name).is_some() {
        cache.register_baked(&baked_kernels(std::slice::from_ref(key)))?;
    }
    cache.get_or_build(key)
}

/// A kernel test's baked pipeline, held with the cache that built it: a pipeline used after its
/// cache (MTL4 compiler, library) is dropped computes garbage.
pub struct BakedPipeline {
    pipeline: ComputePipelineState,
    _cache: SpecializedPipelineCache,
}

impl std::ops::Deref for BakedPipeline {
    type Target = ComputePipelineState;
    fn deref(&self) -> &ComputePipelineState {
        &self.pipeline
    }
}

/// The pipeline of `library`'s kernel `function` with `constants`, baked and built the way the
/// pool builds it — for kernel tests.
pub fn baked_pipeline(
    device: &Device,
    library: &'static str,
    function: &str,
    constants: Vec<ConstantValue>,
) -> Result<BakedPipeline, MetalStreamError> {
    let key = PipelineKey::new(library, Box::leak(Box::from(function)), constants);
    let cache = SpecializedPipelineCache::new(device.clone(), &[])?;
    let pipeline = baked_build(&cache, &key)?;
    Ok(BakedPipeline {
        pipeline,
        _cache: cache,
    })
}

/// The Metal toolchain's identity — the compiler's `--version` and the SDK version — read once per
/// process: part of every kept compile's key, so an Xcode or SDK update recompiles.
fn toolchain() -> &'static str {
    static TOOLCHAIN: OnceLock<String> = OnceLock::new();
    TOOLCHAIN.get_or_init(|| {
        let run = |args: &[&str]| match Command::new("xcrun").args(args).output() {
            Ok(out) => String::from_utf8_lossy(&out.stdout).into_owned(),
            Err(e) => format!("xcrun {args:?}: {e}"),
        };
        run(&["-sdk", "macosx", "metal", "--version"])
            + &run(&["-sdk", "macosx", "--show-sdk-version"])
    })
}

/// `key`'s kernel, compiled alone with its constants defined, with the flags `build.rs` compiles
/// its shader with.
fn compile_baked(key: &PipelineKey) -> Vec<u8> {
    let (library, function) = (key.library_name, key.kernel_name);
    let lib = baked_library(library).unwrap_or_else(|| panic!("bake: `{library}` is not baked"));
    let mut main = String::new();
    for c in &key.constants {
        let (slot, literal) = (c.index, msl_literal(c));
        main += &format!(
            "#define SCRATCHY_CONSTANT_{slot} {literal}\n#define SCRATCHY_SET_{slot} ~, 1\n"
        );
    }
    main += &format!("#define SCRATCHY_KERNEL_{function} ~, 1\n#include \"{library}.metal\"\n");
    let mut flags = vec!["-I", "."];
    flags.extend(MPP_FLAGS.iter().copied().filter(|_| lib.mpp));
    // In a build (`OUT_DIR`), the compile is kept beside the build's other outputs, keyed by
    // every input of it — the Metal toolchain, its source, flags and every file it can include —
    // so a re-expansion that changed none of them reads it back instead of compiling it again.
    let kept = std::env::var_os("OUT_DIR").map(|out| {
        let inputs = [toolchain(), main.as_str()]
            .into_iter()
            .chain(flags.iter().copied());
        let inputs = inputs.chain(BAKED_SOURCES.iter().flat_map(|(path, text)| [*path, *text]));
        let hash = inputs.fold(0xcbf2_9ce4_8422_2325u64, |h, s| {
            let h = s
                .bytes()
                .fold(h, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3));
            (h ^ 0xff).wrapping_mul(0x0100_0000_01b3)
        });
        std::path::Path::new(&out).join(format!("metal-bake/key-{hash:016x}.metallib"))
    });
    if let Some(bytes) = kept.as_ref().and_then(|path| std::fs::read(path).ok()) {
        return bytes;
    }
    let symbol = format!("{library}.{function}");
    let bytes = compile_metallib(&symbol, &main, BAKED_SOURCES, &flags);
    if let Some(path) = kept {
        let seq = SYNTH_SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = path.with_extension(format!("{}.{seq}.tmp", std::process::id()));
        std::fs::create_dir_all(path.parent().expect("metal-bake dir"))
            .and_then(|()| std::fs::write(&tmp, &bytes))
            .and_then(|()| std::fs::rename(&tmp, &path))
            .unwrap_or_else(|e| panic!("bake: keep {}: {e}", path.display()));
    }
    let named = bytes
        .windows(function.len())
        .any(|w| w == function.as_bytes());
    assert!(
        named || bytes.is_empty(),
        "bake: no instantiation of `{library}.metal` is named `{function}`"
    );
    bytes
}

/// `c` as an MSL literal of its declared type; a float as its shortest round-trip decimal, which
/// the compiler reads back to the same bits.
fn msl_literal(c: &ConstantValue) -> String {
    match c.ty {
        ConstantType::UInt => format!("{}u", c.bits),
        ConstantType::Int => format!("({})", c.bits as i32),
        ConstantType::Float => {
            let f = f32::from_bits(c.bits);
            assert!(f.is_finite(), "bake: constant slot {} is {f}", c.index);
            format!("({f:e}f)")
        }
        ConstantType::Bool => (c.bits != 0).to_string(),
    }
}
