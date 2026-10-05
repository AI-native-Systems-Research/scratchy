// SPDX-License-Identifier: Apache-2.0
//! THE BAKE: every kernel of a shader that includes `baked.h`, compiled once per constant set a
//! command carries, its constants compiled in ([`bake`]) — `xcrun metal -c` then
//! `xcrun metallib`, the flow `scratchy-target-metal/build.rs` compiles the standard shaders with.
//!
//! Panics on `xcrun` failure — a bake compile error is a build error that needs to surface, not a
//! runtime soft-fail.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
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
use crate::tape::constants::{TapeVariant, UnboundConstant};
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

/// Per-call sequence, so concurrent compiles in one process (the build script's rayon fan-out
/// shares one `std::process::id()`) get distinct temp dirs and never clobber or delete each other's
/// files.
static SEQ: AtomicU64 = AtomicU64::new(0);

/// FNV-1a over `parts`, each closed by a separator so `["ab", "c"]` and `["a", "bc"]` differ.
fn fnv<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> u64 {
    parts.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, part| {
        let h = (part.iter()).fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3));
        (h ^ 0xff).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The hash of every [`BAKED_SOURCES`] file, path and text.
fn sources_hash() -> u64 {
    static HASH: OnceLock<u64> = OnceLock::new();
    *HASH.get_or_init(|| {
        fnv(BAKED_SOURCES
            .iter()
            .flat_map(|(p, t)| [p.as_bytes(), t.as_bytes()]))
    })
}

/// Every [`BAKED_SOURCES`] file at its path, written once per process — under the build's
/// `OUT_DIR` in a build, else the temp dir — and named by their [`sources_hash`]: every bake
/// compile includes from it.
fn baked_sources() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let root = std::env::var_os("OUT_DIR").map_or_else(std::env::temp_dir, PathBuf::from);
        let dir = root.join(format!("metal-bake/src-{:016x}", sources_hash()));
        if !dir.exists() {
            let seq = SEQ.fetch_add(1, Ordering::Relaxed);
            let tmp = dir.with_extension(format!("{}.{seq}.tmp", std::process::id()));
            for (path, text) in BAKED_SOURCES {
                let path = tmp.join(path);
                std::fs::create_dir_all(path.parent().expect("a source's dir"))
                    .and_then(|()| std::fs::write(&path, text))
                    .unwrap_or_else(|e| panic!("bake: write {}: {e}", path.display()));
            }
            // Another process may have put its copy — the same files — in place first.
            if std::fs::rename(&tmp, &dir).is_err() {
                let _ = std::fs::remove_dir_all(&tmp);
            }
        }
        dir
    })
}

/// Compile `source` — including from [`baked_sources`] — with `flags` to a metallib; empty on a
/// host with no Metal toolchain (only the metal backend reads it).
fn compile_metallib(name: &str, source: &str, flags: &[&str]) -> Vec<u8> {
    let host_os =
        std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| std::env::consts::OS.to_string());
    if host_os != "macos" {
        return Vec::new();
    }
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let tmp_dir = std::env::temp_dir().join(format!("scratchy-bake-{name}-{pid}-{seq}"));
    std::fs::create_dir_all(&tmp_dir)
        .and_then(|()| std::fs::write(tmp_dir.join(format!("{name}.metal")), source))
        .unwrap_or_else(|e| panic!("bake: write {}: {e}", tmp_dir.display()));
    // The compiler runs INSIDE `tmp_dir` on BARE file names and without `-frecord-sources`, whose
    // source archive records the working directory: no build-specific path reaches the bytes, so
    // identical inputs bake identical metallibs.
    let run = |tool: &str, args: &[&str]| {
        let status = Command::new("xcrun")
            .current_dir(&tmp_dir)
            .args(["-sdk", "macosx", tool])
            .args(args)
            .status()
            .unwrap_or_else(|e| panic!("bake: spawn xcrun {tool}: {e}"));
        assert!(
            status.success(),
            "bake: `xcrun {tool}` failed for `{name}` (in {tmp_dir:?})"
        );
    };
    let include = baked_sources().to_str().expect("a UTF-8 bake source dir");
    let (src, air) = (format!("{name}.metal"), format!("{name}.air"));
    let lib = format!("{name}.metallib");
    let mut args = vec!["-O3", "-c", "-I", include];
    args.extend(flags);
    args.extend([src.as_str(), "-o", air.as_str()]);
    run("metal", &args);
    run("metallib", &[air.as_str(), "-o", lib.as_str()]);
    let bytes = std::fs::read(tmp_dir.join(&lib))
        .unwrap_or_else(|e| panic!("bake: read {}: {e}", tmp_dir.join(&lib).display()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    bytes
}

/// `library`, when its kernels compile only in the bake ([`bake`]): their commands' constants
/// reach them compiled in.
pub fn baked_library(library: &str) -> Option<&'static BakedLibrary> {
    BAKED_LIBRARIES.iter().find(|l| l.name == library)
}

/// The key [`bake`] compiles `cmd`'s kernel under for tape `variant` at activation `dtype`, when its
/// library is baked: its variant-bound constants take the variant's values.
pub fn bake_key(
    cmd: &LoweredCommand,
    dtype: MetalDtype,
    variant: TapeVariant,
) -> Result<Option<PipelineKey>, UnboundConstant> {
    let key = match (cmd.kernel, cmd.gemm_dims) {
        (KernelId::Gemm, Some(dims)) => match gemm_pipeline(dtype, dims) {
            Ok((key, _)) => key,
            Err(_) => return Ok(None),
        },
        _ => {
            let constants = ConstantValue::resolve(cmd.constants, variant)?;
            PipelineKey::new(cmd.library, cmd.function, constants)
        }
    };
    Ok(baked_library(key.library_name).map(|_| key))
}

/// One kernel [`bake`] compiled: the metallib holding it — its batch's — and its name there.
#[derive(Clone)]
pub struct BakedEntry {
    pub metallib: Arc<[u8]>,
    pub entry: String,
}

/// What one [`bake`] compiled.
pub struct Bake {
    /// Each key's kernel, in order.
    pub kernels: Vec<BakedEntry>,
    /// The compiles this call ran; an earlier call in this process ran the rest of its batches.
    pub compiled: usize,
}

/// Kernels one bake compile holds at most. A compile starts the compiler and reads the shader's
/// headers once for all of its kernels — a kernel alone costs ~40 ms, one of a batch a few — and a
/// library's batches still spread over the cores.
const BATCH: usize = 32;

/// Compile the kernel each of `keys` names — `function` of a baked library, with the key's
/// constants as `constexpr`s (`shaders/baked.h`). One compile per batch: up to [`BATCH`] kernels
/// of one library, in the order of their constants, each in its own namespace. The batches are a
/// function of the key set alone, so the same keys bake the same metallibs; they compile in
/// parallel, each once per process.
pub fn bake(keys: &[PipelineKey]) -> Bake {
    type Cell = Arc<OnceLock<Arc<[u8]>>>;
    static BAKED: OnceLock<Mutex<HashMap<u64, Cell>>> = OnceLock::new();
    let mut order: Vec<(&PipelineKey, Macros)> = keys.iter().map(|k| (k, macros(k))).collect();
    order.sort_by(|(a, x), (b, y)| (a.library_name, x).cmp(&(b.library_name, y)));
    order.dedup_by(|(a, x), (b, y)| (a.library_name, &*x) == (b.library_name, &*y));
    let runs = order.chunk_by(|(a, _), (b, _)| a.library_name == b.library_name);
    let batches: Vec<&[(&PipelineKey, Macros)]> = runs.flat_map(|r| r.chunks(BATCH)).collect();
    let compiled = AtomicUsize::new(0);
    let baked: Vec<Vec<(&PipelineKey, BakedEntry)>> = (batches.par_iter())
        .map(|batch| {
            let library = batch[0].0.library_name;
            let lib =
                baked_library(library).unwrap_or_else(|| panic!("bake: `{library}` is not baked"));
            let source = batch_source(library, batch);
            let mut flags = vec!["-Wno-modules-import-nested-redundant"];
            flags.extend(MPP_FLAGS.iter().copied().filter(|_| lib.mpp));
            let inputs = [toolchain(), &source, &format!("{:016x}", sources_hash())];
            let hash = fnv(inputs
                .into_iter()
                .chain(flags.iter().copied())
                .map(str::as_bytes));
            let cell = (BAKED.get_or_init(Mutex::default).lock().unwrap())
                .entry(hash)
                .or_default()
                .clone();
            let metallib = cell.get_or_init(|| {
                compiled.fetch_add(1, Ordering::Relaxed);
                kept_compile(hash, library, &source, &flags).into()
            });
            let entries = batch.iter().enumerate().map(|(i, (key, _))| {
                let entry = format!("k{i}::{}", key.kernel_name);
                let named = metallib.windows(entry.len()).any(|w| w == entry.as_bytes());
                assert!(
                    named || metallib.is_empty(),
                    "bake: no instantiation of `{library}.metal` is named `{entry}`"
                );
                let metallib = metallib.clone();
                (*key, BakedEntry { metallib, entry })
            });
            entries.collect()
        })
        .collect();
    let by_key: HashMap<&PipelineKey, BakedEntry> = baked.into_iter().flatten().collect();
    Bake {
        kernels: keys.iter().map(|k| by_key[k].clone()).collect(),
        compiled: compiled.into_inner(),
    }
}

/// The macros `baked.h` reads for one kernel, by name: each of its constants' value and set flag,
/// and the kernel's own flag.
type Macros = Vec<(String, String)>;

/// `key`'s [`Macros`].
fn macros(key: &PipelineKey) -> Macros {
    let set = || "~, 1".to_owned();
    let constants = key.constants.iter().flat_map(|c| {
        let slot = c.index;
        [
            (format!("SCRATCHY_CONSTANT_{slot}"), msl_literal(c)),
            (format!("SCRATCHY_SET_{slot}"), set()),
        ]
    });
    let kernel = (format!("SCRATCHY_KERNEL_{}", key.kernel_name), set());
    constants.chain([kernel]).collect()
}

/// One bake compile of `library`'s kernels `batch`: the guarded headers its shader reaches, once,
/// then per kernel `i` its macros defined around the shader, included in namespace `k<i>`.
fn batch_source(library: &str, batch: &[(&PipelineKey, Macros)]) -> String {
    let mut source: String = guarded_headers(library)
        .iter()
        .map(|h| format!("#include {h}\n"))
        .collect();
    for (i, (_, macros)) in batch.iter().enumerate() {
        let ns = ("SCRATCHY_NS".to_owned(), format!("k{i}"));
        let macros = || macros.iter().chain([&ns]);
        source.extend(macros().map(|(name, value)| format!("#define {name} {value}\n")));
        source += &format!("namespace SCRATCHY_NS {{\n#include \"{library}.metal\"\n}}\n");
        source.extend(macros().map(|(name, _)| format!("#undef {name}\n")));
    }
    source
}

/// The `#pragma once` headers `library`'s shader reaches through its includes, in the order a
/// compile first reads them. A batch includes them ahead of its kernels, so each kernel's namespace
/// re-reads only the shader and the unguarded headers that declare its constants (`baked.h`).
fn guarded_headers(library: &str) -> Vec<String> {
    let text = |path: &str| {
        BAKED_SOURCES
            .iter()
            .find(|(p, _)| *p == path)
            .map(|(_, t)| *t)
    };
    fn walk(
        text: &dyn Fn(&str) -> Option<&'static str>,
        path: &str,
        seen: &mut std::collections::HashSet<String>,
        out: &mut Vec<String>,
    ) {
        let dir = Path::new(path).parent().unwrap_or(Path::new(""));
        let includes = text(path).into_iter().flat_map(str::lines);
        for inc in includes.filter_map(|l| l.trim_start().strip_prefix("#include")) {
            let inc = inc.trim();
            if let Some(system) = inc.strip_prefix('<') {
                let header = format!("<{}>", system.split('>').next().unwrap_or_default());
                if seen.insert(header.clone()) {
                    out.push(header);
                }
                continue;
            }
            let name = inc
                .trim_start_matches('"')
                .split('"')
                .next()
                .unwrap_or_default();
            let beside = dir.join(name).to_string_lossy().into_owned();
            let path = if text(&beside).is_some() {
                beside
            } else {
                name.to_owned()
            };
            if !seen.insert(path.clone()) {
                continue;
            }
            walk(text, &path, seen, out);
            if text(&path).is_some_and(|t| t.contains("#pragma once")) {
                out.push(format!("\"{path}\""));
            }
        }
    }
    let mut out = Vec::new();
    walk(
        &text,
        &format!("{library}.metal"),
        &mut Default::default(),
        &mut out,
    );
    out
}

/// `keys`' kernels, baked, as the statics a baked tape holds
/// ([`ClassedTape::kernels`](crate::tape::lowered::ClassedTape::kernels)) — for kernels lowered or
/// dispatched outside the macro (kernel tests).
pub fn baked_kernels(keys: &[PipelineKey]) -> Vec<BakedKernel> {
    let bake = bake(keys);
    let kernels = keys.iter().zip(bake.kernels);
    kernels
        .map(|(key, kernel)| BakedKernel {
            library: key.library_name,
            function: key.kernel_name,
            constants: baked(key.constants.clone()),
            metallib: baked(kernel.metallib.to_vec()),
            entry: Box::leak(kernel.entry.into_boxed_str()),
        })
        .collect()
}

/// The standard shaders and every baked kernel `tapes` name at `dtype` for tape `variant`, as the
/// pool registers a model's — for tapes lowered outside the macro (tests).
pub fn tape_pipelines(
    device: &Device,
    tapes: &[LoweredMetalTape],
    dtype: MetalDtype,
    variant: TapeVariant,
) -> Result<SpecializedPipelines, MetalStreamError> {
    let commands = tapes.iter().flat_map(|t| t.commands.iter());
    let keys = commands.map(|c| bake_key(&c.command, dtype, variant));
    let keys: Vec<_> = (keys.collect::<Result<Vec<_>, _>>())
        .map_err(|e| MetalStreamError::ShaderCompilationFailed(e.to_string()))?
        .into_iter()
        .flatten()
        .collect();
    let cache = SpecializedPipelineCache::new(device.clone(), &[])?;
    cache.register_baked(&baked_kernels(&keys));
    Ok(SpecializedPipelines::new(Arc::new(cache), variant))
}

/// `key`'s pipeline from `cache`, its kernel baked first — for kernel tests that build several
/// pipelines through one cache.
pub fn baked_build(
    cache: &SpecializedPipelineCache,
    key: &PipelineKey,
) -> Result<ComputePipelineState, MetalStreamError> {
    if baked_library(key.library_name).is_some() {
        cache.register_baked(&baked_kernels(std::slice::from_ref(key)));
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

/// The compile of `source` with `flags`, whose inputs hash to `hash`. In a build (`OUT_DIR`) it is
/// kept beside the build's other outputs under that hash — of every input: the Metal toolchain,
/// the source, its flags and every file it can include — so a re-expansion that changed none of
/// them reads it back instead of compiling it again.
fn kept_compile(hash: u64, library: &str, source: &str, flags: &[&str]) -> Vec<u8> {
    let kept = std::env::var_os("OUT_DIR")
        .map(|out| Path::new(&out).join(format!("metal-bake/batch-{hash:016x}.metallib")));
    if let Some(bytes) = kept.as_ref().and_then(|path| std::fs::read(path).ok()) {
        return bytes;
    }
    // Not `<library>.metal`: the source includes that file, and would find itself first.
    let bytes = compile_metallib(&format!("{library}-batch"), source, flags);
    if let Some(path) = kept {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = path.with_extension(format!("{}.{seq}.tmp", std::process::id()));
        std::fs::create_dir_all(path.parent().expect("metal-bake dir"))
            .and_then(|()| std::fs::write(&tmp, &bytes))
            .and_then(|()| std::fs::rename(&tmp, &path))
            .unwrap_or_else(|e| panic!("bake: keep {}: {e}", path.display()));
    }
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
        ConstantType::KvCap | ConstantType::TqHeads | ConstantType::AttnSplits => panic!(
            "bake: constant slot {} is bound to its tape variant; `bake_key` resolves it",
            c.index
        ),
    }
}
