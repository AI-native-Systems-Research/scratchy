// SPDX-License-Identifier: Apache-2.0
//! ⭐ dxp's OUTPUT, KEPT ACROSS BUILDS — a content-addressed cache of compiled launch groups.
//!
//! A spyre build re-runs the whole bake whenever anything upstream of the models build script
//! changes, and almost every group it then hands `dxp_standalone` is byte-identical to one the last
//! build compiled: a change to one op touches the groups holding that op and nothing else. The
//! in-build memo ([`crate::superdsc_bake`]) already skips a group whose input recurs WITHIN a build;
//! this keeps the answer for the next one.
//!
//! ## The store is keyed by everything the compiler's answer depends on
//!
//! `<store>/<identity>/<input>.bin` ([`DxpCache`]), where the input is the group's own files in the
//! order dxp reads them ([`GroupDigest`]) and the identity ([`DxpIdentity`]) is the compiler:
//!
//!   * **every image the loader maps for it** — the binary AND each shared library, read off the
//!     platform loader's own trace (`DYLD_PRINT_LIBRARIES` / `LD_TRACE_LOADED_OBJECTS`), because
//!     `dxp_standalone` is a thin `main` over ~20 deeptools libraries and a rebuilt `libdxp` leaves
//!     the binary untouched;
//!   * **every data file it reads under `DEEPTOOLS_PATH`** ([`DEEPTOOLS_DATA_DIRS`]);
//!   * **every environment variable deeptools reads** ([`DEEPTOOLS_ENV`] / [`DEEPTOOLS_ENV_PREFIXES`]),
//!     with the values the bake itself sets on the command.
//!
//! Files enter the identity by (path, length, mtime): a rebuilt or reinstalled deeptools rewrites
//! them, and hashing ~360 MB of libraries on every build would cost more than most rebuilds save.
//!
//! ⛔ NO PROBE, NO CACHE. If the loader trace names no image the identity cannot cover the libraries,
//! so [`DxpIdentity::probe`] returns `None` and every group compiles — slower, never stale.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use sha2::{Digest, Sha256};

use crate::superdsc_bake::CompiledGroup;

/// ⭐ A GROUP'S INPUT, AS dxp READS IT — SHA-256 over each file's name and bytes, in read order.
///
/// The in-build memo and the persistent cache share it: equal digests mean equal compiler input.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GroupDigest([u8; 32]);

impl GroupDigest {
    /// Digest `files` in order. Each name and body is length-prefixed so no two different file lists
    /// share a byte stream.
    pub fn of(files: &[(String, String)]) -> GroupDigest {
        Self::hash(files, None)
    }

    /// Digest `files` AND the per-group compile setting `tag` dxp is spawned with — what a group's
    /// compile actually reads when part of its input is per group rather than process-wide (which
    /// [`DxpIdentity`] covers). The tag is length-prefixed after a marker no file list can produce,
    /// so a tagged digest never equals an untagged one.
    pub fn of_compile(files: &[(String, String)], tag: &str) -> GroupDigest {
        Self::hash(files, Some(tag))
    }

    fn hash(files: &[(String, String)], tag: Option<&str>) -> GroupDigest {
        let mut h = Sha256::new();
        for (name, body) in files {
            h.update((name.len() as u64).to_le_bytes());
            h.update(name.as_bytes());
            h.update((body.len() as u64).to_le_bytes());
            h.update(body.as_bytes());
        }
        if let Some(tag) = tag {
            h.update(u64::MAX.to_le_bytes());
            h.update((tag.len() as u64).to_le_bytes());
            h.update(tag.as_bytes());
        }
        GroupDigest(h.finalize().into())
    }
}

impl std::fmt::Display for GroupDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in &self.0[..8] {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// The data subtrees dxp reads under `DEEPTOOLS_PATH` — read off deeptools' own source:
/// `ConstructProgIRHelper.cpp` / `InstructionEstimation.cpp` (opaque templates), `ddl_conversion.cpp`
/// (`ddc/ddl_templates/`), `archDataflowConstr.cpp` + `dsm.cpp` (`perfdsc/HardwareArchMapping/`), and
/// `collenv.cpp` (`share/coll/`). A subtree a deeptools layout lacks contributes nothing.
pub const DEEPTOOLS_DATA_DIRS: [&str; 4] = [
    "dcc/src/Conversion/SentientToProgIR/opaqueTemplates",
    "ddc/ddl_templates",
    "perfdsc/HardwareArchMapping",
    "share/coll",
];

/// Every environment variable deeptools' compile path reads by name (its `dtGetEnv` / `getenv` call
/// sites) that no prefix in [`DEEPTOOLS_ENV_PREFIXES`] already covers. An unset variable and an empty
/// one key differently, the same as they read differently.
pub const DEEPTOOLS_ENV: [&str; 48] = [
    "_MAX_CONTEXT_LEN",
    "_PROMPT_LEN",
    "BO_CUSTOM_L0_POST",
    "BO_CUSTOM_L0_PRE",
    "BO_CUSTOM_L1",
    "BO_CUSTOM_L2",
    "CODEGEN_DUMP_IRS",
    "COMPILATION_MODE",
    "DATA_DEBUG_MODE",
    "DATA_PREC",
    "DATACONV_EXEC_SPACE_SIZE",
    "DDCCOORD",
    "DGP",
    "DGP_DIR",
    "DGP_INPUTS",
    "DGP_MB",
    "DGPLOG",
    "DISABLE_ABOVE_LX_CHUNK_EXPLORE",
    "DISABLE_FOLDED_PERFDSC",
    "DISCARD_ABOVE_LX_SCHEDULE",
    "EAGER_MODE",
    "EN_FOLDED_SDSC",
    "EN_PREFILL_OPT",
    "ENABLE_DT2EXCLUSIVE_PIPELINE",
    "ENABLE_LN32",
    "EXTRA_HBM_SEGMENT_ALLOCATE_FACTOR",
    "FORCE_INF",
    "FORCE_L3_BUFFER_TYPE",
    "FORCE_SYMBOLIC_DSC",
    "FORCE_XRF_ROWS",
    "FP16_EMULATION",
    "HMI_DRM",
    "HOST_ENDIAN_FORMAT",
    "MACHINE_CHECK_ENABLED_UNIT",
    "MARKER_DCC_DVS",
    "MAX_SHAREDPROG_ITERS",
    "PREC_CONFIG",
    "PRIDS_LAYOUT",
    "RANK",
    "RESTICKIFY_DLDSC",
    "SIM_FIDELITY",
    "SKIP_VERIFY",
    "SPARSE_EMULATION",
    "SPYRE_FFDC_DIR",
    "SWA_COMPUTE_OPT",
    "TRAVIS_BRANCH",
    "USE_FIXED_GRP_ID",
    "WORLD_SIZE",
];

/// Prefixes of the deeptools families' own variables (`DT_*`, `DXP_*`, `SEN*`, …) — every
/// `dtGetEnv` name not in [`DEEPTOOLS_ENV`] starts with one of these.
pub const DEEPTOOLS_ENV_PREFIXES: [&str; 27] = [
    "AIU_",
    "COLL_",
    "DBO_",
    "DCC_",
    "DDB_",
    "DEEPRT_",
    "DEEPTOOLS_",
    "DEM_",
    "DIP_",
    "DR5_",
    "DSI_",
    "DSM_",
    "DT_",
    "DUMP_",
    "DXP_",
    "FLEX_",
    "HOSTSEN",
    "PERF",
    "PRINT_",
    "PROG_",
    "RUNDSM_",
    "SDC_",
    "SDSC_",
    "SEN",
    "SET_",
    "VAST",
    "VLLM_DT_",
];

/// ⭐ THE COMPILER, AS A KEY — see the module doc for what it covers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DxpIdentity([u8; 32]);

impl DxpIdentity {
    /// Identify `bin` as it runs under `deeptools`, with `fixed_env` being the variables the bake
    /// sets on every compile (they replace whatever the build's own environment says) and
    /// `per_group` the ones it sets or clears PER GROUP, whose ambient value no child ever sees — so
    /// it is not part of the compiler's identity (each group's own value is in its [`GroupDigest`]).
    ///
    /// `None` when the loader trace names no image — the cache then stays off for this build.
    pub fn probe(
        bin: &Path,
        deeptools: &Path,
        fixed_env: &[(&str, &str)],
        per_group: &[&str],
    ) -> Option<DxpIdentity> {
        let images = loaded_images(bin, deeptools, fixed_env)?;
        let mut h = Sha256::new();
        h.update(b"scratchy-dxp-identity-v1\0");
        let mut stat = |p: &Path, label: &[u8]| {
            h.update(label);
            h.update(p.as_os_str().as_encoded_bytes());
            h.update([0]);
            match std::fs::metadata(p) {
                Ok(m) => {
                    h.update(m.len().to_le_bytes());
                    let t = m
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map_or(0, |d| d.as_nanos());
                    h.update(t.to_le_bytes());
                }
                Err(_) => h.update(b"absent"),
            }
        };
        stat(bin, b"bin\0");
        for img in &images {
            stat(img, b"image\0");
        }
        for sub in DEEPTOOLS_DATA_DIRS {
            let mut files = Vec::new();
            walk_files(&deeptools.join(sub), &mut files);
            files.sort();
            for f in &files {
                stat(f, b"data\0");
            }
        }
        h.update(b"env\0");
        for (k, v) in &identity_env(std::env::vars(), fixed_env, per_group) {
            h.update((k.len() as u64).to_le_bytes());
            h.update(k.as_bytes());
            h.update((v.len() as u64).to_le_bytes());
            h.update(v.as_bytes());
        }
        Some(DxpIdentity(h.finalize().into()))
    }
}

/// The environment a compile runs under, as the identity hashes it: every deeptools variable in
/// `ambient`, minus the ones the bake overrides (`fixed_env`, whose own values replace them) and the
/// ones it sets or clears per group (`per_group`, whose ambient value no child sees), sorted.
fn identity_env(
    ambient: impl Iterator<Item = (String, String)>,
    fixed_env: &[(&str, &str)],
    per_group: &[&str],
) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = ambient
        .filter(|(k, _)| {
            DEEPTOOLS_ENV.contains(&k.as_str())
                || DEEPTOOLS_ENV_PREFIXES.iter().any(|p| k.starts_with(p))
        })
        .filter(|(k, _)| !fixed_env.iter().any(|(f, _)| f == k))
        .filter(|(k, _)| !per_group.contains(&k.as_str()))
        .collect();
    env.extend(
        fixed_env
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string())),
    );
    env.sort();
    env
}

/// Every file under `dir`, recursively; nothing when `dir` does not exist.
fn walk_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        match e.file_type() {
            Ok(t) if t.is_dir() => walk_files(&p, out),
            Ok(_) => out.push(p),
            Err(_) => {}
        }
    }
}

/// The images the platform loader maps for `bin`, read off the loader's own trace. `None` when the
/// trace names none (an unknown platform, a binary the loader refused).
pub(crate) fn loaded_images(
    bin: &Path,
    deeptools: &Path,
    fixed_env: &[(&str, &str)],
) -> Option<Vec<PathBuf>> {
    let mut cmd = Command::new(bin);
    cmd.env("DEEPTOOLS_PATH", deeptools)
        .envs(fixed_env.iter().copied())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // macOS: dyld prints `dyld[<pid>]: <UUID> <path>` per image on stderr, then runs the program —
    // which, given no arguments, prints its usage and exits. Linux: `ld.so` prints the resolved
    // `name => path (addr)` list on stdout and exits WITHOUT running it.
    let text = if cfg!(target_os = "macos") {
        let out = cmd.env("DYLD_PRINT_LIBRARIES", "1").output().ok()?;
        String::from_utf8_lossy(&out.stderr).into_owned()
    } else {
        let out = cmd.env("LD_TRACE_LOADED_OBJECTS", "1").output().ok()?;
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let mut images: Vec<PathBuf> = text
        .lines()
        .filter_map(|l| {
            if cfg!(target_os = "macos") {
                // `dyld[123]: <UUID> /path/to/image`
                let rest = l.strip_prefix("dyld[")?;
                let (_, rest) = rest.split_once("> ")?;
                Some(PathBuf::from(rest.trim()))
            } else {
                // `\tlibfoo.so => /path/libfoo.so (0x…)` or `\t/lib64/ld-linux.so.2 (0x…)`
                let l = l.trim();
                let path = match l.split_once(" => ") {
                    Some((_, r)) => r,
                    None => l,
                };
                let path = path.rsplit_once(" (").map_or(path, |(p, _)| p).trim();
                path.starts_with('/').then(|| PathBuf::from(path))
            }
        })
        .collect();
    images.sort();
    images.dedup();
    (!images.is_empty()).then_some(images)
}

/// ⭐ THE STORE — one directory per compiler identity, one file per compiled group.
///
/// Resolved as `$SCRATCHY_SUPERDSC_CACHE` if set, else `<OUT_DIR>/../../superdsc-dxp-cache` (the
/// target's `build/`, so `cargo clean` takes it with it and two model scopes share one store). Under
/// it, `<identity>/` holds this compiler's groups; every OTHER entry there is a previous compiler's
/// output (a deeptools rebuild, an upgrade) and is WIPED when the store opens — serving one byte of
/// another compiler's program is the one wrong answer here, and wiping is what keeps the store from
/// growing a copy per toolchain.
///
/// Errors are MISSES, never build failures: a store that cannot be read or written costs a compile,
/// which is exactly what not having it costs.
#[derive(Clone, Debug)]
pub struct DxpCache {
    dir: PathBuf,
}

/// Entry-file magic — `SDC1`, little-endian — so a truncated or foreign file reads as a miss.
const ENTRY_MAGIC: u32 = 0x5344_4331;

impl DxpCache {
    /// The store for this build, or `None` in a context with no durable location (no `OUT_DIR` and no
    /// override — a bare `cargo test`), where the in-process memo alone carries the build.
    pub fn for_build(identity: DxpIdentity) -> Option<DxpCache> {
        let base = match std::env::var_os("SCRATCHY_SUPERDSC_CACHE") {
            Some(p) => PathBuf::from(p),
            None => PathBuf::from(std::env::var_os("OUT_DIR")?)
                .join("../..")
                .join("superdsc-dxp-cache"),
        };
        DxpCache::open(&base, identity)
    }

    /// The store under `base` for `identity`, wiping every other identity's (see the type doc).
    pub fn open(base: &Path, identity: DxpIdentity) -> Option<DxpCache> {
        let dir = base.join(identity.hex());
        if let Ok(rd) = std::fs::read_dir(base) {
            for e in rd.flatten() {
                let p = e.path();
                if p == dir {
                    continue;
                }
                if p.is_dir() {
                    let _ = std::fs::remove_dir_all(&p);
                } else {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }
        std::fs::create_dir_all(&dir).ok()?;
        // Crashed `put`s leave `.tmp-` files behind; they are never valid entries.
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                if e.file_name().to_string_lossy().starts_with(".tmp-") {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
        Some(DxpCache { dir })
    }

    fn entry(&self, d: GroupDigest) -> PathBuf {
        self.dir.join(format!("{}.bin", d.hex()))
    }

    /// The compiled group for `d`, or `None` for any reason at all — absent, truncated, foreign,
    /// unreadable. A miss is a compile, which is the status quo ante.
    pub fn get(&self, d: GroupDigest) -> Option<CompiledGroup> {
        let b = std::fs::read(self.entry(d)).ok()?;
        // [magic u32][job_bin_ptr u64][init_len u32][corr_len u32][init][corr]
        if b.len() < 20 || u32::from_le_bytes(b[0..4].try_into().ok()?) != ENTRY_MAGIC {
            return None;
        }
        let job_bin_ptr = u64::from_le_bytes(b[4..12].try_into().ok()?);
        let init_len = u32::from_le_bytes(b[12..16].try_into().ok()?) as usize;
        let corr_len = u32::from_le_bytes(b[16..20].try_into().ok()?) as usize;
        if b.len() != 20 + init_len + corr_len {
            return None;
        }
        Some(CompiledGroup {
            init_binary: b[20..20 + init_len].to_vec(),
            job_bin_ptr,
            correction: b[20 + init_len..].to_vec(),
        })
    }

    /// Best-effort write, atomic by rename so a concurrent `get` sees the old file or the new one,
    /// never a half of either. An image past `u32::MAX` is not stored — the length field cannot spell
    /// it — and the compile it came from is still served this build by the in-process memo.
    pub fn put(&self, d: GroupDigest, g: &CompiledGroup) {
        if g.init_binary.len() > u32::MAX as usize || g.correction.len() > u32::MAX as usize {
            return;
        }
        let mut b = Vec::with_capacity(20 + g.init_binary.len() + g.correction.len());
        b.extend_from_slice(&ENTRY_MAGIC.to_le_bytes());
        b.extend_from_slice(&g.job_bin_ptr.to_le_bytes());
        b.extend_from_slice(&(g.init_binary.len() as u32).to_le_bytes());
        b.extend_from_slice(&(g.correction.len() as u32).to_le_bytes());
        b.extend_from_slice(&g.init_binary);
        b.extend_from_slice(&g.correction);
        let tmp = self.dir.join(format!(
            ".tmp-{}-{}-{:?}",
            d.hex(),
            std::process::id(),
            std::thread::current().id()
        ));
        if std::fs::write(&tmp, &b).is_ok() && std::fs::rename(&tmp, self.entry(d)).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

/// Lower-case hex of a 32-byte digest.
fn hex32(b: &[u8; 32]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

impl GroupDigest {
    /// The full digest as hex — an entry's file name.
    pub fn hex(&self) -> String {
        hex32(&self.0)
    }
}

impl DxpIdentity {
    /// The full identity as hex — the store directory's name.
    pub fn hex(&self) -> String {
        hex32(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⭐ A PER-GROUP KNOB IS NOT THE COMPILER'S. The bake sets or clears
    /// `DISABLE_ABOVE_LX_CHUNK_EXPLORE` on each child itself, so its ambient value reaches no compile —
    /// and must not split the store either: the value each group compiled under is in that group's
    /// own [`GroupDigest`]. Mutation: drop the `per_group` filter and an exported knob changes the
    /// identity, which then WIPES the store on open.
    #[test]
    fn a_per_group_knob_is_left_out_of_the_identity_environment() {
        let fixed = [("DT_PARALLEL_THREADS", "1")];
        let knob = ["DISABLE_ABOVE_LX_CHUNK_EXPLORE"];
        let ambient = |exported: bool| {
            let mut v = vec![("DT_SOMETHING".to_string(), "x".to_string())];
            if exported {
                v.push((
                    "DISABLE_ABOVE_LX_CHUNK_EXPLORE".to_string(),
                    "1".to_string(),
                ));
            }
            v.into_iter()
        };
        assert_eq!(
            identity_env(ambient(false), &fixed, &knob),
            identity_env(ambient(true), &fixed, &knob),
            "an exported per-group knob must not change the identity"
        );
        assert_ne!(
            identity_env(ambient(false), &fixed, &[]),
            identity_env(ambient(true), &fixed, &[]),
            "the knob is a deeptools variable, so without the exclusion it would"
        );
    }
}
