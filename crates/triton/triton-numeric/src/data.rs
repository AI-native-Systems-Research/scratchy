// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! THE REFERENCE DATA, AND WHERE ITS BYTES CAME FROM.
//!
//! `test/numeric/<config>/` holds, per configuration:
//!
//! * `in_<arg>.bin` -- one file per kernel POINTER argument, raw little-endian in the argument's own
//!   dtype. The output argument gets one too, zeroed, because the executor needs an HBM buffer for it.
//! * `ref_out.bin` -- the FIXTURE'S OWN `reference()`, cast to f16.
//! * `meta.json` -- the provenance: shapes, dtypes, seed, the fixture function names, and the torch
//!   version. So the data's origin is IN THE TREE, not in a commit message.
//! * `sha256.txt` -- the bytes' identity, recorded.
//!
//! ⛔ WHY THE BYTES ARE CHECKED IN RATHER THAN GENERATED. There is no torch on a developer machine
//! here; there is torch 2.11.0+cpu on the pod. Reimplementing torch's RNG in Rust to avoid the
//! checked-in files would defeat the point, which is that the reference is computed by THE FIXTURE'S
//! OWN CODE on THE SAME BYTES. A reference re-derived on our side is a second opinion about what the
//! kernel should compute, and where our kernel and the fixture disagree, the FIXTURE is right until
//! proven otherwise -- the disagreement is the finding.

use std::path::{Path, PathBuf};

use ktir_core::dtypes::DType;

use crate::{refuse, Binding, Refusal, Result};

/// `crates/triton` -- this crate sits at `<that>/triton-numeric`.
pub fn spyre_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate sits at crates/triton/triton-numeric")
        .to_path_buf()
}

/// One configuration's data directory.
pub fn dir(config: &str) -> PathBuf {
    spyre_root().join("test-numeric").join(config)
}

/// Where a configuration's bytes were STAGED, if anywhere.
///
/// ⛔ NAMED PER CONFIGURATION FIRST. `TRITON_NUMERIC_STAGED` alone is a convenience for a single
/// staged config; with two staged at once it would point both at one directory and each would then
/// be compared against the other's answer -- and since their extents match, silently pass. The
/// per-config variable is what makes that impossible.
pub fn staged_dir(config: &str) -> Option<PathBuf> {
    let named = format!("TRITON_NUMERIC_STAGED_{}", config.to_uppercase());
    let p = std::env::var(&named)
        .or_else(|_| std::env::var("TRITON_NUMERIC_STAGED"))
        .ok()?;
    let p = PathBuf::from(p);
    p.is_dir().then_some(p)
}

/// HOW ONE TENSOR'S BYTES ARE STORED, which is not always "densely".
///
/// ⛔⛔ THE ENCODING ANNOUNCES ITSELF IN THE FILE EXTENSION AND THE `file` FIELD IS AUTHORITATIVE.
/// A reader that reconstructs `in_<arg>.bin` from the argument name finds no file for an encoded
/// tensor and fails closed, instead of reading a 64 KiB block as an 8 MiB buffer and comparing
/// against whatever it got. So this reader never builds a filename -- it uses `Tensor::file`.
///
/// ⭐ AND EVERY ENCODING HERE IS LOSSLESS IN THE DIRECTION THAT MATTERS. Neither can turn a wrong
/// answer into a right one:
///
///   * `Sparse` expands an absent row to ZERO. If the kernel reads a row the index list never
///     named, it reads zeros and the comparison DIVERGES -- a false FAILURE is possible, a false
///     pass is not. (Measured by the generator: retargeting one id at an absent row moves the
///     output by 46.44 on a +-46.8 range.) Expanding to anything else -- uninitialised memory, a
///     splat, a wrap-around -- destroys that property, so this reader zero-fills and nothing else.
///   * `Replicate` is CHECKED AT GENERATION against the dense buffer bit for bit, and the wrong
///     reading is checked to differ: `repeat_interleave` and `repeat` produce THE SAME SHAPE from
///     the same block and differ only in values, which is exactly how a replication rule asserted
///     in metadata comes to be wrong half the time. `stage_table`'s own docstring records that trap.
#[derive(Debug, Clone)]
pub enum Encoding {
    /// The whole buffer, densely, in row-major order.
    Dense,
    /// `SPRSTBL1`: a 32-byte header (magic, rows u64, cols u64, itemsize u32, n_present u32) then
    /// `n_present` records of `(row_index u32, cols * itemsize bytes)`, ascending by row index,
    /// little-endian. Absent rows are ZERO.
    Sparse { rows: usize, cols: usize, itemsize: usize, n_present: usize },
    /// One block of `block` rows, expanded by `full[i * factor + j] = stored[i]` -- torch's
    /// `repeat_interleave` along axis 0, NOT `repeat`/tiling.
    Replicate { factor: usize, block: usize, axis: usize },
}

/// One tensor's record in `meta.json`.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The `.py` kernel's own parameter name.
    pub arg: String,
    /// The file holding the bytes, AS RECORDED. Never reconstructed from `arg`.
    pub file: String,
    /// The LOGICAL shape the kernel indexes -- the shape after any expansion.
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// True for a buffer the kernel WRITES and the data only has to reserve.
    pub zeroed: bool,
    pub encoding: Encoding,
}

impl Tensor {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// The dense bytes this tensor stands for, expanded from whatever encoding holds them.
    ///
    /// `shas` is the configuration's own recorded `sha256` map, when it has one; every file read
    /// here is checked against it. See [`verify_sha`].
    fn bytes(&self, dir: &Path, shas: Option<&serde_json::Map<String, serde_json::Value>>) -> Result<Vec<u8>> {
        let p = dir.join(&self.file);
        let want = self.elements() * self.dtype.bytes_per_elem();
        match &self.encoding {
            Encoding::Dense => {
                let raw = read_bin(&p, want)?;
                verify_sha(&p, &self.file, &raw, shas)?;
                Ok(raw)
            }
            Encoding::Sparse { rows, cols, itemsize, n_present } => {
                let raw = std::fs::read(&p)
                    .map_err(|e| refuse("data", format!("cannot read {}: {e}", p.display())))?;
                verify_sha(&p, &self.file, &raw, shas)?;
                expand_sparse(&raw, *rows, *cols, *itemsize, *n_present, want, &p)
            }
            Encoding::Replicate { factor, block, axis } => {
                if *axis != 0 {
                    return Err(refuse(
                        "data",
                        format!(
                            "{}: replication on axis {axis} is recorded but this reader only                              expands axis 0. REFUSING rather than expanding the wrong axis, which                              produces a buffer of the right SIZE holding the wrong values.",
                            p.display()
                        ),
                    ));
                }
                let row_bytes = self.shape[1..].iter().product::<usize>()
                    * self.dtype.bytes_per_elem();
                let stored = read_bin(&p, block * row_bytes)?;
                verify_sha(&p, &self.file, &stored, shas)?;
                let mut out = Vec::with_capacity(want);
                // `repeat_interleave`: each stored row repeated `factor` times CONSECUTIVELY. The
                // other reading (`repeat`/tiling: the whole block repeated `factor` times) yields
                // the same length, so length is no check at all and only this loop's shape is.
                for i in 0..*block {
                    let row = &stored[i * row_bytes..(i + 1) * row_bytes];
                    for _ in 0..*factor {
                        out.extend_from_slice(row);
                    }
                }
                if out.len() != want {
                    return Err(refuse(
                        "data",
                        format!(
                            "{}: expanding {block} row(s) by factor {factor} gives {} bytes; the                              recorded shape {:?} needs {want}",
                            p.display(),
                            out.len(),
                            self.shape
                        ),
                    ));
                }
                Ok(out)
            }
        }
    }
}

/// Expand a `SPRSTBL1` file. Absent rows are ZERO -- see [`Encoding`] for why that specific filler
/// is the whole soundness argument and not a convenience.
fn expand_sparse(
    raw: &[u8],
    rows: usize,
    cols: usize,
    itemsize: usize,
    n_present: usize,
    want: usize,
    p: &Path,
) -> Result<Vec<u8>> {
    const HDR: usize = 32;
    if raw.len() < HDR || &raw[0..8] != b"SPRSTBL1" {
        return Err(refuse(
            "data",
            format!("{}: not a SPRSTBL1 sparse table", p.display()),
        ));
    }
    let u64_at = |o: usize| {
        u64::from_le_bytes(raw[o..o + 8].try_into().expect("8 bytes")) as usize
    };
    let u32_at = |o: usize| {
        u32::from_le_bytes(raw[o..o + 4].try_into().expect("4 bytes")) as usize
    };
    // ⭐ THE HEADER IS CROSS-CHECKED AGAINST `meta.json`, both ways. Two statements of the same
    // fact are only safe while something compares them; unchecked they are how a regenerated file
    // comes to be read under the old metadata.
    let (h_rows, h_cols, h_item, h_n) = (u64_at(8), u64_at(16), u32_at(24), u32_at(28));
    if (h_rows, h_cols, h_item, h_n) != (rows, cols, itemsize, n_present) {
        return Err(refuse(
            "data",
            format!(
                "{}: the file's header says rows={h_rows} cols={h_cols} itemsize={h_item}                  n_present={h_n}; meta.json says rows={rows} cols={cols} itemsize={itemsize}                  n_present={n_present}. The bytes and their record disagree.",
                p.display()
            ),
        ));
    }
    let rec = 4 + cols * itemsize;
    let need = HDR + n_present * rec;
    if raw.len() != need {
        return Err(refuse(
            "data",
            format!(
                "{}: {} bytes; a {n_present}-record table of {cols}-wide rows is {need}",
                p.display(),
                raw.len()
            ),
        ));
    }
    let mut out = vec![0u8; want];
    let row_bytes = cols * itemsize;
    let mut last: Option<usize> = None;
    for k in 0..n_present {
        let at = HDR + k * rec;
        let idx = u32_at(at);
        if idx >= rows {
            return Err(refuse(
                "data",
                format!("{}: record {k} names row {idx} of {rows}", p.display()),
            ));
        }
        // Ascending and DISTINCT, as the format states. A repeated index would mean one row's
        // bytes silently overwrite another's, and the file would still be the right length.
        if let Some(prev) = last {
            if idx <= prev {
                return Err(refuse(
                    "data",
                    format!(
                        "{}: record {k} names row {idx} after row {prev}; the format states                          ascending, distinct indices",
                        p.display()
                    ),
                ));
            }
        }
        last = Some(idx);
        out[idx * row_bytes..(idx + 1) * row_bytes]
            .copy_from_slice(&raw[at + 4..at + 4 + row_bytes]);
    }
    Ok(out)
}

/// One configuration's checked-in data, with its provenance.
#[derive(Debug, Clone)]
pub struct Fixture {
    pub config: String,
    /// The fixture module the stimulus and reference came from (`rmsnorm`, `decoder_block`, ...).
    pub fixture: String,
    pub kernel: String,
    pub seed: i64,
    /// The torch build that produced the bytes. Recorded so a later regeneration under a different
    /// build is visible rather than silent.
    pub torch: String,
    /// The `tl.constexpr` values the data was generated at. A test reads its extents FROM HERE so
    /// the tolerance it derives and the bytes it compares cannot disagree about D_MODEL.
    pub constexprs: serde_json::Map<String, serde_json::Value>,
    pub inputs: Vec<Tensor>,
    pub output: Tensor,
    /// Whatever the generator recorded about a reference it could not take from the fixture. Present
    /// only where a fixture HAS no `reference()`; carried so a hand-authored reference cannot pass
    /// for a fixture-supplied one.
    pub reference_note: Option<String>,
    /// ⭐ THE BYTES LIVE IN ANOTHER CONFIGURATION'S DIRECTORY. `embedding_granite_bm128` differs
    /// from `embedding_granite` only in `BLOCK_M`, which never reaches the fixture's `inputs()`, so
    /// their data is byte-identical and stored ONCE. The generator verifies that identity against
    /// the sibling's recorded hashes on every run and refuses to share if they ever differ, so this
    /// is a checked fact rather than a standing assumption.
    pub data_dir: Option<String>,
    /// ⭐ THE BYTES' IDENTITY, AS THIS CONFIGURATION RECORDS IT. `gen_numeric_data.py` writes a
    /// `sha256` map into `meta.json` for exactly the two cases where the bytes are not beside it --
    /// a `data_dir` sibling and a staged directory -- and that is the case where an identity check
    /// is load-bearing rather than decorative. Every file read through either is checked against
    /// this map; see [`verify_sha`].
    pub sha256: Option<serde_json::Map<String, serde_json::Value>>,
    /// The inputs were too large to check in. The `staging` note says where they are and how to
    /// regenerate them. ⛔ AN ABSENT STAGED DIRECTORY IS **NOT** A PASS -- see [`Fixture::staged`].
    pub oversized: bool,
    pub staging: Option<String>,
}

impl Fixture {
    /// Where this configuration's `.bin`/`.sparse`/`.rep` files actually are.
    pub fn data_dir(&self) -> PathBuf {
        match &self.data_dir {
            Some(other) => dir(other),
            None => dir(&self.config),
        }
    }
}

fn dtype_of(s: &str) -> Result<DType> {
    match s {
        "f16" => Ok(DType::F16),
        "f32" => Ok(DType::F32),
        "i32" => Ok(DType::I32),
        "i64" => Ok(DType::I64),
        other => Err(refuse(
            "meta",
            format!("`{other}` is not a dtype this reader knows: expected f16/f32/i32/i64"),
        )),
    }
}

fn field<'a>(v: &'a serde_json::Value, k: &str, at: &str) -> Result<&'a serde_json::Value> {
    v.get(k)
        .ok_or_else(|| refuse("meta", format!("{at} has no `{k}`")))
}

fn usizes(v: &serde_json::Value, at: &str) -> Result<Vec<usize>> {
    let a = v
        .as_array()
        .ok_or_else(|| refuse("meta", format!("{at} is not an array")))?;
    a.iter()
        .map(|e| {
            e.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| refuse("meta", format!("{at} holds a non-integer extent")))
        })
        .collect()
}

fn tensor(v: &serde_json::Value, at: &str) -> Result<Tensor> {
    Ok(Tensor {
        arg: field(v, "arg", at)?
            .as_str()
            .ok_or_else(|| refuse("meta", format!("{at}.arg is not a string")))?
            .to_string(),
        file: field(v, "file", at)?
            .as_str()
            .ok_or_else(|| refuse("meta", format!("{at}.file is not a string")))?
            .to_string(),
        shape: usizes(field(v, "shape", at)?, &format!("{at}.shape"))?,
        dtype: dtype_of(
            field(v, "dtype", at)?
                .as_str()
                .ok_or_else(|| refuse("meta", format!("{at}.dtype is not a string")))?,
        )?,
        zeroed: v.get("zeroed").and_then(|z| z.as_bool()).unwrap_or(false),
        encoding: encoding(v, at)?,
    })
}

/// Which encoding holds this tensor's bytes. `Dense` unless a `sparse` or `replicate` block says
/// otherwise; BOTH present is a refusal rather than a precedence rule, because a precedence rule is
/// a decision a reader makes silently.
fn encoding(v: &serde_json::Value, at: &str) -> Result<Encoding> {
    let sp = v.get("sparse");
    let rp = v.get("replicate");
    match (sp, rp) {
        (Some(_), Some(_)) => Err(refuse(
            "meta",
            format!("{at} records BOTH a `sparse` and a `replicate` encoding"),
        )),
        (Some(s), None) => {
            let n = |k: &str| {
                s.get(k)
                    .and_then(|x| x.as_u64())
                    .map(|x| x as usize)
                    .ok_or_else(|| refuse("meta", format!("{at}.sparse has no integer `{k}`")))
            };
            let fmt = s.get("format").and_then(|f| f.as_str()).unwrap_or_default();
            if fmt != "SPRSTBL1" {
                return Err(refuse(
                    "meta",
                    format!("{at}.sparse format is `{fmt}`; this reader knows SPRSTBL1"),
                ));
            }
            // ⭐ THE FILLER IS PART OF THE CONTRACT, SO IT IS CHECKED RATHER THAN ASSUMED. Only
            // "zero" keeps sparsification unable to mask a wrong gather.
            let absent = s.get("absent_rows").and_then(|f| f.as_str()).unwrap_or_default();
            if absent != "zero" {
                return Err(refuse(
                    "meta",
                    format!(
                        "{at}.sparse says absent rows are `{absent}`, not `zero`. Zero is the                          whole soundness argument: a row the index list never named must read as                          something the kernel cannot have meant, so a wrong gather DIVERGES."
                    ),
                ));
            }
            let full = usizes(
                s.get("full_shape")
                    .ok_or_else(|| refuse("meta", format!("{at}.sparse has no `full_shape`")))?,
                &format!("{at}.sparse.full_shape"),
            )?;
            if full.len() != 2 {
                return Err(refuse(
                    "meta",
                    format!("{at}.sparse.full_shape is {full:?}; this reader expands rank 2"),
                ));
            }
            Ok(Encoding::Sparse {
                rows: full[0],
                cols: full[1],
                itemsize: n("itemsize")?,
                n_present: n("n_present")?,
            })
        }
        (None, Some(r)) => {
            let n = |k: &str| {
                r.get(k)
                    .and_then(|x| x.as_u64())
                    .map(|x| x as usize)
                    .ok_or_else(|| refuse("meta", format!("{at}.replicate has no integer `{k}`")))
            };
            // ⛔ THE RULE NAME IS CHECKED. `repeat_interleave` and `repeat` give THE SAME SHAPE
            // from the same block and differ only in values, so a reader that expands by the wrong
            // one produces a buffer of the right size holding the wrong numbers and no length or
            // shape check anywhere can see it.
            let rule = r.get("rule").and_then(|x| x.as_str()).unwrap_or_default();
            if rule != "repeat_interleave" {
                return Err(refuse(
                    "meta",
                    format!(
                        "{at}.replicate rule is `{rule}`; this reader expands                          `repeat_interleave` only. `repeat`/tiling yields the same shape from the                          same block with different values, so it must never be guessed."
                    ),
                ));
            }
            Ok(Encoding::Replicate {
                factor: n("factor")?,
                block: n("block")?,
                axis: n("axis")?,
            })
        }
        (None, None) => Ok(Encoding::Dense),
    }
}

impl Fixture {
    /// Read one configuration's `meta.json`.
    ///
    /// FAIL CLOSED ON A MISSING DIRECTORY, with the path in the message. A test that treats absent
    /// data as "nothing to check" is a test that reports green for a configuration nobody ran.
    pub fn load(config: &str) -> Result<Fixture> {
        // ⭐ A STAGED CONFIGURATION CARRIES ITS OWN METADATA, and that is the right way round: the
        // metadata describes the bytes, so it belongs WITH them. An in-tree `meta.json` describing
        // data that is not in the tree is exactly the bytes-and-record mismatch this reader refuses
        // everywhere else. So the in-tree directory is tried first (every checked-in configuration),
        // and a staged directory second.
        let d = match dir(config) {
            d if d.join("meta.json").is_file() => d,
            _ => staged_dir(config).unwrap_or_else(|| dir(config)),
        };
        let p = d.join("meta.json");
        let text = std::fs::read_to_string(&p).map_err(|e| {
            refuse(
                "meta",
                format!(
                    "cannot read {}: {e}. The reference data is generated on the pod by \
                     test/numeric/gen_numeric_data.py -- see that script's header.",
                    p.display()
                ),
            )
        })?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| refuse("meta", format!("{} is not JSON: {e}", p.display())))?;

        let got_config = field(&v, "config", "meta.json")?.as_str().unwrap_or_default();
        if got_config != config {
            return Err(refuse(
                "meta",
                format!(
                    "{} says its config is `{got_config}`, but it sits in `{config}/`. A data \
                     directory whose name and contents disagree is data compared against the \
                     wrong program.",
                    p.display()
                ),
            ));
        }

        let inputs = field(&v, "inputs", "meta.json")?
            .as_array()
            .ok_or_else(|| refuse("meta", "`inputs` is not an array"))?
            .iter()
            .enumerate()
            .map(|(i, t)| tensor(t, &format!("inputs[{i}]")))
            .collect::<Result<Vec<_>>>()?;

        Ok(Fixture {
            config: config.to_string(),
            fixture: field(&v, "fixture", "meta.json")?.as_str().unwrap_or_default().to_string(),
            kernel: field(&v, "kernel", "meta.json")?.as_str().unwrap_or_default().to_string(),
            seed: v.get("seed").and_then(|s| s.as_i64()).unwrap_or(-1),
            torch: field(&v, "torch", "meta.json")?.as_str().unwrap_or_default().to_string(),
            constexprs: field(&v, "constexprs", "meta.json")?
                .as_object()
                .cloned()
                .ok_or_else(|| refuse("meta", "`constexprs` is not an object"))?,
            inputs,
            output: tensor(field(&v, "output", "meta.json")?, "output")?,
            reference_note: v
                .get("reference_note")
                .and_then(|s| s.as_str())
                .map(|s| s.to_string()),
            data_dir: v.get("data_dir").and_then(|s| s.as_str()).map(|s| s.to_string()),
            sha256: v.get("sha256").and_then(|m| m.as_object()).cloned(),
            oversized: v.get("oversized").and_then(|b| b.as_bool()).unwrap_or(false),
            staging: v.get("staging").and_then(|s| s.as_str()).map(|s| s.to_string()),
        })
    }

    /// An integer `tl.constexpr` this configuration was generated at.
    ///
    /// ⭐ A TEST READS ITS EXTENTS FROM HERE. Restating `D_MODEL = 4096` in a test file is how a
    /// tolerance derived for one extent comes to gate a comparison at another.
    pub fn int(&self, key: &str) -> Result<i64> {
        self.constexprs
            .get(key)
            .and_then(|v| v.as_i64())
            .ok_or_else(|| {
                refuse(
                    "meta",
                    format!("`{key}` is not an integer constexpr of `{}`", self.config),
                )
            })
    }

    /// A float `tl.constexpr`. Accepts an integer literal too -- JSON writes `12.0` as `12` when the
    /// generator's value happens to be integral, and refusing that would be a schema quibble.
    pub fn float(&self, key: &str) -> Result<f64> {
        self.constexprs
            .get(key)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| {
                refuse("meta", format!("`{key}` is not a numeric constexpr of `{}`", self.config))
            })
    }

    /// The bindings for [`crate::execute`], in `meta.json`'s own `inputs` order -- which the
    /// generator wrote in the `.py` kernel's parameter order, and which [`crate::execute`] checks
    /// against the lowered function's argument count.
    pub fn bindings(&self) -> Result<Vec<Binding>> {
        let d = self.staged_or_data_dir()?;
        self.inputs
            .iter()
            .map(|t| {
                Ok(Binding {
                    name: t.arg.clone(),
                    bytes: t.bytes(&d, self.sha256.as_ref())?,
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                })
            })
            .collect()
    }

    /// The directory to read from, REFUSING an oversized configuration whose bytes were never
    /// staged.
    ///
    /// ⛔⛔ AN ABSENT STAGED DIRECTORY IS NOT A PASS, AND THIS IS THE ONLY PLACE THAT CAN ENFORCE
    /// IT. `swiglu_mlp_granite_flat`'s inputs are 315,621,376 bytes of DENSELY-read weights -- no
    /// sparsification and no replication to exploit -- so they live on the pod and a test must be
    /// POINTED at them with `TRITON_NUMERIC_STAGED=<dir>`. A test that treated the missing
    /// directory as "nothing to check" would report green for the one configuration whose
    /// K-length accumulation is at Granite width, which is the config most likely to be wrong.
    pub fn staged_or_data_dir(&self) -> Result<PathBuf> {
        if !self.oversized {
            return Ok(self.data_dir());
        }
        let var = format!("TRITON_NUMERIC_STAGED_{}", self.config.to_uppercase());
        match staged_dir(&self.config) {
            Some(p) => Ok(p),
            None => Err(refuse(
                "staging",
                format!(
                    "`{}` is OVERSIZED: its inputs are not in the tree and no staged directory was                      given. Set `{var}` (or `TRITON_NUMERIC_STAGED`) to a directory holding them.                      {}",
                    self.config,
                    self.staging.as_deref().unwrap_or("No staging note was recorded."),
                ),
            )),
        }
    }

    /// The fixture's reference, widened to f64.
    ///
    /// The bytes on disk are f16 -- the dtype the kernel writes -- so this is `decode(f16)`, exactly
    /// the widening the executor's own read-back does to produce `Output::data`. Both sides of the
    /// comparison are therefore f16 values held in a wider type, and the comparison itself commits
    /// no rounding of its own.
    pub fn reference(&self) -> Result<Vec<f64>> {
        let d = self.reference_dir()?;
        let p = d.join(&self.output.file);
        let n = self.output.elements();
        let bytes = read_bin(&p, n * self.output.dtype.bytes_per_elem())?;
        verify_sha(&p, &self.output.file, &bytes, self.sha256.as_ref())?;
        decode_f64(&bytes, n, self.output.dtype)
    }

    /// Which directory this configuration's `ref_out.bin` is in.
    ///
    /// ⛔⛔ THE REFERENCE IS THIS CONFIGURATION'S ANSWER, AND READING A NEIGHBOUR'S IS THE FAILURE
    /// THIS FUNCTION EXISTS TO MAKE IMPOSSIBLE. `embedding_granite` and `embedding_granite_bm128`
    /// produce the SAME SHAPE from the SAME BYTES -- so `read_bin`'s length check, every extent
    /// check in this crate and every bracketing control in the test file would pass on the wrong
    /// file, exactly as `one_layer_and_two_layers_are_not_interchangeable` says of the decoders.
    ///
    /// ⭐ SO THE RULE IS: BESIDE ITS OWN `meta.json`, OR A **HASH-VERIFIED** SIBLING, AND NOTHING
    /// ELSE. `gen_numeric_data.py` deletes the data files from a `data_dir` config's directory and
    /// records their `sha256` in its `meta.json` -- after checking byte-identity against the
    /// sibling's `sha256.txt` and refusing to share if it ever stops holding. So the recorded hash
    /// is THIS configuration's statement of what its answer is, and [`Fixture::reference`] requires
    /// the bytes it reads to match it. A sibling whose `ref_out.bin` ever diverged would fail here
    /// rather than be compared.
    ///
    /// ⚖️ AND AN UNVERIFIABLE SIBLING IS A REFUSAL, NOT A FALLBACK. No recorded hash means the
    /// identity is an assumption, and an assumption is what this whole reader refuses elsewhere.
    fn reference_dir(&self) -> Result<PathBuf> {
        if self.oversized {
            return self.staged_or_data_dir();
        }
        let own = dir(&self.config);
        if own.join(&self.output.file).is_file() {
            return Ok(own);
        }
        let Some(sibling) = &self.data_dir else {
            return Err(refuse(
                "data",
                format!(
                    "`{}` has no `{}` of its own and names no `data_dir`",
                    self.config, self.output.file
                ),
            ));
        };
        let recorded = self
            .sha256
            .as_ref()
            .and_then(|m| m.get(&self.output.file))
            .and_then(|v| v.as_str());
        if recorded.is_none() {
            return Err(refuse(
                "data",
                format!(
                    "`{}`'s `{}` lives in `{sibling}/` and its own meta.json records no sha256 for                      it. REFUSING to compare against an unverified sibling: the two share a shape,                      so nothing else in this crate could tell the wrong answer from the right one.",
                    self.config, self.output.file
                ),
            ));
        }
        Ok(dir(sibling))
    }
}

/// Check one file's bytes against the sha256 the configuration recorded for it, when it recorded
/// one.
///
/// ⛔ WHY THE CHECK IS HERE AND NOT IN A SCRIPT. `sha256.txt` in the tree records what the
/// generator wrote; nothing re-read it at TEST time, so a shared or staged file that changed after
/// generation would be compared silently. The two cases where `meta.json` carries a `sha256` map are
/// exactly the two where the bytes are NOT beside their own record -- a `data_dir` sibling and a
/// staged directory -- which is where "the bytes and their record agree" stops being trivially true.
///
/// A file with no recorded hash is NOT an error: most configurations keep their bytes beside their
/// `meta.json` and the generator records no map for them. The refusal for a missing hash where one
/// is REQUIRED lives at its own call site ([`Fixture::reference_dir`]), because only the caller
/// knows whether the identity is load-bearing.
pub fn verify_sha(
    p: &Path,
    file: &str,
    raw: &[u8],
    shas: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Result<()> {
    let Some(want) = shas.and_then(|m| m.get(file)).and_then(|v| v.as_str()) else {
        return Ok(());
    };
    let got = sha256_hex(raw);
    if got != want {
        return Err(refuse(
            "data",
            format!(
                "{} hashes to\n  {got}\nand its own meta.json records\n  {want}\nThese bytes are \
                 not the bytes this configuration was generated against.",
                p.display()
            ),
        ));
    }
    Ok(())
}

/// SHA-256 of `bytes`, lowercase hex.
///
/// ⚖️ WRITTEN HERE RATHER THAN TAKEN AS A DEPENDENCY, for the reason this crate's manifest gives
/// about every edge: the pod is offline and `cargo --offline` has to resolve. This is FIPS 180-4
/// §6.2 with no options and no streaming API -- 60 lines against a `sha2` dependency and its
/// transitive `cpufeatures`/`generic-array` tail. It is checked against the standard's own vectors
/// in [`tests`], which is the whole reason a hand-rolled hash is safe to write: a wrong one fails
/// the vectors, not the harness.
pub fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    // The padded message, block by block, without materialising a second copy of `bytes`: the
    // 384 MiB embedding table is one of the inputs this hashes.
    let bitlen = (bytes.len() as u64).wrapping_mul(8);
    let mut tail = Vec::with_capacity(128);
    tail.push(0x80u8);
    while (bytes.len() + tail.len()) % 64 != 56 {
        tail.push(0);
    }
    tail.extend_from_slice(&bitlen.to_be_bytes());
    let full = bytes.len() / 64;
    let mut block = [0u8; 64];
    for i in 0..full {
        block.copy_from_slice(&bytes[i * 64..(i + 1) * 64]);
        sha256_block(&block, &mut h, &K);
    }
    // The last partial block of `bytes` joined to the padding, then whatever padding is left.
    let rest = &bytes[full * 64..];
    let mut joined = Vec::with_capacity(rest.len() + tail.len());
    joined.extend_from_slice(rest);
    joined.extend_from_slice(&tail);
    for chunk in joined.as_chunks::<64>().0 {
        block.copy_from_slice(chunk);
        sha256_block(&block, &mut h, &K);
    }
    let mut out = String::with_capacity(64);
    for w in h {
        out.push_str(&format!("{w:08x}"));
    }
    out
}

/// One 64-byte block of the SHA-256 compression function.
fn sha256_block(block: &[u8; 64], h: &mut [u32; 8], k: &[u32; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[4 * i],
            block[4 * i + 1],
            block[4 * i + 2],
            block[4 * i + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
        (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(k[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
}

/// Read a raw binary and REFUSE a length that is not the one the shape implies.
///
/// ⛔ THE LENGTH CHECK IS THE POINT. A truncated or stale `.bin` read into a shorter buffer would
/// compare fine over its own length; a `.bin` from a different extent would compare fine over the
/// shorter of the two. Both are silently wrong answers, which is the failure mode this whole crate
/// exists to catch, so the reader refuses before a single element is compared.
pub fn read_bin(p: &Path, expect_bytes: usize) -> Result<Vec<u8>> {
    let bytes = std::fs::read(p)
        .map_err(|e| refuse("data", format!("cannot read {}: {e}", p.display())))?;
    if bytes.len() != expect_bytes {
        return Err(refuse(
            "data",
            format!(
                "{} is {} bytes; its recorded shape and dtype say {expect_bytes}. A comparison \
                 over the shorter of two lengths cannot see a shape defect.",
                p.display(),
                bytes.len()
            ),
        ));
    }
    Ok(bytes)
}

/// Decode raw little-endian bytes of `dtype` to f64.
///
/// f16 goes through `ktir_core::codec::decode`, THE SAME decoder the executor's read-back uses, so
/// the two sides of a comparison cannot disagree about what an f16 bit pattern means.
pub fn decode_f64(bytes: &[u8], n: usize, dtype: DType) -> Result<Vec<f64>> {
    match dtype {
        DType::F16 | DType::F32 => {
            let f32s = ktir_core::codec::decode(bytes, n, dtype);
            Ok(f32s.into_iter().map(|v| v as f64).collect())
        }
        DType::I32 => Ok(bytes
            .as_chunks::<4>().0.iter()
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect()),
        other => Err(refuse(
            "data",
            format!("decoding {other:?} to f64 is not implemented; add it deliberately"),
        )),
    }
}

/// Every configuration that HAS checked-in data, so a sweep can report the rest as MISSING rather
/// than silently covering fewer than it names.
pub fn present() -> Vec<String> {
    triton_ktir_superdsc::cases::ALL
        .iter()
        .filter(|c| dir(c).join("meta.json").is_file())
        .map(|c| c.to_string())
        .collect()
}

/// A one-line provenance banner for a test's output, so a run's report says WHICH bytes it compared.
pub fn banner(f: &Fixture) -> String {
    format!(
        "{} <- test/numeric/{}/  (fixture {}.{}, seed {}, torch {}{})",
        f.config,
        f.config,
        f.fixture,
        f.kernel,
        f.seed,
        f.torch,
        match &f.reference_note {
            Some(n) => format!(", reference: {n}"),
            None => String::new(),
        }
    )
}

#[allow(dead_code)]
fn _refusal_is_used(_: Refusal) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⛔ THE HASH IS ONLY WORTH HAVING IF IT IS THE REAL ONE. FIPS 180-4's own published vectors,
    /// plus the empty string and a 1,000,000-byte input, which is the only case that exercises the
    /// multi-block loop AND a padding block that does not fit beside the message's tail.
    #[test]
    fn sha256_matches_the_published_vectors() {
        for (input, want) in [
            (
                "".to_string(),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                "abc".to_string(),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".to_string(),
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
            (
                "a".repeat(1_000_000),
                "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
            ),
        ] {
            assert_eq!(sha256_hex(input.as_bytes()), want, "input of {} bytes", input.len());
        }
        // THE THREE PADDING BOUNDARIES, from `hashlib.sha256(bytes(n))`: 55 bytes is the last
        // length whose padding fits in one block, 56 is the first that needs a second, and 64 is an
        // exact block followed by a whole padding block. That is where an off-by-one in the
        // `% 64 != 56` loop or in the joined-tail split would live, and none of the four published
        // vectors above touches all three.
        for (n, want) in [
            (55usize, "02779466cdec163811d078815c633f21901413081449002f24aa3e80f0b88ef7"),
            (56, "d4817aa5497628e7c77e6b606107042bbba3130888c5f47a375e6179be789fbb"),
            (64, "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"),
        ] {
            assert_eq!(sha256_hex(&vec![0u8; n]), want, "{n} zero bytes");
        }
    }

    /// ⛔ AND THE CHECK REFUSES, which is the half a hash comparison usually forgets to prove.
    /// `verify_sha` returning `Ok(())` for a file with no recorded hash is the common case, so a
    /// broken implementation that returned `Ok(())` for EVERYTHING would pass every other test in
    /// this crate. Three cases: no record (pass through), the right record (accept), the wrong
    /// record (REFUSE, naming both digests).
    #[test]
    fn a_hash_that_does_not_match_is_refused() {
        let p = Path::new("in_desc_ids.bin");
        let raw = b"not the bytes".to_vec();
        let right = sha256_hex(&raw);
        let mut m = serde_json::Map::new();

        assert!(
            verify_sha(p, "in_desc_ids.bin", &raw, Some(&m)).is_ok(),
            "a file the configuration records no hash for must pass through: most configurations \
             keep their bytes beside their own meta.json and the generator writes no map for them"
        );
        m.insert("in_desc_ids.bin".into(), serde_json::Value::String(right));
        assert!(
            verify_sha(p, "in_desc_ids.bin", &raw, Some(&m)).is_ok(),
            "the recorded hash IS this file's hash and it was refused"
        );
        m.insert("in_desc_ids.bin".into(), serde_json::Value::String("00".repeat(32)));
        let e = verify_sha(p, "in_desc_ids.bin", &raw, Some(&m))
            .expect_err("a hash that does not match MUST refuse, or the check is decoration");
        assert!(
            e.message.contains(&sha256_hex(&raw)) && e.message.contains(&"00".repeat(32)),
            "the refusal must name BOTH digests so a reader can tell a regenerated file from a \
             mis-read one: {e}"
        );
    }

    /// ⭐ AND IT AGREES WITH THE GENERATOR, ON THE BYTES IN THE TREE. A hash that passes the
    /// standard's vectors and disagrees with `gen_numeric_data.py`'s `hashlib.sha256` about a real
    /// file would mean the reader and the record are hashing different things -- a file read in the
    /// wrong mode, say. So this checks one recorded file end to end.
    #[test]
    fn the_recorded_sha_of_a_checked_in_file_reproduces() {
        let f = Fixture::load("embedding_granite_bm128").expect("the bm128 metadata");
        let shas = f.sha256.as_ref().expect("bm128 records its sha256 map, it shares its bytes");
        let file = "in_desc_ids.bin";
        let want = shas.get(file).and_then(|v| v.as_str()).expect(file);
        let raw = std::fs::read(f.data_dir().join(file)).expect("the ids");
        assert_eq!(sha256_hex(&raw), want, "{file}");
    }
}
