// SPDX-License-Identifier: Apache-2.0
//! Every `weights.json` declaration against a real checkpoint. Each fixture under
//! `tests/weight_headers/` is one public checkpoint's safetensors header entries (its tensors outside
//! the layer stacks and the first layer of each layer kind, range-read with no weights downloaded),
//! with the committed configs it is checked under. Each weight a carrier reads is found under the
//! name the loader reads it by (`codegen::safetensors_prefix`; a head's under its
//! `checkpoint_prefix`) and must hold the shape its declaration states under the config's bounds.
//!
//! Where a config names the layout of another conversion of the same weights (MLX's
//! `language_model.model.*` for HF's `model.language_model.*`), the fixture states the renaming
//! (`prefixes`), and a weight a head borrows from its target, the target's name for it (`names`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::classified::{Expr, OpKind, Program, Stmt, WeightId};
use crate::config::ModelParams;
use crate::dim::{Dim, Shape};
use crate::weights_manifest::{ManifestEntryKind, WeightsManifest};

/// A checkpoint's tensors' shapes by name.
type Header = BTreeMap<String, Vec<u64>>;

fn arch_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/arch")
}

/// One fixture checked under one config: what the bake lowers for that model, and the checkpoint.
struct Checked {
    name: String,
    carrier: String,
    model: ModelParams,
    program: Program,
    manifest: WeightsManifest,
    header: Header,
    /// The loader's name prefixes as this checkpoint spells them.
    prefixes: Vec<(String, String)>,
    /// A borrowed weight's name in this checkpoint, by its path.
    names: BTreeMap<String, String>,
    /// The weights this model reads: its unrolled graph's.
    read: std::collections::HashSet<WeightId>,
    /// Those an op reads whole, as a group of tensors (a MoE's experts, a Gated-DeltaNet or
    /// Mamba-2 mixer): their manifest entry names no one tensor's shape, so they are not checked
    /// here.
    groups: std::collections::HashSet<WeightId>,
}

/// The weights `model` reads of the carrier `arch` (its program `program`, as the bake lowers it):
/// those its unrolled graph's nodes take, and those of them an op reads whole as a group.
fn read_by(
    arch: &str,
    program: &Program,
    model: &ModelParams,
) -> (
    std::collections::HashSet<WeightId>,
    std::collections::HashSet<WeightId>,
) {
    use crate::classified::OpKind;
    let inferred = crate::typed_shapes::traced(arch).expect("a typed program");
    let mut prog = program.clone();
    crate::shape::erase_identity_regroupings(&mut prog, &inferred, &model.bounds);
    let cfg = crate::cfg::build_cfg(&prog, model).expect("cfg");
    let fuf = crate::fuf::unroll(&cfg, &inferred).expect("unroll");
    let weights = |nodes: &mut dyn Iterator<Item = &crate::fuf::FufNode>| {
        nodes
            .flat_map(|n| n.inputs.iter())
            .filter_map(|i| match i {
                crate::fuf::FufInput::Weight { id, .. } => Some(*id),
                _ => None,
            })
            .collect()
    };
    let read = weights(&mut fuf.nodes.iter());
    let groups = weights(&mut fuf.nodes.iter().filter(|n| {
        matches!(
            n.op,
            OpKind::Moe
                | OpKind::GemmaMoe
                | OpKind::GptOssMoe
                | OpKind::GatedDeltaNet
                | OpKind::Mamba2
        )
    }));
    (read, groups)
}

/// Every fixture with every config it is checked under (`<carrier>/<stem>`, or a head's
/// `<head carrier>/<target carrier>/<target stem>`: a head's config is its target's).
fn fixtures() -> Vec<Checked> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/weight_headers");
    let mut paths: Vec<_> = (std::fs::read_dir(&root).unwrap())
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    let configs = arch_dir().join("configs");
    let mut out = Vec::new();
    for path in paths {
        let fixture: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let header: Header = (fixture["tensors"].as_object().unwrap().iter())
            .map(|(t, s)| {
                let shape = s
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap())
                    .collect();
                (t.clone(), shape)
            })
            .collect();
        let pairs = |key: &str| -> Vec<(String, String)> {
            (fixture[key].as_array().into_iter().flatten())
                .map(|p| (p[0].as_str().unwrap().into(), p[1].as_str().unwrap().into()))
                .collect()
        };
        let prefixes = pairs("prefixes");
        let names: BTreeMap<String, String> = pairs("names").into_iter().collect();
        for config in fixture["configs"].as_array().unwrap() {
            let config = config.as_str().unwrap();
            let parts: Vec<&str> = config.split('/').collect();
            let (carrier, model) = match parts.as_slice() {
                [head, target, stem] => (
                    head.to_string(),
                    crate::config::load_head_file(
                        &configs.join(target).join(format!("{stem}.json")),
                        &configs.join(head),
                    ),
                ),
                [carrier, stem] => (
                    carrier.to_string(),
                    crate::config::load_file_declared(
                        &configs.join(carrier).join(format!("{stem}.json")),
                    ),
                ),
                _ => panic!("{config}: not a config"),
            };
            let model = model.unwrap_or_else(|e| panic!("{config}: {e:?}"));
            let source =
                std::fs::read_to_string(arch_dir().join(format!("dsl/{carrier}.py"))).unwrap();
            let parsed = crate::parse_python::parse_python_file(&source).unwrap();
            let mut program = crate::classify::carrier_program(&parsed).unwrap();
            // As the bake lowers it for this model (`compile_carrier`).
            if !parsed.vision {
                program.decoder_safetensors_prefix = model.arch.decoder_prefix.clone();
            }
            program.weight_leaf_renames = model.arch.weight_leaf_renames.clone();
            let manifest = crate::weights_manifest::load_or_empty(&configs.join(&carrier)).unwrap();
            let name = format!(
                "{} under {config}",
                path.file_name().unwrap().to_string_lossy()
            );
            let (read, groups) = read_by(&parsed.arch_name, &program, &model);
            out.push(Checked {
                name,
                carrier,
                model,
                program,
                manifest,
                header: header.clone(),
                prefixes: prefixes.clone(),
                names: names.clone(),
                read,
                groups,
            });
        }
    }
    out
}

/// The weights an embedding-style op reads by row (an `[rows, width]` table, stored as declared).
fn tables(program: &Program) -> Vec<WeightId> {
    fn walk(stmts: &[Stmt], out: &mut Vec<WeightId>) {
        for s in stmts {
            match s {
                Stmt::Assign { value, .. } | Stmt::AssignTuple { value, .. } => expr(value, out),
                Stmt::For { body, .. } => walk(body, out),
                Stmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    walk(then_body, out);
                    walk(else_body, out);
                }
            }
        }
    }
    fn expr(e: &Expr, out: &mut Vec<WeightId>) {
        match e {
            Expr::Call { op, args } => {
                if matches!(op, OpKind::Embed | OpKind::PosEmbed) {
                    out.extend(args.iter().filter_map(|a| match a {
                        Expr::Weight { id, .. } => Some(*id),
                        _ => None,
                    }));
                }
                args.iter().for_each(|a| expr(a, out));
            }
            Expr::Mul { lhs, rhs } => {
                expr(lhs, out);
                expr(rhs, out);
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(&program.statements, &mut out);
    out
}

/// Every index a path segment of the header carries (layers, vision blocks).
fn indices(header: &Header) -> Vec<u64> {
    let mut out: Vec<u64> = (header.keys())
        .flat_map(|t| t.split('.').filter_map(|s| s.parse().ok()))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The names the loader reads weight `id` by, as this checkpoint spells them: unindexed, then at
/// each index the header holds.
fn names_of(c: &Checked, id: WeightId) -> Vec<String> {
    let path = c.program.weights.path(id).join(".");
    if let Some(name) = c.names.get(&path) {
        return vec![name.clone()];
    }
    let naming = c.program.weight_naming();
    let raw = c.manifest.kind(c.program.weights.path(id)) == ManifestEntryKind::RawLinear;
    let respell = |name: String| -> String {
        // A head's tensors sit under its `checkpoint_prefix` in its target's checkpoint.
        let name = match &c.model.checkpoint_prefix {
            Some(p) => format!("{p}.{name}"),
            None => name,
        };
        (c.prefixes.iter())
            .find(|(ours, _)| name.starts_with(ours.as_str()))
            .map_or(name.clone(), |(ours, theirs)| {
                format!("{theirs}{}", &name[ours.len()..])
            })
    };
    std::iter::once(None)
        .chain(indices(&c.header).into_iter().map(Some))
        .flat_map(|ix| {
            let index = ix.map(scratchy_tensors::compiler_ids::UnrollIndex);
            let prefix = crate::codegen::safetensors_prefix(naming, &c.model.arch, id, index);
            match raw || path.ends_with("bias") {
                true => vec![respell(prefix)],
                false => vec![respell(format!("{prefix}.weight")), respell(prefix)],
            }
        })
        .collect()
}

/// A stored shape in the manifest's orientation: a matrix stored `[out, in]` read `[in, out]` (an
/// embedding table as stored), a kernel `[out, in, ..]` read `[in · .., out]`; first padded as the
/// loader pads it (`__pad_to_mult8__`, on the stored axis).
fn oriented(c: &Checked, id: WeightId, stored: &[u64]) -> Vec<u64> {
    let path = c.program.weights.path(id).join(".");
    let mut s = stored.to_vec();
    for hint in c.manifest.pad_to_mult8.iter().filter(|h| h.weight == path) {
        if let Some(d) = s.get_mut(hint.dim) {
            *d = d.div_ceil(8) * 8;
        }
    }
    let raw = c.manifest.kind(c.program.weights.path(id)) == ManifestEntryKind::RawLinear;
    match s.as_slice() {
        [rows, cols] if tables(&c.program).contains(&id) || raw => vec![*rows, *cols],
        [out, inp] => vec![*inp, *out],
        [out, rest @ ..] if !rest.is_empty() => vec![rest.iter().product(), *out],
        _ => s,
    }
}

fn eval(c: &Checked, d: &Dim) -> Option<u64> {
    crate::shape::eval_closed_dim(d, &c.model.bounds)
}

/// What the checkpoint says of weight `id` against `want` (its declaration, evaluated): `Ok` when a
/// tensor by one of its names holds it, else what it holds.
fn verdict(c: &Checked, id: WeightId, declared: &Shape, want: &[u64]) -> Result<(), String> {
    let path = c.program.weights.path(id).join(".");
    let found: Vec<(String, Vec<u64>)> = (names_of(c, id).into_iter())
        .filter_map(|n| c.header.get(&n).map(|s| (n.clone(), oriented(c, id, s))))
        .collect();
    if found.iter().any(|(_, s)| s == want) {
        return Ok(());
    }
    // A flat declaration of a tensor the loader reads flat (a conv's base kernel, a codebook): its
    // element count.
    if let [n] = want
        && let Some((_, stored)) =
            (names_of(c, id).iter()).find_map(|nm| c.header.get(nm).map(|s| (nm, s)))
        && stored.len() > 1
    {
        let count: u64 = stored.iter().product();
        return match count == *n {
            true => Ok(()),
            false => Err(format!(
                "`{path}`: declared flat {declared:?} = {n}, the checkpoint holds {stored:?} ({count})"
            )),
        };
    }
    // One projection the DSL reads, its rows several on-disk projections' (`__packed_concats__`):
    // each part as declared, and this one's rows their sum.
    if let Some(parts) = c.manifest.packed_concats.get(&path) {
        let leaf = path.rsplit('.').next().unwrap();
        let mut rows = 0;
        for part in parts {
            let part_leaf = part.rsplit('.').next().unwrap();
            let declared_part =
                (c.manifest.entries.get(part)).ok_or(format!("`{part}`: no declaration"))?;
            let want_part: Vec<u64> = (declared_part.iter())
                .map(|d| eval(c, d))
                .collect::<Option<_>>()
                .ok_or(format!("`{part}`: names a bound the config lacks"))?;
            let held = (names_of(c, id).into_iter())
                .map(|n| n.replace(&format!(".{leaf}."), &format!(".{part_leaf}.")))
                .find_map(|n| c.header.get(&n).map(|s| (n.clone(), s.clone())));
            match held {
                Some((n, s)) => {
                    let s = oriented(c, id, &s);
                    if s != want_part {
                        return Err(format!(
                            "`{part}` ({n}), a part of `{path}`: declared {declared_part:?} = {want_part:?}, the checkpoint holds {s:?}"
                        ));
                    }
                    rows += s.last().copied().unwrap_or(0);
                }
                None => {
                    return Err(format!(
                        "`{part}`, a part of `{path}`: not in the checkpoint"
                    ));
                }
            }
        }
        return match want.last() == Some(&rows) {
            true => Ok(()),
            false => Err(format!(
                "`{path}`: declared {declared:?} = {want:?}, its parts' rows sum to {rows}"
            )),
        };
    }
    // A head drafting over a prefix of its target's vocabulary reads the target's lm_head's first
    // `vocab_size` rows: the stored tensor is the target's, `target_vocab_size` rows.
    if let (Some(target), Some(&rows)) = (
        c.model.bounds.get("target_vocab_size"),
        c.model.bounds.get("vocab_size"),
    ) && path == "lm_head"
        && rows <= *target
    {
        let full: Vec<u64> = (want.iter())
            .map(|&d| if d == rows { *target } else { d })
            .collect();
        if found.iter().any(|(_, s)| *s == full) {
            return Ok(());
        }
    }
    // Packed with its siblings on disk (`__packed_splits__`): the parent's rows are their declared
    // outputs (a bias, their biases), summed, and its input width is this one's.
    let (base, is_bias) = match path.strip_suffix(".bias") {
        Some(base) => (base.to_string(), true),
        None => (path.clone(), false),
    };
    for (parent, parts) in &c.manifest.packed_splits {
        if !parts.contains(&base) {
            continue;
        }
        let leaf = base.rsplit('.').next().unwrap();
        let parent_leaf = parent.rsplit('.').next().unwrap();
        let declared_out = |p: &String| -> Option<u64> {
            let key = if is_bias {
                format!("{p}.bias")
            } else {
                p.clone()
            };
            let s = c.manifest.entries.get(&key)?;
            eval(c, s.last()?)
        };
        let rows: Option<u64> = parts.iter().map(declared_out).sum();
        for n in names_of(c, id) {
            let n = n.replace(&format!(".{leaf}."), &format!(".{parent_leaf}."));
            let n = match n.ends_with(&format!(".{leaf}")) {
                true => format!("{}{parent_leaf}", &n[..n.len() - leaf.len()]),
                false => n,
            };
            if let Some(stored) = c.header.get(&n) {
                let ok = match (stored.as_slice(), rows) {
                    ([r], Some(sum)) => is_bias && *r == sum,
                    ([r, inp], Some(sum)) => !is_bias && *r == sum && want.first() == Some(inp),
                    _ => false,
                };
                return match ok {
                    true => Ok(()),
                    false => Err(format!(
                        "`{path}` packed in {n} {stored:?}: declared {declared:?} = {want:?}, its \
                         siblings' sum {rows:?}"
                    )),
                };
            }
        }
    }
    // A tied head is its embedding table.
    if path == "lm_head" && c.model.tie_word_embeddings && found.is_empty() {
        let embed = (tables(&c.program).first().copied()).ok_or("a tied lm_head and no table")?;
        let table: Vec<(String, Vec<u64>)> = (names_of(c, embed).into_iter())
            .filter_map(|n| c.header.get(&n).map(|s| (n.clone(), s.clone())))
            .collect();
        return match table.first().map(|(n, s)| (n, s.as_slice())) {
            Some((_, [rows, cols])) if want == [*cols, *rows] => Ok(()),
            other => Err(format!(
                "`lm_head` tied: declared {want:?}, the table is {other:?}"
            )),
        };
    }
    Err(match found.first() {
        Some((n, s)) => {
            format!("`{path}` ({n}): declared {declared:?} = {want:?}, the checkpoint holds {s:?}")
        }
        None => format!(
            "`{path}`: none of {} in the checkpoint",
            names_of(c, id)
                .into_iter()
                .take(2)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    })
}

/// Each weight the carrier reads (the groups an op reads whole apart), its declaration under
/// `declare` (the manifest's, or a test's edit of it) evaluated under the model's bounds, against
/// the checkpoint: the mismatches, and how many weights matched.
fn mismatches(c: &Checked, declare: impl Fn(&str, &Shape) -> Shape) -> (Vec<String>, usize) {
    let mut wrong = Vec::new();
    let mut matched = 0;
    for id in c.program.weights.ids() {
        let path = c.program.weights.path(id).join(".");
        if c.groups.contains(&id) || !c.read.contains(&id) {
            continue;
        }
        let Some(declared) = c.manifest.lookup(c.program.weights.path(id)) else {
            wrong.push(format!("`{path}`: no declaration"));
            continue;
        };
        let declared = declare(&path, declared);
        let Some(want) = declared
            .iter()
            .map(|d| eval(c, d))
            .collect::<Option<Vec<u64>>>()
        else {
            wrong.push(format!(
                "`{path}`: {declared:?} names a bound the config lacks"
            ));
            continue;
        };
        match verdict(c, id, &declared, &want) {
            Ok(()) => matched += 1,
            Err(why) => wrong.push(why),
        }
    }
    (wrong, matched)
}

/// Every weight a carrier reads is declared as its checkpoint holds it: for each fixture and each
/// config it is checked under, each of the carrier's weights (found by the loader's name) holds the
/// shape its `weights.json` declaration states under the config's bounds.
#[test]
fn every_weights_json_entry_is_its_checkpoints() {
    let mut failed = Vec::new();
    for c in fixtures() {
        let (wrong, matched) = mismatches(&c, |_, d| d.clone());
        eprintln!("{}: {matched} weights match", c.name);
        if !wrong.is_empty() {
            failed.push(format!(
                "{} ({}):\n  {}",
                c.name,
                c.carrier,
                wrong.join("\n  ")
            ));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Every model each carrier compiles for: its own configs, or a head's targets' that carry it.
fn every_model(carrier: &str) -> Vec<ModelParams> {
    let configs = arch_dir().join("configs");
    let dir = configs.join(carrier);
    let model_configs = |dir: &Path| -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = (std::fs::read_dir(dir).into_iter().flatten().flatten())
            .map(|e| e.path())
            .filter(|p| {
                let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                n.ends_with(".json")
                    && !n.ends_with(".overrides.json")
                    && !matches!(n, "weights.json" | "quantizations.json" | "arch.json")
            })
            .collect();
        paths.sort();
        paths
    };
    let own = model_configs(&dir);
    if !own.is_empty() {
        let source = std::fs::read_to_string(arch_dir().join(format!("dsl/{carrier}.py"))).unwrap();
        let vision = crate::parse_python::parse_python_file(&source)
            .unwrap()
            .vision;
        return (own.iter())
            .map(|p| match vision {
                true => crate::config::load_file_declared_vision(p),
                false => crate::config::load_file_declared(p),
            })
            .map(|m| m.unwrap())
            .collect();
    }
    let spec = crate::config::load_arch_json(&dir).unwrap();
    let mut out = Vec::new();
    for target in &spec.checkpoint_of {
        for p in model_configs(&configs.join(target)) {
            let json: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
            if crate::config::carries_head(&spec, &json) {
                out.push(crate::config::load_head_file(&p, &dir).unwrap());
            }
        }
    }
    out
}

/// Every `weights.json` entry is checked against a checkpoint: each weight a carrier reads is
/// matched by some fixture, unless no committed config of the carrier reads it (a branch no
/// config takes: Nemotron-H's dense-MLP layers, its MTP head's Mamba-2 layers); a packed concat's
/// parts are checked with it; and no entry names a weight its carrier never reads.
#[test]
fn every_weights_json_entry_has_a_checkpoint() {
    let fixtures = fixtures();
    let mut uncovered = Vec::new();
    let mut unread = Vec::new();
    for dir in std::fs::read_dir(arch_dir().join("dsl"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("py"))
    {
        let carrier = dir.file_stem().unwrap().to_string_lossy().to_string();
        let manifest =
            crate::weights_manifest::load_or_empty(&arch_dir().join("configs").join(&carrier))
                .unwrap();
        let source = std::fs::read_to_string(&dir).unwrap();
        let parsed = crate::parse_python::parse_python_file(&source).unwrap();
        let program = crate::classify::carrier_program(&parsed).unwrap();
        let concat_parts: Vec<&String> = manifest.packed_concats.values().flatten().collect();
        // Every committed model's reads.
        let mut reached = std::collections::HashSet::new();
        for model in every_model(&carrier) {
            let mut prog = program.clone();
            if !parsed.vision {
                prog.decoder_safetensors_prefix = model.arch.decoder_prefix.clone();
            }
            reached.extend(read_by(&parsed.arch_name, &prog, &model).0);
        }
        let checked = |id: WeightId| {
            fixtures
                .iter()
                .any(|c| c.carrier == carrier && c.read.contains(&id))
        };
        for entry in manifest.entries.keys() {
            let segs: Vec<&str> = entry.split('.').collect();
            match program.weights.find(&segs) {
                Some(id) if reached.contains(&id) && !checked(id) => {
                    uncovered.push(format!("{carrier}: `{entry}`"))
                }
                Some(id) if !reached.contains(&id) => unread.push(format!("{carrier}: `{entry}`")),
                Some(_) => {}
                None if concat_parts.contains(&entry) => {}
                None => uncovered.push(format!("{carrier}: `{entry}` names no weight it reads")),
            }
        }
    }
    eprintln!("read by no committed config, so checked by none: {unread:?}");
    assert!(
        uncovered.is_empty(),
        "no checkpoint fixture checks:\n  {}",
        uncovered.join("\n  ")
    );
}

/// The check refuses a wrong declaration: in every fixture, each weight of rank 2 the model reads,
/// declared with its two dims swapped (where that changes it), is refused, naming the weight.
#[test]
fn a_weight_declared_with_swapped_dims_is_refused_against_its_checkpoint() {
    let mut refused = 0;
    for c in fixtures() {
        for id in c.program.weights.ids() {
            let path = c.program.weights.path(id).join(".");
            let Some(declared) = c.manifest.lookup(c.program.weights.path(id)) else {
                continue;
            };
            if declared.len() != 2
                || !c.read.contains(&id)
                || eval(&c, &declared[0]) == eval(&c, &declared[1])
            {
                continue;
            }
            let (wrong, _) = mismatches(&c, |p, d| match p == path {
                true => vec![d[1].clone(), d[0].clone()],
                false => d.clone(),
            });
            assert!(
                wrong.len() == 1 && wrong[0].contains(&format!("`{path}`")),
                "{}: `{path}` swapped: {wrong:?}",
                c.name
            );
            refused += 1;
        }
    }
    assert!(refused >= 200, "{refused} swapped weights refused");
}
