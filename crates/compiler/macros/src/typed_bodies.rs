// SPDX-License-Identifier: Apache-2.0
//! Test-only: a program through the typed-Rust emit pass and rustc, as the build script and this
//! crate's compile take every carrier, and then run. So a test's own DSL body gets its shapes from
//! rustc's type checker, as the bake's carriers do, and a program whose shapes are wrong is
//! refused with rustc's own message.
//!
//! The program compiles as a binary of its own: the signature module (`typed_shapes.rs`, its
//! `include!` of the build script's output replaced by this program's emitted module), the dim
//! vocabulary (`dim.rs`) and the two ids the trace keys by. Running it prints the trace.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::classified::{LocalId, Program, WeightId};
use crate::dim::Dim;
use crate::shape::Inferred;
use crate::weights_manifest::WeightsManifest;
use scratchy_forward_compiler_frontend::typed_emit;

/// A DSL body (no decorator), classified as a decoder's, typed against `manifest` and traced.
pub(crate) fn trace_body(src: &str, manifest: &WeightsManifest) -> (Program, Inferred) {
    let ast = crate::parse_python::parse_body(src).expect("parse DSL");
    trace_ast(&ast, src, manifest)
}

/// [`trace_body`] for a body's parsed `ast` (`src` its text, for the emitted comments).
pub(crate) fn trace_ast(
    ast: &crate::ast::Ast,
    src: &str,
    manifest: &WeightsManifest,
) -> (Program, Inferred) {
    let program = crate::classify::classify(ast).expect("classify");
    let shapes = typed(&program, src, "body.py", manifest, &Default::default())
        .unwrap_or_else(|e| panic!("the body does not type:\n{e}"));
    (program, shapes)
}

/// A program of the carrier `stem` (`dsl/<stem>.py`), typed against its own `weights.json` and
/// configs, and traced: for a test that lowers some other program of an arch than the bake's.
pub(crate) fn trace_carrier_program(stem: &str, program: &Program) -> Inferred {
    let arch = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/arch");
    let file = format!("crates/models/arch/dsl/{stem}.py");
    let source = std::fs::read_to_string(arch.join(format!("dsl/{stem}.py"))).expect("carrier");
    let carrier = crate::parse_python::parse_python_file(&source).expect("parse");
    let configs = arch.join("configs").join(stem);
    let manifest = crate::weights_manifest::load_or_empty(&configs).expect("manifest");
    let facts = scratchy_forward_compiler_frontend::carrier_facts::carrier_facts(
        &arch,
        stem,
        carrier.vision,
        &mut Vec::new(),
    );
    typed(program, &source, &file, &manifest, &facts)
        .unwrap_or_else(|e| panic!("{file}: the program does not type:\n{e}"))
}

/// `program` (classified from `source`, the file `file`) through the emit pass and rustc, and run:
/// its trace, or the emit pass's refusal or rustc's errors.
pub(crate) fn typed(
    program: &Program,
    source: &str,
    file: &str,
    manifest: &WeightsManifest,
    facts: &typed_emit::CarrierFacts,
) -> Result<Inferred, String> {
    let carrier = typed_emit::Carrier {
        module: "body",
        arch: "Body",
        file,
        source,
        program,
        manifest,
        facts,
    };
    let mut bounds = BTreeSet::new();
    let module = typed_emit::emit(&carrier, &mut bounds)?;
    let emitted = format!(
        "{}\n{module}\n{}",
        typed_emit::emit_markers(&bounds)?,
        typed_emit::emit_dispatch(&[("Body".to_string(), "body".to_string())]),
    );
    let dir = scratch_dir();
    let main = dir.join("main.rs");
    std::fs::write(&main, standalone(&emitted)).expect("write the program");
    let bin = dir.join("body");
    let out = std::process::Command::new(env!("TYPED_SHAPES_RUSTC"))
        .args([
            "--edition",
            "2024",
            "--crate-type",
            "bin",
            "--crate-name",
            "body",
        ])
        // Wide enough that a diagnostic quotes the emitted line whole, its carrier comment too.
        .arg("--diagnostic-width=1000")
        .arg("-o")
        .arg(&bin)
        .arg(&main)
        .output()
        .expect("run rustc");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    let run = std::process::Command::new(&bin).output().expect("run");
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
    Ok(parse_trace(&String::from_utf8_lossy(&run.stdout)))
}

/// A fresh directory under this crate's `OUT_DIR` for one program.
fn scratch_dir() -> PathBuf {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let dir = Path::new(env!("OUT_DIR"))
        .join("typed-bodies")
        .join(format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// The binary's source: the signatures with `emitted` in place of the build script's output, the
/// dim vocabulary, the ids, and a `main` printing the trace.
fn standalone(emitted: &str) -> String {
    let src = Path::new(env!("CARGO_MANIFEST_DIR"));
    let read = |p: &str| std::fs::read_to_string(src.join(p)).expect(p);
    let include = "include!(concat!(env!(\"OUT_DIR\"), \"/typed_carriers.rs\"));";
    let signatures = read("src/typed_shapes.rs");
    assert!(
        signatures.contains(include),
        "typed_shapes.rs includes the build script's output"
    );
    format!(
        "mod dim {{\n{dim}\n}}\n\
         mod classified {{\n\
         \x20   #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]\n\
         \x20   pub struct LocalId(pub u32);\n\
         \x20   #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]\n\
         \x20   pub struct WeightId(pub u32);\n\
         }}\n\
         mod typed_shapes {{\n{signatures}\n}}\n\
         {MAIN}",
        dim = read("../frontend/src/dim.rs"),
        signatures = signatures.replace(include, emitted),
    )
}

/// The binary's `main`: one line per local (`L id dims`), weight (`W id dims`) and nested value
/// (`N target n dims`), each dim in [`parse_dim`]'s notation, then the program's locals' names in
/// `LocalId` order (`P name`).
const MAIN: &str = r#"
fn dim(d: &dim::Dim) -> String {
    let list = |v: &[dim::Dim]| v.iter().map(dim).collect::<Vec<_>>().join(",");
    match d {
        dim::Dim::Lit(n) => n.to_string(),
        dim::Dim::Bound(b) => b.clone(),
        dim::Dim::Mul(v) => format!("*({})", list(v)),
        dim::Dim::Add(v) => format!("+({})", list(v)),
        dim::Dim::Div(a, b) => format!("/({},{})", dim(a), dim(b)),
    }
}

fn shape(s: &[dim::Dim]) -> String {
    s.iter().map(dim).collect::<Vec<_>>().join(" ")
}

fn main() {
    let shapes = typed_shapes::traced("Body").expect("typed");
    for (id, s) in &shapes.locals {
        println!("L {} {}", id.0, shape(s));
    }
    for (id, s) in &shapes.weights {
        println!("W {} {}", id.0, shape(s));
    }
    for ((target, n), s) in &shapes.nested {
        println!("N {} {} {}", target.0, n, shape(s));
    }
    for name in &shapes.program {
        println!("P {name}");
    }
}
"#;

/// The trace the binary printed.
fn parse_trace(text: &str) -> Inferred {
    let mut shapes = Inferred::default();
    for line in text.lines() {
        let mut words = line.split(' ');
        let kind = words.next().expect("a record");
        let mut id = || -> u32 { words.next().expect("an id").parse().expect("an id") };
        match kind {
            "L" => {
                let local = LocalId(id());
                shapes.locals.insert(local, parse_shape(words));
            }
            "W" => {
                let weight = WeightId(id());
                shapes.weights.insert(weight, parse_shape(words));
            }
            "N" => {
                let target = LocalId(id());
                let n = id();
                shapes.nested.insert((target, n), parse_shape(words));
            }
            "P" => shapes
                .program
                .push(words.next().expect("a local's name").to_string()),
            other => panic!("unknown trace record `{other}`"),
        }
    }
    shapes
}

fn parse_shape<'a>(words: impl Iterator<Item = &'a str>) -> Vec<Dim> {
    (words.filter(|w| !w.is_empty()))
        .map(|w| {
            let (d, rest) = parse_dim(w);
            assert!(rest.is_empty(), "trailing `{rest}` in `{w}`");
            d
        })
        .collect()
}

/// One dim: a literal, a bound's name, or `*(..)`, `+(..)`, `/(a,b)` over dims; and what follows.
fn parse_dim(s: &str) -> (Dim, &str) {
    fn list(mut s: &str) -> (Vec<Dim>, &str) {
        let mut out = Vec::new();
        loop {
            let (d, rest) = parse_dim(s);
            out.push(d);
            match rest.as_bytes().first() {
                Some(b',') => s = &rest[1..],
                Some(b')') => return (out, &rest[1..]),
                _ => panic!("a dim list ends at `)`: `{rest}`"),
            }
        }
    }
    if let Some(rest) = s.strip_prefix("*(") {
        let (v, rest) = list(rest);
        (Dim::Mul(v), rest)
    } else if let Some(rest) = s.strip_prefix("+(") {
        let (v, rest) = list(rest);
        (Dim::Add(v), rest)
    } else if let Some(rest) = s.strip_prefix("/(") {
        let (mut v, rest) = list(rest);
        assert_eq!(v.len(), 2, "a quotient has two dims");
        let den = v.pop().expect("denominator");
        let num = v.pop().expect("numerator");
        (Dim::Div(Box::new(num), Box::new(den)), rest)
    } else {
        let end = s.find([',', ')']).unwrap_or(s.len());
        let (word, rest) = s.split_at(end);
        let d = match word.parse() {
            Ok(n) => Dim::Lit(n),
            Err(_) => Dim::Bound(word.to_string()),
        };
        (d, rest)
    }
}
