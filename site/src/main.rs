//! Renders the GitHub Pages site into site/_site:
//!
//!   /                   the landing page
//!   /architectures.html every model architecture's DSL, diffable
//!   /book/              docs/*.md + CONTRIBUTING.md, rendered client-side by zero-md
//!   /metal.html         Metal benchmark runs from site/data/metal/
//!
//! Every page is rendered here, at build time, with Dioxus's server-side
//! renderer: the browser gets HTML, Carbon's web components, and a few lines
//! of script per page — no wasm.
//!
//! Usage, from the repo root:
//!
//!   cargo run --release --manifest-path site/Cargo.toml            render once
//!   cargo run --release --manifest-path site/Cargo.toml -- serve   render, serve on
//!       http://localhost:8000/, and re-render (pages reload) on every change;
//!       `--port N` picks another port, `--no-open` skips opening a browser

mod archs;
mod book;
mod carbon;
mod chrome;
mod highlight;
mod landing;
mod metal;
mod serve;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Where the site's inputs are and where it is written.
pub struct Site {
    /// site/: styles, favicon, content, data.
    pub root: PathBuf,
    /// The repo root, for the docs and the DSL files.
    pub repo: PathBuf,
    /// site/_site/, rebuilt from empty every run.
    pub out: PathBuf,
}

/// Static files copied as they are.
const ASSETS: [&str; 2] = ["styles.css", "favicon.png"];

/// `n` with thousands separators, `1,580`.
pub fn thousands(n: usize) -> String {
    thousands_f(n as f64, 0)
}

/// `v` to `decimals` places with thousands separators, `12,345.7`.
pub fn thousands_f(v: f64, decimals: usize) -> String {
    let s = format!("{v:.decimals$}");
    let (sign, s) = s.strip_prefix('-').map_or(("", s.as_str()), |r| ("-", r));
    let (int, frac) = s.split_once('.').map_or((s, None), |(i, f)| (i, Some(f)));
    let mut grouped = String::new();
    for (i, c) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(c);
    }
    match frac {
        Some(f) => format!("{sign}{grouped}.{f}"),
        None => format!("{sign}{grouped}"),
    }
}

fn copy(from: &Path, to: &Path) -> Result<(), String> {
    fs::copy(from, to)
        .map(|_| ())
        .map_err(|e| format!("{}: {e}", from.display()))
}

fn build(site: &Site) -> Result<(), String> {
    match fs::remove_dir_all(&site.out) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("{}: {e}", site.out.display())),
    }
    fs::create_dir_all(site.out.join("book"))
        .map_err(|e| format!("{}: {e}", site.out.display()))?;

    for asset in ASSETS {
        copy(&site.root.join(asset), &site.out.join(asset))?;
    }
    fs::write(site.out.join("index.html"), landing::page()?).map_err(|e| e.to_string())?;
    book::build(site)?;
    println!("{}", archs::build(site)?);
    println!("{}", metal::build(site)?);

    let broken = book::broken_links(site)?;
    if !broken.is_empty() {
        return Err(broken
            .iter()
            .map(|b| format!("broken link: {b}"))
            .chain(["link check failed".to_string()])
            .collect::<Vec<_>>()
            .join("\n"));
    }
    println!("site assembled at {}", site.out.display());
    Ok(())
}

/// What to do, from the command line.
enum Command {
    /// Render the site once.
    Build,
    /// Render, serve, and re-render on every change.
    Serve { port: u16, open: bool },
}

const USAGE: &str = "usage: scratchy-site [serve [--port N] [--no-open]]";

fn parse(args: &[String]) -> Result<Command, String> {
    let Some((first, rest)) = args.split_first() else {
        return Ok(Command::Build);
    };
    if first != "serve" {
        return Err(USAGE.to_string());
    }
    let (mut port, mut open) = (8000, true);
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--port" => {
                port = it
                    .next()
                    .and_then(|p| p.parse().ok())
                    .ok_or_else(|| format!("--port needs a port number\n{USAGE}"))?;
            }
            "--no-open" => open = false,
            _ => return Err(USAGE.to_string()),
        }
    }
    Ok(Command::Serve { port, open })
}

fn main() -> ExitCode {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let site = Site {
        repo: root
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| root.join("..")),
        out: root.join("_site"),
        root,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let run = parse(&args).and_then(|cmd| match cmd {
        Command::Build => build(&site),
        Command::Serve { port, open } => serve::run(&site, port, open),
    });
    match run {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::thousands_f;

    #[test]
    fn thousands_groups_the_integer_part_only() {
        assert_eq!(thousands_f(1234567.891, 1), "1,234,567.9");
        assert_eq!(thousands_f(999.0, 0), "999");
        assert_eq!(thousands_f(-1000.0, 2), "-1,000.00");
    }
}
