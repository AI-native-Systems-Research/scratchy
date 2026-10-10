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

const STYLES: &str = "styles.css";
const FAVICON: &str = "favicon.png";

/// The static files, under site/; each is published under its content hash.
const ASSETS: [&str; 2] = [STYLES, FAVICON];

/// A file's published name: `<stem>.<hash>.<ext>`, the hash (64-bit FNV-1a)
/// of its bytes, so the name changes exactly when the file does.
fn hashed_name(name: &str, bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    match name.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}.{hash:016x}.{ext}"),
        None => format!("{name}.{hash:016x}"),
    }
}

/// Copies one static file into the site under its hashed name, which it returns.
fn publish(site: &Site, name: &str) -> Result<String, String> {
    let from = site.root.join(name);
    let bytes = fs::read(&from).map_err(|e| format!("{}: {e}", from.display()))?;
    let published = hashed_name(name, &bytes);
    let to = site.out.join(&published);
    fs::write(&to, bytes).map_err(|e| format!("{}: {e}", to.display()))?;
    Ok(published)
}

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

fn build(site: &Site) -> Result<(), String> {
    match fs::remove_dir_all(&site.out) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("{}: {e}", site.out.display())),
    }
    fs::create_dir_all(site.out.join("book"))
        .map_err(|e| format!("{}: {e}", site.out.display()))?;

    let assets = chrome::Published {
        styles: publish(site, STYLES)?,
        favicon: publish(site, FAVICON)?,
    };
    fs::write(site.out.join("index.html"), landing::page(&assets)?).map_err(|e| e.to_string())?;
    book::build(site, &assets)?;
    println!("{}", archs::build(site, &assets)?);
    println!("{}", metal::build(site, &assets)?);

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
    use super::{hashed_name, thousands_f};

    #[test]
    fn a_published_name_changes_exactly_when_its_bytes_do() {
        let a = hashed_name("styles.css", b"body { color: red }");
        assert_eq!(a, hashed_name("styles.css", b"body { color: red }"));
        assert_ne!(a, hashed_name("styles.css", b"body { color: blue }"));
        assert!(
            a.starts_with("styles.") && a.ends_with(".css") && a.len() == "styles..css".len() + 16
        );
    }

    #[test]
    fn thousands_groups_the_integer_part_only() {
        assert_eq!(thousands_f(1234567.891, 1), "1,234,567.9");
        assert_eq!(thousands_f(999.0, 0), "999");
        assert_eq!(thousands_f(-1000.0, 2), "-1,000.00");
    }
}
