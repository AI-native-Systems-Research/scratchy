//! `serve`: render the site, serve _site/ over HTTP, and re-render whenever an
//! input changes; open pages reload themselves.
//!
//! A change to the renderer's own source (or to what it compiles in) rebuilds
//! the binary and replaces this process with the new one, on the same port.

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime};

use notify_debouncer_mini::new_debouncer;
use notify_debouncer_mini::notify::RecursiveMode;

use crate::{ASSETS, Site, archs, book, metal};

/// How long the watcher waits for a burst of writes (an editor's save, a
/// `git checkout`) to settle before acting on it once.
const SETTLE: Duration = Duration::from_millis(200);

/// Where open pages listen for "reload".
const RELOAD_PATH: &str = "/__reload";

/// Added to every page served (never to the built site): reload on a
/// re-render, and after a restart once the server is back.
const RELOAD_JS: &str = "<script>
(function () {
  var es = new EventSource('/__reload'), lost = false;
  es.onmessage = function () { location.reload(); };
  es.onerror = function () { lost = true; };
  es.onopen = function () { if (lost) location.reload(); };
})();
</script>
";

const TYPES: [(&str, &str); 7] = [
    ("html", "text/html; charset=utf-8"),
    ("css", "text/css; charset=utf-8"),
    ("js", "text/javascript; charset=utf-8"),
    ("json", "application/json"),
    ("md", "text/markdown; charset=utf-8"),
    ("png", "image/png"),
    ("svg", "image/svg+xml"),
];

/// What a change means for the running server.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
enum Change {
    /// Something a render reads: render again.
    Content,
    /// The renderer itself, or a file it compiles in: rebuild and restart.
    Source,
}

/// Every input, and what changing it means. Content comes from the same
/// tables the render reads, so nothing it reads goes unwatched.
fn inputs(site: &Site) -> Vec<(PathBuf, Change)> {
    let mut v: Vec<(PathBuf, Change)> = book::sources()
        .map(|s| (site.repo.join(s), Change::Content))
        .chain(ASSETS.iter().map(|a| (site.root.join(a), Change::Content)))
        .collect();
    v.push((site.repo.join(archs::DSL), Change::Content));
    v.push((site.root.join(metal::DATA), Change::Content));
    for s in ["src", "theme", "Cargo.toml", "Cargo.lock"] {
        v.push((site.root.join(s), Change::Source));
    }
    v
}

/// What a changed path means, if anything.
fn classify(inputs: &[(PathBuf, Change)], path: &Path) -> Option<Change> {
    inputs
        .iter()
        .filter(|(p, _)| path.starts_with(p))
        .map(|(_, c)| *c)
        .reduce(|a, b| if a > b { a } else { b })
}

/// Size and modification time of every file under the inputs.
type Fingerprint = BTreeMap<PathBuf, (u64, Option<SystemTime>)>;

fn fingerprint(inputs: &[(PathBuf, Change)]) -> Fingerprint {
    fn walk(path: &Path, into: &mut Fingerprint) {
        let Ok(meta) = fs::metadata(path) else { return };
        if meta.is_dir() {
            for entry in fs::read_dir(path).into_iter().flatten().flatten() {
                walk(&entry.path(), into);
            }
        } else {
            into.insert(path.to_path_buf(), (meta.len(), meta.modified().ok()));
        }
    }
    let mut fp = Fingerprint::new();
    for (path, _) in inputs {
        walk(path, &mut fp);
    }
    fp
}

pub fn run(site: &Site, port: u16, open: bool) -> Result<(), String> {
    // Serve even a failed render: the fix is one save away.
    if let Err(e) = crate::build(site) {
        eprintln!("{e}");
    }
    let listener =
        TcpListener::bind(("127.0.0.1", port)).map_err(|e| format!("port {port}: {e}"))?;
    let clients: Arc<Mutex<Vec<TcpStream>>> = Arc::default();
    {
        let (out, clients) = (site.out.clone(), Arc::clone(&clients));
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (out, clients) = (out.clone(), Arc::clone(&clients));
                thread::spawn(move || {
                    if let Err(e) = handle(stream, &out, &clients) {
                        eprintln!("serve: {e}");
                    }
                });
            }
        });
    }

    let url = format!("http://localhost:{port}/");
    println!("serving {} at {url} (ctrl-c stops)", site.out.display());
    if open {
        let opener = if cfg!(target_os = "macos") {
            "open"
        } else {
            "xdg-open"
        };
        if Command::new(opener).arg(&url).status().is_err() {
            println!("open a browser to {url}");
        }
    }

    let inputs = inputs(site);
    let (tx, rx) = std::sync::mpsc::channel();
    let mut debouncer = new_debouncer(SETTLE, tx).map_err(|e| e.to_string())?;
    // A file is watched through its directory: editors save by replacing the
    // file, which would orphan a watch on the file itself. One watch per
    // directory, recursive if any input needs it.
    let mut dirs: BTreeMap<&Path, bool> = BTreeMap::new();
    for (path, _) in &inputs {
        let (dir, recursive) = if path.is_dir() {
            (path.as_path(), true)
        } else {
            (path.parent().unwrap_or(path), false)
        };
        *dirs.entry(dir).or_default() |= recursive;
    }
    for (dir, recursive) in dirs {
        let mode = if recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };
        debouncer
            .watcher()
            .watch(dir, mode)
            .map_err(|e| format!("watch {}: {e}", dir.display()))?;
    }

    // Events only wake the loop; what changed is read off the inputs' own
    // fingerprints. macOS reports events on files a render merely copies, so
    // trusting events re-renders forever.
    let mut seen = fingerprint(&inputs);
    for events in rx {
        events.map_err(|e| e.to_string())?;
        let now = fingerprint(&inputs);
        let changed = now
            .iter()
            .filter(|(path, stamp)| seen.get(*path) != Some(stamp))
            .map(|(path, _)| path)
            .chain(seen.keys().filter(|path| !now.contains_key(*path)))
            .filter_map(|path| classify(&inputs, path))
            .reduce(|a, b| if a > b { a } else { b });
        seen = now;
        match changed {
            None => {}
            Some(Change::Content) => match crate::build(site) {
                Ok(()) => reload(&clients),
                Err(e) => eprintln!("{e}"),
            },
            Some(Change::Source) => restart(site, port)?,
        }
    }
    Ok(())
}

/// Rebuild the renderer and become it, keeping the port; open pages reconnect
/// and reload. A failed build keeps this one running.
fn restart(site: &Site, port: u16) -> Result<(), String> {
    println!("renderer changed: rebuilding");
    let mut cargo = Command::new(env!("CARGO"));
    cargo
        .arg("build")
        .arg("--manifest-path")
        .arg(site.root.join("Cargo.toml"));
    if !cfg!(debug_assertions) {
        cargo.arg("--release");
    }
    match cargo.status() {
        Ok(s) if s.success() => {
            let exe = std::env::current_exe().map_err(|e| e.to_string())?;
            // exec only returns on failure.
            let e = Command::new(exe)
                .args(["serve", "--port", &port.to_string(), "--no-open"])
                .exec();
            Err(format!("restart: {e}"))
        }
        Ok(_) => {
            eprintln!("renderer build failed; still serving the previous one");
            Ok(())
        }
        Err(e) => Err(format!("cargo: {e}")),
    }
}

fn reload(clients: &Mutex<Vec<TcpStream>>) {
    if let Ok(mut clients) = clients.lock() {
        clients.retain_mut(|c| c.write_all(b"data: reload\n\n").is_ok());
    }
}

fn handle(stream: TcpStream, out: &Path, clients: &Mutex<Vec<TcpStream>>) -> io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request = String::new();
    reader.read_line(&mut request)?;
    // The headers say nothing this server needs.
    let mut header = String::new();
    while reader.read_line(&mut header)? > 2 {
        header.clear();
    }
    let mut stream = stream;
    let mut parts = request.split_whitespace();
    let (method, target) = (
        parts.next().unwrap_or_default(),
        parts.next().unwrap_or_default(),
    );
    if method != "GET" && method != "HEAD" {
        return respond(
            &mut stream,
            "405 Method Not Allowed",
            "text/plain",
            b"GET only\n",
            true,
        );
    }
    let path = target.split(['?', '#']).next().unwrap_or_default();
    if path == RELOAD_PATH {
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\n\r\n")?;
        if let Ok(mut clients) = clients.lock() {
            clients.push(stream);
        }
        return Ok(());
    }
    let Some(file) = resolve(out, path) else {
        return respond(
            &mut stream,
            "404 Not Found",
            "text/plain",
            b"not found\n",
            method == "GET",
        );
    };
    let ext = file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();
    let mime = TYPES
        .iter()
        .find(|(e, _)| *e == ext)
        .map_or("application/octet-stream", |(_, m)| m);
    let mut body = fs::read(&file)?;
    if ext == "html" {
        let text = String::from_utf8_lossy(&body);
        let at = text.rfind("</body>").unwrap_or(text.len());
        body = format!("{}{RELOAD_JS}{}", &text[..at], &text[at..]).into_bytes();
    }
    respond(&mut stream, "200 OK", mime, &body, method == "GET")
}

fn respond(
    stream: &mut TcpStream,
    status: &str,
    mime: &str,
    body: &[u8],
    with_body: bool,
) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    if with_body {
        stream.write_all(body)?;
    }
    Ok(())
}

/// The file a URL path names under `out`: percent-decoded, never above it,
/// a directory meaning its index.html.
fn resolve(out: &Path, path: &str) -> Option<PathBuf> {
    let path = percent_decode(path)?;
    let mut file = out.to_path_buf();
    for seg in path.split('/').filter(|s| !s.is_empty() && *s != ".") {
        if seg == ".." {
            return None;
        }
        file.push(seg);
    }
    if file.is_dir() {
        file.push("index.html");
    }
    file.is_file().then_some(file)
}

fn percent_decode(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let hex = std::str::from_utf8(b.get(i + 1..i + 3)?).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_never_leaves_the_site() {
        let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        assert_eq!(resolve(&out, "/main.rs"), Some(out.join("main.rs")));
        assert_eq!(resolve(&out, "/../Cargo.toml"), None);
        assert_eq!(resolve(&out, "/%2e%2e/Cargo.toml"), None);
    }

    #[test]
    fn source_outranks_content() {
        let inputs = vec![
            (PathBuf::from("/s"), Change::Content),
            (PathBuf::from("/s/src"), Change::Source),
        ];
        assert!(classify(&inputs, Path::new("/s/src/main.rs")) == Some(Change::Source));
        assert!(classify(&inputs, Path::new("/s/data.json")) == Some(Change::Content));
        assert!(classify(&inputs, Path::new("/elsewhere")).is_none());
    }
}
