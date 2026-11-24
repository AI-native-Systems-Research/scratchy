//! P11, lock and comment hygiene.

use crate::comments;
use clippy_utils::diagnostics::span_lint_and_help;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::{declare_lint, declare_lint_pass};
use std::path::Path;

declare_lint! {
    /// P11: no escape-hatch constructors. A checked type has one checking constructor, and it is
    /// the only way in, so a `from_untyped` or `conflated_with_*` beside it is a second way in.
    pub P11_ESCAPE_HATCH_CONSTRUCTOR,
    Forbid,
    "an escape-hatch constructor beside a checked type's checking constructor"
}

declare_lint! {
    /// P11: comments describe the code as it is, in the present tense. A comment that narrates
    /// the code's history (a tombstone, a move) describes code that is no longer there.
    pub P11_TOMBSTONE_COMMENT,
    Forbid,
    "a comment that narrates history instead of describing the code"
}

declare_lint! {
    /// P11: a comment never points at scratch files. A `/tmp/` path is gone on the next boot,
    /// and a plan or report there is not part of the repository.
    pub P11_TMP_PATH_COMMENT,
    Forbid,
    "a comment that cites a /tmp/ path"
}

declare_lint! {
    /// P11: no lint level in a `[lints]` table. For a purity lint, a level set there is a
    /// command-line opt-out of a rule, which a `Forbid` lint can't refuse from source.
    pub P11_LINT_OPT_OUT,
    Forbid,
    "a [lints] table that sets a purity lint's level"
}

declare_lint_pass!(P11 => [P11_ESCAPE_HATCH_CONSTRUCTOR, P11_TOMBSTONE_COMMENT, P11_TMP_PATH_COMMENT, P11_LINT_OPT_OUT]);

pub static LINTS: &[&rustc_lint::Lint] = &[
    P11_ESCAPE_HATCH_CONSTRUCTOR,
    P11_TOMBSTONE_COMMENT,
    P11_TMP_PATH_COMMENT,
    P11_LINT_OPT_OUT,
];

/// The package's manifest, then the nearest manifest above it with a `[workspace]` table.
fn manifests(manifest_dir: &Path) -> Vec<std::path::PathBuf> {
    let mut found = vec![manifest_dir.join("Cargo.toml")];
    let workspace = manifest_dir
        .ancestors()
        .skip(1)
        .map(|d| d.join("Cargo.toml"))
        .find(|m| {
            std::fs::read_to_string(m).is_ok_and(|t| {
                t.parse::<toml::Table>()
                    .is_ok_and(|t| t.contains_key("workspace"))
            })
        });
    found.extend(workspace);
    found
}

const TOMBSTONE_HELP: &str = "say what the code does now, or delete the comment";
const TMP_HELP: &str =
    "cite the in-tree source, or use a placeholder such as <out-dir> in a command example";

/// The text of a comment token, without its markers.
fn comment_body(token: &str) -> &str {
    if let Some(line) = token.strip_prefix("//") {
        line.strip_prefix(['/', '!']).unwrap_or(line)
    } else {
        let inner = token.strip_prefix("/*").unwrap_or(token);
        let inner = inner.strip_suffix("*/").unwrap_or(inner);
        inner.strip_prefix(['*', '!']).unwrap_or(inner)
    }
}

/// Runs both comment checks over one comment's text, at `span`. `suffix` locates comments that
/// have no span of their own (a manifest's), and is empty otherwise.
pub fn check_comment(cx: &EarlyContext<'_>, span: rustc_span::Span, body: &str, suffix: &str) {
    if let Some(phrase) = comments::tombstone(body) {
        span_lint_and_help(
            cx,
            P11_TOMBSTONE_COMMENT,
            span,
            format!(
                "P11: this comment narrates history (`{}`); comments describe the code as it is, in the present tense{suffix}",
                phrase.trim_end_matches(['`', ' '])
            ),
            None,
            TOMBSTONE_HELP,
        );
    }
    if comments::tmp_path(body).is_some() {
        span_lint_and_help(
            cx,
            P11_TMP_PATH_COMMENT,
            span,
            format!(
                "P11: this comment cites a path under the temp directory; a comment never points at scratch files{suffix}"
            ),
            None,
            TMP_HELP,
        );
    }
}

/// What one file's tokens hold for P11: its comments, and the names of the `fn`s it defines.
#[derive(Default)]
struct Found {
    comments: Vec<(rustc_span::Span, String)>,
    fns: Vec<(rustc_span::Span, String)>,
}

/// Every comment in one source file, and every name a `fn` keyword defines, each with its span.
/// Read from tokens, so code no build compiles (tests, a disabled feature) is read too.
fn file_tokens(file: &rustc_span::SourceFile, found: &mut Found) {
    let Some(src) = file.src.as_deref() else {
        return;
    };
    let mut offset = 0u32;
    let mut after_fn = false;
    for token in rustc_lexer::tokenize(src, rustc_lexer::FrontmatterAllowed::No) {
        let len = token.len;
        let text = &src[offset as usize..(offset + len) as usize];
        let lo = file.start_pos + rustc_span::BytePos(offset);
        let span = rustc_span::Span::with_root_ctxt(lo, lo + rustc_span::BytePos(len));
        match token.kind {
            rustc_lexer::TokenKind::LineComment { .. }
            | rustc_lexer::TokenKind::BlockComment { .. } => {
                found.comments.push((span, comment_body(text).to_owned()));
            }
            rustc_lexer::TokenKind::Whitespace => {}
            rustc_lexer::TokenKind::Ident => {
                if after_fn {
                    found.fns.push((span, text.to_owned()));
                }
                after_fn = text == "fn";
            }
            _ => after_fn = false,
        }
        offset += len;
    }
}

declare_lint_pass!(P11Track => []);

/// Before expansion, while rustc still collects its dep-info: records every file the owner target
/// reads (the package's files and manifests), so cargo re-runs these lints when one of them
/// changes, as it does for the files the crate compiles. The early pass runs after dep-info is
/// written, too late to record them itself.
impl EarlyLintPass for P11Track {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, _: &rustc_ast::Crate) {
        let Some(manifest_dir) = std::env::var_os("CARGO_MANIFEST_DIR") else {
            return;
        };
        let manifest_dir = Path::new(&manifest_dir);
        let sess = cx.sess();
        let manifest = std::fs::read_to_string(manifest_dir.join("Cargo.toml")).unwrap_or_default();
        let owner = comments::owner_target(&manifest, |p| manifest_dir.join(p).exists());
        if !crate_root(sess, manifest_dir).is_some_and(|root| owns_package(sess, &owner, &root)) {
            return;
        }
        let mut depinfo = sess.file_depinfo.borrow_mut();
        for path in comments::package_files(manifest_dir)
            .into_iter()
            .chain(manifests(manifest_dir))
        {
            depinfo.insert(rustc_span::Symbol::intern(&path.to_string_lossy()));
        }
    }
}

/// Whether this crate is its package's owner target, the one that checks the whole package.
pub(crate) fn owns_package(
    sess: &rustc_session::Session,
    owner: &Option<std::path::PathBuf>,
    root: &Path,
) -> bool {
    !sess.opts.test && owner.as_deref() == Some(root)
}

/// The crate root, relative to its package's manifest directory where it is under it.
pub(crate) fn crate_root(
    sess: &rustc_session::Session,
    manifest_dir: &Path,
) -> Option<std::path::PathBuf> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let root = cwd.join(sess.local_crate_source_file()?.local_path()?);
    Some(
        root.strip_prefix(manifest_dir)
            .map(Path::to_path_buf)
            .unwrap_or(root),
    )
}

/// The tokens this crate checks. The package's owner target (`comments::owner_target`) reads
/// every file of the package, compiled or not; a loose single-file crate reads the files it
/// compiles; every other target of a package reads none, since its owner already did.
fn checked_tokens(cx: &EarlyContext<'_>, manifest_dir: &Path) -> Found {
    let sess = cx.sess();
    let cwd = std::env::current_dir().unwrap_or_default();
    let Some(root) = crate_root(sess, manifest_dir) else {
        return Found::default();
    };
    let manifest = std::fs::read_to_string(manifest_dir.join("Cargo.toml")).unwrap_or_default();
    let owner = comments::owner_target(&manifest, |p| manifest_dir.join(p).exists());
    let mut found = Found::default();
    if owns_package(sess, &owner, &root) {
        for path in comments::package_files(manifest_dir) {
            if let Ok(file) = sess.source_map().load_file(&path) {
                file_tokens(&file, &mut found);
            }
        }
    } else if comments::loose(&root) {
        for file in sess.source_map().files().iter() {
            let rustc_span::FileName::Real(real) = &file.name else {
                continue;
            };
            let local = !file.is_imported()
                && real
                    .local_path()
                    .is_some_and(|p| comments::scanned(&cwd.join(p), manifest_dir));
            if local {
                file_tokens(file, &mut found);
            }
        }
    }
    found
}

fn escape_hatch(name: &str) -> bool {
    name == "from_untyped" || name.starts_with("conflated_with_")
}

impl EarlyLintPass for P11 {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, krate: &rustc_ast::Crate) {
        let Some(manifest_dir) = std::env::var_os("CARGO_MANIFEST_DIR") else {
            return;
        };
        let manifest_dir = Path::new(&manifest_dir);
        let found = checked_tokens(cx, manifest_dir);
        for (span, body) in &found.comments {
            check_comment(cx, *span, body, "");
        }
        for (span, name) in found.fns.iter().filter(|(_, name)| escape_hatch(name)) {
            span_lint_and_help(
                cx,
                P11_ESCAPE_HATCH_CONSTRUCTOR,
                *span,
                format!(
                    "P11: `{name}` is an escape-hatch constructor; a checked type has one checking constructor, the only way in"
                ),
                None,
                "delete it, and mint the value through the type's checking constructor",
            );
        }
        let manifest = std::fs::read_to_string(manifest_dir.join("Cargo.toml")).unwrap_or_default();
        let owner = comments::owner_target(&manifest, |p| manifest_dir.join(p).exists());
        let owns = crate_root(cx.sess(), manifest_dir)
            .is_some_and(|root| owns_package(cx.sess(), &owner, &root));
        if owns {
            let at = krate.spans.inner_span.shrink_to_lo();
            for (line, body) in comments::toml_comments(&manifest) {
                check_comment(cx, at, body, &format!(" (Cargo.toml:{line})"));
            }
            for path in manifests(manifest_dir) {
                let text = std::fs::read_to_string(&path).unwrap_or_default();
                for name in comments::lint_opt_outs(&text) {
                    span_lint_and_help(
                        cx,
                        P11_LINT_OPT_OUT,
                        at,
                        format!(
                            "P11: `{}` sets the level of `{name}`; no site opts out of a purity rule",
                            path.display()
                        ),
                        None,
                        "delete the entry, and fix the code the lint flags",
                    );
                }
            }
        }
    }
}
