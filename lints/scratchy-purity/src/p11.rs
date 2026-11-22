//! P11, lock and comment hygiene.

use crate::comments;
use clippy_utils::diagnostics::span_lint_and_help;
use std::path::Path;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::{declare_lint, declare_lint_pass};

declare_lint! {
    /// P11: no escape-hatch constructors. A checked type has one checking constructor, and it is
    /// the only way in, so a `from_untyped` or `conflated_with_*` beside it is a second way in.
    pub P11_ESCAPE_HATCH_CONSTRUCTOR,
    Deny,
    "an escape-hatch constructor beside a checked type's checking constructor"
}

declare_lint! {
    /// P11: comments describe the code as it is, in the present tense. A comment that narrates
    /// the code's history (a tombstone, a move) describes code that is no longer there.
    pub P11_TOMBSTONE_COMMENT,
    Deny,
    "a comment that narrates history instead of describing the code"
}

declare_lint! {
    /// P11: a comment never points at scratch files. A `/tmp/` path is gone on the next boot,
    /// and a plan or report there is not part of the repository.
    pub P11_TMP_PATH_COMMENT,
    Deny,
    "a comment that cites a /tmp/ path"
}

declare_lint_pass!(P11 => [P11_ESCAPE_HATCH_CONSTRUCTOR, P11_TOMBSTONE_COMMENT, P11_TMP_PATH_COMMENT]);

pub static LINTS: &[&rustc_lint::Lint] =
    &[P11_ESCAPE_HATCH_CONSTRUCTOR, P11_TOMBSTONE_COMMENT, P11_TMP_PATH_COMMENT];

const TOMBSTONE_HELP: &str = "say what the code does now, or delete the comment";
const TMP_HELP: &str = "cite the in-tree source, or use a placeholder such as <out-dir> in a command example";

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
                "P11: this comment narrates history (`{phrase}`); comments describe the code as it is, in the present tense{suffix}"
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
                "P11: this comment cites a scratch path (`/tmp/`); a comment never points at scratch files{suffix}"
            ),
            None,
            TMP_HELP,
        );
    }
}

/// Every comment in one source file, each with its span.
fn file_comments(file: &rustc_span::SourceFile, found: &mut Vec<(rustc_span::Span, String)>) {
    let Some(src) = file.src.as_deref() else {
        return;
    };
    let mut offset = 0u32;
    for token in rustc_lexer::tokenize(src, rustc_lexer::FrontmatterAllowed::No) {
        let len = token.len;
        if matches!(
            token.kind,
            rustc_lexer::TokenKind::LineComment { .. } | rustc_lexer::TokenKind::BlockComment { .. }
        ) {
            let text = &src[offset as usize..(offset + len) as usize];
            let lo = file.start_pos + rustc_span::BytePos(offset);
            let span = rustc_span::Span::with_root_ctxt(lo, lo + rustc_span::BytePos(len));
            found.push((span, comment_body(text).to_owned()));
        }
        offset += len;
    }
}

/// Whether this crate is its package's owner target, the one that checks the whole package.
fn owns_package(sess: &rustc_session::Session, owner: &Option<std::path::PathBuf>, root: &Path) -> bool {
    !sess.opts.test && owner.as_deref() == Some(root)
}

/// The crate root, relative to its package's manifest directory where it is under it.
fn crate_root(sess: &rustc_session::Session, manifest_dir: &Path) -> Option<std::path::PathBuf> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let root = cwd.join(sess.local_crate_source_file()?.local_path()?);
    Some(root.strip_prefix(manifest_dir).map(Path::to_path_buf).unwrap_or(root))
}

/// The comments this crate checks. The package's owner target (`comments::owner_target`) reads
/// every file of the package, compiled or not; a loose single-file crate reads the files it
/// compiles; every other target of a package reads none, since its owner already did.
fn checked_comments(cx: &EarlyContext<'_>, manifest_dir: &Path) -> Vec<(rustc_span::Span, String)> {
    let sess = cx.sess();
    let cwd = std::env::current_dir().unwrap_or_default();
    let Some(root) = crate_root(sess, manifest_dir) else {
        return Vec::new();
    };
    let manifest = std::fs::read_to_string(manifest_dir.join("Cargo.toml")).unwrap_or_default();
    let owner = comments::owner_target(&manifest, |p| manifest_dir.join(p).exists());
    let mut found = Vec::new();
    if owns_package(sess, &owner, &root) {
        for path in comments::package_files(manifest_dir) {
            if let Ok(file) = sess.source_map().load_file(&path) {
                file_comments(&file, &mut found);
            }
        }
    } else if comments::loose(&root) {
        for file in sess.source_map().files().iter() {
            let rustc_span::FileName::Real(real) = &file.name else {
                continue;
            };
            let local = !file.is_imported() && real.local_path().is_some_and(|p| comments::scanned(&cwd.join(p), manifest_dir));
            if local {
                file_comments(file, &mut found);
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
        for (span, body) in checked_comments(cx, manifest_dir) {
            check_comment(cx, span, &body, "");
        }
        let manifest = std::fs::read_to_string(manifest_dir.join("Cargo.toml")).unwrap_or_default();
        let owner = comments::owner_target(&manifest, |p| manifest_dir.join(p).exists());
        let owns = crate_root(cx.sess(), manifest_dir).is_some_and(|root| owns_package(cx.sess(), &owner, &root));
        if owns {
            for (line, body) in comments::toml_comments(&manifest) {
                check_comment(cx, krate.spans.inner_span.shrink_to_lo(), body, &format!(" (Cargo.toml:{line})"));
            }
        }
    }

    fn check_fn(
        &mut self,
        cx: &EarlyContext<'_>,
        kind: rustc_ast::visit::FnKind<'_>,
        _: rustc_span::Span,
        _: rustc_ast::NodeId,
    ) {
        let rustc_ast::visit::FnKind::Fn(_, _, f) = kind else {
            return;
        };
        let ident = f.ident;
        if !escape_hatch(ident.as_str()) || ident.span.in_external_macro(cx.sess().source_map()) {
            return;
        }
        span_lint_and_help(
            cx,
            P11_ESCAPE_HATCH_CONSTRUCTOR,
            ident.span,
            format!(
                "P11: `{ident}` is an escape-hatch constructor; a checked type has one checking constructor, the only way in"
            ),
            None,
            "delete it, and mint the value through the type's checking constructor",
        );
    }
}
