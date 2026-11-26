//! P7, a real compiler ladder: no model, arch or stem name decides behavior.

use crate::{comments, exceptions, names, p11};
use clippy_utils::diagnostics::span_lint_and_help;
use rustc_ast::token::{self, LitKind};
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_ast::visit::{self, AssocCtxt, Visitor};
use rustc_ast::{
    AssocItem, AssocItemKind, AttrKind, Attribute, BinOpKind, Expr, ExprKind, Item, ItemKind,
    MetaItemInner, ModKind, Pat, PatKind, Ty, TyKind,
};
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::{SourceFile, Span, Symbol};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

declare_lint! {
    /// P7: no model, arch or stem name in logic. A string equal to a name the workspace declares
    /// (a models-crate feature, a config's `architectures` or `model_type`, a DSL arch file, an
    /// arch's family) decides nothing against an arch or stem value: not as a `match` or `let`
    /// pattern, an `==` operand, an allowlist member, or a `.contains` / `.starts_with` argument.
    /// The fact the name stands for is derived from the config, once. A checkpoint field whose
    /// values happen to be spelled like a model (a GGUF tokenizer kind) is that field, not an
    /// arch value, and is not this lint's business.
    pub P7_MODEL_NAME_IN_LOGIC,
    Forbid,
    "a model, arch or stem name deciding behavior"
}

declare_lint_pass!(P7 => [P7_MODEL_NAME_IN_LOGIC]);

pub static LINTS: &[&rustc_lint::Lint] = &[P7_MODEL_NAME_IN_LOGIC];

const HELP: &str = "state the fact the model name stands for (a shape, a flag the config carries) and decide on that";

/// What P7 reads from the workspace: its names, and its P7 exceptions (or why they don't parse).
struct Workspace {
    names: BTreeSet<String>,
    exempt: Result<Vec<String>, String>,
}

/// The workspace above `manifest_dir`, read once per compiler process.
fn workspace(manifest_dir: &Path) -> &'static Option<(PathBuf, Workspace)> {
    static WORKSPACE: OnceLock<Option<(PathBuf, Workspace)>> = OnceLock::new();
    WORKSPACE.get_or_init(|| {
        let root = names::workspace_root(manifest_dir)?;
        let text = std::fs::read_to_string(root.join(exceptions::FILE)).unwrap_or_default();
        let exempt = exceptions::parse(&text).map(|all| {
            all.into_iter()
                .filter(|e| e.rule == "P7")
                .map(|e| e.item)
                .collect()
        });
        let names = names::load(&root);
        Some((root, Workspace { names, exempt }))
    })
}

/// One file P7 checks: its path, its source file in the compiler's source map, and its
/// crate-qualified module path.
type Checked = (PathBuf, Arc<SourceFile>, Vec<String>);

/// The files this crate checks: the owner target reads every file of its package, a loose
/// single-file crate the files it compiles, and every other target none (see
/// `p11::checked_tokens`).
fn checked_files(cx: &EarlyContext<'_>, manifest_dir: &Path) -> Vec<Checked> {
    let sess = cx.sess();
    let Some(root) = p11::crate_root(sess, manifest_dir) else {
        return Vec::new();
    };
    let manifest = std::fs::read_to_string(manifest_dir.join("Cargo.toml")).unwrap_or_default();
    let owner = comments::owner_target(&manifest, |p| manifest_dir.join(p).exists());
    if p11::owns_package(sess, &owner, &root) {
        let krate = lib_name(&manifest).unwrap_or_else(|| crate_name(cx));
        let src = manifest_dir.join(root.parent().unwrap_or(Path::new("")));
        comments::package_files(manifest_dir)
            .into_iter()
            .filter_map(|path| {
                let file = sess.source_map().load_file(&path).ok()?;
                let module = module_path(&krate, &path, &src);
                Some((path, file, module))
            })
            .collect()
    } else if comments::loose(&root) {
        let cwd = std::env::current_dir().unwrap_or_default();
        let krate = crate_name(cx);
        let root = manifest_dir.join(&root);
        let src = root.parent().unwrap_or(Path::new("")).to_path_buf();
        sess.source_map()
            .files()
            .iter()
            .filter(|f| !f.is_imported())
            .filter_map(|f| match &f.name {
                rustc_span::FileName::Real(real) => {
                    real.local_path().map(|p| (cwd.join(p), Arc::clone(f)))
                }
                _ => None,
            })
            .filter(|(p, _)| comments::scanned(p, manifest_dir))
            .map(|(path, file)| {
                let module = if path == root {
                    vec![krate.clone()]
                } else {
                    module_path(&krate, &path, &src)
                };
                (path, file, module)
            })
            .collect()
    } else {
        Vec::new()
    }
}

/// The package's library crate name: `[lib] name`, else the package name with `-` as `_`.
fn lib_name(manifest: &str) -> Option<String> {
    let table: toml::Table = manifest.parse().ok()?;
    let lib = table
        .get("lib")
        .and_then(|l| l.get("name"))
        .and_then(toml::Value::as_str);
    let package = table
        .get("package")
        .and_then(|p| p.get("name"))
        .and_then(toml::Value::as_str);
    lib.or(package).map(|n| n.replace('-', "_"))
}

fn crate_name(cx: &EarlyContext<'_>) -> String {
    let sess = cx.sess();
    sess.opts.crate_name.clone().unwrap_or_else(|| {
        sess.local_crate_source_file()
            .and_then(|f| {
                f.local_path()
                    .and_then(|p| p.file_stem())
                    .map(|s| s.to_string_lossy().into_owned())
            })
            .unwrap_or_default()
    })
}

/// The module path of the file at `path`, for a crate rooted in `src`: `src/a/b.rs` and
/// `src/a/b/mod.rs` are `krate::a::b`, and the root file is `krate`.
pub fn module_path(krate: &str, path: &Path, src: &Path) -> Vec<String> {
    let mut module = vec![krate.to_owned()];
    let Ok(rel) = path.strip_prefix(src) else {
        return module;
    };
    let parts: Vec<String> = rel
        .with_extension("")
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    let last = parts.len().saturating_sub(1);
    for (i, part) in parts.into_iter().enumerate() {
        let root_file =
            i == last && (part == "mod" || (i == 0 && (part == "lib" || part == "main")));
        if !root_file {
            module.push(part);
        }
    }
    module
}

/// Whether a `cfg` predicate holds only under `test`: `test` itself, or an `all(..)` with it.
fn test_only(meta: &MetaItemInner) -> bool {
    let Some(item) = meta.meta_item() else {
        return false;
    };
    match item.name() {
        Some(name) if name.as_str() == "test" => item.is_word(),
        Some(name) if name.as_str() == "all" => item
            .meta_item_list()
            .is_some_and(|list| list.iter().any(test_only)),
        _ => false,
    }
}

/// Whether an item's attributes make it test code: `#[test]` (any path ending in `test`), or a
/// `#[cfg(..)]` that holds only under `test`.
fn is_test(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        let AttrKind::Normal(normal) = &attr.kind else {
            return false;
        };
        let last = normal
            .item
            .path
            .segments
            .last()
            .map(|s| s.ident.name.as_str().to_owned());
        match last.as_deref() {
            Some("test") => true,
            Some("cfg") if normal.item.path.segments.len() == 1 => attr
                .meta_item_list()
                .is_some_and(|list| list.len() == 1 && test_only(&list[0])),
            _ => false,
        }
    })
}

/// The `#[path = ".."]` an item carries.
fn path_attr(attrs: &[Attribute]) -> Option<String> {
    attrs
        .iter()
        .find(|a| a.has_name(Symbol::intern("path")))
        .and_then(Attribute::value_str)
        .map(|s| s.as_str().to_owned())
}

/// A string literal's span and text, looking through `&` and parentheses.
fn str_lit(e: &Expr) -> Option<(Span, Symbol)> {
    match &e.kind {
        ExprKind::Lit(token::Lit {
            kind: LitKind::Str | LitKind::StrRaw(_),
            symbol,
            ..
        }) => Some((e.span, *symbol)),
        ExprKind::AddrOf(_, _, inner) | ExprKind::Paren(inner) => str_lit(inner),
        _ => None,
    }
}

/// The string literals of an array expression (`["a", "b"]` or `&["a", "b"]`).
fn str_array(e: &Expr) -> Option<Vec<(Span, Symbol)>> {
    match &e.kind {
        ExprKind::Array(items) => Some(items.iter().filter_map(|i| str_lit(i)).collect()),
        ExprKind::AddrOf(_, _, inner) | ExprKind::Paren(inner) => str_array(inner),
        _ => None,
    }
}

/// Whether a type is `&str` (any lifetime).
fn is_str_ref(ty: &Ty) -> bool {
    match &ty.kind {
        TyKind::Ref(_, mt) => {
            matches!(&mt.ty.kind, TyKind::Path(None, p) if p.segments.len() == 1 && p.segments[0].ident.name.as_str() == "str")
        }
        TyKind::Paren(inner) => is_str_ref(inner),
        _ => false,
    }
}

/// Whether a type is a slice or array of `&str`, borrowed or not.
fn is_str_list(ty: &Ty) -> bool {
    match &ty.kind {
        TyKind::Ref(_, mt) => is_str_list(&mt.ty),
        TyKind::Slice(inner) | TyKind::Array(inner, _) => is_str_ref(inner),
        TyKind::Paren(inner) => is_str_list(inner),
        _ => false,
    }
}

/// The name a membership test reads its list from: `LIST` in `LIST.contains(..)`,
/// `LIST.iter().any(..)` and the like.
fn list_name(receiver: &Expr) -> Option<Symbol> {
    match &receiver.kind {
        ExprKind::Path(None, path) => path.segments.last().map(|s| s.ident.name),
        ExprKind::MethodCall(call)
            if ["iter", "into_iter", "as_slice"].contains(&call.seg.ident.name.as_str()) =>
        {
            list_name(&call.receiver)
        }
        ExprKind::AddrOf(_, _, inner) | ExprKind::Paren(inner) => list_name(inner),
        _ => None,
    }
}

const MEMBERSHIP: &[&str] = &[
    "contains",
    "any",
    "all",
    "position",
    "find",
    "binary_search",
];
const TEXT_MATCH: &[&str] = &[
    "contains",
    "starts_with",
    "ends_with",
    "strip_prefix",
    "strip_suffix",
];

/// What the walk over a package finds.
#[derive(Default)]
struct Findings {
    /// Each deciding literal.
    hits: Vec<(Span, Symbol)>,
    /// The `&str`-list consts and statics, by name, with their literals.
    lists: BTreeMap<Symbol, Vec<(Span, Symbol)>>,
    /// The names a membership test reads a list from.
    members: HashSet<Symbol>,
    /// The files a `#[cfg(test)] mod x;` declares, and the module directories under them.
    test_files: Vec<PathBuf>,
}

struct Walk<'a> {
    psess: &'a rustc_session::parse::ParseSess,
    exempt: &'a [String],
    /// The crate-qualified path of the item being walked.
    path: Vec<String>,
    /// The directory the current module's `mod x;` files are in.
    mod_dir: PathBuf,
    /// The locals of each enclosing `fn` bound from an arch or stem value.
    locals: Vec<HashSet<Symbol>>,
    found: Findings,
}

impl Walk<'_> {
    fn exempted(&self) -> bool {
        let path = self.path.join("::");
        self.exempt.iter().any(|e| exceptions::covers(e, &path))
    }

    fn hit(&mut self, lit: Option<(Span, Symbol)>) {
        self.found.hits.extend(lit);
    }

    fn list(&mut self, ident: Symbol, ty: &Ty, expr: Option<&Expr>) {
        if is_str_list(ty)
            && let Some(items) = expr.and_then(str_array)
        {
            self.found.lists.entry(ident).or_default().extend(items);
        }
    }

    /// A `matches!(scrutinee, pattern)`: its pattern decides when its scrutinee is an arch or
    /// stem value.
    fn matches_macro(&mut self, tokens: &TokenStream) {
        let mut trees = tokens.iter();
        let scrutinee: Vec<TokenTree> = trees
            .by_ref()
            .take_while(|t| !matches!(t, TokenTree::Token(tok, _) if tok.kind == token::Comma))
            .cloned()
            .collect();
        let pattern: Vec<TokenTree> = trees.cloned().collect();
        let mut parser =
            rustc_parse::parser::Parser::new(self.psess, TokenStream::new(scrutinee), None);
        let scrutinee = match parser.parse_expr() {
            Ok(e) => e,
            Err(e) => {
                e.cancel();
                return;
            }
        };
        let mut parser =
            rustc_parse::parser::Parser::new(self.psess, TokenStream::new(pattern), None);
        match parser.parse_pat_allow_top_guard(
            None,
            rustc_parse::parser::RecoverComma::No,
            rustc_parse::parser::RecoverColon::No,
            rustc_parse::parser::CommaRecoveryMode::LikelyTuple,
        ) {
            Ok(pat) => {
                self.decide(&scrutinee, &pat);
                self.visit_expr(&scrutinee);
                self.visit_pat(&pat);
            }
            Err(e) => e.cancel(),
        }
    }

    /// Whether an expression is an arch or stem value: it reads one (a binding, field or method
    /// whose name says arch, architecture, stem, family or model type, or a local bound from
    /// one), through borrows, conversions, `?` and fallbacks.
    fn arch_valued(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Path(None, path) => path.segments.last().is_some_and(|s| {
                arch_word(s.ident.name.as_str())
                    || (path.segments.len() == 1
                        && self.locals.iter().any(|l| l.contains(&s.ident.name)))
            }),
            ExprKind::Field(base, ident) => {
                arch_word(ident.name.as_str()) || self.arch_valued(base)
            }
            ExprKind::MethodCall(call) => {
                let method = call.seg.ident.name.as_str();
                arch_word(method)
                    || self.arch_valued(&call.receiver)
                    || ((method.starts_with("or") || method.starts_with("unwrap_or"))
                        && call.args.iter().any(|a| self.arch_valued(a)))
            }
            ExprKind::Call(f, _) => self.arch_valued(f),
            ExprKind::Closure(c) => self.arch_valued(&c.body),
            ExprKind::Tup(items) => items.iter().any(|i| self.arch_valued(i)),
            ExprKind::AddrOf(_, _, inner)
            | ExprKind::Paren(inner)
            | ExprKind::Unary(_, inner)
            | ExprKind::Try(inner)
            | ExprKind::Cast(inner, _)
            | ExprKind::Index(inner, _, _) => self.arch_valued(inner),
            _ => false,
        }
    }

    /// Whether a membership test's argument reads an arch or stem value: the value itself, or a
    /// predicate (`|f| *f == arch`) that compares one.
    fn mentions_arch(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Binary(_, l, r) => self.mentions_arch(l) || self.mentions_arch(r),
            ExprKind::Closure(c) => self.mentions_arch(&c.body),
            ExprKind::Block(block, _) => block.stmts.last().is_some_and(|s| match &s.kind {
                rustc_ast::StmtKind::Expr(e) => self.mentions_arch(e),
                _ => false,
            }),
            ExprKind::Paren(inner) => self.mentions_arch(inner),
            _ => self.arch_valued(e),
        }
    }

    /// Binds every name `pat` binds as an arch or stem value, when `from` is one.
    fn bind(&mut self, pat: &Pat, from: &Expr) {
        if self.arch_valued(from) {
            let mut names = Bindings::default();
            names.visit_pat(pat);
            if let Some(locals) = self.locals.last_mut() {
                locals.extend(names.0);
            }
        }
    }

    /// The string literals of `pat` decide when `scrutinee` is an arch or stem value.
    fn decide(&mut self, scrutinee: &Expr, pat: &Pat) {
        if self.arch_valued(scrutinee) {
            let mut lits = PatLits::default();
            lits.visit_pat(pat);
            self.found.hits.extend(lits.0);
        }
    }

    /// Walks a `fn` body with its own locals.
    fn fn_scope(&mut self, walk: impl FnOnce(&mut Self)) {
        self.locals.push(HashSet::new());
        walk(self);
        self.locals.pop();
    }

    fn item_scope(
        &mut self,
        name: Option<String>,
        attrs: &[Attribute],
        walk: impl FnOnce(&mut Self),
    ) {
        if is_test(attrs) {
            return;
        }
        let pushed = name.is_some();
        self.path.extend(name);
        if !self.exempted() {
            walk(self);
        }
        if pushed {
            self.path.pop();
        }
    }
}

/// The words that name an arch or stem value: `arch`, `arch_hint`, `hf_arches`, `stem`,
/// `model_type`, `family` and the like, as whole `_`-separated words.
fn arch_word(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let words: Vec<&str> = lower.split('_').collect();
    words.iter().any(|w| {
        [
            "arch",
            "arches",
            "architecture",
            "architectures",
            "stem",
            "stems",
            "family",
        ]
        .contains(w)
    }) || words.windows(2).any(|w| w == ["model", "type"])
}

/// The names a pattern binds.
#[derive(Default)]
struct Bindings(Vec<Symbol>);

impl<'ast> Visitor<'ast> for Bindings {
    fn visit_pat(&mut self, p: &'ast Pat) {
        if let PatKind::Ident(_, ident, _) = &p.kind {
            self.0.push(ident.name);
        }
        visit::walk_pat(self, p);
    }

    fn visit_expr(&mut self, _: &'ast Expr) {}
}

/// The string literals a pattern matches, outside its guards.
#[derive(Default)]
struct PatLits(Vec<(Span, Symbol)>);

impl<'ast> Visitor<'ast> for PatLits {
    fn visit_pat(&mut self, p: &'ast Pat) {
        if let PatKind::Expr(e) = &p.kind {
            self.0.extend(str_lit(e));
        }
        visit::walk_pat(self, p);
    }

    fn visit_expr(&mut self, _: &'ast Expr) {}
}

fn ty_name(ty: &Ty) -> Option<String> {
    match &ty.kind {
        TyKind::Path(_, path) => path.segments.last().map(|s| s.ident.to_string()),
        TyKind::Ref(_, mt) => ty_name(&mt.ty),
        _ => None,
    }
}

impl<'ast> Visitor<'ast> for Walk<'_> {
    fn visit_item(&mut self, item: &'ast Item) {
        if let ItemKind::Mod(_, ident, kind) = &item.kind {
            let name = ident.to_string();
            match kind {
                ModKind::Unloaded => {
                    if is_test(&item.attrs) {
                        let parent = self.mod_dir.clone();
                        match path_attr(&item.attrs) {
                            Some(p) => self.found.test_files.push(parent.join(p)),
                            None => {
                                self.found
                                    .test_files
                                    .push(parent.join(format!("{name}.rs")));
                                self.found.test_files.push(parent.join(&name));
                            }
                        }
                    }
                }
                ModKind::Loaded(..) => {
                    let dir = self.mod_dir.clone();
                    if is_test(&item.attrs) {
                        self.found.test_files.push(dir.join(&name));
                    }
                    self.mod_dir = dir.join(&name);
                    self.item_scope(Some(name), &item.attrs, |w| visit::walk_item(w, item));
                    self.mod_dir = dir;
                }
            }
            return;
        }
        let name = match &item.kind {
            ItemKind::Fn(f) => Some(f.ident.to_string()),
            ItemKind::Trait(t) => Some(t.ident.to_string()),
            ItemKind::Impl(i) => ty_name(&i.self_ty),
            ItemKind::Const(c) => Some(c.ident.to_string()),
            ItemKind::Static(s) => Some(s.ident.to_string()),
            _ => None,
        };
        if let ItemKind::MacCall(_) = &item.kind {
            return;
        }
        self.item_scope(name, &item.attrs, |w| {
            match &item.kind {
                ItemKind::Const(c) => w.list(c.ident.name, &c.ty, c.body.as_deref()),
                ItemKind::Static(s) => w.list(s.ident.name, &s.ty, s.expr.as_deref()),
                _ => {}
            }
            w.fn_scope(|w| visit::walk_item(w, item));
        });
    }

    fn visit_assoc_item(&mut self, item: &'ast AssocItem, ctxt: AssocCtxt) {
        let name = match &item.kind {
            AssocItemKind::Fn(f) => Some(f.ident.to_string()),
            AssocItemKind::Const(c) => Some(c.ident.to_string()),
            AssocItemKind::MacCall(_) => return,
            _ => None,
        };
        self.item_scope(name, &item.attrs, |w| {
            if let AssocItemKind::Const(c) = &item.kind {
                w.list(c.ident.name, &c.ty, c.body.as_deref());
            }
            w.fn_scope(|w| visit::walk_assoc_item(w, item, ctxt));
        });
    }

    fn visit_expr(&mut self, e: &'ast Expr) {
        if is_test(&e.attrs) {
            return;
        }
        match &e.kind {
            ExprKind::Binary(op, l, r) if matches!(op.node, BinOpKind::Eq | BinOpKind::Ne) => {
                if self.arch_valued(l) {
                    self.hit(str_lit(r));
                }
                if self.arch_valued(r) {
                    self.hit(str_lit(l));
                }
            }
            ExprKind::Match(scrutinee, arms, _) => {
                for arm in arms {
                    self.decide(scrutinee, &arm.pat);
                    self.bind(&arm.pat, scrutinee);
                }
            }
            ExprKind::Let(pat, scrutinee, ..) => {
                self.decide(scrutinee, pat);
                self.bind(pat, scrutinee);
            }
            ExprKind::ForLoop(f) => self.bind(&f.pat, &f.iter),
            ExprKind::MethodCall(call) => {
                let method = call.seg.ident.name.as_str();
                if TEXT_MATCH.contains(&method) && self.arch_valued(&call.receiver) {
                    self.hit(call.args.first().and_then(|a| str_lit(a)));
                }
                if MEMBERSHIP.contains(&method) && call.args.iter().any(|a| self.mentions_arch(a)) {
                    self.found.members.extend(list_name(&call.receiver));
                    if let Some(items) = str_array(&call.receiver) {
                        self.found.hits.extend(items);
                    }
                }
                // A closure over an arch list's items reads arch values.
                if self.arch_valued(&call.receiver) {
                    for arg in &call.args {
                        if let ExprKind::Closure(c) = &arg.kind {
                            for param in &c.fn_decl.inputs {
                                self.bind(&param.pat, &call.receiver);
                            }
                        }
                    }
                }
            }
            ExprKind::MacCall(mac) => {
                if mac
                    .path
                    .segments
                    .last()
                    .is_some_and(|s| s.ident.name.as_str() == "matches")
                {
                    self.matches_macro(&mac.args.tokens);
                }
                return;
            }
            _ => {}
        }
        visit::walk_expr(self, e);
    }

    fn visit_stmt(&mut self, s: &'ast rustc_ast::Stmt) {
        if let rustc_ast::StmtKind::MacCall(mac) = &s.kind {
            if is_test(&mac.attrs) {
                return;
            }
            if mac
                .mac
                .path
                .segments
                .last()
                .is_some_and(|s| s.ident.name.as_str() == "matches")
            {
                self.matches_macro(&mac.mac.args.tokens);
            }
            return;
        }
        visit::walk_stmt(self, s);
        if let rustc_ast::StmtKind::Let(local) = &s.kind
            && let Some(init) = local.kind.init()
        {
            self.bind(&local.pat, init);
        }
    }
}

/// The directory a file's `mod x;` declarations resolve in: its own directory for a root or
/// `mod.rs` file, else a directory named for it.
fn mod_dir(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or(Path::new("")).to_path_buf();
    match path.file_stem().and_then(|s| s.to_str()) {
        Some("lib" | "main" | "mod" | "build") | None => parent,
        Some(stem) => parent.join(stem),
    }
}

/// Whether a file is test code: under the package's `tests` directory, or a test module's file.
fn test_file(path: &Path, manifest_dir: &Path, test_files: &[PathBuf]) -> bool {
    path.strip_prefix(manifest_dir)
        .is_ok_and(|rest| rest.starts_with("tests"))
        || test_files
            .iter()
            .any(|t| path == t || (t.extension().is_none() && path.starts_with(t)))
}

/// Walks the non-test code of `files`, and returns the deciding name literals it holds.
fn scan(
    psess: &rustc_session::parse::ParseSess,
    files: &[Checked],
    manifest_dir: &Path,
    ws: &Workspace,
    exempt: &[String],
) -> Vec<(Span, String)> {
    let mut parsed = Vec::new();
    for (path, file, module) in files {
        // The file's own name and text: the source map hands back the file it already holds,
        // so the spans are the ones every other pass reports at.
        let Some(src) = file.src.as_deref() else {
            continue;
        };
        let Ok(mut parser) = rustc_parse::new_parser_from_source_str(
            psess,
            file.name.clone(),
            src.to_owned(),
            rustc_parse::lexer::StripTokens::ShebangAndFrontmatter,
        )
        .map_err(|errs| errs.into_iter().for_each(|e| e.cancel())) else {
            continue;
        };
        match parser.parse_crate_mod() {
            Ok(krate) => parsed.push((path.clone(), module.clone(), krate)),
            Err(e) => e.cancel(),
        }
    }
    let mut walk = Walk {
        psess,
        exempt,
        path: Vec::new(),
        mod_dir: PathBuf::new(),
        locals: vec![HashSet::new()],
        found: Findings::default(),
    };
    // A first walk finds the test modules, so every file they declare is known before the
    // walk that keeps its findings.
    for (path, _, krate) in &parsed {
        walk.mod_dir = mod_dir(path);
        for item in &krate.items {
            walk.visit_item(item);
        }
    }
    let test_files = std::mem::take(&mut walk.found.test_files);
    walk.found = Findings::default();
    for (path, module, krate) in &parsed {
        if test_file(path, manifest_dir, &test_files) || is_test(&krate.attrs) {
            continue;
        }
        walk.path = module.clone();
        walk.mod_dir = mod_dir(path);
        if walk.exempted() {
            continue;
        }
        for item in &krate.items {
            walk.visit_item(item);
        }
    }
    let Findings {
        mut hits,
        lists,
        members,
        ..
    } = walk.found;
    for (name, items) in lists {
        if members.contains(&name) {
            hits.extend(items);
        }
    }
    let mut out: Vec<(Span, String)> = hits
        .into_iter()
        .map(|(span, sym)| (span, sym.as_str().to_owned()))
        .filter(|(_, s)| ws.names.contains(s))
        .collect();
    out.sort_by_key(|(span, _)| (span.lo(), span.hi()));
    out.dedup_by_key(|(span, _)| (span.lo(), span.hi()));
    out
}

/// A parse session that shares the compiler's source map, so spans point into the files the
/// diagnostics render, and drops its own diagnostics: a file that is no module (an `include!`
/// fragment) is skipped, never reported.
fn silent_psess(sess: &rustc_session::Session) -> rustc_session::parse::ParseSess {
    let dcx = rustc_errors::DiagCtxt::new(Box::new(rustc_errors::emitter::SilentEmitter));
    rustc_session::parse::ParseSess::with_dcx(dcx, sess.psess.clone_source_map())
}

declare_lint_pass!(P7Track => []);

/// Records the files P7 reads beside the package's (the names' sources and the exceptions), so
/// cargo re-runs the lint when one changes. See `p11::P11Track` for why this is a pre-expansion
/// pass.
impl EarlyLintPass for P7Track {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, _: &rustc_ast::Crate) {
        let Some(manifest_dir) = std::env::var_os("CARGO_MANIFEST_DIR") else {
            return;
        };
        let manifest_dir = Path::new(&manifest_dir);
        let sess = cx.sess();
        let manifest = std::fs::read_to_string(manifest_dir.join("Cargo.toml")).unwrap_or_default();
        let owner = comments::owner_target(&manifest, |p| manifest_dir.join(p).exists());
        if !p11::crate_root(sess, manifest_dir)
            .is_some_and(|root| p11::owns_package(sess, &owner, &root))
        {
            return;
        }
        let Some(root) = names::workspace_root(manifest_dir) else {
            return;
        };
        let mut depinfo = sess.file_depinfo.borrow_mut();
        for path in names::sources(&root)
            .into_iter()
            .chain([root.join(exceptions::FILE)])
        {
            depinfo.insert(Symbol::intern(&path.to_string_lossy()));
        }
    }
}

impl EarlyLintPass for P7 {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, krate: &rustc_ast::Crate) {
        let Some(manifest_dir) = std::env::var_os("CARGO_MANIFEST_DIR") else {
            return;
        };
        let manifest_dir = Path::new(&manifest_dir);
        let Some((_, ws)) = workspace(manifest_dir) else {
            return;
        };
        let files = checked_files(cx, manifest_dir);
        if files.is_empty() {
            return;
        }
        let exempt: &[String] = match &ws.exempt {
            Ok(exempt) => exempt,
            Err(why) => {
                span_lint_and_help(
                    cx,
                    P7_MODEL_NAME_IN_LOGIC,
                    krate.spans.inner_span.shrink_to_lo(),
                    format!("P7: `{}` does not parse: {why}", exceptions::FILE),
                    None,
                    "give each [[exception]] its rule, item and approved, and nothing else",
                );
                &[]
            }
        };
        let psess = silent_psess(cx.sess());
        for (span, name) in scan(&psess, &files, manifest_dir, ws, exempt) {
            span_lint_and_help(
                cx,
                P7_MODEL_NAME_IN_LOGIC,
                span,
                format!(
                    "P7: `{name}` is a model name deciding behavior; derive the fact from the config in one shared place, and name the math, not the model"
                ),
                None,
                HELP,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_files_module_path_follows_the_module_tree() {
        let src = Path::new("/p/src");
        let at = |f: &str| module_path("k", &src.join(f), src).join("::");
        assert_eq!(at("lib.rs"), "k");
        assert_eq!(at("main.rs"), "k");
        assert_eq!(at("arch_registry.rs"), "k::arch_registry");
        assert_eq!(at("a/mod.rs"), "k::a");
        assert_eq!(at("a/b.rs"), "k::a::b");
        assert_eq!(at("a/lib.rs"), "k::a::lib");
    }

    #[test]
    fn mod_declarations_resolve_beside_root_files_and_under_others() {
        assert_eq!(mod_dir(Path::new("/p/src/lib.rs")), Path::new("/p/src"));
        assert_eq!(mod_dir(Path::new("/p/src/a/mod.rs")), Path::new("/p/src/a"));
        assert_eq!(mod_dir(Path::new("/p/src/a.rs")), Path::new("/p/src/a"));
    }

    #[test]
    fn test_files_are_the_tests_dir_and_test_modules() {
        let m = Path::new("/p");
        let tests = [
            PathBuf::from("/p/src/a/tests.rs"),
            PathBuf::from("/p/src/a/tests"),
        ];
        assert!(test_file(Path::new("/p/tests/t.rs"), m, &tests));
        assert!(test_file(Path::new("/p/src/a/tests.rs"), m, &tests));
        assert!(test_file(Path::new("/p/src/a/tests/b.rs"), m, &tests));
        assert!(!test_file(Path::new("/p/src/a/testsuite.rs"), m, &tests));
        assert!(!test_file(Path::new("/p/src/a.rs"), m, &tests));
    }

    #[test]
    fn the_package_lib_name_is_the_lib_then_the_package() {
        assert_eq!(
            lib_name("[package]\nname = \"scratchy-compiler\""),
            Some("scratchy_compiler".into())
        );
        assert_eq!(
            lib_name("[package]\nname = \"a-b\"\n[lib]\nname = \"c\""),
            Some("c".into())
        );
        assert_eq!(lib_name(""), None);
    }
}
