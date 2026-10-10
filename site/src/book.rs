//! book/: docs/*.md and CONTRIBUTING.md, rendered client-side by zero-md.
//!
//! The Markdown under docs/ stays the single source of truth. Each chapter is
//! copied (internal links rewritten to point at the built .html shells) into
//! _site/book/, next to a thin Carbon UI Shell page that renders it via
//! <zero-md>.

use std::fs;
use std::path::Path;

use dioxus::prelude::*;

use crate::carbon::{Module, SideNav, SideNavLink, SideNavMenu};
use crate::chrome::{self, Head, Library, Root, Tab, Theme};
use crate::{Site, repo};

pub struct Chapter {
    pub title: &'static str,
    /// Source, relative to the repo root.
    src: &'static str,
    /// Destination under book/, without extension.
    pub slug: &'static str,
}

const fn ch(title: &'static str, src: &'static str, slug: &'static str) -> Chapter {
    Chapter { title, src, slug }
}

pub const CHAPTERS: [Chapter; 6] = [
    ch("Introduction", "site/content/introduction.md", "index"),
    ch("Building", "docs/BUILD.md", "BUILD"),
    ch("The compiler", "docs/COMPILER.md", "COMPILER"),
    ch("Adding a model architecture", "docs/MODELS.md", "MODELS"),
    ch(
        "Spyre on OpenShift",
        "docs/spyre/KUBERNETES.md",
        "spyre/KUBERNETES",
    ),
    ch("Contributing", "CONTRIBUTING.md", "CONTRIBUTING"),
];

pub const BLOGS: [Chapter; 2] = [
    ch(
        "What “Reuse” Means in the Age of AI",
        "docs/blogs/REUSE.md",
        "blogs/REUSE",
    ),
    ch(
        "Systems Must Evolve to be Compilers",
        "docs/blogs/EVOLVE.md",
        "blogs/EVOLVE",
    ),
];

/// Internal links: markdown source -> built .html shell. The docs/-prefixed
/// forms come before the bare ones.
const LINK_REWRITES: [(&str, &str); 11] = [
    ("](docs/BUILD.md)", "](BUILD.html)"),
    ("](docs/COMPILER.md)", "](COMPILER.html)"),
    ("](docs/MODELS.md)", "](MODELS.html)"),
    ("](docs/spyre/KUBERNETES.md)", "](spyre/KUBERNETES.html)"),
    ("](BUILD.md)", "](BUILD.html)"),
    ("](COMPILER.md)", "](COMPILER.html)"),
    ("](MODELS.md)", "](MODELS.html)"),
    ("](spyre/KUBERNETES.md)", "](spyre/KUBERNETES.html)"),
    ("](CONTRIBUTING.md)", "](CONTRIBUTING.html)"),
    (
        "](CLAUDE.md)",
        concat!("](", repo!("/blob/main/CLAUDE.md"), ")"),
    ),
    (
        "](LICENSE)",
        concat!("](", repo!("/blob/main/LICENSE"), ")"),
    ),
];

const CONTENT_CSS: &str = include_str!("../theme/docs-content.css");
/// Builds the "On this page" list from the rendered chapter's h2/h3s.
const TOC_JS: &str = include_str!("book_toc.js");

fn nav_links(chapters: &[Chapter], current: &str, root: Root) -> Element {
    rsx! {
        for c in chapters {
            SideNavLink { href: "{root}book/{c.slug}.html", active: c.slug == current, "{c.title}" }
        }
    }
}

fn build_chapter(site: &Site, c: &Chapter) -> Result<(), String> {
    // book/<slug>.html sits one directory per path segment below _site/.
    let root = Root(c.slug.matches('/').count() + 1);
    let src = site.repo.join(c.src);
    let md = LINK_REWRITES.iter().fold(
        fs::read_to_string(&src).map_err(|e| format!("{}: {e}", src.display()))?,
        |text, (from, to)| text.replace(from, to),
    );
    let dest = site.out.join("book").join(c.slug);
    if let Some(dir) = dest.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    write(&dest.with_extension("md"), &md)?;

    let name = Path::new(c.slug)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(c.slug);
    let body = rsx! {
        {chrome::header(root, Some(Tab::Docs))}

        SideNav { label: "Docs navigation",
            {nav_links(&CHAPTERS, c.slug, root)}
            SideNavMenu { title: "Blogs", {nav_links(&BLOGS, c.slug, root)} }
        }

        div { class: "docs-layout",
            main { class: "docs-content",
                zero-md { "src": "{name}.md",
                    template { "data-append": "", style { dangerous_inner_html: CONTENT_CSS } }
                }
            }
            nav { class: "page-toc", "aria-label": "On this page" }
        }

        script { dangerous_inner_html: TOC_JS }
    };
    let title = format!("{} — scratchy", c.title);
    let page = chrome::document(
        Head {
            title: &title,
            description: None,
            og: None,
            root,
            modules: &[Module::UiShell],
            libraries: &[Library::ZeroMd],
            theme: Theme::FollowSystem,
        },
        body,
    );
    write(&dest.with_extension("html"), &page)
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Every chapter's source, relative to the repo root.
pub fn sources() -> impl Iterator<Item = &'static str> {
    CHAPTERS.iter().chain(&BLOGS).map(|c| c.src)
}

pub fn build(site: &Site) -> Result<(), String> {
    CHAPTERS
        .iter()
        .chain(&BLOGS)
        .try_for_each(|c| build_chapter(site, c))
}

/// Every `](target)` in `text`, as markdown's inline-link syntax reads it.
fn link_targets(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find("](") {
        let after = &rest[at + 2..];
        match after.find(')') {
            Some(0) | None => rest = &rest[at + 1..],
            Some(end) => {
                out.push(&after[..end]);
                rest = &after[end + 1..];
            }
        }
    }
    out
}

/// Every relative markdown link under book/ must resolve to a file there.
/// Returns the broken ones, as "page -> target".
pub fn broken_links(site: &Site) -> Result<Vec<String>, String> {
    let mut broken = Vec::new();
    for c in CHAPTERS.iter().chain(&BLOGS) {
        let md = site.out.join("book").join(c.slug).with_extension("md");
        let text = fs::read_to_string(&md).map_err(|e| format!("{}: {e}", md.display()))?;
        for target in link_targets(&text) {
            if ["http://", "https://", "#", "mailto:"]
                .iter()
                .any(|p| target.starts_with(p))
            {
                continue;
            }
            let file = target.split(['#', '?']).next().unwrap_or_default();
            if file.is_empty() {
                continue;
            }
            if !md.parent().is_some_and(|dir| dir.join(file).exists()) {
                broken.push(format!("book/{}.md -> {target}", c.slug));
            }
        }
    }
    Ok(broken)
}

#[cfg(test)]
mod tests {
    use super::link_targets;

    #[test]
    fn link_targets_reads_inline_links_only() {
        assert_eq!(
            link_targets("see [a](x.md#s) and [b]() and [c](https://e)"),
            ["x.md#s", "https://e"]
        );
    }
}
