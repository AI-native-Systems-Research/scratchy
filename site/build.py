#!/usr/bin/env python3
"""Assemble the GitHub Pages site into site/_site:

    /            the hand-written landing page
    /book/       docs/*.md + CONTRIBUTING.md, rendered client-side by zero-md

The Markdown under docs/ stays the single source of truth. Each chapter is
copied (with internal links rewritten to point at the built .html shells)
into site/_site/book/, alongside a thin Carbon UI Shell page that renders it
via <zero-md>. There is no server-side markdown-to-HTML step here — mdBook is
gone; the browser renders the .md file at request time.

Usage: site/build.py
"""
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SITE = HERE / "_site"
BOOK = SITE / "book"

# (title, source path relative to repo root, dest slug under book/, without extension)
CHAPTERS = [
    ("Introduction", "site/src/introduction.md", "index"),
    ("Building", "docs/BUILD.md", "BUILD"),
    ("The compiler", "docs/COMPILER.md", "COMPILER"),
    ("Adding a model architecture", "docs/MODELS.md", "MODELS"),
    ("Spyre on OpenShift", "docs/spyre/KUBERNETES.md", "spyre/KUBERNETES"),
    ("Contributing", "CONTRIBUTING.md", "CONTRIBUTING"),
]
BLOG_CHAPTERS = [
    ("What “Reuse” Means in the Age of AI", "docs/blogs/REUSE.md", "blogs/REUSE"),
]
ALL_CHAPTERS = CHAPTERS + BLOG_CHAPTERS

REPO_BLOB = "https://github.com/AI-native-Systems-Research/scratchy/blob/main"

# Internal-link rewrites: markdown source links -> built .html slugs.
# Order matters: the docs/-prefixed forms must be rewritten before the bare
# forms, and both must run before the absolute-GitHub-link special cases.
LINK_REWRITES = [
    (r"\]\(docs/BUILD\.md\)", "](BUILD.html)"),
    (r"\]\(docs/COMPILER\.md\)", "](COMPILER.html)"),
    (r"\]\(docs/MODELS\.md\)", "](MODELS.html)"),
    (r"\]\(docs/spyre/KUBERNETES\.md\)", "](spyre/KUBERNETES.html)"),
    (r"\]\(BUILD\.md\)", "](BUILD.html)"),
    (r"\]\(COMPILER\.md\)", "](COMPILER.html)"),
    (r"\]\(MODELS\.md\)", "](MODELS.html)"),
    (r"\]\(spyre/KUBERNETES\.md\)", "](spyre/KUBERNETES.html)"),
    (r"\]\(CONTRIBUTING\.md\)", "](CONTRIBUTING.html)"),
    (r"\]\(CLAUDE\.md\)", f"]({REPO_BLOB}/CLAUDE.md)"),
    (r"\]\(LICENSE\)", f"]({REPO_BLOB}/LICENSE)"),
]

DOCS_CONTENT_CSS = (HERE / "theme" / "docs-content.css").read_text()

HEAD_COMMON = """\
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — scratchy</title>
<link rel="icon" type="image/png" href="{root}favicon.png">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@carbon/styles@1/css/styles.min.css">
<link rel="stylesheet" href="{root}styles.css">
<script type="module" src="https://1.www.s81c.com/common/carbon/web-components/tag/v2/latest/ui-shell.min.js"></script>
<script type="module" src="https://cdn.jsdelivr.net/npm/zero-md@3?register"></script>
<script>
(function () {{
  var mq = window.matchMedia('(prefers-color-scheme: dark)');
  function apply(dark) {{
    document.documentElement.classList.remove('cds--g100', 'cds--white');
    document.documentElement.classList.add(dark ? 'cds--g100' : 'cds--white');
  }}
  apply(mq.matches);
  mq.addEventListener('change', function (e) {{ apply(e.matches); }});
}})();
</script>
"""

PAGE_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
{head}
</head>
<body>

<cds-header aria-label="scratchy">
  <cds-header-menu-button button-label-active="Close menu" button-label-inactive="Open menu"></cds-header-menu-button>
  <cds-header-name href="{root}" prefix="▚">scratchy</cds-header-name>
  <cds-header-nav menu-bar-label="scratchy navigation">
    <cds-header-nav-item href="{root}book/index.html">Docs</cds-header-nav-item>
    <cds-header-nav-item href="{root}book/COMPILER.html">Compiler</cds-header-nav-item>
    <cds-header-nav-item href="https://github.com/AI-native-Systems-Research/scratchy">GitHub</cds-header-nav-item>
  </cds-header-nav>
</cds-header>

<cds-side-nav aria-label="Docs navigation" class="docs-side-nav">
  <cds-side-nav-items>
{nav_links}
    <cds-side-nav-menu title="Blogs" expanded>
{nav_blog_links}
    </cds-side-nav-menu>
  </cds-side-nav-items>
</cds-side-nav>

<div class="docs-layout">
  <main class="docs-content">
    <zero-md src="{md_src}">
      <template data-append><style>{docs_css}</style></template>
    </zero-md>
  </main>
  <nav class="page-toc" aria-label="On this page"></nav>
</div>

<script>
(function () {{
  var main = document.querySelector('.docs-content zero-md');
  var toc = document.querySelector('.page-toc');
  main.addEventListener('zero-md-rendered', function () {{
    var root = main.shadowRoot;
    var headings = root.querySelectorAll('.markdown-body h2, .markdown-body h3');
    toc.innerHTML = '';
    if (!headings.length) return;
    var list = document.createElement('ul');
    headings.forEach(function (h) {{
      var li = document.createElement('li');
      li.className = h.tagName.toLowerCase();
      var a = document.createElement('a');
      a.href = '#' + h.id;
      a.textContent = h.textContent;
      a.addEventListener('click', function (e) {{
        e.preventDefault();
        main.goto('#' + h.id);
      }});
      li.appendChild(a);
      list.appendChild(li);
    }});
    toc.appendChild(list);
  }});
}})();
</script>

</body>
</html>
"""


def rewrite_links(text: str) -> str:
    for pattern, repl in LINK_REWRITES:
        text = re.sub(pattern, repl, text)
    return text


def nav_items(chapters, current_slug, root):
    lines = []
    for title, _src, slug in chapters:
        active = " active" if slug == current_slug else ""
        lines.append(
            f'      <cds-side-nav-link href="{root}book/{slug}.html"{active}>{title}</cds-side-nav-link>'
        )
    return "\n".join(lines)


def build_chapter(title, src_rel, slug):
    src = ROOT / src_rel
    depth = slug.count("/") + 1  # book/<slug>.html is depth levels below site/_site/
    root = "../" * depth

    md_text = rewrite_links(src.read_text())
    dest_md = BOOK / f"{slug}.md"
    dest_html = BOOK / f"{slug}.html"
    dest_md.parent.mkdir(parents=True, exist_ok=True)
    dest_md.write_text(md_text)

    head = HEAD_COMMON.format(title=title, root=root)
    nav_links = nav_items(CHAPTERS, slug, root)
    nav_blog_links = nav_items(BLOG_CHAPTERS, slug, root)
    page = PAGE_TEMPLATE.format(
        head=head,
        root=root,
        nav_links=nav_links,
        nav_blog_links=nav_blog_links,
        md_src=f"{Path(slug).name}.md",
        docs_css=DOCS_CONTENT_CSS,
    )
    dest_html.write_text(page)


def check_markdown_links():
    """Every relative markdown link must resolve to a file under book/."""
    broken = False
    link_re = re.compile(r"\]\(([^)]+)\)")
    for md in BOOK.rglob("*.md"):
        for m in link_re.finditer(md.read_text()):
            target = m.group(1)
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = target.split("#")[0].split("?")[0]
            if not target:
                continue
            resolved = (md.parent / target).resolve()
            if not resolved.exists():
                print(f"broken link: {md.relative_to(SITE)} -> {m.group(1)}", file=sys.stderr)
                broken = True
    return broken


def main():
    shutil.rmtree(SITE, ignore_errors=True)
    BOOK.mkdir(parents=True)

    for asset in ("index.html", "styles.css", "favicon.png"):
        shutil.copy(HERE / asset, SITE / asset)

    for title, src_rel, slug in ALL_CHAPTERS:
        build_chapter(title, src_rel, slug)

    if check_markdown_links():
        print("link check failed", file=sys.stderr)
        sys.exit(1)

    print(f"site assembled at {SITE}")


if __name__ == "__main__":
    main()
