#!/usr/bin/env bash
# Assemble the GitHub Pages site into site/_site:
#   /            the hand-written landing page
#   /book/       docs/*.md + CONTRIBUTING.md, rendered by mdBook
#
# The Markdown under docs/ stays the single source of truth; it is copied into
# site/src/ (gitignored) just before the book is built.
#
# Usage: site/build.sh   [requires mdbook on PATH]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"

# Chapter sources, in the order SUMMARY.md lists them: <repo path>:<src path>
chapters=(
  "docs/BUILD.md:BUILD.md"
  "docs/COMPILER.md:COMPILER.md"
  "docs/MODELS.md:MODELS.md"
  "docs/spyre/KUBERNETES.md:spyre/KUBERNETES.md"
  "CONTRIBUTING.md:CONTRIBUTING.md"
  "docs/blogs/REUSE.md:blogs/REUSE.md"
)

# Links that are correct relative to the repo root (how GitHub renders them) but
# not relative to the book. Rewritten in the copies only, never in the sources.
repo='https://github.com/AI-native-Systems-Research/scratchy/blob/main'

for entry in "${chapters[@]}"; do
  src="$root/${entry%%:*}"
  dst="$here/src/${entry##*:}"
  mkdir -p "$(dirname "$dst")"
  sed -e 's|](docs/BUILD\.md)|](BUILD.md)|g' \
      -e 's|](docs/COMPILER\.md)|](COMPILER.md)|g' \
      -e 's|](docs/MODELS\.md)|](MODELS.md)|g' \
      -e 's|](docs/spyre/KUBERNETES\.md)|](spyre/KUBERNETES.md)|g' \
      -e "s|](CLAUDE\.md)|]($repo/CLAUDE.md)|g" \
      -e "s|](LICENSE)|]($repo/LICENSE)|g" \
      "$src" > "$dst"
done

rm -rf "$here/book" "$here/_site"
mdbook build "$here"

mkdir -p "$here/_site"
cp "$here/index.html" "$here/styles.css" "$here/favicon.png" "$here/og-image.png" "$here/_site/"
cp -R "$here/book" "$here/_site/book"

# Generated straight from the DSL sources, so the architectures page cannot
# drift from what the compiler actually reads.
python3 "$here/build-archs.py" "$root" "$here/_site/architectures.html"

# Every local href must resolve inside _site, or the copies drifted from the
# sources and the deploy would ship dead links.
broken=0
while IFS= read -r page; do
  dir="$(dirname "$page")"
  while IFS= read -r href; do
    # Site-absolute hrefs (mdBook's 404 page) only resolve once deployed.
    case "$href" in http*|/*|\#*|mailto:*) continue ;; esac
    target="$dir/${href%%[#?]*}"
    [ -z "${href%%[#?]*}" ] && continue
    case "$target" in */) target="${target}index.html" ;; esac
    if [ ! -e "$target" ]; then
      echo "broken link: ${page#"$here/_site/"} -> $href" >&2
      broken=1
    fi
  done < <(grep -o 'href="[^"]*"' "$page" | sed 's/href="//;s/"$//')
done < <(find "$here/_site" -name '*.html')
[ "$broken" -eq 0 ] || { echo "link check failed" >&2; exit 1; }

echo "site assembled at $here/_site"
