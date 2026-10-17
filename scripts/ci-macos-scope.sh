#!/usr/bin/env bash
# ⭐ THE CRATES NO macOS CI JOB COMPILES, AND THE TWO CHECKS THAT READ THE LIST.
#
#   ci-macos-scope.sh changed <base> <head>
#       Prints `macos=false` when every file the diff touches belongs to one of these crates, so a
#       PR that changes only them skips both macOS jobs; `macos=true` otherwise. A file belongs to
#       the deepest crate directory holding it, and a file in no crate (Cargo.lock, .github/, docs)
#       needs the macOS jobs.
#   ci-macos-scope.sh lock
#       Run last in each macOS job. Fails if the job compiled one of these crates: a new dependency
#       edge into one turns the macOS jobs red, instead of letting a PR that changes only that crate
#       skip the jobs that build it. CI's rust-cache drops every workspace crate's build output
#       before saving, so a workspace crate's `.fingerprint` entry is one this job compiled.
set -euo pipefail

# One crate directory per line: the Spyre target and its triton re-host crates. Not ktir-core,
# ktir-emulator or ktir-optimizer: the tests job runs those.
MACOS_UNBUILT='crates/targets/spyre
crates/targets/spyre/builder
crates/targets/spyre/bundle
crates/targets/spyre/flex-rs
crates/targets/spyre/ktir/ktir-superdsc
crates/triton/splice
crates/triton/triton-frontend
crates/triton/triton-ktir
crates/triton/triton-ktir-superdsc
crates/triton/triton-numeric'

changed() {
    # Every crate directory, deepest first, so the first one holding a file is its owner:
    # ktir-core's files are ktir-core's, though they sit under crates/targets/spyre too.
    local crates
    crates=$(git ls-files '*/Cargo.toml' | sed 's|/Cargo.toml$||' \
        | awk -F/ '{ print NF "\t" $0 }' | sort -rn | cut -f2-)
    local files
    files=$(git diff --name-only --no-renames "$1" "$2")
    if [ -z "$files" ]; then
        echo "no changed files" >&2
        echo macos=true
        return
    fi
    local file owner dir
    while IFS= read -r file; do
        owner=""
        while IFS= read -r dir; do
            case "$file" in "$dir"/*) owner=$dir; break ;; esac
        done <<<"$crates"
        if [ -z "$owner" ] || ! grep -qxF "$owner" <<<"$MACOS_UNBUILT"; then
            echo "the macOS jobs build ${owner:-what is outside every crate}: $file" >&2
            echo macos=true
            return
        fi
    done <<<"$files"
    echo "every changed file is in a crate no macOS job compiles" >&2
    echo macos=false
}

lock() {
    # The listed directories' package names, from cargo's own manifest paths.
    local metadata packages names dir name
    metadata=$(cargo metadata --no-deps --format-version 1)
    packages=$(jq -r --arg root "$PWD/" \
        '.packages[] | "\(.manifest_path | ltrimstr($root) | rtrimstr("/Cargo.toml"))\t\(.name)"' <<<"$metadata")
    names=""
    while IFS= read -r dir; do
        name=$(awk -F'\t' -v dir="$dir" '$1 == dir { print $2 }' <<<"$packages")
        if [ -z "$name" ]; then
            echo "scripts/ci-macos-scope.sh lists $dir, which holds no workspace crate" >&2
            exit 1
        fi
        names="$names${names:+|}$name"
    done <<<"$MACOS_UNBUILT"
    local compiled
    compiled=$(find "$(jq -r .target_directory <<<"$metadata")" -mindepth 3 -maxdepth 3 -type d \
        -path '*/.fingerprint/*' \
        | grep -E "/\.fingerprint/($names)-[0-9a-f]{16}\$" || true)
    if [ -n "$compiled" ]; then
        echo "this job compiled a crate scripts/ci-macos-scope.sh lists as one no macOS job compiles:" >&2
        echo "$compiled" >&2
        echo "a PR changing only that crate would skip this job: drop it from the list, or the edge that pulls it in" >&2
        exit 1
    fi
    echo "compiled none of: ${names//|/, }"
}

case "${1:-}" in
    changed) changed "$2" "$3" ;;
    lock) lock ;;
    *) echo "usage: $0 changed <base> <head> | lock" >&2; exit 2 ;;
esac
