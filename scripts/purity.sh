#!/usr/bin/env bash
# The purity lints (lints/scratchy-purity, the purity skill's rules as rustc lints), run by dylint.
# CI, the Claude Code hooks and hand runs all call this one runner.
#
#     scripts/purity.sh                   every scope this host builds (--scope all)
#     scripts/purity.sh --scope <scope>   root | metal | metal-all | spyre | spyre-hw | all
#     scripts/purity.sh --file <path>     the package that owns <path>
#
# Scopes, with the features CI's clippy uses for each:
#   root       the root workspace without the five metal crates (any host)
#   metal      the five metal crates, one model (macOS)
#   metal-all  the five metal crates, every model, MTP heads, TurboQuant and vision (macOS, CI)
#   spyre      backends/spyre (any host)
#   spyre-hw   backends/spyre-hw (a card host: a real ../senlib-rs checkout beside this tree)
#   all        root, spyre, metal where macOS, spyre-hw where a card host
#
# Exit 0 when clean. Exit 1 with the findings on stderr, or when cargo-dylint is missing.
# A --file outside every scanned package exits 0 with one line saying so.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DYLINT_VERSION=6.4.0
METAL_CRATES=(scratchy-target-metal scratchy-target-metal-cli scratchy-target-metal-compiler
    scratchy-target-metal-models scratchy-target-metal-worker)
ROOT_FEATURES=scratchy-models/granite-3.1-8b-instruct
SPYRE_FEATURES=scratchy-models/granite-3.1-8b-instruct
# The card workspace's own alias for the models crate; it panics at build time with zero models.
SPYRE_HW_FEATURES=model/granite-3.1-8b-instruct,snt1p0
METAL_FEATURES=scratchy-models/qwen3-0.6b
METAL_ALL_FEATURES=scratchy-models/all,scratchy-models/mtp,scratchy-target-metal-models/turboquant,scratchy-target-metal-models/vision

say() { echo "scratchy-purity: $*" >&2; }

if ! cargo dylint --version 2>/dev/null | grep -q "${DYLINT_VERSION}"; then
    say "cargo-dylint ${DYLINT_VERSION} is missing; run cargo install cargo-dylint dylint-link --version ${DYLINT_VERSION} --locked"
    exit 1
fi

macos() { [[ "$(uname -s)" == Darwin ]]; }
card_host() { [[ -e "${ROOT}/../senlib-rs/.git" ]]; }

# Runs dylint in one workspace directory with cargo-check arguments; findings go to stderr.
lint() { # dir args...
    local dir=$1; shift
    local log rc
    log=$(mktemp)
    (cd "${ROOT}/${dir}" && cargo dylint --all -- "$@") >"${log}" 2>&1
    rc=$?
    if (( rc != 0 )); then
        grep -E -A12 '^error' "${log}" >&2 || cat "${log}" >&2
    fi
    rm -f "${log}"
    return "${rc}"
}

# The card workspaces resolve a sibling senlib-rs checkout; off a card host, the stub stands in.
resolve_senlib() { [[ -e "${ROOT}/../senlib-rs" ]] || "${ROOT}/scripts/ci-senlib-stub.sh" >/dev/null; }

metal_packages() { local c; for c in "${METAL_CRATES[@]}"; do printf -- '-p\n%s\n' "${c}"; done; }
metal_excludes() { local c; for c in "${METAL_CRATES[@]}"; do printf -- '--exclude\n%s\n' "${c}"; done; }

scope() { # name
    local args=()
    case "$1" in
        root)
            resolve_senlib
            while IFS= read -r a; do args+=("${a}"); done < <(metal_excludes)
            lint . --workspace "${args[@]}" --features "${ROOT_FEATURES}" ;;
        metal|metal-all)
            macos || { say "the metal crates build only on macOS; skipped"; return 0; }
            resolve_senlib
            while IFS= read -r a; do args+=("${a}"); done < <(metal_packages)
            local features=${METAL_FEATURES}
            [[ "$1" == metal-all ]] && features=${METAL_ALL_FEATURES}
            lint . "${args[@]}" --features "${features}" ;;
        spyre)
            resolve_senlib
            lint backends/spyre --workspace --features "${SPYRE_FEATURES}" ;;
        spyre-hw)
            card_host || { say "backends/spyre-hw builds only on a card host; skipped"; return 0; }
            # Nothing here links, so the build script's FFI link directives are skipped.
            SCRATCHY_SKIP_SENDNN_CXX=1 lint backends/spyre-hw --workspace --features "${SPYRE_HW_FEATURES}" ;;
        all)
            local rc=0 s
            for s in root spyre metal spyre-hw; do scope "${s}" || rc=1; done
            return "${rc}" ;;
        *) say "unknown scope: $1"; return 2 ;;
    esac
}

# The package owning a file: its nearest manifest with a [package] table, below the repo root.
owning_manifest() { # path
    local d
    d=$(cd "$(dirname "$1")" 2>/dev/null && pwd) || return 1
    while [[ "${d}" == "${ROOT}"* && "${d}" != "${ROOT}" ]]; do
        if [[ -f "${d}/Cargo.toml" ]] && grep -q '^\[package\]' "${d}/Cargo.toml"; then
            echo "${d}/Cargo.toml"; return 0
        fi
        d=$(dirname "${d}")
    done
    return 1
}

# The --features argument for one package: the scope's model, where the package selects models.
features_for() { # manifest features
    grep -q 'scratchy-models' "$1" && printf -- '--features\n%s\n' "$2"
}

file() { # path
    local path=$1 manifest name ws
    [[ "${path}" == /* ]] || path="${PWD}/${path}"
    if ! manifest=$(owning_manifest "${path}"); then
        say "${path#"${ROOT}"/} is in no package; nothing to lint"; return 0
    fi
    name=$(sed -n '/^\[package\]/,/^\[/{s/^name *= *"\(.*\)"/\1/p;}' "${manifest}" | head -1)
    resolve_senlib
    ws=$(cargo locate-project --workspace --message-format plain --manifest-path "${manifest}" 2>/dev/null) || {
        say "${manifest#"${ROOT}"/} is in no workspace this runner lints; nothing to lint"; return 0
    }
    ws=$(dirname "${ws}")
    local feat=()
    case "${ws#"${ROOT}"}" in
        "")
            local c
            for c in "${METAL_CRATES[@]}"; do
                if [[ "${c}" == "${name}" ]]; then
                    macos || { say "${name} builds only on macOS; skipped"; return 0; }
                    while IFS= read -r a; do feat+=("${a}"); done < <(features_for "${manifest}" "${METAL_FEATURES}")
                    lint . -p "${name}" ${feat[@]+"${feat[@]}"}; return
                fi
            done
            while IFS= read -r a; do feat+=("${a}"); done < <(features_for "${manifest}" "${ROOT_FEATURES}")
            lint . -p "${name}" ${feat[@]+"${feat[@]}"} ;;
        /backends/spyre)
            while IFS= read -r a; do feat+=("${a}"); done < <(features_for "${manifest}" "${SPYRE_FEATURES}")
            lint backends/spyre -p "${name}" ${feat[@]+"${feat[@]}"} ;;
        # The card workspace's members need its model alias, so a file there lints the workspace.
        /backends/spyre-hw) scope spyre-hw ;;
        *) say "${manifest#"${ROOT}"/} is in no workspace this runner lints; nothing to lint"; return 0 ;;
    esac
}

case "${1:-}" in
    "") scope all ;;
    --scope) scope "${2:?--scope needs a scope}" ;;
    --file) file "${2:?--file needs a path}" ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) say "unknown argument: $1"; exit 2 ;;
esac
