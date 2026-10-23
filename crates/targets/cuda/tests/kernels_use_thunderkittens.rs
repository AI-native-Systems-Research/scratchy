// SPDX-License-Identifier: Apache-2.0
//! Every CUDA kernel is written with ThunderKittens: a check of the kernels' source text, with no
//! toolchain (AI-native-Systems-Research/scratchy#845; the rule is
//! `aa-personal-space/cuda-moonshot/PLAN-tk100.md` §1.2-1.5).
//!
//! A kernel is a `.cu` file directly under `kernels/` (the vendored `kernels/thunderkittens/` is
//! not one). It uses ThunderKittens when it passes every rule below. Each failure prints
//! `path:line RULE what`, and a test fails only after listing every violation it found.
//!
//! | rule | what it refuses |
//! |---|---|
//! | T1 | a kernel that does not include `"kittens.cuh"`, or includes anything but `"kittens.cuh"`, `"prototype.cuh"`, `"scratchy_ops.cuh"`, `<climits>`, `<cstdint>`, `<cmath>`, `<type_traits>` |
//! | T2 | a `*` in a `__global__` parameter list; a line declaring a pointer with no initialiser (a raw pointer field) |
//! | T3 | the constructs plain CUDA kernels move and convert data with: warp shuffles and votes, atomics, `__ldg`, PTX loads and stores, `__nv_bfloat16` and its conversions, `make_uint4`, `reinterpret_cast` to a vector type |
//! | T4 | a math intrinsic in a kernel: those live in `kernels/scratchy_ops.cuh` |
//! | T5 | `asm` anywhere but an exact `(file, PTX string)` pair of [`ASM_ALLOWLIST`] |
//! | T6 | `__shared__` of anything but TK semaphores, `st`/`sv` tiles and vectors, or the `extern __shared__` buffer TK's allocators carve |
//! | T7 | a kernel file with no kernel, or a `__global__` body that names no TK scope, tile, vector or view |
//! | T8 | a `.cu`/`.cuh` outside `kernels/` (but the vendored tree), a `.cuh` there but the op header, a `__global__` in the op header |
//! | T9 | a T3/T5 construct in the op header outside a `// TK-GAP(<name>)` block, or a gap name not on [`TK_GAPS`] |
//! | T10 | the checker itself: plain-CUDA fixtures must be refused, a TK fixture accepted |
//!
//! T1-T4, T6 and T7 read the text with its comments blanked (string literals kept); T5 and the
//! gap markers read it raw. Tokens match on word boundaries, as `cfg_neutrality.rs`'s `has_word`
//! does, except the prefixes `__shfl_` and `atomic`, which match anywhere.
//!
//! Kernels not yet rewritten are on [`NOT_YET_PORTED`]: each must still fail some rule (so the
//! list cannot go stale), and every kernel off it must pass every rule. Each port removes its
//! kernels from the list.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// The kernels not yet rewritten with ThunderKittens. Each port PR removes its own.
const NOT_YET_PORTED: &[&str] = &[
    "add_bf16.cu",
    "attn_decode_bf16.cu",
    "embed_bf16.cu",
    "gate_apply_bf16.cu",
    "gate_split_bf16.cu",
    "gdn_conv1d.cu",
    "gdn_gating.cu",
    "gdn_rms_norm_gated.cu",
    "gdn_scan.cu",
    "gemm_narrow_bf16.cu",
    "gemv_bf16.cu",
    "kv_append_bf16.cu",
    "rms_norm_bf16.cu",
    "rope_bf16.cu",
    "silu_mul_bf16.cu",
];

/// The inline PTX a kernel may hold, as `(file, PTX string)`: the async-proxy fence after the
/// prefill's consumers zero a partial V tile, which TK at the pin emits only inside
/// `warpgroup::mma_fence` and its TMA stores, never on its own.
const ASM_ALLOWLIST: &[(&str, &str)] =
    &[("attn_prefill_bf16.cu", "fence.proxy.async.shared::cta;\n")];

/// The op header's gap helpers: operations TK has no map for, written once against TK types.
const TK_GAPS: &[&str] = &["rv_ternary_map"];

/// The one extension header, directly under `kernels/`.
const OP_HEADER: &str = "scratchy_ops.cuh";

/// The includes a kernel may have besides `"kittens.cuh"`.
const PROJECT_INCLUDES: &[&str] = &["kittens.cuh", "prototype.cuh", "scratchy_ops.cuh"];
const SYSTEM_INCLUDES: &[&str] = &["climits", "cstdint", "cmath", "type_traits"];

/// T3: banned anywhere in a kernel or the op header (outside a gap), on word boundaries.
const T3_WORDS: &[&str] = &[
    "__ballot_sync",
    "__match_any_sync",
    "__any_sync",
    "__all_sync",
    "__ldg",
    "cp.async",
    "ld.global",
    "st.global",
    "make_uint4",
    "__nv_bfloat16",
    "__bfloat162float",
    "__float2bfloat16",
    "__floats2bfloat162_rn",
    "__bfloat1622float2",
    "__float22bfloat162_rn",
];
/// T3: banned prefixes, matched anywhere.
const T3_PREFIXES: &[&str] = &["__shfl_", "atomic"];
/// T3: the types a `reinterpret_cast<` may not name (after `const`/`volatile`).
const T3_CASTS: &[&str] = &["uint4", "uint2", "float4", "float2", "__nv_bfloat162"];

/// T4: the math intrinsics only the op header may call.
const T4_WORDS: &[&str] = &[
    "__expf",
    "expf",
    "exp2f",
    "log1pf",
    "rsqrtf",
    "fmaf",
    "__fmul_rn",
    "__fadd_rn",
    "fmaxf",
];

/// T7: what a `__global__` body names when it uses TK, each at the start of a token.
const T7_TOKENS: &[&str] = &[
    "warp::",
    "warpgroup::",
    "group<",
    "tma::",
    "rt_",
    "rv_",
    "st_",
    "sv_",
    "gl<",
];

/// T10's fixture (a): today's `add_bf16.cu` kernel and launcher, trimmed. Plain CUDA.
const FIXTURE_PLAIN_ADD: &str = r##"// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace {
constexpr int VEC = 8;
constexpr int THREADS = 256;

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&f)[VEC]) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
    for (int i = 0; i < VEC / 2; ++i) {
        const float2 v = __bfloat1622float2(h[i]);
        f[2 * i] = v.x;
        f[2 * i + 1] = v.y;
    }
}

__global__ void __launch_bounds__(THREADS)
add_bf16_kernel(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
                __nv_bfloat16* __restrict__ out, int vecs) {
    const int v = blockIdx.x * THREADS + threadIdx.x;
    if (v >= vecs) {
        return;
    }
    float fa[VEC], fb[VEC];
    load8(a + size_t(v) * VEC, fa);
    load8(b + size_t(v) * VEC, fb);
    for (int i = 0; i < VEC; ++i) {
        fa[i] = __fadd_rn(fa[i], fb[i]);
    }
}
}  // namespace

extern "C" int scratchy_cuda_add_bf16(const void* a, const void* b, void* out, uint32_t rows,
                                      uint32_t cols, void* stream) {
    const __nv_bfloat16* ap = static_cast<const __nv_bfloat16*>(a);
    void* args[] = {&ap};
    return static_cast<int>(cudaLaunchKernel(add_bf16_kernel, dim3(1), dim3(THREADS), args, 0,
                                             static_cast<cudaStream_t>(stream)));
}
"##;

/// T10's fixture (b): TK included, raw pointers and an intrinsic used, nothing of TK touched.
const FIXTURE_TK_IN_NAME_ONLY: &str = r##"#include "kittens.cuh"

namespace {
__global__ void exp_ker(const float* x, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = expf(x[i]);
    }
}
}  // namespace
"##;

/// T10's fixture (c): one `__grid_constant__` struct of `gl` views, a body of `warp::load` and
/// `warp::store` on an `rv_fl`. Accepted by every rule.
const FIXTURE_TK: &str = r##"#include "kittens.cuh"

#include <cstdint>

using namespace kittens;

namespace {
using row_gl = gl<float, 1, 1, -1, -1>;
struct copy_globals {
    row_gl x;
    row_gl y;
};

__global__ void copy_ker(const __grid_constant__ copy_globals g) {
    rv_fl<256> row;
    row_gl y = g.y;
    warp::load(row, g.x, {0, 0, int(blockIdx.x), 0});
    warp::store(y, row, {0, 0, int(blockIdx.x), 0});
}
}  // namespace
"##;

/// One rule a source breaks, at a line.
#[derive(Debug)]
struct Violation {
    line: usize,
    rule: &'static str,
    what: String,
}

/// Which rules a file is held to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A `.cu` kernel: T1-T7.
    Kernel,
    /// `scratchy_ops.cuh`: its includes, T2, T6, no kernel (T8), and T3/T5 only in gaps (T9).
    OpHeader,
}

/// A source three ways: as written; with its comments blanked (`code`); and with its comments
/// and the contents of its string and character literals blanked (`bare`). Blanking keeps every
/// newline, so a line number is the same in all three.
struct Source {
    raw: String,
    code: String,
    bare: String,
}

impl Source {
    fn new(raw: &str) -> Self {
        enum State {
            Code,
            Line,
            Block,
            Literal(char),
        }
        let blank = |c: char| if c == '\n' { '\n' } else { ' ' };
        let (mut code, mut bare) = (String::new(), String::new());
        let chars: Vec<char> = raw.chars().collect();
        let (mut state, mut i) = (State::Code, 0);
        while i < chars.len() {
            let (c, next) = (chars[i], chars.get(i + 1).copied());
            match state {
                State::Code if c == '/' && next == Some('/') => state = State::Line,
                State::Code if c == '/' && next == Some('*') => {
                    state = State::Block;
                    code.push_str("  ");
                    bare.push_str("  ");
                    i += 2;
                    continue;
                }
                State::Code => {
                    if c == '"' || c == '\'' {
                        state = State::Literal(c);
                    }
                    code.push(c);
                    bare.push(c);
                    i += 1;
                    continue;
                }
                State::Line if c == '\n' => state = State::Code,
                State::Line => {}
                State::Block if c == '*' && next == Some('/') => {
                    state = State::Code;
                    code.push_str("  ");
                    bare.push_str("  ");
                    i += 2;
                    continue;
                }
                State::Block => {}
                State::Literal(_) if c == '\\' && next.is_some() => {
                    let n = next.unwrap_or(' ');
                    code.extend([c, n]);
                    bare.extend([' ', blank(n)]);
                    i += 2;
                    continue;
                }
                State::Literal(q) => {
                    code.push(c);
                    bare.push(if c == q { c } else { blank(c) });
                    if c == q {
                        state = State::Code;
                    }
                    i += 1;
                    continue;
                }
            }
            // A comment character, or the newline that ends a line comment.
            code.push(blank(c));
            bare.push(blank(c));
            i += 1;
        }
        Source {
            raw: raw.to_string(),
            code,
            bare,
        }
    }
}

fn is_word(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphanumeric()
}

/// Every offset of `needle` in `hay`, bounded by non-word characters on the left and, when
/// `right` is set, on the right too.
fn find(hay: &str, needle: &str, right: bool) -> Vec<usize> {
    let h = hay.as_bytes();
    (hay.match_indices(needle))
        .map(|(i, _)| i)
        .filter(|&i| {
            let before = i > 0 && is_word(h[i - 1]);
            let end = i + needle.len();
            let after = right && end < h.len() && is_word(h[end]);
            !before && !after
        })
        .collect()
}

/// `needle` on word boundaries both sides.
fn words(hay: &str, needle: &str) -> Vec<usize> {
    find(hay, needle, true)
}

/// The 1-based line of byte `at` of `text`.
fn line_of(text: &str, at: usize) -> usize {
    text[..at].matches('\n').count() + 1
}

/// The identifier (letters, digits, `_` and `::`) at the start of `s`, and the rest.
fn ident(s: &str) -> (&str, &str) {
    let end = s
        .find(|c: char| !(c == '_' || c == ':' || c.is_ascii_alphanumeric()))
        .unwrap_or(s.len());
    s.split_at(end)
}

fn violation(line: usize, rule: &'static str, what: impl Into<String>) -> Violation {
    Violation {
        line,
        rule,
        what: what.into(),
    }
}

/// T1: the includes. A kernel includes `"kittens.cuh"` and only the allowed headers; the op header
/// includes `"kittens.cuh"` alone.
fn t1_includes(src: &Source, kind: Kind) -> Vec<Violation> {
    let mut out = Vec::new();
    let mut kittens = false;
    for (n, line) in src.code.lines().enumerate() {
        let Some(rest) = line.trim_start().strip_prefix('#') else {
            continue;
        };
        let Some(target) = rest.trim_start().strip_prefix("include") else {
            continue;
        };
        let target = target.trim();
        let (quoted, name) = match target.chars().next() {
            Some('"') => (true, target.trim_matches('"')),
            Some('<') => (false, target.trim_start_matches('<').trim_end_matches('>')),
            _ => (false, target),
        };
        kittens |= quoted && name == "kittens.cuh";
        let allowed = match (kind, quoted) {
            (Kind::Kernel, true) => PROJECT_INCLUDES.contains(&name),
            (Kind::Kernel, false) => SYSTEM_INCLUDES.contains(&name),
            (Kind::OpHeader, true) => name == "kittens.cuh",
            (Kind::OpHeader, false) => false,
        };
        if !allowed {
            out.push(violation(n + 1, "T1", format!("includes {target}")));
        }
    }
    if !kittens {
        out.push(violation(1, "T1", "does not include \"kittens.cuh\""));
    }
    out
}

/// Whether a line declares a pointer with no initialiser, the shape of a raw pointer field: the
/// issue's `^\s*(const\s+)?[\w:<>, ]+\*\s*(__restrict__\s*)?\w+\s*;`.
fn declares_bare_pointer(line: &str) -> bool {
    let t = line.trim_start();
    // `[\w:<>, ]+`: the type, `const` included.
    let ty = |c: char| c == '_' || c.is_ascii_alphanumeric() || ":<>, ".contains(c);
    let run = t.find(|c: char| !ty(c)).unwrap_or(t.len());
    let Some(rest) = t[run..].strip_prefix('*').filter(|_| run > 0) else {
        return false;
    };
    let rest = rest.trim_start();
    let rest = rest
        .strip_prefix("__restrict__")
        .map_or(rest, str::trim_start);
    // `\w+\s*;`: one name, then the semicolon.
    let name = rest
        .find(|c: char| !(c == '_' || c.is_ascii_alphanumeric()))
        .unwrap_or(rest.len());
    name > 0 && rest[name..].trim_start().starts_with(';')
}

/// T2: no `*` from a `__global__` to the `{` (or `;`) after its parameters, and no raw pointer
/// field.
fn t2_pointers(src: &Source) -> Vec<Violation> {
    let mut out = Vec::new();
    for at in words(&src.bare, "__global__") {
        let rest = &src.bare[at..];
        let end = rest.find(['{', ';']).unwrap_or(rest.len());
        if let Some(star) = rest[..end].find('*') {
            out.push(violation(
                line_of(&src.bare, at + star),
                "T2",
                "a `__global__` parameter list holds a raw pointer",
            ));
        }
    }
    for (n, line) in src.bare.lines().enumerate() {
        if declares_bare_pointer(line) {
            out.push(violation(
                n + 1,
                "T2",
                format!(
                    "declares a raw pointer with no initialiser: `{}`",
                    line.trim()
                ),
            ));
        }
    }
    out
}

/// T3: the plain-CUDA data movement and conversion constructs, as `(line, token)`.
fn t3_hits(src: &Source) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for w in T3_WORDS {
        out.extend(
            words(&src.code, w)
                .into_iter()
                .map(|at| (at, w.to_string())),
        );
    }
    for p in T3_PREFIXES {
        out.extend((src.code.match_indices(p)).map(|(at, _)| (at, p.to_string())));
    }
    for (at, _) in src.code.match_indices("reinterpret_cast<") {
        let mut rest = src.code[at + "reinterpret_cast<".len()..].trim_start();
        loop {
            let (word, after) = ident(rest);
            if word == "const" || word == "volatile" {
                rest = after.trim_start();
                continue;
            }
            if T3_CASTS.contains(&word) {
                out.push((at, format!("reinterpret_cast<{word}")));
            }
            break;
        }
    }
    let mut lines: Vec<(usize, String)> = (out.into_iter())
        .map(|(at, what)| (line_of(&src.code, at), what))
        .collect();
    lines.sort();
    lines
}

/// T4: math intrinsics, which belong in the op header.
fn t4_intrinsics(src: &Source) -> Vec<Violation> {
    let mut out: Vec<Violation> = Vec::new();
    for w in T4_WORDS {
        for at in words(&src.code, w) {
            out.push(violation(
                line_of(&src.code, at),
                "T4",
                format!("calls `{w}`, which only `{OP_HEADER}` may"),
            ));
        }
    }
    out
}

/// Every `asm` of the raw text, as `(line, its PTX string)`; `None` when no string literal
/// follows the `(`. Adjacent literals are joined and their escapes decoded.
fn asm_strings(raw: &str) -> Vec<(usize, Option<String>)> {
    let mut out = Vec::new();
    let mut at: Vec<usize> = ["asm", "__asm", "__asm__"]
        .iter()
        .flat_map(|w| words(raw, w))
        .collect();
    at.sort_unstable();
    for a in at {
        let (word, mut rest) = ident(&raw[a..]);
        let _ = word;
        rest = rest.trim_start();
        for q in ["volatile", "__volatile__"] {
            if let Some(r) = rest.strip_prefix(q) {
                rest = r.trim_start();
            }
        }
        let mut ptx: Option<String> = None;
        if let Some(r) = rest.strip_prefix('(') {
            rest = r;
            loop {
                rest = rest.trim_start();
                let Some(r) = rest.strip_prefix('"') else {
                    break;
                };
                let mut s = ptx.take().unwrap_or_default();
                let mut chars = r.char_indices();
                let mut end = r.len();
                while let Some((i, c)) = chars.next() {
                    match c {
                        '"' => {
                            end = i + 1;
                            break;
                        }
                        '\\' => match chars.next().map(|(_, e)| e) {
                            Some('n') => s.push('\n'),
                            Some('t') => s.push('\t'),
                            Some(e) => s.push(e),
                            None => {}
                        },
                        c => s.push(c),
                    }
                }
                ptx = Some(s);
                rest = &r[end..];
            }
        }
        out.push((line_of(raw, a), ptx));
    }
    out
}

/// T5: every `asm` is an allowlisted `(file, PTX string)` pair.
fn t5_asm(name: &str, src: &Source) -> Vec<Violation> {
    (asm_strings(&src.raw).into_iter())
        .filter(|(_, ptx)| !(ptx.as_deref()).is_some_and(|p| ASM_ALLOWLIST.contains(&(name, p))))
        .map(|(line, ptx)| {
            violation(
                line,
                "T5",
                match ptx {
                    Some(p) => format!("inline PTX {p:?} is not on the allowlist"),
                    None => "inline assembly with no PTX string".to_string(),
                },
            )
        })
        .collect()
}

/// T6: `__shared__` declares TK semaphores, `st`/`sv` objects, or TK's `extern __shared__` buffer.
fn t6_shared(src: &Source) -> Vec<Violation> {
    let mut out = Vec::new();
    for at in words(&src.bare, "__shared__") {
        let before = src.bare[..at].trim_end();
        let external = before.ends_with("extern")
            && !before[..before.len() - "extern".len()]
                .bytes()
                .last()
                .is_some_and(is_word);
        let mut rest = src.bare[at + "__shared__".len()..].trim_start();
        if let Some(r) = rest.strip_prefix("alignas") {
            rest = r.find(')').map_or("", |e| r[e + 1..].trim_start());
        }
        let (ty, after) = ident(rest);
        let ty_short = ty.strip_prefix("kittens::").unwrap_or(ty);
        let ok = if external {
            let (name, tail) = ident(after.trim_start());
            ["int", "alignment_dummy"].contains(&ty_short)
                && !name.is_empty()
                && tail.trim_start().starts_with('[')
                && tail.trim_start()[1..].trim_start().starts_with(']')
        } else {
            ty_short == "semaphore"
                || ty_short.starts_with("st_")
                || ty_short.starts_with("sv_")
                || (["st", "sv"].contains(&ty_short) && after.trim_start().starts_with('<'))
        };
        if !ok {
            out.push(violation(
                line_of(&src.bare, at),
                "T6",
                format!("`__shared__` of `{ty}`, not a TK semaphore, tile or vector"),
            ));
        }
    }
    out
}

/// The byte range of the body `{ … }` that opens at `open`, by brace matching.
fn body_of(text: &str, open: usize) -> &str {
    let mut depth = 0usize;
    for (i, b) in text.bytes().enumerate().skip(open) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &text[open..=i];
                }
            }
            _ => {}
        }
    }
    &text[open..]
}

/// T7: the file defines a kernel, and every `__global__` body names TK.
fn t7_uses_tk(src: &Source) -> Vec<Violation> {
    let mut out = Vec::new();
    let globals = words(&src.bare, "__global__");
    if globals.is_empty() && !src.bare.contains("lcf::kernel<") {
        out.push(violation(
            1,
            "T7",
            "defines no `__global__` and instantiates no `lcf::kernel<`",
        ));
    }
    for at in globals {
        let rest = &src.bare[at..];
        let Some(open) = rest
            .find(['{', ';'])
            .filter(|&e| rest.as_bytes()[e] == b'{')
        else {
            continue;
        };
        let body = body_of(&src.bare, at + open);
        if !T7_TOKENS.iter().any(|t| !find(body, t, false).is_empty()) {
            out.push(violation(
                line_of(&src.bare, at),
                "T7",
                "a `__global__` body names no TK scope, tile, vector or view",
            ));
        }
    }
    out
}

/// The op header's `// TK-GAP(<name>) begin` … `end` blocks, as `(name, first line, last line)`,
/// and the T9 violations of their markers.
fn gaps(raw: &str) -> (Vec<(String, usize, usize)>, Vec<Violation>) {
    let (mut blocks, mut bad) = (Vec::new(), Vec::new());
    let mut open: Option<(String, usize)> = None;
    for (n, line) in raw.lines().enumerate() {
        let Some(rest) = line.trim().strip_prefix("// TK-GAP(") else {
            continue;
        };
        let Some((name, mark)) = rest.split_once(')') else {
            bad.push(violation(n + 1, "T9", "a malformed TK-GAP marker"));
            continue;
        };
        match (mark.trim(), open.take()) {
            ("begin", None) => open = Some((name.to_string(), n + 1)),
            ("end", Some((begun, from))) if begun == name => {
                blocks.push((begun, from, n + 1));
            }
            (m, was) => {
                bad.push(violation(
                    n + 1,
                    "T9",
                    format!("TK-GAP({name}) {m} does not close an open `begin` of the same name"),
                ));
                open = was;
            }
        }
    }
    if let Some((name, from)) = open {
        bad.push(violation(
            from,
            "T9",
            format!("TK-GAP({name}) begin is never ended"),
        ));
    }
    (blocks, bad)
}

/// T9: in the op header, every T3 or T5 construct sits in a gap, and the gaps are [`TK_GAPS`].
fn t9_gaps(src: &Source) -> Vec<Violation> {
    let (blocks, mut out) = gaps(&src.raw);
    let inside = |line: usize| blocks.iter().any(|(_, a, b)| (*a..=*b).contains(&line));
    for (line, what) in t3_hits(src) {
        if !inside(line) {
            out.push(violation(
                line,
                "T9",
                format!("`{what}` outside a TK-GAP block"),
            ));
        }
    }
    for (line, _) in asm_strings(&src.raw) {
        if !inside(line) {
            out.push(violation(line, "T9", "`asm` outside a TK-GAP block"));
        }
    }
    let names: BTreeSet<&str> = blocks.iter().map(|(n, ..)| n.as_str()).collect();
    let want: BTreeSet<&str> = TK_GAPS.iter().copied().collect();
    if names != want {
        out.push(violation(
            1,
            "T9",
            format!("the TK-GAP names are {names:?}, not the allowlist's {want:?}"),
        ));
    }
    out
}

/// Every rule `kind` holds a source to, its violations by line.
fn check(name: &str, text: &str, kind: Kind) -> Vec<Violation> {
    let src = Source::new(text);
    let mut out = t1_includes(&src, kind);
    out.extend(t2_pointers(&src));
    out.extend(t6_shared(&src));
    match kind {
        Kind::Kernel => {
            out.extend(
                (t3_hits(&src).into_iter())
                    .map(|(line, what)| violation(line, "T3", format!("uses `{what}`"))),
            );
            out.extend(t4_intrinsics(&src));
            out.extend(t5_asm(name, &src));
            out.extend(t7_uses_tk(&src));
        }
        Kind::OpHeader => {
            out.extend(t9_gaps(&src));
            for at in words(&src.bare, "__global__") {
                out.push(violation(
                    line_of(&src.bare, at),
                    "T8",
                    format!("`{OP_HEADER}` defines a `__global__`"),
                ));
            }
        }
    }
    out.sort_by_key(|v| (v.line, v.rule));
    out
}

/// `path:line RULE what`, one per violation.
fn lines(path: &str, found: &[Violation]) -> Vec<String> {
    (found.iter())
        .map(|v| format!("{path}:{} {} {}", v.line, v.rule, v.what))
        .collect()
}

fn rules(found: &[Violation]) -> BTreeSet<&'static str> {
    found.iter().map(|v| v.rule).collect()
}

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn kernels_dir() -> PathBuf {
    crate_dir().join("kernels")
}

/// Every kernel: the `.cu` files directly under `kernels/`, sorted, with their text.
fn kernels() -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = (fs::read_dir(kernels_dir()).expect("read kernels/"))
        .map(|e| e.expect("a kernels/ entry").path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "cu"))
        .map(|p| {
            let name = p
                .file_name()
                .expect("a file name")
                .to_string_lossy()
                .into_owned();
            let text = fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {name}: {e}"));
            (name, text)
        })
        .collect();
    out.sort();
    out
}

/// Every kernel off [`NOT_YET_PORTED`] passes every rule.
#[test]
fn every_ported_kernel_passes_every_rule() {
    let mut failures = Vec::new();
    for (name, text) in kernels() {
        if NOT_YET_PORTED.contains(&name.as_str()) {
            continue;
        }
        let found = check(&name, &text, Kind::Kernel);
        let path = format!("kernels/{name}");
        if found.is_empty() {
            eprintln!("{path}: passes T1-T7 PASS");
        }
        failures.extend(lines(&path, &found));
    }
    for f in &failures {
        eprintln!("{f}");
    }
    assert!(
        failures.is_empty(),
        "{} violation(s) of the ThunderKittens-only rule: see the lines above",
        failures.len()
    );
}

/// Every kernel on [`NOT_YET_PORTED`] exists and still fails some rule; the rules each fails are
/// printed, for its port's issue to quote.
#[test]
fn every_kernel_not_yet_ported_still_fails() {
    let all = kernels();
    let names: BTreeSet<&str> = all.iter().map(|(n, _)| n.as_str()).collect();
    let mut problems = Vec::new();
    let listed: BTreeSet<&str> = NOT_YET_PORTED.iter().copied().collect();
    if listed.len() != NOT_YET_PORTED.len() {
        problems.push("NOT_YET_PORTED names a kernel twice".to_string());
    }
    for name in NOT_YET_PORTED {
        if !names.contains(name) {
            problems.push(format!(
                "NOT_YET_PORTED names {name}, which is not a kernel"
            ));
        }
    }
    for (name, text) in &all {
        if !listed.contains(name.as_str()) {
            continue;
        }
        let found = check(name, text, Kind::Kernel);
        let failed: Vec<&str> = rules(&found).into_iter().collect();
        eprintln!(
            "kernels/{name} (not yet ported) fails {}",
            if failed.is_empty() {
                "nothing".to_string()
            } else {
                failed.join(" ")
            }
        );
        if found.is_empty() {
            problems.push(format!(
                "kernels/{name} passes every rule: remove it from NOT_YET_PORTED"
            ));
        }
    }
    for p in &problems {
        eprintln!("{p}");
    }
    assert!(
        problems.is_empty(),
        "NOT_YET_PORTED is stale: see the lines above"
    );
}

/// T10: the checker refuses plain CUDA, and refuses TK included but unused, by the rules each
/// breaks, and accepts a kernel written with TK.
#[test]
fn the_checker_refuses_plain_cuda_and_accepts_a_tk_kernel() {
    let mut pass = true;
    let cases: [(&str, &str, &[&str]); 3] = [
        (
            "fixture_plain_add.cu",
            FIXTURE_PLAIN_ADD,
            &["T1", "T2", "T3", "T4", "T7"],
        ),
        (
            "fixture_tk_in_name_only.cu",
            FIXTURE_TK_IN_NAME_ONLY,
            &["T2", "T4", "T7"],
        ),
        ("fixture_tk.cu", FIXTURE_TK, &[]),
    ];
    for (name, text, want) in cases {
        let found = check(name, text, Kind::Kernel);
        let got = rules(&found);
        let want: BTreeSet<&str> = want.iter().copied().collect();
        let ok = if want.is_empty() {
            found.is_empty()
        } else {
            want.is_subset(&got)
        };
        eprintln!(
            "T10 {name}: refused by {got:?}, expected {} {}",
            if want.is_empty() {
                "none".to_string()
            } else {
                format!("at least {want:?}")
            },
            if ok { "PASS" } else { "FAIL" }
        );
        for l in lines(name, &found) {
            eprintln!("    {l}");
        }
        pass &= ok;
    }
    assert!(
        pass,
        "the checker does not refuse what it should, or refuses a TK kernel"
    );
}

/// T1, T2, T6, T8 and T9 on the op header: it includes only `"kittens.cuh"`, defines no kernel,
/// and keeps every T3/T5 construct in a named gap, the gaps being exactly [`TK_GAPS`].
#[test]
fn the_op_header_keeps_its_gaps_named() {
    let path = kernels_dir().join(OP_HEADER);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {OP_HEADER}: {e}"));
    let found = check(OP_HEADER, &text, Kind::OpHeader);
    let (blocks, _) = gaps(&text);
    let names: Vec<&str> = blocks.iter().map(|(n, ..)| n.as_str()).collect();
    eprintln!("kernels/{OP_HEADER}: TK-GAP blocks {names:?} (allowlist {TK_GAPS:?})");
    let failures = lines(&format!("kernels/{OP_HEADER}"), &found);
    for f in &failures {
        eprintln!("{f}");
    }
    assert!(
        failures.is_empty(),
        "the op header breaks the rule: see the lines above"
    );
}

/// T5's allowlist names only PTX its file holds: an entry its kernel no longer has is stale.
#[test]
fn every_inline_ptx_allowlist_entry_is_used() {
    let mut stale = Vec::new();
    for (file, ptx) in ASM_ALLOWLIST {
        let text = fs::read_to_string(kernels_dir().join(file))
            .unwrap_or_else(|e| panic!("read the allowlisted {file}: {e}"));
        let used = (asm_strings(&text).into_iter()).any(|(_, p)| p.as_deref() == Some(*ptx));
        eprintln!("T5 allowlist ({file}, {ptx:?}): used={used}");
        if !used {
            stale.push(format!("kernels/{file} holds no `asm` of {ptx:?}"));
        }
    }
    assert!(stale.is_empty(), "stale allowlist entries: {stale:?}");
}

/// Every `.cu`/`.cuh` file under `dir`, recursively, but the vendored tree.
fn cuda_sources(dir: &Path, vendored: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display())) {
        let path = entry.expect("a directory entry").path();
        if path == vendored {
            continue;
        }
        if path.is_dir() {
            cuda_sources(&path, vendored, out);
        } else if path.extension().is_some_and(|e| e == "cu" || e == "cuh") {
            out.push(path);
        }
    }
}

/// T8: the crate's CUDA sources are the kernels and the op header directly under `kernels/`
/// (`build.rs` compiles exactly the top-level `.cu` files, so the checked set is the built set),
/// and the vendored tree.
#[test]
fn the_crate_holds_cuda_sources_only_under_kernels() {
    let (root, kernels) = (crate_dir(), kernels_dir());
    let mut found = Vec::new();
    cuda_sources(&root, &kernels.join("thunderkittens"), &mut found);
    found.sort();
    let mut bad = Vec::new();
    for path in &found {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .display()
            .to_string();
        let top = path.parent() == Some(kernels.as_path());
        let header = path.extension().is_some_and(|e| e == "cuh");
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
        if !top {
            bad.push(format!(
                "{rel} T8 a CUDA source outside kernels/ (or below it)"
            ));
        } else if header && name.as_deref() != Some(OP_HEADER) {
            bad.push(format!(
                "{rel} T8 a header under kernels/ other than {OP_HEADER}"
            ));
        }
    }
    eprintln!(
        "T8: {} CUDA sources under kernels/ (vendored tree excluded), {} outside",
        found.len(),
        bad.len()
    );
    for b in &bad {
        eprintln!("{b}");
    }
    assert!(
        bad.is_empty(),
        "CUDA sources outside kernels/: see the lines above"
    );
}
