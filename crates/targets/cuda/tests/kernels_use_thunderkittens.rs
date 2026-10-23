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
//! | T2 | a `*` in a `__global__` parameter list; a line declaring a pointer with no initialiser, however many declarators it has; a `using`/`typedef` of a pointer type; any pointer member of a kernel's argument struct |
//! | T3 | the constructs plain CUDA kernels move and convert data with: warp shuffles and votes, atomics, `__ldg`, PTX loads and stores, `__nv_bfloat16` and its conversions, `make_*` vector constructors, a `reinterpret_cast` or C-style cast to a scalar or vector pointer (but `(int*)&__shm[0]` as a TK allocator's argument) |
//! | T4 | a math intrinsic in a kernel: any `__name(` call but synchronisation, and any libm function (`sqrtf`, `std::exp2`) outside a host launcher; those live in `kernels/scratchy_ops.cuh` |
//! | T5 | `asm` anywhere but an exact `(file, PTX string)` pair of [`ASM_ALLOWLIST`], each at most as often as it is listed |
//! | T6 | `__shared__` of anything but TK semaphores, `st`/`sv` tiles and vectors, or the `extern __shared__` buffer TK's allocators carve |
//! | T7 | a kernel file with no kernel, or a `__global__` body that names no TK scope, nor a TK tile, vector or view type as a whole identifier |
//! | T8 | a `.cu`/`.cuh` outside `kernels/` (but the vendored tree), a `.cuh` there but the op header, a `__global__` in the op header |
//! | T9 | a T3/T5 construct in the op header outside a `// TK-GAP(<name>)` block, or a gap name not on [`TK_GAPS`] |
//! | T10 | the checker itself: plain-CUDA fixtures must be refused, a TK fixture accepted |
//!
//! T1-T4, T6 and T7 read the text with its comments blanked (string literals kept); T5 and the
//! gap markers read it raw. Listed tokens match on word boundaries, as `cfg_neutrality.rs`'s
//! `has_word` does, except the prefixes `__shfl_` and `atomic`, which match anywhere; the CUDA
//! intrinsic families (conversions, cache hints, warp collectives, math) match by pattern, so a
//! spelling no list names is refused too.
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

/// T4: the `__name(` calls a kernel may make: synchronisation and the launch bounds.
const T4_ALLOWED_DUNDER: &[&str] = &["__syncthreads", "__syncwarp", "__launch_bounds__"];

/// T4: the libm and CUDA math functions, by stem: each is refused unqualified or `std::`/`::`
/// qualified, and with an `f` or `l` suffix (`sqrtf`, `tanhl`), outside a host launcher.
/// Integer `min`/`max`/`abs` are index math, not on the list.
const LIBM: &[&str] = &[
    "exp",
    "exp2",
    "exp10",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "logb",
    "ilogb",
    "sqrt",
    "rsqrt",
    "cbrt",
    "rcbrt",
    "hypot",
    "rhypot",
    "norm",
    "rnorm",
    "norm3d",
    "rnorm3d",
    "norm4d",
    "rnorm4d",
    "pow",
    "powi",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "sincos",
    "sinpi",
    "cospi",
    "sincospi",
    "erf",
    "erfc",
    "erfinv",
    "erfcinv",
    "erfcx",
    "normcdf",
    "normcdfinv",
    "lgamma",
    "tgamma",
    "j0",
    "j1",
    "jn",
    "y0",
    "y1",
    "yn",
    "fma",
    "fmin",
    "fmax",
    "fabs",
    "fdim",
    "fmod",
    "remainder",
    "remquo",
    "floor",
    "ceil",
    "trunc",
    "round",
    "rint",
    "nearbyint",
    "lround",
    "llround",
    "lrint",
    "llrint",
    "frexp",
    "ldexp",
    "scalbn",
    "scalbln",
    "modf",
    "copysign",
    "nextafter",
    "nan",
    "isnan",
    "isinf",
    "isfinite",
    "signbit",
    "saturate",
    "fdivide",
];

/// T7: TK's scopes, as a body names them (`warp::load`).
const T7_SCOPES: &[&str] = &["warp::", "warpgroup::", "group<", "tma::", "kittens::"];
/// T7: TK's tile, vector and view types, matched as whole identifiers (`rt_fl<16, 64>`, not
/// `rt_dummy`).
const T7_TYPES: &[&str] = &[
    "rt",
    "rv",
    "st",
    "sv",
    "gl",
    "crt",
    "crv",
    "cst",
    "csv",
    "rt_bf",
    "rt_fl",
    "rt_hf",
    "rt_int",
    "rt_int8",
    "rt_uint8",
    "rt_fp8e4m3",
    "rt_fp8e5m2",
    "rt_fp8e8m0",
    "rv_bf",
    "rv_fl",
    "rv_hf",
    "st_bf",
    "st_fl",
    "st_hf",
    "st_int",
    "st_int8",
    "st_uint8",
    "st_fp8e4m3",
    "st_fp8e5m2",
    "st_fp8e8m0",
    "sv_bf",
    "sv_fl",
    "sv_hf",
    "sv_int",
    "sv_int8",
    "sv_uint8",
    "sv_fp8e4m3",
    "sv_fp8e5m2",
    "sv_fp8e8m0",
    "crt_bf",
    "crt_fl",
    "crt_hf",
    "crv_bf",
    "crv_fl",
    "crv_hf",
    "cst_bf",
    "cst_fl",
    "cst_hf",
    "csv_bf",
    "csv_fl",
    "csv_hf",
    "col_vec",
    "row_vec",
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

/// T10's verifier fixtures: one construct each that slipped past the first version of the
/// checker (PR #13's review), in an otherwise conforming kernel, with the rule that must refuse
/// it. `{}` is replaced by the construct, inside the body of a `__global__` that names TK.
const GAP_KERNEL: &str = r##"#include "kittens.cuh"

#include <cstdint>

using namespace kittens;

namespace {
using row_gl = gl<float, 1, 1, -1, -1>;
struct gap_globals {
    row_gl x;
    row_gl y;
};

__global__ void gap_ker(const __grid_constant__ gap_globals g) {
    rv_fl<256> row;
    warp::load(row, g.x, {0, 0, int(blockIdx.x), 0});
    float w = 1.0f;
    {}
}
}  // namespace
"##;

/// `(name, statement in the body, the rule that must refuse it)`.
const GAP_STATEMENTS: &[(&str, &str, &str)] = &[
    (
        "bf16_rn_conversion",
        "auto b = __float2bfloat16_rn(w);",
        "T3",
    ),
    (
        "bf16_rz_conversion",
        "auto b = __float2bfloat16_rz(w);",
        "T3",
    ),
    (
        "bf16_to_float",
        "float f = __bfloat162float(__float2bfloat16(w));",
        "T3",
    ),
    ("bfloat16_family", "auto b = __bfloat16_as_ushort(w);", "T3"),
    ("half_conversion", "auto h = __half2float(w);", "T3"),
    ("int_conversion", "float f = __int2float_rn(int(w));", "T3"),
    ("sqrtf", "w = sqrtf(w);", "T4"),
    ("exp2_no_suffix", "w = exp2(w);", "T4"),
    ("std_sqrt", "w = std::sqrt(w);", "T4"),
    ("tanhf", "w = tanhf(w);", "T4"),
    ("logf", "w = logf(w);", "T4"),
    ("fmaf_rn", "w = __fmaf_rn(w, w, w);", "T4"),
    ("fadd_rz", "w = __fadd_rz(w, w);", "T4"),
    ("fmul_rd", "w = __fmul_rd(w, w);", "T4"),
    ("fdividef", "w = __fdividef(w, 3.0f);", "T4"),
    ("fast_logf", "w = __logf(w);", "T4"),
    ("hmul", "auto h = __hmul(w, w);", "T4"),
    ("ldcs", "w = __ldcs(&w);", "T3"),
    ("stcs", "__stcs(&w, w);", "T3"),
    ("ldcg", "w = __ldcg(&w);", "T3"),
    ("stcg", "__stcg(&w, w);", "T3"),
    ("ldlu", "w = __ldlu(&w);", "T3"),
    ("ldcv", "w = __ldcv(&w);", "T3"),
    ("stwb", "__stwb(&w, w);", "T3"),
    ("stwt", "__stwt(&w, w);", "T3"),
    (
        "reduce_add_sync",
        "int s = __reduce_add_sync(0xffffffffu, int(w));",
        "T3",
    ),
    (
        "reduce_max_sync",
        "int s = __reduce_max_sync(0xffffffffu, int(w));",
        "T3",
    ),
    ("syncwarp_vote", "unsigned v = __activemask();", "T3"),
    ("c_cast_float4", "float4 v = *(const float4*)(&w);", "T3"),
    (
        "c_cast_shared",
        "extern __shared__ int __shm[]; float* s = (float*)__shm;",
        "T3",
    ),
    ("c_cast_uintptr", "float* p = (float*)uintptr_t(0);", "T3"),
    (
        "reinterpret_scalar",
        "float* p = reinterpret_cast<float*>(&w);",
        "T3",
    ),
    (
        "shm_int_cast_stored",
        "extern __shared__ int __shm[]; int* p = (int*)&__shm[0];",
        "T3",
    ),
    (
        "shm_int_cast_passed_elsewhere",
        "extern __shared__ int __shm[]; consume((int*)&__shm[0]);",
        "T3",
    ),
    (
        "shm_int_cast_offset_into_allocator",
        "extern __shared__ int __shm[]; tma_swizzle_allocator al((int*)&__shm[4]);",
        "T3",
    ),
];

/// Statements that must pass in the same kernel: TK's own idiom for carving shared memory, the
/// one C-style cast the rules allow.
const GAP_ACCEPTED: &[(&str, &str)] = &[
    (
        "shm_tma_swizzle_allocator",
        "extern __shared__ int __shm[]; tma_swizzle_allocator al((int*)&__shm[0]);",
    ),
    (
        "shm_shared_allocator_bare",
        "extern __shared__ int __shm[]; kittens::shared_allocator<16> al((int*)__shm);",
    ),
];

/// `(name, a whole kernel source, the rule that must refuse it)`: the verifier's constructs that
/// live outside a body.
const GAP_SOURCES: &[(&str, &str, &str)] = &[
    (
        "two_pointer_declarators",
        r##"#include "kittens.cuh"
using namespace kittens;
namespace {
using row_gl = gl<float, 1, 1, -1, -1>;
struct gap_globals {
    row_gl x;
    float *y, *z;
};
__global__ void gap_ker(const __grid_constant__ gap_globals g) {
    rv_fl<256> row;
    warp::load(row, g.x, {0, 0, 0, 0});
}
}  // namespace
"##,
        "T2",
    ),
    (
        "initialised_pointer_field",
        r##"#include "kittens.cuh"
using namespace kittens;
namespace {
using row_gl = gl<float, 1, 1, -1, -1>;
struct gap_globals {
    row_gl x;
    const float* y = nullptr;
};
__global__ void gap_ker(const __grid_constant__ gap_globals g) {
    rv_fl<256> row;
    warp::load(row, g.x, {0, 0, 0, 0});
}
}  // namespace
"##,
        "T2",
    ),
    (
        "pointer_alias_parameter",
        r##"#include "kittens.cuh"
using namespace kittens;
namespace {
using fptr = const float*;
__global__ void gap_ker(fptr p) {
    rv_fl<256> row;
    warp::zero(row);
}
}  // namespace
"##,
        "T2",
    ),
    (
        "pointer_typedef_parameter",
        r##"#include "kittens.cuh"
using namespace kittens;
namespace {
typedef float* fptr;
__global__ void gap_ker(fptr p) {
    rv_fl<256> row;
    warp::zero(row);
}
}  // namespace
"##,
        "T2",
    ),
    (
        "t7_identifier_prefix_only",
        r##"#include "kittens.cuh"
namespace {
struct gap_globals {
    int n;
};
__global__ void gap_ker(const __grid_constant__ gap_globals g) {
    int rt_dummy = g.n, st_x = 0, sv_y = 0, rv_z = 0;
    (void)rt_dummy; (void)st_x; (void)sv_y; (void)rv_z;
}
}  // namespace
"##,
        "T7",
    ),
    (
        "second_allowlisted_fence",
        r##"#include "kittens.cuh"
using namespace kittens;
namespace {
using row_gl = gl<float, 1, 1, -1, -1>;
struct gap_globals {
    row_gl x;
};
__global__ void gap_ker(const __grid_constant__ gap_globals g) {
    rv_fl<256> row;
    warp::load(row, g.x, {0, 0, 0, 0});
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
}  // namespace
"##,
        "T5",
    ),
];

/// The verifier's whole scratch kernel, `zz_game.cu`, as the review lists it: every construct at
/// once. It must be refused by T2, T3, T4 and T7.
const FIXTURE_VERIFIER_GAME: &str = r##"#include "kittens.cuh"

#include <cstdint>

using namespace kittens;

namespace {
using fptr = const float*;
struct game_globals {
    const float* x = nullptr;
    float *y, *z;
    uintptr_t addr;
};

__global__ void game_ker(const __grid_constant__ game_globals g) {
    extern __shared__ int __shm[];
    float* s = (float*)__shm;
    int rt_dummy = 0, st_x = 0;
    float w = *(const float4*)(g.x + 4 * threadIdx.x) .x;
    w = sqrtf(w) + exp2(w) + __fmaf_rn(w, w, w) + __fdividef(w, 2.0f) + __logf(w) + tanhf(w);
    auto h = __hmul(__float2half(w), __float2half(w));
    auto b = __float2bfloat16_rn(w);
    w += __ldcs(g.x) + float(__reduce_add_sync(0xffffffffu, rt_dummy + st_x));
    __stcs(g.y, w);
    float* p = (float*)g.addr;
    s[0] = w + float(b) + float(h) + *p;
}

__global__ void game_alias_ker(fptr q) {
    (void)q;
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

/// Every identifier of `text` (`[A-Za-z_][A-Za-z0-9_]*`, a number literal's letters excluded),
/// with its byte offset.
fn identifiers(text: &str) -> Vec<(usize, &str)> {
    let b = text.as_bytes();
    let (mut out, mut i) = (Vec::new(), 0);
    while i < b.len() {
        if b[i] == b'_' || b[i].is_ascii_alphabetic() {
            let start = i;
            while i < b.len() && is_word(b[i]) {
                i += 1;
            }
            out.push((start, &text[start..i]));
        } else if b[i].is_ascii_digit() {
            // A number (`1.0f`, `0x7FC1`, `1e-6`): its letters are no identifier.
            while i < b.len() && (is_word(b[i]) || b[i] == b'.') {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    out
}

/// The namespace an identifier at `at` is qualified with: `None` unqualified, `Some("")` for a
/// global `::name`, else the last qualifier (`std` for `std::sqrt`, `warp` for `warp::exp2`).
fn qualifier(text: &str, at: usize) -> Option<&str> {
    let before = text[..at].trim_end().strip_suffix("::")?.trim_end();
    let start = before
        .rfind(|c: char| !(c == '_' || c.is_ascii_alphanumeric()))
        .map_or(0, |i| i + 1);
    Some(&before[start..])
}

/// Whether the text after byte `end` opens a call: the next non-space character is `(`.
fn called(text: &str, end: usize) -> bool {
    text[end..].trim_start().starts_with('(')
}

/// Whether `s` starts with the word `w`.
fn starts_with_word(s: &str, w: &str) -> bool {
    s.strip_prefix(w)
        .is_some_and(|r| !r.bytes().next().is_some_and(is_word))
}

/// The byte ranges of the bodies of `extern "C"` functions (and blocks): the host launchers,
/// which may use what kernels may not (`aa-personal-space/cuda-moonshot/PLAN-tk100.md` §1.3).
fn host_ranges(bare: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for at in words(bare, "extern") {
        if !bare[at + "extern".len()..].trim_start().starts_with('"') {
            continue;
        }
        let rest = &bare[at..];
        let mut depth = 0i32;
        let open = rest.char_indices().find_map(|(i, c)| {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                ';' if depth == 0 => return Some(None),
                '{' if depth == 0 => return Some(Some(i)),
                _ => {}
            }
            None
        });
        if let Some(Some(open)) = open {
            let body = body_of(bare, at + open);
            out.push((at + open, at + open + body.len()));
        }
    }
    out
}

/// Whether identifier `name` is one of the constructs T3 refuses by pattern, and which.
fn t3_family(name: &str) -> Option<&'static str> {
    let lower = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_lowercase());
    if let Some(rest) = name.strip_prefix("__") {
        // Warp collectives: every `__shfl*`, every `__*_sync`, and the mask they take.
        if rest.starts_with("shfl") || rest.ends_with("_sync") || rest == "activemask" {
            return Some("a warp shuffle, vote or reduction");
        }
        // Cache-hinted global loads and stores: `__ldg`, `__ldcs`, `__stwt`, ...
        if let Some(hint) = rest.strip_prefix("ld").or_else(|| rest.strip_prefix("st"))
            && (1..=2).contains(&hint.len())
            && lower(hint)
        {
            return Some("a cache-hinted global load or store");
        }
        // The CUDA bf16/half types and every conversion: `__nv_*`, `__half*`, `__bfloat16*`,
        // `__<type>2<type>*` (`__float2bfloat16_rn`, `__int2float_rn`), `__<type>_as_<type>`.
        let conversion = rest
            .split_once('2')
            .is_some_and(|(from, to)| lower(from.trim_end_matches('s')) && !to.is_empty());
        if rest.starts_with("nv_")
            || rest.starts_with("half")
            || rest.starts_with("bfloat16")
            || conversion
            || rest.contains("_as_")
        {
            return Some("a CUDA bf16/half type or a conversion intrinsic");
        }
    }
    // The CUDA vector constructors: `make_uint4`, `make_float2`, ...
    if let Some(rest) = name.strip_prefix("make_")
        && rest.ends_with(['1', '2', '3', '4'])
        && lower(&rest[..rest.len() - 1])
    {
        return Some("a CUDA vector constructor");
    }
    None
}

/// Whether identifier `name` names a scalar or CUDA vector type: what a cast may not make a
/// pointer or reference to (memory read as anything but a TK type).
fn scalar_type(name: &str) -> bool {
    const FUNDAMENTAL: &[&str] = &[
        "void",
        "char",
        "signed",
        "unsigned",
        "short",
        "int",
        "long",
        "float",
        "double",
        "bool",
        "size_t",
        "ptrdiff_t",
        "uintptr_t",
        "intptr_t",
        "bf16",
        "bf16_2",
        "half",
        "half_2",
    ];
    let vector = |s: &str| {
        s.strip_suffix(['1', '2', '3', '4']).is_some_and(|base| {
            let base = base.strip_prefix('u').unwrap_or(base);
            [
                "char", "short", "int", "long", "longlong", "float", "double",
            ]
            .contains(&base)
        })
    };
    let fixed = |s: &str| {
        (s.strip_prefix("uint").or_else(|| s.strip_prefix("int")))
            .and_then(|r| r.strip_suffix("_t"))
            .is_some_and(|w| ["8", "16", "32", "64"].contains(&w))
    };
    FUNDAMENTAL.contains(&name) || vector(name) || fixed(name) || name.starts_with("__nv_")
}

/// TK's shared-memory allocators: the types whose constructor takes the `extern __shared__`
/// buffer as an `int*` (`common/util.cuh`).
const TK_ALLOCATORS: &[&str] = &["shared_allocator", "tma_allocator", "tma_swizzle_allocator"];

/// Whether `text` ends with the name of a TK allocator type, `kittens::` qualified or not, with or
/// without template arguments (`shared_allocator<1024>`).
fn ends_with_allocator(text: &str) -> bool {
    let text = text.trim_end();
    let text = match text.strip_suffix('>') {
        Some(t) => t.rfind('<').map_or(t, |i| t[..i].trim_end()),
        None => text,
    };
    let start = text
        .rfind(|c: char| !(c == '_' || c == ':' || c.is_ascii_alphanumeric()))
        .map_or(0, |i| i + 1);
    let name = &text[start..];
    TK_ALLOCATORS.contains(&name.strip_prefix("kittens::").unwrap_or(name))
}

/// Whether the cast opening at byte `at` of `bare` is TK's own idiom for carving shared memory
/// (`aa-personal-space/cuda-moonshot/PLAN-tk100.md` §1.2 item 5), as upstream's `mha_h100.cu`
/// spells it: exactly `(int*)&__shm[0]` or `(int*)__shm` (spaces aside), as the whole argument of
/// a TK allocator's constructor, `tma_swizzle_allocator al((int*)&__shm[0]);`. Stored anywhere
/// else, or passed to anything else, the cast is refused like any other.
fn allocator_argument(bare: &str, at: usize) -> bool {
    // The callee: `(` right before the cast, after `ALLOCATOR name` or `ALLOCATOR`.
    let Some(callee) = bare[..at].trim_end().strip_suffix('(') else {
        return false;
    };
    let callee = callee.trim_end();
    let name_start = callee
        .rfind(|c: char| !(c == '_' || c.is_ascii_alphanumeric()))
        .map_or(0, |i| i + 1);
    if !(ends_with_allocator(callee) || ends_with_allocator(&callee[..name_start])) {
        return false;
    }
    // The argument: from the cast to the `)` that closes the call, which then ends the statement.
    let mut depth = 0i32;
    let close = bare[at..].char_indices().find_map(|(i, c)| {
        match c {
            '(' => depth += 1,
            ')' if depth == 0 => return Some(at + i),
            ')' => depth -= 1,
            _ => {}
        }
        None
    });
    let Some(close) = close else {
        return false;
    };
    let arg: String = bare[at..close].split_whitespace().collect();
    (arg == "(int*)&__shm[0]" || arg == "(int*)__shm")
        && bare[close + 1..].trim_start().starts_with(';')
}

/// Every C-style cast to a pointer or reference type in `bare`, `(T*)x` or `(const T&)x`, as
/// `(offset, what)`, but TK's allocator idiom ([`allocator_argument`]). A `(` preceded by a word,
/// `)`, `]` or `>` opens a call or a declarator, not a cast.
fn c_style_casts(bare: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for (at, _) in bare.match_indices('(') {
        let before = bare[..at].trim_end().bytes().last();
        if before.is_some_and(|b| is_word(b) || b")]>".contains(&b)) {
            continue;
        }
        let inner = &bare[at + 1..];
        let Some(close) = inner.find(')') else {
            continue;
        };
        let ty = inner[..close].trim();
        let head = ty.trim_end_matches(|c: char| c == '*' || c == '&' || c.is_whitespace());
        let head = head
            .trim_end_matches("const")
            .trim_end_matches("volatile")
            .trim_end();
        let star = &ty[head.len()..];
        let type_like = head.chars().all(|c| {
            c == '_'
                || c == ':'
                || c == '<'
                || c == '>'
                || c == ','
                || c == ' '
                || c.is_ascii_alphanumeric()
        });
        let named = identifiers(head)
            .iter()
            .any(|(_, w)| !["const", "volatile"].contains(w));
        if type_like && named && star.contains(['*', '&']) && !allocator_argument(bare, at) {
            out.push((at, format!("the C-style cast `({ty})`")));
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
                .map(|at| (at, format!("uses `{w}`"))),
        );
    }
    for p in T3_PREFIXES {
        out.extend((src.code.match_indices(p)).map(|(at, _)| (at, format!("uses `{p}`"))));
    }
    let mut lines: Vec<(usize, String)> = (out.into_iter())
        .map(|(at, what)| (line_of(&src.code, at), what))
        .collect();
    let mut bare = Vec::new();
    for (at, name) in identifiers(&src.bare) {
        // A name of the issue's own list is reported once, by that list.
        if T3_WORDS.contains(&name) || T3_PREFIXES.iter().any(|p| name.starts_with(p)) {
            continue;
        }
        if let Some(family) = t3_family(name) {
            bare.push((at, format!("uses `{name}`, {family}")));
        }
    }
    for (at, _) in src.bare.match_indices("reinterpret_cast<") {
        let mut rest = src.bare[at + "reinterpret_cast<".len()..].trim_start();
        while let Some(r) = ["const", "volatile"]
            .iter()
            .find_map(|q| starts_with_word(rest, q).then(|| &rest[q.len()..]))
        {
            rest = r.trim_start();
        }
        let (word, _) = ident(rest);
        if T3_CASTS.contains(&word) || scalar_type(word) {
            bare.push((at, format!("uses `reinterpret_cast<{word}`")));
        }
    }
    bare.extend(c_style_casts(&src.bare));
    lines.extend((bare.into_iter()).map(|(at, what)| (line_of(&src.bare, at), what)));
    lines.sort();
    lines.dedup();
    lines
}

/// T4: math intrinsics and libm functions, which belong in the op header: the listed names
/// anywhere; any other `__name(` call but synchronisation and launch bounds anywhere; and a libm
/// function, unqualified or `std::`/`::`-qualified, anywhere but a host launcher's body. A call
/// qualified with any other namespace (`warp::exp2`) is TK's or the op header's.
fn t4_intrinsics(src: &Source) -> Vec<Violation> {
    let mut hits: BTreeSet<(usize, String)> = BTreeSet::new();
    for w in T4_WORDS {
        for at in words(&src.code, w) {
            hits.insert((line_of(&src.code, at), w.to_string()));
        }
    }
    let host = host_ranges(&src.bare);
    let in_host = |at: usize| host.iter().any(|&(a, b)| (a..b).contains(&at));
    for (at, name) in identifiers(&src.bare) {
        // `x.exp(…)`, `p->exp(…)`: a member of a TK object, not a free function.
        let before = src.bare[..at].trim_end();
        let member = before.ends_with('.') || before.ends_with("->");
        if member || !called(&src.bare, at + name.len()) || t3_family(name).is_some() {
            continue;
        }
        let dunder = name.starts_with("__")
            && name.as_bytes().get(2).is_some_and(u8::is_ascii_lowercase)
            && !T4_ALLOWED_DUNDER.contains(&name);
        let stem = |n: &str| {
            LIBM.contains(&n)
                || n.strip_suffix(['f', 'l'])
                    .is_some_and(|s| LIBM.contains(&s))
        };
        let libm = stem(name)
            && matches!(qualifier(&src.bare, at), None | Some("") | Some("std"))
            && !in_host(at);
        if dunder || libm {
            hits.insert((line_of(&src.bare, at), name.to_string()));
        }
    }
    (hits.into_iter())
        .map(|(line, w)| {
            violation(
                line,
                "T4",
                format!("calls `{w}`, which only `{OP_HEADER}` may"),
            )
        })
        .collect()
}

/// Words that open a statement which is not a declaration, though a type-like run and a `*`
/// follow (`return a * b;`).
const NOT_DECLARATIONS: &[&str] = &[
    "return",
    "case",
    "throw",
    "delete",
    "else",
    "goto",
    "co_return",
];

/// Whether statement `stmt` (its `;` removed) declares a pointer: a type, then a declarator list
/// one of whose declarators is `*name` (any `*`s, `const`, `volatile` and `__restrict__`), `name`
/// followed by nothing, an array bound, or, when `init` allows it, an initialiser. Covers the
/// issue's `^\s*(const\s+)?[\w:<>, ]+\*\s*(__restrict__\s*)?\w+\s*;` and the multi-declarator
/// `float *y, *z;`. A `*` between two operands (`st_bf<64, 64*N>`, `a * b`) declares nothing: no
/// name follows it.
fn declares_pointer(stmt: &str, init: bool) -> bool {
    let t = stmt.trim();
    if NOT_DECLARATIONS.iter().any(|k| starts_with_word(t, k)) {
        return false;
    }
    // `[\w:<>, ]+`: the type, `const` included, and any declarators before the first `*`.
    let ty = |c: char| c == '_' || c.is_ascii_alphanumeric() || ":<>, ".contains(c);
    let run = t.find(|c: char| !ty(c)).unwrap_or(t.len());
    if run == 0 || !t[run..].starts_with('*') || identifiers(&t[..run]).is_empty() {
        return false;
    }
    t[run..].split(',').any(|d| {
        let Some(mut r) = d.trim().strip_prefix('*') else {
            return false;
        };
        loop {
            r = r.trim_start().trim_start_matches('*').trim_start();
            match ["const", "volatile", "__restrict__"]
                .iter()
                .find(|q| starts_with_word(r, q))
            {
                Some(q) => r = &r[q.len()..],
                None => break,
            }
        }
        let name = r
            .find(|c: char| !(c == '_' || c.is_ascii_alphanumeric()))
            .unwrap_or(r.len());
        let starts = r
            .bytes()
            .next()
            .is_some_and(|b| b == b'_' || b.is_ascii_alphabetic());
        // Past the name, any array bounds; then nothing, or an initialiser where `init` allows.
        let mut after = r[name..].trim_start();
        while let Some(b) = after.strip_prefix('[') {
            after = b.find(']').map_or("", |e| b[e + 1..].trim_start());
        }
        starts && (after.is_empty() || (init && after.starts_with('=')))
    })
}

/// Whether a `*` at byte `star` of `ty` (a type, with no declarator name) is a pointer's, not a
/// product's: nothing that could be an operand follows it.
fn pointer_star(ty: &str, star: usize) -> bool {
    let rest = ty[star + 1..].trim_start();
    rest.is_empty()
        || rest.starts_with(['>', ',', ')', ';', '*', '&', '['])
        || ["const", "volatile", "__restrict__"]
            .iter()
            .any(|q| starts_with_word(rest, q))
}

/// Every `using NAME = TYPE;` and `typedef TYPE NAME;` of `bare` whose type is a pointer (or holds
/// one), as `(offset, name)`: a pointer behind an alias leaves no `*` where the alias is used.
fn pointer_aliases(bare: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for at in words(bare, "using") {
        let rest = bare[at + "using".len()..].trim_start();
        if starts_with_word(rest, "namespace") {
            continue;
        }
        let (name, after) = ident(rest);
        let Some(ty) = after.trim_start().strip_prefix('=') else {
            continue;
        };
        let ty = &ty[..ty.find(';').unwrap_or(ty.len())];
        if (ty.match_indices('*')).any(|(s, _)| pointer_star(ty, s)) {
            out.push((at, name.to_string()));
        }
    }
    for at in words(bare, "typedef") {
        let rest = &bare[at + "typedef".len()..];
        let stmt = &rest[..rest.find(';').unwrap_or(rest.len())];
        let Some(&(last, name)) = identifiers(stmt).last() else {
            continue;
        };
        let ty = &stmt[..last];
        if (ty.match_indices('*')).any(|(s, _)| pointer_star(ty, s)) {
            out.push((at, name.to_string()));
        }
    }
    out
}

/// Every `struct NAME` / `class NAME` of `bare` with a body, as `NAME -> body offset`.
fn structs(bare: &str) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    for kw in ["struct", "class"] {
        for at in words(bare, kw) {
            let (name, after) = ident(bare[at + kw.len()..].trim_start());
            if name.is_empty() {
                continue;
            }
            let start = bare.len() - after.len();
            let Some(open) = bare[start..].find(['{', ';', '(', ')', '>']) else {
                continue;
            };
            if bare.as_bytes()[start + open] == b'{' {
                out.push((name.to_string(), start + open));
            }
        }
    }
    out
}

/// T2: no `*` from a `__global__` to the `{` (or `;`) after its parameters; no line declaring a
/// pointer with no initialiser, however many declarators it has; no `using`/`typedef` of a
/// pointer type; and no pointer member, initialised or not, in a kernel's argument struct (every
/// struct of the file a `__global__` names before its body, or, when the file instantiates
/// `lcf::kernel<`, its `globals`), nor in a struct of the file such a struct holds.
fn t2_pointers(src: &Source) -> Vec<Violation> {
    let mut out = Vec::new();
    let mut roots: Vec<String> = Vec::new();
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
        roots.extend(
            identifiers(&rest[..end])
                .into_iter()
                .map(|(_, w)| w.to_string()),
        );
    }
    if src.bare.contains("lcf::kernel<") {
        roots.push("globals".to_string());
    }
    for (n, line) in src.bare.lines().enumerate() {
        if let Some(stmt) = line.trim_end().strip_suffix(';')
            && declares_pointer(stmt, false)
        {
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
    for (at, name) in pointer_aliases(&src.bare) {
        out.push(violation(
            line_of(&src.bare, at),
            "T2",
            format!("`{name}` is an alias of a pointer type"),
        ));
    }
    let defined = structs(&src.bare);
    let mut seen: BTreeSet<String> = BTreeSet::new();
    while let Some(name) = roots.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        for (_, open) in defined.iter().filter(|(n, _)| *n == name) {
            let body = body_of(&src.bare, *open);
            let inner = &body[1..body.len().saturating_sub(1)];
            let mut from = 0;
            for stmt in inner.split(';') {
                if declares_pointer(stmt.trim_start_matches(['{', '}']), true) {
                    out.push(violation(
                        line_of(
                            &src.bare,
                            open + 1 + from + stmt.len() - stmt.trim_start().len(),
                        ),
                        "T2",
                        format!(
                            "the kernel argument struct `{name}` holds a pointer: `{}`",
                            stmt.trim()
                        ),
                    ));
                }
                from += stmt.len() + 1;
            }
            roots.extend(identifiers(inner).into_iter().map(|(_, w)| w.to_string()));
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
        let (_, mut rest) = ident(&raw[a..]);
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

/// T5: every `asm` is an allowlisted `(file, PTX string)` pair, each at most as often as the pair
/// is listed.
fn t5_asm(name: &str, src: &Source) -> Vec<Violation> {
    let mut left: Vec<(&str, &str)> = ASM_ALLOWLIST.to_vec();
    let mut out = Vec::new();
    for (line, ptx) in asm_strings(&src.raw) {
        let listed = (ptx.as_deref()).is_some_and(|p| ASM_ALLOWLIST.contains(&(name, p)));
        let slot = (ptx.as_deref()).and_then(|p| left.iter().position(|&e| e == (name, p)));
        match (slot, ptx) {
            (Some(i), _) => {
                left.remove(i);
            }
            (None, Some(p)) if listed => out.push(violation(
                line,
                "T5",
                format!("inline PTX {p:?} appears more often than the allowlist lists it"),
            )),
            (None, Some(p)) => out.push(violation(
                line,
                "T5",
                format!("inline PTX {p:?} is not on the allowlist"),
            )),
            (None, None) => out.push(violation(line, "T5", "inline assembly with no PTX string")),
        }
    }
    out
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

/// T7: the file defines a kernel, and every `__global__` body names TK: one of its scopes
/// ([`T7_SCOPES`], on a left word boundary) or one of its types as a whole identifier
/// ([`T7_TYPES`]).
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
        let scope = T7_SCOPES.iter().any(|t| !find(body, t, false).is_empty());
        let ty = identifiers(body).iter().any(|(_, w)| T7_TYPES.contains(w));
        if !(scope || ty) {
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
                format!("{what}, outside a TK-GAP block"),
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
            out.extend((t3_hits(&src).into_iter()).map(|(line, what)| violation(line, "T3", what)));
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

/// One T10 case: `name`'s source must be refused by at least `want`, or by nothing when `want` is
/// empty. Prints its line and every violation found.
fn self_test(name: &str, text: &str, want: &[&str]) -> bool {
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
    ok
}

/// T10: the checker refuses plain CUDA, and refuses TK included but unused, by the rules each
/// breaks, and accepts a kernel written with TK.
#[test]
fn the_checker_refuses_plain_cuda_and_accepts_a_tk_kernel() {
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
    let mut pass = true;
    for (name, text, want) in cases {
        pass &= self_test(name, text, want);
    }
    assert!(
        pass,
        "the checker does not refuse what it should, or refuses a TK kernel"
    );
}

/// T10: every construct of PR #13's review that slipped past the first checker is refused, alone
/// in an otherwise conforming kernel, by its rule; the conforming kernel itself, with no
/// construct and with TK's shared-memory allocator idiom, passes; and the verifier's whole
/// scratch kernel is refused by T2, T3, T4 and T7.
#[test]
fn the_checker_refuses_every_construct_the_review_slipped_past_it() {
    let mut pass = self_test("gap_kernel_empty.cu", &GAP_KERNEL.replace("{}", ""), &[]);
    for (name, stmt) in GAP_ACCEPTED {
        let text = GAP_KERNEL.replace("{}", stmt);
        pass &= self_test(&format!("gap_{name}.cu"), &text, &[]);
    }
    for (name, stmt, rule) in GAP_STATEMENTS {
        let text = GAP_KERNEL.replace("{}", stmt);
        pass &= self_test(&format!("gap_{name}.cu"), &text, &[rule]);
    }
    for (name, text, rule) in GAP_SOURCES {
        pass &= self_test(&format!("gap_{name}.cu"), text, &[rule]);
    }
    pass &= self_test(
        "fixture_verifier_game.cu",
        FIXTURE_VERIFIER_GAME,
        &["T2", "T3", "T4", "T7"],
    );
    assert!(
        pass,
        "a construct the review found is not refused by its rule: see the lines above"
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
