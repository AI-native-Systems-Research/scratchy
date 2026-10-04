# triton-frontend — bridge one: Python kernel source → TTIR, in Rust

Takes the text of a `@triton.jit` kernel plus its signature and constexpr bindings, and
produces an in-memory `ttir::Module`. **No Python runs. Nothing is shelled out to.**

```
cargo test --features ruff          # 59 tests: golden diff, controls, divergence, status,
                                    #           gather, transcendental width, slicing, fusion
cargo test --no-default-features    # 6 tests: zero dependencies, no Python parser
```

The dependency-free run is not a stub. It holds the crate to: every raw golden parsing
(all sixteen, including the 647-line causal-attention one and the `si32` descriptor in
`embedding`), print -> parse being structurally idempotent, type spellings round-tripping,
the two target presets actually differing, and two planted-difference controls of its own.
Only the step that turns Python TEXT into an AST needs `ruff`, so the tests that start from
source are `#![cfg(feature = "ruff")]`.

## Status, per fixture configuration

Machine-checked by `tests/fixture_status.rs` — this table is generated from the same
expectations that test asserts, so it cannot drift silently.

| configuration | status | ops matched | target |
|---|---|---|---|
| `vector_add` | **MATCHES** | 23 | either |
| `mul` | **MATCHES** | 23 | either |
| `bias_add_f32` | refused, **and so is Triton's** | — | no golden exists; see below |
| `swiglu_mlp` | **MATCHES** | 91 | either |
| `swiglu_mlp_granite` | **MATCHES** | 91 | either |
| `swiglu_mlp_tiledk` | **MATCHES** | 91 | either |
| `embedding` | **MATCHES** | 35 | either |
| `embedding_granite` | **MATCHES** | 35 | either |
| `rmsnorm` | **MATCHES** | 61 | either |
| `rmsnorm_granite` | **MATCHES** | 61 | either |
| `rope` | **MATCHES** | 82 | either |
| `rope_granite_q` | **MATCHES** | 82 | either |
| `rope_granite_kv` | **MATCHES** | 82 | either |
| `decoder_layer` | **MATCHES** | 278 | either |
| `decoder_two_layers` | **MATCHES** | 320 | either |
| `attention_flash_noncausal` | **MATCHES** | 287 | `upstream_gpu` |
| `attention_flash_causal` | **MATCHES** | 426 | `upstream_gpu` |

**Every configuration the Python toolchain can compile now matches its golden**, field by
field: op name, result types **and shapes**, every attribute (constant values, compare
predicates, axis / range / divisibility attributes, `tt.call` callee symbols), operand
dataflow normalized to `arg N` / `op K result R`, block and region nesting, and the `loc`
names that carry the Python variable. Op-count histograms agree exactly.

### The indirect address: what `tt.descriptor_gather` cost to add

`embedding.py` is the first fixture whose ADDRESS IS DATA — a token id read from memory —
and it is the one item in this round that closed a gap rather than confirming one. Three
changes, all small, and one of them was invisible until the golden existed:

1. **`semantic::descriptor_gather`**, a port of `semantic.py:1143` including all five of its
   `assert`s. Each assert in the oracle is a REFUSAL, so each is reproduced and each names
   the rule it enforces; `tests/gather.rs` holds the four reachable ones against Triton's own
   measured messages, plus an accepting control so the refusals mean something.
2. **`gather` in the census's `VALUE_METHODS`**, and a `call_method` arm. Note the ORDER
   there: Triton validates the descriptor and the index vector before it turns `y_offset`
   into an IR value, so the constant a literal `0` produces lands immediately before the
   gather.
3. **`si32`.** A descriptor over an `!tt.ptr<i32>` prints its BLOCK element type with
   Triton's signedness — `!tt.tensordesc<64xsi32>` — while the same line prints the pointee
   signless as `<i32>`. Every earlier golden is f16, where signedness cannot show, so the
   reader had never met the spelling and the golden did not parse at all until it did.
   `ttir::desc_block_elem` is the one place it is printed and it records the measurement
   (`*u32 -> ui32`, `*i16 -> si16`, `*i8 -> si8`, `*i64 -> si64`, `*fp32 -> f32`; an i1
   descriptor Triton refuses outright, so no spelling for one exists).

### RMSNorm found a fail-closed gap that no fixture could reach

`rmsnorm.py` needed one new `tl.*` target (`rsqrt` → `math.rsqrt`, three lines of dispatch)
and nothing else — every other construct it uses was already in. But writing its f32 island
deliberately exposed something: **`semantic::math_unary` accepted any float**, while Triton
decorates `exp`, `exp2` AND `rsqrt` with `@_check_dtype(dtypes=["fp32", "fp64"])` and refuses
each of them on f16 (`ValueError: Expected dtype ['fp32', 'fp64'] but got fp16`).

So this front end would have compiled `tl.rsqrt(x_f16)` — and `tl.exp(x_f16)` — that the
oracle rejects. No fixture reaches it from the wrong side, because both existing users
(`swiglu_mlp.py`'s sigmoid, this kernel's reciprocal-sqrt) widen first; that is precisely why
the gap survived. The gate is now in `math_unary` with Triton's own wording, and
`tests/math_width.rs` is the control: three refusals plus the island form that must still
compile.

### RoPE needed NO new construct — and found the worst gap of the three

`rope.py` compiles with the constructs already in: four descriptor loads (two of them the
same descriptor at column offsets `0` and `HALF`), two stores, five multiplies. At head_dim
128 in f16 each half of `rotate_half` is **exactly one 64-lane stick**, so the rotation is a
whole-stick swap and there is nothing to permute.

Getting there meant measuring two forms rather than assuming:

| form | Triton | why not |
|---|---|---|
| `x1 = x[:, :HALF]` | **refuses** — `ValueError: unsupported tensor index` | `tensor.__getitem__` accepts only `None` and a full `:` |
| `tl.reshape` → `tl.permute` → `tl.split` | **compiles** (`tt.reshape`, `tt.trans {order = array<i32: 0, 2, 1>}`, two-result `tt.split`) | three constructs the bridge lacks, a rank-3 tile no fixture has, and a permutation ACROSS the lane axis, where the only attested lane permutations are the reduction fold's `NFWD0`/`NFWD2` within 8 |
| two descriptor loads at `0` and `HALF` | **compiles**, and is what the fixture uses | no new construct, no data movement |

**And the first row is where the bug was.** `codegen`'s slice arm returned `Val::Slice` for
ANY slice, and `subscript` treats `Val::Slice` as a no-op — so `x[:, :64]` COMPILED here and
yielded the whole 128-wide tile. A silently wrong answer, for a kernel the oracle refuses
outright, invisible because every earlier subscript in the tree is `[:, None]` or `[None, :]`.
The bounds are now checked and the refusal names the construct; `tests/slicing.rs` covers five
bounded spellings plus the two accepted forms, so a blanket refusal of subscripts would fail
it too.

### The fusion experiment, and the mangling bug it exposed

`decoder_block.py` holds a whole Granite decoder block as one `@triton.jit` function and two
kernels that call it — once, and twice. **The bridge accepts both**, field by field against
their raw goldens. Measured at `M = 64`, `D_MODEL = 128`, `D_FF = 256`:

| | one layer | two layers, fused |
|---|---|---|
| raw ops (`make_ir`) | 278 | 320 |
| kernel arguments after descriptor flattening | 14 | 23 |
| widest function boundary | 57 args | 57 args |
| ops after `make_ttir` (inliner, canonicalizer, CSE) | 134 | 242 |

Two layers is **+42 raw ops, not +278**, because the layer body is emitted ONCE and called
twice: both calls mangle to the same symbol, since the constexprs are identical and only the
descriptor values differ. So a raw-op count does not measure the work in a fused kernel — the
post-pass count does, and there the inliner duplicates the body. `tests/fusion.rs` asserts all
of it, including "one body, two calls", so the +42 cannot be read the wrong way.

**The bug it exposed was in float mangling.** `rms_norm_eps = 1e-05` is a constexpr, and a
float constexpr is mangled with Python's `repr` INTO A SYMBOL NAME. Rust's `Display` never
uses exponent notation, so we produced `c0.00001` where the oracle has `c1e-05` — one
formatting rule, reported as three findings (a missing function, an extra function, and a
`tt.call` callee mismatch). `mangle::py_float_repr` now reproduces CPython's `float_repr`,
including its fixed/exponent switch at `-4 < decpt <= 16`. Every float constexpr in the tree
until now (`1.0`, `0.22`, `0.0078125`) sat inside the fixed window, which is why nothing
caught it. The same name also has to be read back: MLIR **quotes** a symbol that is not a bare
identifier, and `c1e-05` contains a `-`, so `ttir::parse::unquote_symbol` and `print`'s
`symbol()` are the two halves of that.

### Why attention is diffed at `upstream_gpu`

The goldens come from the real Triton, which HAS the `f16 / f16 → f32` promotion this crate
deliberately turns off for Spyre. Fifteen configurations never reach a bare `/` on f16 —
`swiglu_mlp.py` calls `tl.fdiv` precisely to dodge it — so they match under **either** target,
which `every_matching_fixture_is_target_independent` asserts.

`attention_flash.py`'s epilogue `acc = acc / l_i[:, None]` is the one bare f16 divide in any
fixture. Diffed at `upstream_gpu` it matches exactly, which proves the **port** faithful.
`tests/divergence.rs::the_attention_epilogue_divide_is_the_whole_divergence` then asserts that
at `spyre` the difference is **exactly two fewer `arith.extf` and one fewer `arith.truncf`**,
with the `arith.divf` still present and every other op count unchanged — which proves the
**divergence** real and localized. Using one target for both jobs would have lost one of the
two facts. `swiglu_mlp.py`'s own delta 6 predicted this in prose; it is now measured.

## Three findings that correct the brief

**1. The oracle is `make_ir`, not `make_ttir`.** `triton.compile` reaches TTIR in two steps
and only the first is this bridge's job:

```
ASTSource.make_ir(...)   <- code_generator.py's AST walk.        BRIDGE ONE
backend.make_ttir(...)   <- inliner, canonicalizer, ttir-combine,
                            reorder-broadcast, cse, symbol-dce   (compiler.py:115)
```

The `.ttir.mlir` goldens already in the tree are POST-pass. Diffing a port of the AST walk
against them would demand it also reproduce MLIR canonicalization, constant CSE and function
inlining — none of which are AST lowering. `tools/gen_ttir_goldens.py` now writes both
checkpoints; `.ttir_raw.mlir` is the one the diff uses. The difference is not cosmetic:

| configuration | raw lines | post-pass lines |
|---|---|---|
| `vector_add` | 48 | 38 |
| `swiglu_mlp` | 166 | 96 |
| `embedding` | 59 | 43 |
| `rmsnorm` | 115 | 75 |
| `rope` | 125 | 72 |
| `decoder_layer` | 587 | 434 |
| `decoder_two_layers` | 658 | 659 |
| `attention_flash_noncausal` | 482 | 218 |
| `attention_flash_causal` | 647 | 319 |

**2. `bias_add_f32` is NOT usable for this bridge, and cannot be.** The brief said all five
fixtures have ttir goldens "even where they have no KTIR, since ttir is upstream of those
failures". That is true of `vector_add` and `mul`, whose failure is in `PlanCorelets`. It is
false of `bias_add_f32`, which dies **inside `make_ir`** — that is, inside bridge one:

```
NameError: Cannot access global variable BIAS from within @jit'ed function. Triton
kernels can only access global variables that are instanstiated as constexpr
(`x = triton.language.constexpr(42)`).
```

So there is no `bias_add_f32.ttir_raw.mlir` and there never will be one. This crate
reproduces the same refusal, quoting Triton's wording, and `tests/fixture_status.rs` asserts
that a golden does **not** exist for it — compiling a kernel Triton refuses would make this
front end more permissive than its own oracle, which is a silent-wrong-answer risk, not a
feature. `tools/gen_ttir_goldens.py` records it as an asserted XFAIL: if it ever starts
compiling, the generator says `SURPRISE` and exits nonzero.

**3. The census's headline number counts SPELLINGS, not implementations.** Re-run over all
eight fixtures it is **39 distinct call targets** across 9 jit functions (33 AST node types),
and that is 22 implementations: `a_desc.load` and `x_desc.load` are one function reached
through different receivers. Classified: 16 `tl.*` functions, 1 builtin (`float`), 1 user jit
function (`_attn_fwd_inner`), 1 method on an unnamed receiver (`to`), and 20
receiver-qualified methods. `src/py/census.rs` carries the table — and the arithmetic, because
the previous version of it summed to 32 while claiming 33.

## No text boundaries in the pipeline

The product is a **value**. Bridge two consumes `ttir::Module` directly. Text exists in
exactly two places and both are test instruments:

- `ttir::print` renders a module, for the golden diff and for reading by eye;
- `ttir::parse` reads TTIR text, for one purpose only: loading the goldens the Python
  toolchain emits so `diff` can compare against them.

**The instruments are held to each other, and doing that found five printer defects.**
`no_parser.rs::print_then_parse_round_trips_structurally` used to round-trip `vector_add`
only — an elementwise kernel with no calls, no loops, no regions and no second block. Pointed
at `embedding` and `decoder_layer` as well, it caught: `tt.call` printed with its callee in the
attribute dict (unreadable, so NO module with a user call round-tripped, attention included);
`scf.for` printed in the generic form, which loses the induction variable and the carries;
a function's SECOND block dropped entirely, taking the `ub.poison` and `tt.return` of every
generated helper with it; a region's entry-block label omitted, which lost `tt.reduce`'s
combiner arguments; an integer `axis` filtered out by key alongside `get_program_id`'s
`Attr::Axis`, so every `tt.expand_dims` and `tt.reduce` printed without one; `tt.trans`'s
`order` printed as `[1 : i32, 0 : i32]` instead of `array<i32: 1, 0>`, which reads back as a
string; and a six-digit float mantissa that does not round-trip an f32 `1e-05`. None of it
affects the golden diff — that compares values, not text — but all of it affects what a human
reads when a diff fails, which is the printer's entire job.

Neither is reachable from the lowering path. Printing a module so another stage can parse it
back would be the bug this note exists to prevent.

### The swappable boundary

The shared vocabulary between the two bridges is confined to **one module**: `src/ttir/mod.rs`
(types, attributes, ops, regions, a value arena — no builder logic, no semantics, no Python,
no target policy). Whether the KTIR types downstream become scratchy's vendored `ktir-core`
or stay ours is an open decision, so replacing this vocabulary must be a contained change.

What depends on it: `semantic` and `codegen` construct it; `ttir::print` / `ttir::parse`
render and read it (tests only); `diff` compares two of them; bridge two consumes it.
`src/py` never mentions a TTIR type and `src/ttir` never mentions a Python one — they meet
only in `src/codegen.rs`.

## The diff has controls

A structural diff that passes while comparing nothing is the failure mode that matters, so
`tests/golden_diff.rs` plants a difference of each kind and asserts it is caught **and
named**: op name, result element type, result shape at the same element type, attribute
value, compare predicate, operand swap (same ops and types, different dataflow), a removed
op, a `loc` name, a function name — plus a control that an unparsable golden **fails** rather
than passing, and a baseline control that the unmutated comparison is clean (without which
none of the others prove anything).

One of these found a real gap: the diff compared the op's `loc` name and only fell back to
the result's, so a wrong result name was invisible whenever the op's was right. It now
compares both.

## The one deliberate divergence from Triton's semantics

`computation_type_impl` promotes `f16 / f16 → f32`, and its stated reason is that `/` and `%`
"do not exist natively in PTX for fp16". **That is a PTX fact and it is false on this
device**: `OpFuncs::REALDIV` is bound by `broadcast_ops.ddl` and `arith.divf` already maps to
`"realdiv"` in `../triton-superdsc/triton-superdsc-lower/src/opmap.rs`. Ported
target-conditional, off for Spyre, documented at the switch in
`src/target.rs::Target::div_promotes_narrow_floats`.

`tl.fdiv` being the same `create_fdiv` with `arithmetic_check` off is the evidence that this
is front-end policy and not a semantic necessity.

**No fixture covers it**, and that is not an oversight — the fixtures were written to work
around the promotion, so `swiglu_mlp.py` calls `tl.fdiv` and never a bare `/` on f16. The
goldens therefore contain no evidence either way, and `tests/divergence.rs` exercises the
switch directly against a kernel written for the purpose, asserting **both** directions
(Spyre keeps f16; upstream widens both operands and yields f32) plus that the switch does not
leak into multiplication.

## Why `ruff_python_parser`, and why behind a feature

Gated because this workspace's other crates are dependency-free so `cargo test --offline`
works on the pod, which has no crates.io access: `ruff_python_parser` pulls 84 packages
(measured — `cargo add ruff_python_parser ruff_python_ast` then `cargo fetch` reports
"Locking 84 packages"). `--no-default-features` builds and tests clean with zero
dependencies — `tests/no_parser.rs` is what it runs.

Used at all because Python's grammar is not the part of this job worth owning. This reverses
a stalled earlier skeleton's decision to hand-roll a ~1500-line parser. The scope control
that matters is not at the parse step — it is `src/py/census.rs`, which refuses out-of-census
constructs **by name at their source line**. Parsing generously and refusing precisely beats
parsing narrowly and failing vaguely, because the refusal can name the construct *and* say
where it is.

Only `src/py/ruff_adapter.rs` ever sees a ruff type.

### Two refusal layers, deliberately ordered

1. `py::ruff_adapter` refuses Python with no place in our AST (`while`, `lambda`, a
   comprehension), naming the CPython node type. It returns on the first one — there is
   nothing to build.
2. `py::census::check` then walks the built AST and reports **all** remaining violations at
   once, so one run names all the work.

## Faithfulness notes worth keeping

The port reproduces Triton's **dead ops in place**, because the raw oracle contains them and
an op-count diff would otherwise fail. `pid * BLOCK` emits twelve ops:

- `binary_op_type_checking_impl` materializes the scalar `64` as a constant for type
  inspection, then `scalar_constant` materializes it **again** at the promoted type, leaving
  the first dead;
- `binary_op_sanitize_overflow_impl` emits two `extsi` to i64, the widened `muli`, two bound
  constants, two `cmpi` and an `andi` **before** the multiply it guards;
- `device_assert` emits nothing, because it returns early unless `options.debug` is on
  (`semantic.py:1810`) — which is why that `andi` is dead.

Our SSA names come out identical to the golden's, down to MLIR's `_0`…`_12` disambiguation
counter, though the diff does not depend on that.

Locations: MLIR columns are **1-based** where CPython's `col_offset` is 0-based, so a loc is
`(node's file line, col_offset + 1)`. Verified against the golden rather than assumed —
`tl.program_id(0)` sits at 0-based column 10 of `vector_add.py` line 47 and the golden says
`47:11`.

## What is left

Nothing on the fixtures. Every construct they reach is lowered and every golden they have
matches. What is refused, and refused BY NAME rather than half-built, is what no fixture
exercises:

- a `for` loop with a **negative** step (Triton flips the bounds and recomputes the induction
  variable inside the body);
- a `for` loop with a `tl.range` **scheduling keyword** (`num_stages`, `flatten`, …) — these
  become `scf.for` attributes, and silently dropping one would change performance invisibly;
- an `if` on a **runtime** value (an `scf.if`, or `visit_if_top_level`'s block split when the
  branch contains a `return`);
- **more than one `return`** in a function (`decide_return_type` reduces the return types to a
  common type across them);
- a descriptor or tuple **carried through a loop** (its handles would have to be flattened into
  `iter_args` and reassembled from the block arguments);
- `tl.max(..., return_indices=True)`, `keep_dims=True`, `tl.sum(dtype=…)`, `axis=None`;
- `tl.dot` on **integer** operands (a different zero and an i32 result);
- a **bounded tensor slice** (`x[:, :64]`) — refused, because TRITON refuses it; only `None`
  and a full `:` are subscript items there. This one used to compile into the whole tile, so it
  is on this list as a FIXED BUG rather than as unbuilt work (`tests/slicing.rs`);
- `tl.exp` / `tl.math.exp2` / `tl.rsqrt` on **f16** — refused, again because Triton's
  `@_check_dtype` refuses it; the widen / call / truncate island is the only spelling
  (`tests/math_width.rs`);
- `tl.reshape`, `tl.permute` and `tl.split` — the other expressible form of `rotate_half`, and
  the one a MULTI-HEAD fused block would need for the reverse operation. There is **no
  concatenate** here either, which is why `decoder_block.py` is one head: assembling per-head
  attention outputs into one `[M, D_MODEL]` tile has no spelling yet;
- everything outside the census: any other `tl.*` target, `while`, `lambda`, comprehensions,
  `*args` / `**kwargs`, chained assignment, multiple comparison.

## Extending it

Add the fixture, then run `tools/gen_ttir_goldens.py` to produce its `.ttir_raw.mlir`, then add
it to `tests/fixture_status.rs`. If it needs a construct from the refused list, the compile
error names the construct and its source line — start there. `tests/no_parser.rs` asserts that
every golden in the tree parses, so a new one that the reader cannot handle fails loudly rather
than quietly comparing less.
