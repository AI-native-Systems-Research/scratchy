<!-- SPDX-License-Identifier: Apache-2.0 -->
# The chat format DSL — specification

This specifies a small language for declaring a model's **prompt format**, and the
compiler that turns one declaration into a Rust renderer at build time. The request
path then carries no template and no template engine.

An earlier approach compiled each model's HuggingFace Jinja template directly. That was
measured and rejected: it covered 5 of 33 fleet templates, its complexity grew with
every model generation (368 B for the smallest template, 18,924 B for the largest), and
the newest models increasingly ship no template at all to compile. The reframe that
replaced it: **the complexity lives in the Jinja encoding, not in the prompt format.**
`namespace()` exists because Jinja variables don't survive a loop iteration;
`.split('</think>')` because it has no structured data. Every model author invents a
different workaround for the same limitations, which is why templates look so unalike
while producing such similar output.

Everything below derives from a survey of **23 upstream templates** drawn from the
architectures in `crates/models/arch/configs/` — 64,648 B of Jinja, rendered through real
minijinja and classified by **output**. Where this spec claims something about what
models do, it is a measurement, and the measurement is named.

Precisely what that sample covers, since template count and architecture count are not
the same number: 23 templates, of which **21 are distinct by content** (two vision
variants share a template byte-for-byte, as do two MoE variants). They cover **17 of the
25 architecture families** directly; 7 more are accounted for by sharing a template
lineage with one of those or by not being chat models at all; and **Command-R is
unreachable** — its template is gated on every mirror tried, which is the one real hole
(§11). `cargo run -p scratchy-chat-template-survey -- coverage` reconciles this against
the live `configs/` tree, so the claim cannot quietly rot as architectures are added.

---

## 1. The model

For 20 of the 23 surveyed templates the rendered prompt is

```
render(msgs, gp) = head(role₀)
                 + content₀
                 + sep(role₀, role₁) + content₁
                 + …
                 + sep(role_{n-2}, role_{n-1}) + content_{n-1}
                 + tail(role_{n-1}, gp)
```

where `sep` depends on the two roles **alone** — not on position, not on conversation
length, not on any other message. For 21 of 23 it further factors as
`sep[a → b] == close[a] + open[b]`.

So the whole language rests on four declared quantities:

| quantity | definition |
|---|---|
| `open[role]` | bytes before a message's content |
| `close[role]` | bytes after a message's content |
| `prelude` | bytes before the first message's `open` |
| `genprompt` | bytes after the last `close`, when `add_generation_prompt` is set |

and the renderer is a flat run of `push_str`:

```
prelude? + Σᵢ ( open[roleᵢ] + contentᵢ + close[roleᵢ] ) + genprompt?
```

Three refinements cover the rest of the sample; §4 specifies each.

- A **pair override** `sep[a → b]`, which wins over `close[a] + open[b]`. Needed by the
  two templates that coalesce consecutive same-role turns.
- A **system attachment** rule, for formats where the system block rides on the last
  user turn rather than being wrapped in place.
- A **reasoning** rule, for formats where the trailing assistant turn carries a
  `<think>` block and earlier ones are stripped.

### 1.1 Why `genprompt` is its own quantity

It is not "re-emit `open[assistant]`", because the two measurably differ. deepseek-v2
opens a *history* assistant turn with `"Assistant: "` and emits `"Assistant:"` as the
generation prompt — no trailing space. A language that reused one string for both would
be wrong by one byte, in the position that decides the model's first generated token.

---

## 2. Grammar

A declaration is a UTF-8 text file, **line-oriented**, one directive per line. There are
no expressions, no conditionals and no loops: anything a template would compute, this
language declares.

```ebnf
file        = { line } ;
line        = [ directive ] [ comment ] newline ;
comment     = "#" { any-char-but-newline } ;
directive   = use | open | close | sep | prelude | genprompt
            | tools | toolresult | reasoning | system-attach
            | reject | validated | content-parts ;

use         = "use" pattern-name ;
open        = "open"  role-sel string ;
close       = "close" role-sel string ;
sep         = "sep"   role "->" role string ;
prelude     = "prelude" { prelude-part } ;
prelude-part= "bos" | "system_default" string ;
genprompt   = "genprompt" string ;

tools       = "tools" placement serializer ;
placement   = "head" | "last_user" ;

toolresult  = "toolresult" ( "role" role
                           | "wrap" role string string
                           | "reject" string ) ;

reasoning   = "reasoning" "trailing_assistant" string string
              { reasoning-opt } ;
reasoning-opt = "skip_turns" "matching" string ;

system-attach = "system" "attach_last_user" "join" string
                "on_missing" ( "drop" | "reject" string ) ;

reject      = "reject" condition string ;
condition   = "system" | "system_not_first" | "non_alternating"
            | "empty_messages" | "tool_role" ;

content-parts = "content_parts" ( "reject" string
                                 | "join" string { part-rule } ) ;
part-rule   = "part" part-kind string ;
part-kind   = "text" | "image" ;

validated   = "validated_against" string ;

role        = "system" | "user" | "assistant" | "tool" ;
role-sel    = role | "*" ;
string      = '"' { char | escape } '"' ;
escape      = "\\" ( "n" | "t" | "r" | "\\" | '"' | "u{" hex+ "}" ) ;
```

Indentation is insignificant except that `reasoning-opt` lines must follow their
`reasoning` directive. Blank lines and comments are ignored.

### 2.1 Closed vocabulary

An unrecognised directive, role, placement, serializer, condition or escape is a
**build error**. There is no pass-through, no `raw` directive and no escape hatch.

That property is the reason this language is worth having, and it is load-bearing rather
than aspirational: the survey exists to establish that prompt formats *can* be said in a
closed vocabulary. A declaration either says something the compiler understands exactly,
or the build stops.

### 2.2 Precedence

Directives apply in file order, and a later directive **overrides** an earlier one for
the same key. `use` is therefore just an expansion a model can patch:

```
use chatml                              # sets open/close/genprompt for all roles
open assistant "<start_of_turn>model\n" # …then override one of them
```

This is a single mechanism, not two. There is deliberately no "rename role" directive —
a role whose wire name differs from its emitted tag (`assistant` → `model`) is expressed
by overriding that role's `open` string, which is what the measured table contains
anyway.

`sep` outranks `close`/`open`: where `sep a -> b` is declared, those exact bytes are
emitted between an `a` turn and a `b` turn, and the participating `close[a]` / `open[b]`
are not.

### 2.3 Patterns

`use <pattern>` expands to a fixed set of directives. Patterns are **compiler built-ins,
not files** — they are shared vocabulary, so they live in the compiler where they are
type-checked and tested once rather than copied into every declaration.

| pattern | expansion |
|---|---|
| `chatml` | `open <r> "<\|im_start\|>{r}\n"`, `close * "<\|im_end\|>\n"`, `genprompt "<\|im_start\|>assistant\n"` |
| `role_tags` | `open <r> "<\|{r}\|>\n"`, `close * "<\|end\|>\n"`, `genprompt "<\|assistant\|>\n"` |

8 of the 23 surveyed templates are ChatML and 3 are per-role tags, so these two cover
half the sample. More may be added as models land; the bar proposed in §11 is two or
more models sharing an expansion *exactly*.

> **A measured constraint on patterns.** Within one model family, `qwen3` strips
> reasoning from history while `qwen3-moe` and `qwen3-next` keep it — adjacent
> generations, materially different semantics. Patterns therefore capture *shared
> grammar*, never *model lineage*, and `use` must never be keyed on an architecture
> name.

---

## 3. Worked derivations

The test of the model is whether it reproduces the measured tables. Three, spanning the
easy case, the verbose case and the hard case.

### 3.1 smollm2 — 368 B of Jinja

```
use chatml
prelude system_default "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
```

Derivation against the measured table:

| measured | derived |
|---|---|
| `head[user]` = `"<\|im_start\|>system\nYou are …<\|im_end\|>\n<\|im_start\|>user\n"` | `prelude + open[user]` ✓ |
| `head[system]` = `"<\|im_start\|>system\n"` | `open[system]`, prelude suppressed by an explicit system message ✓ |
| `sep[user→assistant]` = `"<\|im_end\|>\n<\|im_start\|>assistant\n"` | `close[user] + open[assistant]` ✓ |
| `tail[user, gp]` = `"<\|im_end\|>\n<\|im_start\|>assistant\n"` | `close[user] + genprompt` ✓ |
| `tail[user, !gp]` = `"<\|im_end\|>\n"` | `close[user]` ✓ |

### 3.2 granite-3.3 — 4,566 B of Jinja

```
open  user      "<|start_of_role|>user<|end_of_role|>"
open  assistant "<|start_of_role|>assistant<|end_of_role|>"
open  system    "<|start_of_role|>system<|end_of_role|>"
open  tool      "<|start_of_role|>tool<|end_of_role|>"
close *         "<|end_of_text|>\n"
genprompt       "<|start_of_role|>assistant<|end_of_role|>"

prelude system_default "Knowledge Cutoff Date: April 2024.\nToday's Date: {date}.\nYou are Granite, developed by IBM. You are a helpful AI assistant."
tools      head granite_json
toolresult role tool
reject     empty_messages "Messages list cannot be empty"
```

### 3.3 gemma-4 — 18,924 B of Jinja

The hardest template in the sample, and the one whose output grammar turns out to be
among the flattest:

```
open  user      "<|turn>user\n"
open  assistant "<|turn>model\n"
open  system    "<|turn>system\n"
close *         "<turn|>\n"
genprompt       "<|turn>model\n<|channel>thought\n<channel|>"

prelude bos
sep assistant -> assistant ""      # consecutive assistant turns coalesce
sep user      -> user      ""
tools head gemma_json
content_parts join "" part image "<start_of_image>" part text "{text}"
```

**18,924 B of Jinja, nine declared lines.** Its 18 KB buys nothing that appears in the
output — which is the central claim of this design, stated as a ratio.

---

## 4. The three refinements

### 4.1 Pair override — `sep`

```
sep assistant -> assistant ""
```

Emits exactly those bytes between an `assistant` turn and the next `assistant` turn,
suppressing `close[assistant]` and `open[assistant]`. An empty string therefore
concatenates the two contents into a single turn — measured behaviour for two of the
surveyed formats when given two same-role messages in a row.

That input is one a careless reader would call degenerate, and a real client can send
it. Declaring the behaviour is mandatory, not optional: an undeclared same-role pair
falls back to `close + open`, which is a different prompt.

### 4.2 System attachment

```
system attach_last_user join "\n\n" on_missing drop
```

The system message is not wrapped in place. Its content is prefixed to the **last user
turn's** content, joined by `join`. Measured: `"[INST] <system>\n\n<user>[/INST]"`.

`on_missing` is **required**, and it is where this spec takes its firmest position.
When there is no user turn to attach to, the upstream template silently drops the system
prompt entirely — measured, for three of the role sequences probed. A declaration must
therefore say out loud which it wants:

- `on_missing drop` — reproduce upstream exactly, data loss included.
- `on_missing reject "<message>"` — refuse the request instead.

There is no default. A declaration that omits `on_missing` fails to compile, naming the
measured behaviour.

Byte-exact parity with upstream is the right default — silent prompt drift is
undetectable downstream, and the whole test strategy (§8) rests on it. But "byte-exact
with a known-broken template" and "a correct prompt" are different goals, and this is
the one place in the format vocabulary where they visibly conflict. Making the loss
*declared* keeps parity as the tested property while making it impossible to ship the
loss unknowingly. It is mechanical rather than left to reviewer vigilance.

### 4.3 Reasoning

```
reasoning trailing_assistant "<think>\n" "\n</think>\n\n"
  skip_turns matching "<tool_response>"
```

Semantics, exactly as measured:

1. Find the index of the last `user` turn whose content does **not** both start and end
   with the `skip_turns matching` string. Call it `L`.
2. An `assistant` turn at index > `L` is a **trailing** turn. Its reasoning text is
   emitted wrapped in the two delimiters, before its content — and emitted *even when
   the reasoning is empty*, producing `"<think>\n\n</think>\n\n"`.
3. An `assistant` turn at index ≤ `L` has its reasoning **stripped**: the delimiters and
   the text between them are removed and not emitted.

Reasoning text comes from the typed context's `reasoning` field when set, otherwise from
the content between the two delimiters.

Three things make this worth declaring rather than hand-writing. Step 1 is a reverse
scan over the whole list before any byte is emitted, so the rule is genuinely two-pass.
Step 1 keys on a **content pattern**, not on a role. And step 3 deletes content the
client sent — the single case in the survey where a format's output is not a function of
each message locally. A hand translation of one template's rules scored 0 of 6 against a
mechanical translation's 8 of 8, and this is exactly the shape of rule a human reader
drops.

---

## 5. The typed context

The generated renderer takes a **typed struct**, not a JSON value. No
`serde_json::Value` in the signature, no dynamic dispatch, no string-keyed lookup at
request time. Every byte emitted is either a declared literal or the output of a named
serializer, so there is no need for a dynamic value layer at all.

```rust
pub struct ChatRequest<'a> {
    pub messages: &'a [Message<'a>],
    pub tools: &'a [Tool<'a>],
    pub add_generation_prompt: bool,
    pub bos_token: &'a str,
    pub eos_token: &'a str,
    /// Pre-formatted date for `{date}` in a prelude. The caller owns the clock;
    /// the renderer stays a pure function of its input.
    pub date: &'a str,
}

pub struct Message<'a> {
    pub role: Role,
    pub content: Content<'a>,
    pub tool_calls: &'a [ToolCall<'a>],
    pub reasoning: Option<&'a str>,
}

/// Closed. An unknown wire role is a request-validation error, not a render error.
pub enum Role { System, User, Assistant, Tool }

pub enum Content<'a> {
    Text(&'a str),
    Parts(&'a [Part<'a>]),
}

pub enum Part<'a> { Text(&'a str), Image }

pub struct ToolCall<'a> {
    pub id: &'a str,
    pub name: &'a str,
    /// Verbatim as received. §6.2 explains why this is not parsed.
    pub arguments: &'a str,
}
```

### 5.1 This widens the serving-side message type

`crates/serving/api/src/chat_template.rs` currently renders from
`TemplateMessage { role: String, content: String }`, which cannot express a tool call, a
reasoning block or multi-part content. Any model needing §4.3, §6.2 or `content_parts`
is unrepresentable until that type is widened to the shape above, so it is a prerequisite
rather than a later refinement.

> Separately: `/tokenize` and `/chat/completions` build their message lists by different
> paths, which produces divergent prompts between the two endpoints independently of
> which renderer runs. That is a pre-existing defect, not introduced by this design, but
> it should be fixed before parity measurements from either endpoint are trusted.

---

## 6. Serializers and the parity traps

### 6.1 `tojson` is not `serde_json`

Tool declarations are serialized by a named built-in serializer. Each must reproduce
Jinja's `tojson` **byte-exactly**, which no standard JSON writer does:

| char | `tojson` | `serde_json` |
|---|---|---|
| `<` | `<` | `<` |
| `>` | `>` | `>` |
| `&` | `&` | `&` |
| `'` | `'` | `'` |

`"` `/` `\` `\n` `\t` `\u{7f}` `\u{2028}` and non-ASCII all agree. The divergence being
*narrow* is what makes it easy to miss — `serde_json::to_string` looks obviously correct
and is wrong for any tool description containing markup or an apostrophe, which is
entirely ordinary.

Two further measured facts the serializers must honour:

- **`indent=4` is not `to_string_pretty`.** minijinja indents 4 spaces, serde_json 2.
  One surveyed template needs the pretty form on one line and the compact form on
  another, so a single model requires both modes.
- **`tojson` sorts object keys**, even for an insertion-ordered map. `serde_json` agrees
  only because `preserve_order` is off. Nothing in the workspace enables it today — but
  Cargo features unify across the whole dependency graph, so one future transitive
  dependency turning it on would silently change tool bytes. The serializer must
  therefore sort keys **itself** rather than inherit ordering, and a test must pin that.
  A comment is not protection against a feature unified in from elsewhere.

Named serializers, each the exact shape one or more measured models emit: `granite_json`
(pretty, indent 4), `qwen_xml`, `llama_json`, `mistral_json`, `gemma_json`.

### 6.2 Tool-call arguments pass through verbatim

Given `arguments` as a JSON *string* — what real clients send — one surveyed model emits
it `tojson`-escaped while the next generation of the same family emits it raw. Same
input, same family, different bytes. So `ToolCall::arguments` is `&str`, carried
verbatim, and whether it is re-escaped is the serializer's decision, declared per model.
Parsing arguments into a value and re-serializing would silently normalise them and
erase the distinction.

### 6.3 The risk is concentrated by construction

Because there is no generic "render this value" operation, a value-formatting difference
cannot appear in arbitrary positions — only inside the named serializers of §6.1. A
concentrated risk is testable exhaustively, and that is the main safety argument for
declaring formats rather than interpreting templates.

---

## 7. File layout and discovery

```
crates/models/arch/configs/<arch>/<stem>.chat         # the declaration
crates/models/arch/configs/<arch>/<stem>.chat.jinja   # upstream Jinja, TEST FIXTURE ONLY
```

**Per model, not per arch**, and not as a style choice: the single `llama` arch holds
smollm2 (368 B), tinyllama (410 B) and llama-3.2 (3,827 B), whose formats are unrelated.
The survey added a second, independent reason — formats diverge *within* a family across
adjacent generations (§2.3).

Discovery reuses the existing mechanism in
`crates/models/arch/scratchy-forwards.rs::emit_chat_templates`, scanning `*.chat`
instead of `*.chat.jinja`. Its established behaviour is kept: gate on the same per-stem
feature `config.rs` uses (`CARGO_FEATURE_<STEM>`), emit one module per model into
`$OUT_DIR`, register it with one `inventory::submit!`.

The `.chat.jinja` file stays checked in, no longer compiled, as the oracle's input (§8).
That is the only reason to keep it.

### 7.1 A model with no declaration

While the template interpreter is still present, a model with no `.chat` falls back to
it. A CI step lists which enabled models lack a declaration, so the remaining gap stays
visible rather than being discovered at the end.

Once the interpreter is removed, a missing declaration becomes a build error, matching
how an unsupported architecture already fails: a `model/<stem>` feature then requires
its format declaration. Making it an error earlier would block every model not yet
migrated, so the ordering matters.

---

## 8. Proving it: the differential oracle

`minijinja` is a `[dev-dependency]` only. Every declaration is proved by rendering the
model's **real upstream Jinja** and diffing byte-for-byte against the generated
renderer.

The oracle's environment must match the production interpreter exactly — `trim_blocks`,
`lstrip_blocks`, pycompat, `raise_exception`, `strftime_now` on a fixed clock — or it is
testing a different language than the one that shipped.

Two requirements, both for measured reasons:

**Generated message shapes, not a fixed matrix.** When both sides derive from the same
Jinja AST, equivalence is mechanical and any divergence is a codegen bug. Here a human
reads upstream Jinja and re-expresses it as a declaration — the step that scored 0 of 6
in the one case it was measured, against 8 of 8 for mechanical translation. A fixed
matrix cannot contain the edge case the human missed. The generator must cover: every
role sequence up to length 5 including repeats, system in first / middle / last position,
empty content, multi-part content, tool calls with and without content, consecutive tool
results, and inline `<think>` in both leading and trailing assistant turns.

**Errors must match too.** 5 of the 23 surveyed templates raise on an ordinary tool
result; one raises on any system message at all; one rejects a tool-call ID that is not
9 alphanumeric characters. A renderer that *accepts* input the real template rejects is
a parity failure in the direction nobody tests for. The oracle asserts `Ok ⟺ Ok` and
`Err ⟺ Err`; §11 raises how closely the messages themselves must agree.

### 8.1 Drift detection

A declaration is not derived from the upstream template, so it cannot be checked against
it by byte equality. Instead a declaration records the upstream hash it was authored
against:

```
validated_against "sha256:9f2c…"
```

compared at startup against the resolved template, and **warning** rather than failing —
a refactored template can produce byte-identical output, and failing on that would be
worse than the drift it reports. The differential test pins the real guarantee at test
time, where a mismatch can be investigated rather than merely survived.

`CompiledTemplate.source` carries the declaration's own text, keeping its diagnostic
value: a log line can say exactly which format was used.

---

## 9. Error messages

Compile errors are build failures carrying file, line and column. They are part of this
spec because they are the entire interface when a declaration is wrong.

An unsupported or unrecognised construct **fails the build; it never warns and skips.**
Skipping would silently fall back to the interpreter and make the declared path look as
though it worked.

```
configs/llama/smollm2-135m.chat:3:1
  unknown directive `genprmopt`
  did you mean `genprompt`?

configs/mistral/mistral-7b.chat:8:1
  `system attach_last_user` requires `on_missing`
  the upstream template DROPS the system prompt when no user turn follows it.
  write `on_missing drop` to reproduce that, or
        on_missing reject "..." to refuse the request instead.

configs/granite/granite-3.3-8b.chat:5:7
  unknown serializer `granite_pretty`
  known: granite_json, qwen_xml, llama_json, mistral_json, gemma_json

configs/qwen3/qwen3-8b.chat
  no `close` for role `tool`, which `toolresult role tool` requires
```

Runtime errors use the existing `TemplateError`: `Raised` for a declared `reject`,
`BadValue` for a request the format cannot represent.

---

## 10. Leaf-crate discipline

`scratchy-chat-template-compiler` keeps the constraints that let it be published
standalone:

- no `scratchy-*` dependency;
- no `inventory` in its source — it emits a plain `CompiledTemplate` and stops;
- builds standalone via `--manifest-path`;
- renaming the crate is a one-line change.

The DSL parser, the patterns and the emitter are the crate's own business. Discovery,
feature gating and registration stay on scratchy's side.

### 10.1 A consequence worth naming

Because every `open` / `close` / `genprompt` / `prelude` string is a compile-time
constant, its **token IDs are compile-time constants too**. Pre-tokenizing the literal
chunks at build time and splicing token IDs — rather than rendering a string and
re-tokenizing it per request — becomes reachable. Nothing in this spec depends on that,
and it is not specified here; it is noted because the typed context of §5 is what makes
it possible, and a future change should not have to rediscover why.

---

## 11. Open design questions

1. **§4.2's mandatory `on_missing`** is the proposed resolution of reproducing upstream
   data loss: make the loss declared rather than silent. It is the one place this spec
   deliberately makes a declaration longer than upstream behaviour strictly requires.
2. **§8.1's warn-rather-than-fail** drift policy.
3. **How closely must error *messages* agree?** §8 requires `Err ⟺ Err`. Requiring
   byte-identical messages is stricter than any model needs and couples us to upstream
   wording; requiring only "both failed" could mask a rejection for the wrong reason.
4. **Command-R is unmeasured** — its template is gated on every mirror tried, making it
   the only one of 25 architecture families the survey could not reach. It is an older
   architecture with an unusual tool format, so it should not be assumed to land in the
   majority bucket. Either the template is obtained through an authenticated fetch, or
   the architecture is declared out of scope. Deciding after the grammar is fixed is the
   expensive order.
5. **The bar for adding a pattern** (§2.3): proposed as two or more models sharing an
   expansion *exactly*. A pattern that merely *nearly* fits is how a closed vocabulary
   starts leaking.
