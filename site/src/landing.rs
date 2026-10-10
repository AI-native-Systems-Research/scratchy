//! index.html, the landing page.

use dioxus::prelude::*;

use crate::carbon::{
    Button, ButtonKind, ClickableTile, CodeSnippet, Column, Grid, Gutter, Heading, Link,
    Orientation, Page, Section, SideNav, Span, Stack, Tile,
};
use crate::chrome::{self, Content, Head, OpenGraph, REPO, Root, Theme};
use crate::{highlight, repo};

/// The opening of crates/models/arch/dsl/llama.py, as the landing page quotes it.
const LLAMA: &str = "@forward
def llama():
    hidden_states = embed(input_ids, embed_tokens)
    for layer in range(num_hidden_layers):
        normed = rmsnorm(hidden_states, input_layernorm[layer])
        q = gemm(normed, self_attn.q_proj[layer])
        k = gemm(normed, self_attn.k_proj[layer])
        v = gemm(normed, self_attn.v_proj[layer])
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
        attn = attention(q, k, v, kv_cache[layer], block_table)
        oproj = gemm(attn, self_attn.o_proj[layer])
        hidden_states = add(oproj, hidden_states)

        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer])
        gate = silu(gemm(normed2, mlp.gate_proj[layer]))
        up = gemm(normed2, mlp.up_proj[layer])
        # ...";

/// A page section: its heading, an optional line under it, then its content.
fn section(heading: &str, sub: Option<Element>, body: Element) -> Element {
    rsx! {
        Section { level: 2,
            Stack { gap: 5,
                Heading { "{heading}" }
                if let Some(sub) = sub { p { {sub} } }
                {body}
            }
        }
    }
}

/// One card of the "what you write" triple: its step, a heading, a line of
/// explanation, and a link to the real file.
fn ingredient(n: u8, heading: &str, link: &str, name: &str, body: Element) -> Element {
    rsx! {
        Column { span: Span::THIRD,
            Tile {
                Section { level: 4,
                    Stack { gap: 3,
                        Heading { "{n}. {heading}" }
                        p { {body} }
                        Link { href: "{link}", code { "{name}" } }
                    }
                }
            }
        }
    }
}

fn number(value: &str, unit: &str, label: &str, body: Element) -> Element {
    rsx! {
        Column { span: Span::THIRD,
            Tile {
                Section { level: 3,
                    Stack { gap: 3,
                        Heading { "{value}\u{202f}{unit}" }
                        strong { "{label}" }
                        p { {body} }
                    }
                }
            }
        }
    }
}

fn doc(href: &str, heading: &str, body: &str) -> Element {
    rsx! {
        Column { span: Span::QUARTER,
            ClickableTile { href,
                Section { level: 4,
                    Stack { gap: 3,
                        Heading { "{heading}" }
                        p { "{body}" }
                    }
                }
            }
        }
    }
}

fn target(flag: &str, what: &str) -> Element {
    rsx! {
        Column { span: Span::THIRD,
            Tile {
                Stack { gap: 2,
                    strong { code { "{flag}" } }
                    p { "{what}" }
                }
            }
        }
    }
}

pub fn page() -> Result<String, String> {
    let body = rsx! {
        {chrome::header(Root(0), None)}
        // No side nav of its own: only the header's links, opened from the
        // header's menu button on small screens.
        SideNav { label: "scratchy", top: chrome::top_links(Root(0), None), menu_only: true }

        Content {
        Page {
          Stack { gap: 11,
            // The hero: no section around it, so its heading is the page's h1.
            Stack { gap: 7,
                Heading { "A hyper-specializing" br {} "inference stack compiler" }
                p {
                    "Fifty years of software reuse let the compiler delete and specialize, but never "
                    em { "restructure" }
                    ". AI knocks that wall down — and once ideas are the unit of reuse, every project gets to be bespoke. "
                    strong { "Scratchy is what that looks like for inference." }
                }
                Grid { gutter: Gutter::Cards,
                    {ingredient(1, "A forward DSL",
                        repo!("/blob/main/crates/models/arch/dsl/gemma4-moe.py#L9"), "gemma4-moe",
                        rsx! { "The entire forward pass of an architecture, written as the math." })}
                    {ingredient(2, "A model config",
                        repo!("/blob/main/crates/models/arch/configs/gemma4-moe/gemma-4-26b-a4b-it.json"), "gemma-4-26b-a4b-it",
                        rsx! { "The stock " code { "config.json" } " for one instance of that architecture." })}
                    {ingredient(3, "A quantization",
                        repo!("/blob/main/crates/models/quantization/presets/fp8-dynamic-per-channel.json"), "fp8-dynamic-per-channel",
                        rsx! { "One preset, which " em { "replaces" } " the dense emission rather than adding to it." })}
                }
                Stack { gap: 4, orientation: Orientation::Horizontal,
                    Button { kind: ButtonKind::Primary, href: "book/BUILD.html", "Build it" }
                    Button { kind: ButtonKind::Tertiary, href: "book/COMPILER.html", "How it works" }
                }
            }

            {section("Blogs", Some(rsx! { "Longer-form writing on why the stack is shaped the way it is." }), rsx! {
                Grid { gutter: Gutter::Cards,
                    Column { span: Span::HALF,
                        ClickableTile { href: "book/blogs/REUSE.html",
                            Section { level: 4,
                                Stack { gap: 3,
                                    Heading { "What “Reuse” Means in the Age of AI" }
                                    p {
                                        "Shared libraries, source modules, bundlers, AoT-compiled crates — every era let \
                                         the toolchain specialize a little more, and every one of them stopped at the \
                                         same wall. What happens when the unit of reuse becomes an idea, and what that \
                                         bought us on IBM Spyre."
                                    }
                                }
                            }
                        }
                    }
                    Column { span: Span::HALF,
                        ClickableTile { href: "book/blogs/EVOLVE.html",
                            Section { level: 4,
                                Stack { gap: 3,
                                    Heading { "Systems Must Evolve to be Compilers" }
                                    p {
                                        "Here is the hot take: AI coding agents will eliminate the software system as a \
                                         concept. In its place every solution will be bespoke, span the full stack, and \
                                         will be generated from scratch."
                                    }
                                }
                            }
                        }
                    }
                }
            })}

            {section("Key numbers", None, rsx! {
                Grid { gutter: Gutter::Cards,
                    {number("30", "Mi", "AoT binary, Metal", rsx! { "100\u{202f}Mi for Spyre, 250\u{202f}Mi for CUDA." })}
                    {number("330", "Mi", "Docker image, Spyre", rsx! { "The runtime is the binary. There is no graph executor to ship." })}
                    {number("300", "ms", "Warm start, Apple Silicon",
                        rsx! { em { "Independent of model size." } " 12\u{202f}s for 8B on Spyre." })}
                }
            })}

            {section("Write the math once", None, rsx! {
                Grid { gutter: Gutter::Cards,
                    Column { span: Span::NARROW,
                        Stack { gap: 5,
                            p {
                                "Most inference engines hand-write a Python/CUDA module per architecture, then rely on \
                                 a runtime graph executor to pick kernels. Scratchy takes the opposite approach: you \
                                 write the forward declaratively, and the compiler emits the whole pass ahead of time."
                            }
                            p {
                                "Rust's Turing-complete procedural macros run the entire pipeline at expansion — op \
                                 tape, reroll, layer classes, slot coloring, lifetimes, barriers, arena sizes, kernel \
                                 selection — and emit static tapes of each target's own types."
                            }
                            p {
                                strong { "Everything is a constant." }
                                " That turns complex analysis into arithmetic, and hands the optimizer enough \
                                 constants to cut register pressure in the kernels that matter."
                            }
                            Link { href: "book/COMPILER.html", "Read the compiler deep dive" }
                        }
                    }
                    Column { span: Span::WIDE,
                        Stack { gap: 3,
                            p { code { "crates/models/arch/dsl/llama.py" } }
                            CodeSnippet { {highlight::block(LLAMA)} }
                            Link { href: "architectures.html", "See all 25 architectures, side by side" }
                        }
                    }
                }
            })}

            {section("The hard case: IBM Spyre", None, rsx! {
                Grid { gutter: Gutter::Cards,
                    Column { span: Span::NARROW,
                        Stack { gap: 5,
                            p {
                                "Scratchy's first target is not CUDA. It is the "
                                a { href: "https://research.ibm.com/blog/spyre-for-z", "IBM Spyre AIU" }
                                " — and that is the real test, because novel silicon is where the library era has \
                                 nothing to offer you."
                            }
                            p {
                                "Each Spyre core has a 2\u{202f}MiB scratchpad, of which 1,677,721 bytes are yours. \
                                 Every tile of every operation in the forward pass must fit. Overflow it and you do \
                                 not get an error message: you get "
                                code { "DtException 1535" }
                                " on the card, minutes later, about a tile you can no longer inspect. The dominant \
                                 cost of new hardware isn't writing kernels — it's the feedback loop from a nameless \
                                 on-card fault back to the line of math that caused it."
                            }
                            p {
                                "Because scratchy knows every tile size at compile time, that fault becomes a "
                                strong { code { "cargo build" } " error on your laptop" }
                                ". A general-purpose runtime structurally cannot do this: it doesn't know the shapes \
                                 until it is already running, on the card, where the only channel back to you is an \
                                 integer."
                            }
                        }
                    }
                    Column { span: Span::WIDE,
                        Stack { gap: 5,
                            p {
                                "Supporting silicon with no ecosystem behind it took about as much Rust as supporting \
                                 CUDA, and it cost "
                                em { "zero lines of model code" }
                                ". The same 25 architectures, the same 1,580 lines of DSL, the same 22-line LLaMA. An \
                                 8B model boots in 12\u{202f}s from a 330\u{202f}Mi image."
                            }
                            p {
                                "Reuse-by-code quietly puts a "
                                em { "population threshold" }
                                " on what hardware is allowed to exist — “support” means a vendor maintaining a \
                                 general backend inside someone else's general framework until the market justifies \
                                 the headcount. Reuse-by-idea drops that threshold to one team with a compiler."
                            }
                        }
                    }
                }
            })}

            {section("Targets", Some(rsx! { "Pick one backend per build; they are mutually exclusive." }), rsx! {
                Grid { gutter: Gutter::Cards,
                    {target("-Fspyre", "IBM Spyre AIU · requires the Spyre build toolkit")}
                    {target("-Fcuda", "NVIDIA · requires the CUDA build toolkit")}
                    {target("-Fmetal", "Apple Silicon · requires macOS")}
                }
            })}

            {section("Getting started", Some(rsx! {
                "Install a recent "
                a { href: "https://rustup.rs/", "Rust toolchain" }
                ", then name your target, your model and your quant. Every build names its own scope — naming zero \
                 models is a build-time panic, not a silent empty binary."
            }), rsx! {
                Stack { gap: 3,
                    p { "Apple Silicon · Llama 3.2 3B · MLX 4-bit" }
                    CodeSnippet {
                        span { class: "c-cm", "# compile a server specialized for exactly this triple" }
                        "\ncargo build -F metal,model/llama-3.2-3b,quant/mlx --release\n\n"
                        span { class: "c-cm", "# run it" }
                        "\n./target/release/scr chat mlx-community/Llama-3.2-3B-Instruct-4bit \\\n    -q "
                        span { class: "c-s", "\"why is the sky blue?\"" }
                    }
                }
                Stack { gap: 3,
                    p { code { "model/<stem>" } " — one checked-in config. Widen with " code { "model/<arch>" } " or " code { "model/all" } "." }
                    p { code { "quant/<preset>" } " — compiles that quantization " em { "instead of" } " dense/bf16." }
                    p { code { "-Fserve" } " " code { "-Fbench" } " — beyond the default " code { "scr chat" } " CLI." }
                }
                Link { href: "book/BUILD.html", "Full feature-scoping mechanics" }
            })}

            {section("Documentation", None, rsx! {
                Grid { gutter: Gutter::Cards,
                    {doc("book/BUILD.html", "Building", "Feature scoping, model and quant selection, air-gapped builds.")}
                    {doc("book/COMPILER.html", "The compiler", "The whole-forward DSL, and what the proc macro actually emits.")}
                    {doc("architectures.html", "Model architectures", "All 25 supported models in the DSL, diffable two at a time.")}
                    {doc("book/MODELS.html", "Adding an architecture", "The recipe for teaching scratchy a model it has never seen.")}
                    {doc("book/spyre/KUBERNETES.html", "Spyre on OpenShift", "Building and shipping an image for the AIU.")}
                    {doc("book/CONTRIBUTING.html", "Contributing", "PR workflow, CI gates, and the architecture invariants.")}
                    {doc(REPO, "Source", "The repository, issues, and pull requests.")}
                }
            })}

            footer {
                Stack { gap: 3,
                    p {
                        "Apache License 2.0 · an "
                        a { href: "https://ai-native-systems-research.github.io/ai-native-systems-research/", "AI-native Systems Research" }
                        " project"
                    }
                    p {
                        "Practising what it argues: scratchy's serving algorithms — paged KV cache, continuous \
                         batching, prefix caching — derive from "
                        a { href: "https://github.com/vllm-project/vllm", "vLLM" }
                        ", and its Spyre hardware model from IBM's "
                        code { "torch-spyre" }
                        " and KTIR. Both are Apache\u{a0}2.0, and both are credited by file and line at each \
                         derivation site rather than in a NOTICE file nobody reads."
                    }
                }
            }
          }
        }
        }
    };
    chrome::document(
        Head {
            title: "Scratchy — A Hyper-specializing Inference Stack Compiler",
            description: Some(
                "Once ideas are the unit of reuse, every project gets to be bespoke. Scratchy compiles one model, \
                 one quantization and one device into an inference server that exists for nothing else — 25 \
                 architectures in 1,580 lines, because everything is a constant.",
            ),
            og: Some(OpenGraph {
                title: "Scratchy — A Hyper-specializing Inference Stack Compiler",
                description: "Fifty years of reuse let the compiler delete and specialize, never restructure. AI \
                              knocks that wall down — and once ideas are the unit of reuse, every project gets \
                              to be bespoke.",
            }),
            root: Root(0),
            libraries: &[],
            theme: Theme::FollowSystem,
        },
        body,
    )
}
