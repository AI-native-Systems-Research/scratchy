# Scratchy: A Hyper-specializing Inference Stack Compiler

This repository is an experiment in leveraging the awesomeness of Rust
to build a **full-stack compiler**. **Scratchy** is such a compiler,
one that builds inference runtimes *from scratch* — or as close to
that as is possible. Scratchy takes as input a triple:

- a [DSL](https://en.wikipedia.org/wiki/Domain-specific_language) for
  the entire forward of a model architecture;
  e.g. [**gemma4-moe**](crates/models/arch/dsl/gemma4-moe.rs.in#L58)
- the config.json for a given instance of that architecture;
  e.g. [**gemma4-moe-26b-a4b-it**](crates/models/arch/configs/gemma4-moe/gemma-4-26b-a4b-it.json)
- the JSON config for a given quantization;
  e.g. [**fp8-dynamic-per-channel**](crates/models/quantization/presets/fp8-dynamic-per-channel.json)

Given that triple input, Scratchy generates an inference server
specialized for that input.  Scratchy extensively utilizes Rust's
Turing complete procedural-macros to avoid much of the complexity of
writing a compiler, to allow for rapid integration of new
architectures, and to generate faster specialized code.  In scratchy,
everything is a constant. This turns complex analysis into arithmetic,
greatly simplifying our code. It also can enable more constant-prop
optimizations, reducing register pressure in key kernels.

Scratchy's initial design point is [IBM Spyre
AIU](https://research.ibm.com/blog/spyre-for-z), but also supports
NVIDIA CUDA and Apple Silicon.

## Key Numbers

- Very small AoT binaries, e.g. 30Mi for Metal, 100Mi for Spyre, 250Mi for Cuda.
- Very small docker images, e.g. 330Mi for Spyre.
- Fast startup time, e.g. 300ms warm startup on Apple Silicon *independent of model size*; 12s for 8B on Spyre.

## Getting Started

Install a recent [Rust toolchain](https://rustup.rs/). Then, pick your
target via [feature
flag](https://doc.rust-lang.org/cargo/reference/features.html):
`-Fcuda` requires the CUDA build toolkit; `-Fmetal` requires macOS;
`-Fspyre` requires the Spyre build toolkit.  For example, you can
compile support for Apple Silicon, the Llama 3.2 3B models, and the
MLX 4-bit quant:

```bash
cargo build -F metal,model/llama-3.2-3b,quant/mlx --release
```

By default, only the simple `scr chat` CLI command is compiled
in. Provide `-Fserve` or `-Fbench` to bring more features into the
binary. To run a quick test of the build:

```bash
./target/release/scr chat mlx-community/Llama-3.2-3b-Instruct-4bit -q "why is the sky blue?"
```

Note the convention for selecting models and quants:
- `model/<stem>` — one checked-in model config, e.g. `model/qwen2.5-7b`
  (`model/<arch>` for every config in an arch, `model/all` for everything).
- `quant/<preset>` — compiles that quantization instead of dense/bf16
  (e.g. `quant/mlx` for every MLX affine int4 preset at once).
- `hf-completions` — shell tab completion over the real HuggingFace ids this
  build can actually run (`scr completions zsh --install`). **On by default**;
  it resolves that list against huggingface.co at build time, so an air-gapped
  build wants `--no-default-features`. See [`docs/BUILD.md`](docs/BUILD.md).

### Deep Dives

- See [`docs/BUILD.md`](docs/BUILD.md) for the full feature-scoping
mechanics.
- See [`docs/COMPILER.md`](docs/COMPILER.md) for more information on the procmacro approach.
- See [`docs/MODELS.md`](docs/MODELS.md) if you are interested in adding support for a new model architecture.
- See [`docs/spyre/KUBERNETES.md`](docs/spyre/KUBERNETES.md) for help using OpenShift to build an image for Spyre.
- See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the PR workflow, CI gates, and architecture invariants.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
