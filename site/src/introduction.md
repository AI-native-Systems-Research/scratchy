# Scratchy

**Scratchy** is a full-stack compiler that builds inference runtimes *from
scratch* — or as close to that as is possible. It takes a triple:

- a **DSL** for the entire forward of a model architecture;
- the **`config.json`** for a given instance of that architecture;
- the **JSON config** for a given quantization.

Given that triple, scratchy generates an inference server specialized for that
input and nothing else. Rust's Turing-complete procedural macros run the whole
pipeline at expansion time, so **everything is a constant** — complex analysis
becomes arithmetic, and the optimizer gets enough constants to cut register
pressure in the kernels that matter.

Scratchy's initial design point is [IBM Spyre AIU], but it also supports NVIDIA
CUDA and Apple Silicon.

[IBM Spyre AIU]: https://research.ibm.com/blog/spyre-for-z

## Where to start

| If you want to… | Read |
| --- | --- |
| build and run the CLI | [Building](BUILD.md) |
| understand the proc-macro approach | [The compiler](COMPILER.md) |
| teach scratchy a new architecture | [Adding a model architecture](MODELS.md) |
| ship an image for Spyre | [Spyre on OpenShift](spyre/KUBERNETES.md) |
| open a PR | [Contributing](CONTRIBUTING.md) |

## Quick start

Install a recent [Rust toolchain](https://rustup.rs/), then name your target,
your model and your quant — every build names its own scope, and naming zero
models is a build-time panic rather than a silent empty binary:

```bash
cargo build -F metal,model/llama-3.2-3b,quant/mlx --release

./target/release/scr chat mlx-community/Llama-3.2-3B-Instruct-4bit \
    -q "why is the sky blue?"
```

Pick exactly one backend: `-Fmetal` (macOS), `-Fcuda` (CUDA toolkit), or
`-Fspyre` (Spyre toolkit). See [Building](BUILD.md) for the full feature-scoping
mechanics.

## License

Apache License 2.0 — see [LICENSE](https://github.com/AI-native-Systems-Research/scratchy/blob/main/LICENSE).
