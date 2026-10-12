# Adding a model architecture (or a model to one)

How to add one is [`docs/MODELS.md`](../../../../docs/MODELS.md)'s "Recipe for a new model
architecture" (feature, verbatim config, `weights.json`, DSL, golden). This is what to check after,
each item something that has gone wrong.

## 1. The config is how everything finds the checkpoint

- `configs/<arch>/<stem>.json` is the checkpoint's `config.json`, verbatim. Nothing else names a
  repo: the build matches cached checkpoints to compiled configs by `architectures[0]`,
  `hidden_size`, `num_hidden_layers` and `vocab_size` (`text_config`'s when nested). That match is
  how a checkpoint's tokenizer gets baked.
- Before measuring anything, find the real checkpoints a stem serves and confirm their `config.json`
  matches it on those four fields. Fetch just that file; there's no need to pull the model.

## 2. A quantized checkpoint loads only with a matching preset

- "no compiled model variant matches this checkpoint" means no (stem, quant preset) pair in the build
  describes its on-disk layout. Find the preset in `configs/<arch>/quantizations.json`, or add one
  there, with per-checkpoint drift in `<stem>-<preset>.overrides.json`. Build with
  `quant/<preset>`; `quant/mlx` covers every MLX affine preset.
- Check that it loads on main too: a checkpoint that fails on both is a model-support gap, not your
  regression.

## 3. The tokenizer: nothing to declare, two things to check

- There's no per-model step. The build bakes the tokenizer of every Hub-cache checkpoint that matches
  a compiled config (`scratchy-forwards.rs` `bake_tokenizers`,
  `crates/core/model/src/baked_tokenizer.rs`). Never add per-model lists, sidecar files or special
  cases: a mechanism that covers only the model being measured is a benchmark hack.
- **Check that it bakes.** With the checkpoint's `tokenizer.json` and `config.json` in a Hub cache
  (`HF_HUB_CACHE` selects which), run
  `cargo test --release -p scratchy-core-model --lib bakes_every_cached_tokenizer -- --ignored --nocapture`.
  A `refused … <why>` line names what `BakedBpe` doesn't reproduce yet: a non-BPE model, dropout, a
  non-empty prefix/suffix, or ids that aren't exactly `0..n`. Extend `baked_tokenizer` (bake's
  encode/decode self-check proves the extension) rather than let it fall back to runtime loading
  unnoticed.
- **Check that it's used.** A run logs `Tokenizer compiled in (tokenizer.json <digest>)`. Without
  that line, the model parses `tokenizer.json` at startup, which costs ~50–300 ms.
- The build bakes only what's in the builder's Hub cache, so pull the model first, then build.

## 4. Measure startup the way it's judged

- The number is the `startup` field of
  `scr chat <model> -q "write about the history of france" --max-tokens 64 --bench`, which is engine
  ready. Its `TTFT` is timed after an untimed warmup, so neither field sees first-request work. Don't
  report a first-request win as a startup win.
- Compare against main built in its own `CARGO_TARGET_DIR`, in ABBA order (main, PR, PR, main). A fixed
  order biases whichever binary runs second.
- Discard each new binary's first launch, because it compiles shaders. Measure only while no other
  `scr`, mlx or `cargo` job is running.
- Check greedy parity (`--temperature 0`) against main.
- Read the timeline to see what gates startup: `try_load in`, then `Resolved model architecture` →
  `Model:` (that gap is the wait for the tokenizer), then `init engine … took`.

## Pitfall

- `ls` here prints columns, so `ls | grep -v x` drops whole rows of names. Use `find`.
