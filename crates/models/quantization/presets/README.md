# Quantization presets

Each `<preset>.json` here carries the `quantization_config` block
for one well-known quant method + knob preset (e.g.
`gptq-sym-desc_act` = AutoGPTQ symmetric 4-bit with activation
ordering at group_size=128).

Per-arch `<arch>/quantizations.json` is a map from preset name →
list of dense-base stems the preset applies to:

```json
{
  "quantizations": {
    "awq-gemm": ["llama-3.2-1b"],
    "gptq-sym-desc_act": ["tinyllama-1.1b"]
  }
}
```

The config loader deep-merges each listed preset onto the matching
`<size>.json` dense base to synthesize a compiled variant named
`<size>-<preset>` (e.g. `llama-3.2-1b` + `awq-gemm` →
`llama-3.2-1b-awq-gemm`). Sizes are enumerated explicitly rather
than auto-fanning across every dense base — not every (arch, size,
quant) combo has a real HF repo, and synthesizing unused variants
multiplicatively compounds release-build times through the
proc-macro + emitted-code pipeline.

When a specific HF repo for a `<size>, <preset>` combo diverges
from the dense base in non-quant ways (e.g. TinyLlama-1.1B-Chat-
v0.3-GPTQ ships `vocab_size: 32003` vs the dense base's 32000),
drop a `<arch>/<size>-<preset>.overrides.json` alongside the
dense base. Its top-level fields deep-merge last, after the
preset.

Adding a new quant method: drop a new `<preset>.json` here + one
entry in each applicable arch's `quantizations.json` referencing
the specific `<size>` it applies to. No per-variant full-config
copies needed.

Today's presets:
- `awq-gemm` — AutoAWQ 4-bit, group_size=128, zero_point=true.
- `gptq-sym` — AutoGPTQ 4-bit symmetric, group_size=128, desc_act=false.
- `gptq-sym-desc_act` — as above but desc_act=true (activation ordering).
- `ct-int4-sym` — compressed-tensors INT4 symmetric, group_size=128.
- `bnb-nf4-dq` — bitsandbytes NF4, double quantization.
