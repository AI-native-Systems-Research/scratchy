# Copyright (c) 2026 IBM Corporation. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Granite's INPUT EMBEDDING: a row gather from the vocabulary table, times 12.

    inputs_embeds = embed_tokens(input_ids) * config.embedding_multiplier

(`GraniteModel.forward`; `embedding_multiplier = 12.0` in
`ibm-granite/granite-3.3-8b-instruct`'s config.json, read from
`crates/models/arch/configs/granite/granite-3.3-8b-instruct.json` rather than recalled.)

THIS IS THE ONLY FIXTURE IN THE TREE WITH INDIRECT ADDRESSING, and that is why it exists.
The other seven all address memory affinely: an offset computed from `tl.program_id` and
constexpr block sizes. Here the ROW INDEX IS DATA -- a token id read from memory -- so the
address depends on a loaded value. In TTIR that is `tt.descriptor_gather`, and it is a
DIFFERENT OP from `tt.descriptor_load`, not a load with a fancier offset:

    %ids  = tt.descriptor_load  %ids_desc[%off]     : !tt.tensordesc<64xsi32> -> tensor<64xi32>
    %rows = tt.descriptor_gather %tab_desc[%ids, %y] : (!tt.tensordesc<1x4096xf16>,
                                                        tensor<64xi32>, i32)
                                                     -> tensor<64x4096xf16>

WHAT TRITON'S OWN SEMANTIC LAYER REQUIRES OF IT (`python/triton/language/semantic.py`'s
`descriptor_gather`, read at the definition -- every one of these is an `assert` there, so
violating one is a refusal, not a slow path):

  * the descriptor is 2D and its block has EXACTLY ONE ROW (`block_shape[0] == 1`). The
    gather's whole shape comes from the index vector, so a multi-row block would be
    ambiguous. That is why `tab_desc`'s block is `[1, D_MODEL]` and not `[BLOCK_M, D_MODEL]`.
  * `x_offsets` is 1D with dtype int16 or int32 -- so the id buffer is `*i32`, and the ids
    reach the op as a `tensor<BLOCK_M x i32>` straight off a descriptor load.
  * `x_offsets.shape[0] >= 8`. BLOCK_M is 64.
  * `block_shape[1] >= 32 // bitwidth * 8`, which for f16 is 16 columns. D_MODEL is 128 in
    the small configuration and 4096 for Granite.

THE DELTA, and the justification for each item
==============================================

1. THE SCALE IS constexpr, NOT A RUNTIME ARGUMENT (SPYRE-SPECIFIC). Same constraint as
   `attention_flash.py`'s delta 9, for the same measured reason: `EMB_SCALE` multiplies a
   tile, so it reaches a COMPUTE OP AS AN OPERAND, and on this device a compute group can
   only read values it LOADS -- the scale has to arrive through memory as a caller-seeded
   splat buffer, which `--spyre-carried-values-to-memory` routes only when the VALUE is
   known at compile time. `embedding_multiplier` is a config constant, so unlike a runtime
   softmax scale this costs nothing: it is 12.0 for every launch of this model.

2. V / D_MODEL / N_TOK ARE constexpr (SPYRE-SPECIFIC), for `attention_flash.py`'s delta 7
   reason: affine loop bounds are static and the emitter derives the work division from the
   descriptor extents, so a runtime extent has no faithful lowering. Spyre compiles one
   kernel per shape.

3. THE VOCABULARY EXTENT IS NOT A MULTIPLE OF 64, and that is a HOST obligation rather
   than a kernel change. Granite's `vocab_size` is 49159 -- not 49152, and not a multiple
   of the 64-element f16 stick. The gathered ROW is D_MODEL wide (4096 = 64 sticks), so the
   stick-axis extent the emitter checks is D_MODEL and the vocabulary axis is only ever
   indexed one row at a time. Nothing in this kernel therefore needs the vocabulary padded;
   the note is here so a reader does not "fix" 49159 to 49152 and silently drop seven rows.

4. NO PADDING-INDEX HANDLING. HF's `nn.Embedding` takes a `padding_idx` and Granite's
   config carries `pad_token_id = 0`, but `padding_idx` affects only the BACKWARD pass (it
   zeroes that row's gradient); the forward is a plain row lookup for every id, including
   the pad id. This is inference-only, so there is nothing to reproduce. `reference()`
   below is `torch.nn.functional.embedding` with no `padding_idx`, which is exactly what
   HF's forward does.

5. OUT-OF-RANGE IDS ARE NOT CHECKED, and the reason is that Triton's descriptor gather
   does not offer a check to reproduce -- the TMA-style descriptor's own out-of-bounds
   behaviour is "values outside the tensor bounds are filled with zeros" (the `.load`
   docstring), and `padding_option="zero"` is `make_tensor_descriptor`'s default. A caller
   that passes an id >= V therefore gets a zero row, not a fault. `inputs()` generates ids
   in `[0, V)` and the host is the layer that owns validating a tokenizer's output.

NOT CHANGED: the gather itself, the row width, and the one multiply. There is no reduction,
no accumulator, and no loop -- this kernel is deliberately the smallest thing that exercises
indirect addressing, so that when it fails the failure is about the gather.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def embedding_fwd(desc_ids, desc_table, desc_o,  #
                  N_TOK: tl.constexpr, V: tl.constexpr,  # SPYRE: delta 2
                  D_MODEL: tl.constexpr,  #
                  BLOCK_M: tl.constexpr,  #
                  EMB_SCALE: tl.constexpr,  # SPYRE: delta 1
                  ):
    start_m = tl.program_id(0)
    # The token ids for this block. `*i32` because `descriptor_gather` accepts int16 or
    # int32 index vectors and nothing else.
    ids_desc = tl.make_tensor_descriptor(desc_ids, shape=[N_TOK], strides=[1],
                                         block_shape=[BLOCK_M])
    # ONE ROW per block, because the gather's row count comes from the index vector.
    table_desc = tl.make_tensor_descriptor(desc_table, shape=[V, D_MODEL],
                                          strides=[D_MODEL, 1],
                                          block_shape=[1, D_MODEL])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[N_TOK, D_MODEL],
                                       strides=[D_MODEL, 1],
                                       block_shape=[BLOCK_M, D_MODEL])

    offs_m = start_m * BLOCK_M
    ids = ids_desc.load([offs_m])
    # THE INDIRECTION. `y_offset` is 0: the row is taken whole, so the gather starts at
    # column 0 of each selected row.
    rows = table_desc.gather(ids, 0)
    # `* embedding_multiplier` (delta 1). f16 throughout: the constexpr float is materialized
    # at the tile's dtype, so no widening island appears here.
    o_desc.store([offs_m, 0], rows * EMB_SCALE)


# --- the Spyre configuration ---------------------------------------------------
# Granite-3.3 8B, from its config.json: vocab_size 49159, hidden_size 4096,
# embedding_multiplier 12.0.
GRANITE = dict(v=49159, d_model=4096)
EMB_MULTIPLIER = 12.0
BLOCK_M = 64

SIGNATURE = {
    "desc_ids": "*i32", "desc_table": "*fp16", "desc_o": "*fp16",
    "N_TOK": "constexpr", "V": "constexpr", "D_MODEL": "constexpr",
    "BLOCK_M": "constexpr", "EMB_SCALE": "constexpr",
}


def constexprs(n_tok=256, v=512, d_model=128, emb_scale=EMB_MULTIPLIER):
    """One configuration. Shapes are constexpr (delta 2), so a shape change is a recompile."""
    return {"N_TOK": n_tok, "V": v, "D_MODEL": d_model,
            "BLOCK_M": BLOCK_M, "EMB_SCALE": emb_scale}


def inputs(seed: int = 0, n_tok=256, v=512, d_model=128):
    """Randomized ids in [0, V) and an f16 table. The ids are i32, as the gather requires."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, v, (n_tok,), generator=g, dtype=torch.int32)
    table = torch.randn(v, d_model, generator=g, dtype=torch.float16)
    return ids, table


def reference(ids: torch.Tensor, table: torch.Tensor,
              emb_scale: float = EMB_MULTIPLIER) -> torch.Tensor:
    """Native PyTorch reference, matching `GraniteModel.forward`'s first two lines:

        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds * self.config.embedding_multiplier

    `F.embedding` is HF's forward exactly -- no `padding_idx` (delta 4).
    """
    return torch.nn.functional.embedding(ids.long(), table) * emb_scale
