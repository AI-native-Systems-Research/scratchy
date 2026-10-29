# SPDX-License-Identifier: Apache-2.0
"""What torch-spyre's SDPA planner chooses for a shape -- the source of the `sdpa_*` tilings.

    python3 torch_spyre_sdpa_plan.py <torch-spyre checkout> B,H,Hkv,Lq,Lk,D [...]

`_select_sdpa_tiling` (torch_spyre/_inductor/decompositions.py) is plain Python over ints, but
importing torch_spyre needs its device extension. So this runs the planner's own source in
isolation: the module's helper prefix -- from after its import block up to the first registered
decomposition -- is exec'd with the package-relative imports stubbed. Nothing the planner computes
is restated here. The stubs are the inputs it reads from the device: 32 cores (`config.sencores`'s
default), a 64-element f16 stick, and the frontend's LX reservation (`_lx_planning_size`: 2 MiB x
(1 - DXP_LX_FRAC_AVAIL = 0.2), truncated, rounded up to 128 B).

A causal shape is planned with torch-spyre's own mask shape ([1, 1, Lq, Lk], unsqueezed at the GQA
group axis); Lq = 1 is planned unmasked, the decode shape.
"""

import dataclasses
import logging
import math
import os
import sys
import threading
import types
import typing

import torch

NUM_CORES = 32
LX_PLANNING_BYTES = ((int(2 * 1024 * 1024 * (1 - 0.2)) + 127) // 128) * 128


def planner(checkout):
    path = os.path.join(checkout, "torch_spyre", "_inductor", "decompositions.py")
    lines = open(path).read().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("logger = ")) + 1
    end = next(i for i, ln in enumerate(lines) if ln.startswith("@register_spyre_decompositions"))
    mod = types.ModuleType("torch_spyre_sdpa_plan_exec")
    mod.__dict__.update(
        dataclasses=dataclasses, math=math, threading=threading, torch=torch,
        Any=typing.Any, Callable=typing.Callable, Optional=typing.Optional,
        Sequence=typing.Sequence, Union=typing.Union,
        Unsupported=type("Unsupported", (Exception,), {}),
        config=types.SimpleNamespace(sencores=NUM_CORES, dxp_lx_frac_avail=0.2),
        get_inductor_logger=logging.getLogger, logger=logging.getLogger("plan"),
        get_elem_in_stick=lambda dtype: 64, get_device_dtype=None, DataFormats=None,
        MAX_QUERY_BLOCK=None, STICK=64, SlidingWindowPlan=None, check_window_read=None,
        plan_sliding_window=None, query_blocking=None, rejection_reason=None,
        DEVICE_NAME="spyre", FP8_E4M3FN_MAX=448.0, FP8_E4M3FN_MIN=-448.0,
    )
    if sys.version_info < (3, 10):
        # `zip(strict=)` is 3.10+; the planner only zips equal-length tuples.
        mod.zip = lambda *a, strict=False: zip(*a)
    sys.modules[mod.__name__] = mod
    src = "from __future__ import annotations\n" + "\n".join(lines[start:end])
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod._select_sdpa_tiling


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    select = planner(sys.argv[1])
    for arg in sys.argv[2:]:
        b, h, hkv, lq, lk, d = (int(x) for x in arg.split(","))
        mask = (1, 1, 1, lq, lk) if hkv != h else (1, 1, lq, lk)
        t = select(batch_size=b, num_heads=h, num_kvheads=hkv, max_seqlen_q=lq,
                   max_seqlen_kv=lk, head_dim=d, element_size=2, num_cores=NUM_CORES,
                   lx_budget_bytes=LX_PLANNING_BYTES,
                   mask_shapes=(mask,) if lq > 1 else ())
        print(f"B={b} H={h} Hkv={hkv} Lq={lq} Lk={lk} D={d}: {t.strategy}, "
              f"q_tile {t.q_tile_size} x{t.num_q_tiles}, kv_block {t.kv_block_size} "
              f"x{t.num_kv_blocks}, head_tiles {t.num_head_tiles}, "
              f"group_tiles {t.num_group_tiles} ({t.reason})")


if __name__ == "__main__":
    main()
