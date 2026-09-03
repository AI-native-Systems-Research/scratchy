# cuBLAS bf16 GEMM benchmark

Standalone benchmark of the LLaMA 1B prefill GEMM shapes against cuBLAS, used
to find the empirical hardware ceiling on each platform. Measures per-shape
TFLOPS and `% of dense bf16 peak`, plus the total per-forward GEMM time so
you know what your megakernel needs to beat.

## Build & run

```bash
PATH=/usr/local/cuda-12.9/bin:$PATH nvcc -O3 -arch=sm_89 -lcublas bench.cu -o bench
./bench [seq_len]   # default seq_len=1024
```

Update the `l4_bf16_peak_tflops` constant in `bench.cu` for other GPUs (e.g.
L40S = 362, A100 bf16 = 312, H100 = 989, etc.).

## L4 reference numbers (`./bench 1024`)

```
name          M      K      N      lat_us        TFLOPS    % peak    1L *16(ms)
qkv        1024   2048   2304    98.212 us       98.40    81.3%       1.571
o          1024   2048   2048    97.679 us       87.94    72.7%       1.563
gate       1024   2048   8192   408.489 us       84.11    69.5%       6.536
up         1024   2048   8192   422.513 us       81.32    67.2%       6.760
down       1024   8192   2048   441.226 us       77.87    64.4%       7.060
```

Total cuBLAS GEMM time @ seq=1024 (all 16 layers): **23.49 ms** (68.3% of L4
peak). The TK megakernel polyalgorithm currently sits at ~55 ms full-pass at
seq=1024 — about half cuBLAS efficiency on the GEMM portion alone, with room
to close before hitting the hardware ceiling.
