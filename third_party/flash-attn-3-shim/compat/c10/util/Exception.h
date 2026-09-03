// Stub for c10/util/Exception.h. Provides TORCH_CHECK / TORCH_INTERNAL_ASSERT
// so vllm-flash-attention-3 sources compile without pulling in libtorch.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <sstream>

#define TORCH_CHECK(cond, ...) \
  do { if (!(cond)) { fprintf(stderr, "TORCH_CHECK failed: %s\n", #cond); abort(); } } while (0)

#define TORCH_INTERNAL_ASSERT(cond, ...) TORCH_CHECK(cond, __VA_ARGS__)
#define TORCH_WARN_ONCE(...) do {} while (0)
#define TORCH_WARN(...) do {} while (0)
