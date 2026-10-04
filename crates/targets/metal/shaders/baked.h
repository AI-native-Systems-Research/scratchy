// SPDX-License-Identifier: Apache-2.0
//
// A shader whose constants are compiled in. Every value the model, the bucket or the tape fixes
// reaches its kernels as a `constexpr`: the bake (`scratchy_target_metal::aot::bake`) compiles the
// kernel a command names once per distinct constant set, defining `SCRATCHY_CONSTANT_<slot>` for
// every slot of the command's typed constants (`tape/kernel_constants.rs`) and
// `SCRATCHY_KERNEL_<symbol>` for the one kernel it compiles. A shader that includes this header
// is compiled by the bake only (`build.rs`), and declares no function constant: the pipeline built
// from a baked kernel sets none.

#pragma once

// `name`, the value of constant slot `slot`.
#define SCRATCHY_CONSTANT(type, name, slot) constant constexpr type name = SCRATCHY_CONSTANT_##slot

#define SCRATCHY_CAT_(a, b) a##b
#define SCRATCHY_CAT(a, b) SCRATCHY_CAT_(a, b)
#define SCRATCHY_SECOND_(a, b, ...) b
#define SCRATCHY_SECOND(...) SCRATCHY_SECOND_(__VA_ARGS__)
#define SCRATCHY_IF_0(...)
#define SCRATCHY_IF_1(...) __VA_ARGS__
#define SCRATCHY_PICK_0(set, unset) unset
#define SCRATCHY_PICK_1(set, unset) set

// `name`, the value of constant slot `slot` when the bake sets it (`name##_SET`), else zero: a
// slot only some of a shader's kernels carry, or one a kernel reads only when set.
#define SCRATCHY_CONSTANT_OPTIONAL(type, name, slot)                                       \
  constant constexpr bool name##_SET = SCRATCHY_SECOND(SCRATCHY_SET_##slot, 0, ~);        \
  constant constexpr type name = SCRATCHY_CAT(SCRATCHY_PICK_, SCRATCHY_SECOND(             \
      SCRATCHY_SET_##slot, 0, ~))(SCRATCHY_CONSTANT_##slot, type())

// `1` in the compile the bake made for kernel `sym`, else `0` — usable in `#if`.
#define SCRATCHY_COMPILES(sym) SCRATCHY_SECOND(SCRATCHY_KERNEL_##sym, 0, ~)

// The explicit instantiation `fn` under host name `sym`, emitted only in the compile the bake made
// for `sym`: there `SCRATCHY_KERNEL_<sym>` expands to `~, 1` and the probe's second element is
// `1`; anywhere else the probe is the bare token and its second element is `0`.
#define SCRATCHY_KERNEL(sym, ...)                                                   \
  SCRATCHY_CAT(SCRATCHY_IF_, SCRATCHY_COMPILES(sym))(                               \
      template [[host_name(#sym)]] [[kernel]] decltype(__VA_ARGS__) __VA_ARGS__;)
