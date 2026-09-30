// SPDX-License-Identifier: Apache-2.0
// How every shader is compiled: at build time, by the build machine's `xcrun metal`, with its
// toolchain's defaults (the language standard among them) — the dispatch kernels by `build.rs`,
// which `include!`s this file, and each generated decode megakernel by the bake, so the one
// kernel's bodies compile exactly as their dispatch kernels do.

/// `xcrun` arguments compiling MSL to AIR (then `-c <source> -o <air>`): `-O3` and flat recorded
/// sources, as MLX ships.
pub const MSL_TO_AIR: &[&str] = &["-sdk", "macosx", "metal", "-O3", "-frecord-sources=flat"];

/// `xcrun` arguments linking AIR into a metallib (then `<air> -o <metallib>`).
pub const AIR_TO_METALLIB: &[&str] = &["-sdk", "macosx", "metallib"];
