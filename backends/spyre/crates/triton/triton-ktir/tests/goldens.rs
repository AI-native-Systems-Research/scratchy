// SPDX-License-Identifier: Apache-2.0
//! The integration tests that read the C++ toolchain's goldens, as one test binary over the
//! golden plumbing they share (`common`). Each module is the `tests/<name>.rs` beside this file.

mod common;

mod consumer_contract;
mod golden_ktir;
mod golden_sched;
mod grid_unroll;
mod refusals;
