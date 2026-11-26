//! The integration tests over the shipped fixtures, as one test binary over the fixture plumbing
//! they share (`common`). Each module is the `tests/<name>.rs` beside this file.

mod common;

mod batched_dot;
mod dtype_compare;
mod fixture_status;
mod fusion;
mod gather;
mod golden_diff;
mod make_ttir;
mod math_width;
mod scatter;
mod slicing;
