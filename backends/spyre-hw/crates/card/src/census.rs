// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE ON-CARD ALLOC-CENSUS WINDOW — the steady-state number flex-rs's lib
//! tests cannot give: a real serve's decode steps, counted on the card host.
//!
//! The instrument is flex-rs's `alloc_census` (the same one its zero-alloc law
//! tests run under): the binary registers [`flex_rs::alloc_census::CensusAllocator`]
//! as its global allocator (the CLI's `alloc-census` feature), and this module
//! frames the window. When `SCRATCHY_ALLOC_CENSUS_STEPS=N` is set, the N decode
//! steps collected after the FIRST collect run inside
//! [`flex_rs::alloc_census::arm`]..[`flex_rs::alloc_census::disarm`], and the
//! report lands on stderr at the Nth: per-step allocations and bytes — the
//! number the zero-alloc law claims is zero.
//!
//! The window opens at the first COLLECT, not the first submit: the step-1
//! submit (the first bind, the tail of staging) is warmup by every serving
//! measurement this tree has, and the law is about the steady step. `arm()`
//! resets the counters, so nothing before the window counts; the step running
//! when `disarm()` takes the report is inside it.
//!
//! Instrumentation only: the env var frames the window and changes nothing
//! else. The allocator forwards every call verbatim to the system allocator
//! (flex-rs's `CensusAllocator`), so the serving path is byte-identical with
//! the census on or off — an unset or unparsable var is no window at all.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The window's step budget, read once from `SCRATCHY_ALLOC_CENSUS_STEPS`.
/// `None` — unset or unparsable or zero — is no window at all, the common case.
fn window() -> Option<usize> {
    static WINDOW: OnceLock<Option<usize>> = OnceLock::new();
    *WINDOW.get_or_init(|| {
        std::env::var("SCRATCHY_ALLOC_CENSUS_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 1)
    })
}

/// Steps collected inside the window so far.
static STEPS: AtomicUsize = AtomicUsize::new(0);

/// One collected decode step — the serving loop's drain calls this at the top
/// of every collect (before the fence), so the window spans the fence, the
/// pick, and every submit between collects: the whole steady step.
pub fn collect_step() {
    let Some(n) = window() else {
        return;
    };
    let i = STEPS.fetch_add(1, Ordering::Relaxed);
    if i == 0 {
        flex_rs::alloc_census::arm();
    }
    if i + 1 == n {
        let report = flex_rs::alloc_census::disarm();
        eprintln!("[alloc-census] {n} collected decode steps: {report}");
    }
}
