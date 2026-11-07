// SPDX-License-Identifier: Apache-2.0
//! A per-test watchdog for the GPU test binaries: a kernel that never finishes leaves its test
//! thread spinning in a synchronize, so libtest would wait forever. Each binary's `h100()` arms
//! it; past [`LIMIT`] it names the test on stderr and aborts the process, so the hang fails.

use std::cell::OnceCell;
use std::io::Write;
use std::sync::mpsc;
use std::time::Duration;

/// The longest one GPU test may run: about 2.5× the slowest measured on the H100 (246 s).
pub const LIMIT: Duration = Duration::from_secs(600);

thread_local! {
    /// The calling test's end: dropped with its thread (pass or panic), which stops its watchdog.
    static DONE: OnceCell<mpsc::Sender<()>> = const { OnceCell::new() };
}

/// [`arm_for`] at [`LIMIT`].
pub fn arm() {
    arm_for(LIMIT);
}

/// Start the calling thread's watchdog, once per thread: if the thread (libtest's thread for one
/// test, named after it) is still running after `limit`, print its name and abort.
pub fn arm_for(limit: Duration) {
    DONE.with(|done| {
        done.get_or_init(|| {
            let (tx, rx) = mpsc::channel::<()>();
            let name = std::thread::current()
                .name()
                .unwrap_or("<unnamed>")
                .to_owned();
            std::thread::spawn(move || {
                if rx.recv_timeout(limit) == Err(mpsc::RecvTimeoutError::Timeout) {
                    // Straight to stderr: libtest's capture (which a spawned thread inherits)
                    // would hold the line, and the abort would lose it.
                    let _ = writeln!(
                        std::io::stderr(),
                        "WATCHDOG: test `{name}` still running after {} s: aborting",
                        limit.as_secs()
                    );
                    std::process::abort();
                }
            });
            tx
        });
    });
}
