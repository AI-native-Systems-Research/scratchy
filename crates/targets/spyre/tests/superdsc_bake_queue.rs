// SPDX-License-Identifier: Apache-2.0
//! Laws of the bounded builder work queue (`superdsc_bake`). These live here rather than in a
//! `#[cfg(test)]` module because the crate's inline test module does not compile under `superdsc`
//! alone — an inline test would silently never run.

use scratchy_target_spyre::superdsc_bake::{
    Bake, COMPILE_WIDTH, DxpTool, IN_FLIGHT, MAX_STAGED_BYTES, StageRoot,
};

/// ⭐ THE DISK BOUND AND THE COMPILE WIDTH ARE SEPARATE NUMBERS.
///
/// They were one constant, which made it unsettable: raising it to get dxp parallelism raised peak
/// scratch by the same factor, and lowering it to bound scratch throttled the compile. They limit
/// different resources, so collapsing them back into one is a regression this pins.
#[test]
fn the_disk_bound_is_not_the_compile_width() {
    // Bytes vs processes — not even the same unit, which is the point.
    assert!(
        MAX_STAGED_BYTES >= 64 * 1024 * 1024,
        "too small to hold a wide group"
    );
    assert!(
        MAX_STAGED_BYTES <= 4 * 1024 * 1024 * 1024,
        "a bound this large is not a bound"
    );
    assert!(
        COMPILE_WIDTH > 1,
        "serial compiles are what this queue exists to avoid"
    );
    assert!(
        COMPILE_WIDTH <= 128,
        "one dxp process per group; do not fork hundreds"
    );
    // The channel only has to outrun the workers so a finished one never waits on the producer.
    assert!(
        IN_FLIGHT >= COMPILE_WIDTH,
        "channel shallower than the worker pool starves it"
    );
}

/// ⭐ THE STAGING ROOT IS LOCAL AND OVERRIDABLE, AND IT IS NOT A CACHE.
///
/// The per-op json is dxp's input and nothing else's: it is written where files are cheap, compiled,
/// and deleted. Per-process so two concurrent cargo units never share one, and under the temp dir so
/// nothing outlives the build.
#[test]
fn staging_is_per_process_and_env_overridable() {
    let under = std::env::temp_dir().join(format!("stagetest_{}", std::process::id()));
    // SAFETY: single-threaded test, value read once by `resolve` below.
    unsafe { std::env::set_var("SCRATCHY_SUPERDSC_STAGE", &under) };
    let s = StageRoot::resolve();
    assert!(
        s.path().starts_with(&under),
        "override ignored: {}",
        s.path().display()
    );
    assert!(
        s.path()
            .to_string_lossy()
            .contains(&std::process::id().to_string()),
        "staging must be per-process: {}",
        s.path().display()
    );
    assert!(
        s.group_dir("deadbeefdeadbeef", 7)
            .ends_with("deadbeefdeadbeef/group_7")
    );
    unsafe { std::env::remove_var("SCRATCHY_SUPERDSC_STAGE") };
}

/// Absent the override, staging is the temp dir — never the old bundle cache location.
#[test]
fn default_staging_is_the_temp_dir() {
    let s = StageRoot::resolve();
    assert!(
        s.path().starts_with(std::env::temp_dir()),
        "default staging must be the temp dir, got {}",
        s.path().display()
    );
}

/// NO dxp ⇒ NO queue, so the emitter writes json where a cardless build expects it and compiles
/// nothing. A capability probe, not a behaviour flag: one code path, taken whenever the tool exists.
#[test]
fn without_the_tool_there_is_no_queue() {
    if std::env::var_os("DEEPTOOLS_PATH").is_none() {
        assert!(DxpTool::resolve().is_none(), "no SDK ⇒ no tool");
        assert!(Bake::start().is_none(), "no tool ⇒ no queue");
    }
}
