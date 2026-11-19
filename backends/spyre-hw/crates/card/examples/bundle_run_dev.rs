// SPDX-License-Identifier: Apache-2.0

//! Bring-up driver for [`scratchy_target_spyre_card::bundle_run`], so iterating on it does not rebuild
//! the whole CLI.
//!
//! ```bash
//! cargo run --release --example bundle_run_dev --features spyre-hw -- \
//!   <dir> t0=x.bin t1=w.bin --out t2=o.bin
//! ```
//!
//! `--time <n>` (before the bindings) arms the timing leg: after the correctness launch, `n`
//! more launches of the same bundle, each timed launch-to-completion, the median printed as a
//! `TIME runs=<n> median_ns=… min_ns=… max_ns=…` line.
//!
//! The four markers that ride on the input list are the same ones `scr bundle run --input` takes:
//! `stick:t<id>=<rows>`, `raw:t<id>`, `const:t<id>=<value>` and `fp32:t<id>=<path>` (a resident
//! f32 const, staged verbatim through `bind_const_fp32`). An embedding gather, for instance:
//!
//! ```bash
//! ... <dir> t0=tokens_i32.bin raw:t0 t1=table.bin stick:t1=256 t2=out.bin stick:t2=256 \
//!   const:t4294967275=12.0 --out t2=o.bin
//! ```

fn main() -> anyhow::Result<()> {
    // ⭐ WITHOUT A SUBSCRIBER, flex's COMPLETION THREAD has nowhere to write its CB-error lines — the
    // `hmi_syndrome` / `qgi_syndrome` / `job_count` that name WHY a launch was rejected. Every run so
    // far reported a bare `sync rc=-1` for exactly this reason.
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_writer(std::io::stderr)
        .init();

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(dir) = argv.first() else {
        anyhow::bail!("usage: bundle_run_dev <dir> [--time <n>] [t<id>=<in>] ... [--out t<id>=<out>] ...");
    };
    let mut inputs: Vec<String> = Vec::new();
    let mut outputs: Vec<String> = Vec::new();
    let mut next_is_out = false;
    let mut timing_runs = 0u32;
    let mut rest = argv[1..].iter();
    while let Some(a) = rest.next() {
        if a == "--out" {
            next_is_out = true;
            continue;
        }
        if a == "--time" {
            let Some(n) = rest.next() else {
                anyhow::bail!("--time wants a run count");
            };
            timing_runs = n.parse().map_err(|e| anyhow!("--time {n}: {e}"))?;
            continue;
        }
        if next_is_out {
            outputs.push(a.clone());
            next_is_out = false;
        } else {
            inputs.push(a.clone());
        }
    }
    scratchy_target_spyre_card::bundle_run::run_with_timing(
        std::path::Path::new(dir),
        &inputs,
        &outputs,
        timing_runs,
    )
}
