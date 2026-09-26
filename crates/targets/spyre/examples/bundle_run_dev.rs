// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! Bring-up driver for [`scratchy_target_spyre::bundle_run`], so iterating on it does not rebuild
//! the whole CLI.
//!
//! ```bash
//! cargo run --release --example bundle_run_dev --features spyre-hw -- \
//!   <dir> t0=x.bin t1=w.bin --out t2=o.bin
//! ```
//!
//! The three markers that ride on the input list are the same ones `scr bundle run --input` takes:
//! `stick:t<id>=<rows>`, `raw:t<id>` and `const:t<id>=<value>`. An embedding gather, for instance:
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
        anyhow::bail!("usage: bundle_run_dev <dir> [t<id>=<in>] ... [--out t<id>=<out>] ...");
    };
    let mut inputs: Vec<String> = Vec::new();
    let mut outputs: Vec<String> = Vec::new();
    let mut next_is_out = false;
    for a in &argv[1..] {
        if a == "--out" {
            next_is_out = true;
            continue;
        }
        if next_is_out {
            outputs.push(a.clone());
            next_is_out = false;
        } else {
            inputs.push(a.clone());
        }
    }
    scratchy_target_spyre::bundle_run::run(std::path::Path::new(dir), &inputs, &outputs)
}
