// SPDX-License-Identifier: Apache-2.0

//! Bring-up driver for [`scratchy_target_spyre_card::bundle_run`], so iterating on it does not rebuild
//! the whole CLI.
//!
//! ```bash
//! cargo run --release --example bundle_run_dev --features spyre-hw -- \
//!   <dir> t0=x.bin t1=w.bin --out t2=o.bin
//! ```
//!
//! `--abba <dirA> <dirB> --rounds <n> --a <bindings …> --b <bindings …>` runs the noise-protocol
//! A/B instead: both bundles staged in this one process, `n` ABBA blocks of refill-free timed
//! launches, one `TIME-SAMPLE side=A|B round=<r> ns=<n>` line per launch (see
//! `bundle_run::run_abba`). Each side takes its OWN binding list (the two forms' op inputs can
//! differ). Both dirs copy of one bundle — the same list twice — makes the A/A noise floor.
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
    if argv.first().map(String::as_str) == Some("--abba") {
        let mut rest = argv[1..].iter();
        let (Some(dir_a), Some(dir_b)) = (rest.next(), rest.next()) else {
            anyhow::bail!(
                "--abba wants two bundle dirs: --abba <dirA> <dirB> --rounds <n> --a <bindings …> --b <bindings …>"
            );
        };
        let mut rounds: u32 = 0;
        let mut inputs_a: Vec<String> = Vec::new();
        let mut inputs_b: Vec<String> = Vec::new();
        let mut cur: Option<&mut Vec<String>> = None;
        while let Some(a) = rest.next() {
            if a == "--rounds" {
                let Some(n) = rest.next() else {
                    anyhow::bail!("--rounds wants a round count");
                };
                rounds = n
                    .parse()
                    .map_err(|e| anyhow::anyhow!("--rounds {n}: {e}"))?;
                cur = None;
                continue;
            }
            if a == "--a" {
                cur = Some(&mut inputs_a);
                continue;
            }
            if a == "--b" {
                cur = Some(&mut inputs_b);
                continue;
            }
            let Some(list) = cur.as_deref_mut() else {
                anyhow::bail!("bindings must follow --a or --b: {a}");
            };
            list.push(a.clone());
        }
        if rounds == 0 {
            anyhow::bail!("--abba wants --rounds <n> with n ≥ 1 — each round is one ABBA block");
        }
        return scratchy_target_spyre_card::bundle_run::run_abba(
            std::path::Path::new(dir_a),
            std::path::Path::new(dir_b),
            &inputs_a,
            &inputs_b,
            rounds,
        );
    }
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
    scratchy_target_spyre_card::bundle_run::run(std::path::Path::new(dir), &inputs, &outputs)
}
