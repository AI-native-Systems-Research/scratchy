// SPDX-License-Identifier: Apache-2.0
//! Every shared constant a kernel reads (`scratchy::NAME` in `kernels/*.cu` and `*.cuh`) is one
//! the generated header defines (`kernel_consts::cuda_header`), so a kernel cannot name a
//! constant Rust does not have. Only nvcc compiles the kernels; this catches a wrong name without
//! it. No toolchain and no GPU.

use std::collections::BTreeSet;
use std::path::Path;

use scratchy_target_cuda::kernel_consts::cuda_header;

/// The names after `namespace` `ns` in `text`: each `ns` followed by an upper-case identifier.
fn upper_names_after<'t>(text: &'t str, ns: &str) -> impl Iterator<Item = &'t str> {
    text.match_indices(ns).filter_map(move |(at, _)| {
        let rest = &text[at + ns.len()..];
        let end = rest
            .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
            .unwrap_or(rest.len());
        let name = &rest[..end];
        (name.len() > 1 && name.starts_with(|c: char| c.is_ascii_uppercase())).then_some(name)
    })
}

#[test]
fn every_shared_constant_a_kernel_reads_is_in_the_header() {
    let header = cuda_header();
    let defined: BTreeSet<&str> = header
        .lines()
        .filter_map(|l| l.strip_prefix("inline constexpr uint32_t "))
        .map(|l| l.split([' ', '[']).next().unwrap_or(""))
        .collect();
    assert!(
        defined.contains("GEMV_CHUNK"),
        "the header defines nothing:\n{header}"
    );

    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
    let mut read = 0usize;
    for entry in std::fs::read_dir(&dir).expect("kernels/ is readable") {
        let path = entry.expect("a kernels/ entry").path();
        let kernel = matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("cu" | "cuh")
        );
        if !kernel {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("a kernel source is UTF-8");
        for name in upper_names_after(&text, "scratchy::") {
            read += 1;
            assert!(
                defined.contains(name),
                "{} reads `scratchy::{name}`, which `kernel_consts` does not define",
                path.display()
            );
        }
    }
    assert!(read > 0, "no kernel reads a shared constant");
    eprintln!("{read} shared-constant reads in the kernels, each defined in the header");
}
