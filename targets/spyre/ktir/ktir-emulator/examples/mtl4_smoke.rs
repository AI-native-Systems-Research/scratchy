// MTL4 dispatch-path smoke test: exercises every entry point migrated off
// classic `queue.commandBuffer()` (clippy.toml's `disallowed-methods`) onto
// the local `mtl4` MTL4 dispatch helper, checking each against a naive CPU
// reference. Run: cargo run --features metal --example mtl4_smoke
use ktir_emulator::metal::{Epilogue, NaxGemm, run_nax_matmul_tile};

fn cpu_matmul(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn assert_close(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut max_err = 0.0f32;
    for (g, w) in got.iter().zip(want) {
        max_err = max_err.max((g - w).abs());
    }
    assert!(
        max_err <= tol,
        "{name}: max abs err {max_err} exceeds tol {tol}"
    );
    println!("{name}: OK (max abs err {max_err})");
}

fn main() {
    // run_nax_matmul_tile: the fixed 16x16 . 16x32 NAX tile kernel.
    let a16: Vec<f32> = (0..16 * 16)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let b16: Vec<f32> = (0..16 * 32)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();
    let got = run_nax_matmul_tile(&a16, &b16).expect("run_nax_matmul_tile");
    let want = cpu_matmul(16, 16, 32, &a16, &b16);
    assert_close("run_nax_matmul_tile", &got, &want, 5e-2);

    let gemm = NaxGemm::new().expect("NaxGemm::new");

    // run (-> run_epi): general GEMM via the reduce/kernel dispatch path.
    let (m, k, n) = (37usize, 65usize, 41usize);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
    let got = gemm.run(m, k, n, &a, &b).expect("NaxGemm::run");
    let want = cpu_matmul(m, k, n, &a, &b);
    assert_close("NaxGemm::run", &got, &want, 5e-1);

    // matmul_unified (zero-copy path) — check against the same reference.
    let ua = gemm.unified_from(&a).unwrap();
    let ub = gemm.unified_from(&b).unwrap();
    let mut uc = gemm.unified(m * n).unwrap();
    gemm.matmul_unified(m, k, n, &ua, &ub, &mut uc, None, Epilogue::NONE, false)
        .expect("matmul_unified");
    assert_close("matmul_unified", uc.as_slice(), &want, 5e-1);

    // gemv_unified / gemv: m=1 GEMV path.
    let x: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let got = gemm.gemv(k, n, &x, &b, false).expect("gemv");
    let want = cpu_matmul(1, k, n, &x, &b);
    assert_close("gemv", &got, &want, 5e-1);

    // run_chain, single step (isolates the Batch machinery from the
    // barrier/ping-pong dependency).
    let got1 = gemm
        .run_chain(
            m,
            &a,
            &[ktir_emulator::metal::ChainStep {
                k,
                n,
                b: &b,
                e: None,
                epi: Epilogue::NONE,
            }],
        )
        .expect("run_chain (1 step)");
    let want1 = cpu_matmul(m, k, n, &a, &b);
    assert_close("run_chain (1 step)", &got1, &want1, 5e-1);

    // run_chain: two chained matmuls in one MTL4 command buffer/encoder,
    // exercising the inter-dispatch barrier (step 2 reads step 1's output).
    let (n2,) = (23usize,);
    let b2: Vec<f32> = (0..n * n2).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let step1_out = cpu_matmul(m, k, n, &a, &b);
    let want_chain = cpu_matmul(m, n, n2, &step1_out, &b2);
    let got = gemm
        .run_chain(
            m,
            &a,
            &[
                ktir_emulator::metal::ChainStep {
                    k,
                    n,
                    b: &b,
                    e: None,
                    epi: Epilogue::NONE,
                },
                ktir_emulator::metal::ChainStep {
                    k: n,
                    n: n2,
                    b: &b2,
                    e: None,
                    epi: Epilogue::NONE,
                },
            ],
        )
        .expect("run_chain");
    assert_close("run_chain", &got, &want_chain, 5e-1);

    // run_batched: 3 independent GEMMs dispatched together.
    let batch = 3usize;
    let a_batched: Vec<f32> = (0..batch * m * k)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.03)
        .collect();
    let b_batched: Vec<f32> = (0..batch * k * n)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
        .collect();
    let got = gemm
        .run_batched(batch, m, k, n, &a_batched, &b_batched)
        .expect("run_batched");
    let mut want_batched = Vec::with_capacity(batch * m * n);
    for bi in 0..batch {
        want_batched.extend(cpu_matmul(
            m,
            k,
            n,
            &a_batched[bi * m * k..(bi + 1) * m * k],
            &b_batched[bi * k * n..(bi + 1) * k * n],
        ));
    }
    assert_close("run_batched", &got, &want_batched, 5e-1);

    // gpu_time_seconds: the MTL4 Batch path with NO inter-dispatch dependency
    // (every iteration overwrites the same output identically).
    let secs = gemm
        .gpu_time_seconds(m, k, n, &a, &b, 4)
        .expect("gpu_time_seconds");
    assert!(secs >= 0.0 && secs.is_finite(), "gpu_time_seconds: {secs}");
    println!("gpu_time_seconds: OK ({secs}s over 4 iters)");

    println!("ALL MTL4 SMOKE CHECKS PASSED");
}
