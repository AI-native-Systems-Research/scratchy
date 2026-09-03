// SPDX-License-Identifier: Apache-2.0
//! Retile — the THIRTEENTH island: the weight-staging PRODUCER==CONSUMER proof for a matmul KERNEL. A
//! kernel `[in=a, out=b]` sticked on `out` lives on-device as `[b/64, a, 64]`. Staging must WRITE each
//! device cell `(t, i, s)` from the host element it represents — host row `i`, host col `t·64+s` — and the
//! matmul READS that cell via `dev_off`. If the staging's flat write position and `dev_off`'s read position
//! disagree, the weight is staged scrambled ⇒ garbage (the exact class of the multi-day K-cache Kᵀ bug,
//! here for kernel weights). This proves they AGREE by construction.
//!
//! ISLAND obligation (Kani, concrete `b`, symbolic device coords, ~<2s): `device_flat(t,i,s) ==
//! dev_off([a,b], 1, [i, t·64+s])` for every in-range coord — the flat cell staging writes is exactly the
//! cell the matmul reads. `dev_off`'s `/64`,`%64` are concrete-divisor (`b` concrete) ⇒ cheap.

use crate::tiled_ir::STK;

/// The HOST row-major offset the staging gathers for device kernel coord `(t,i,s)` — host row `i`, host col
/// `t·64+s`, in a `[a,b]` row-major buffer. (The host-side read position; not asserted here, documents the map.)
pub fn retile_host_off(b: u32, t: u32, i: u32, s: u32) -> u32 {
    i * b + (t * STK + s)
}

/// The DEVICE flat offset of kernel coord `(t,i,s)` in the re-tiled `[b/64, a, 64]` layout: `t·(a·64) +
/// i·64 + s`. This is where staging WRITES the cell.
pub fn device_flat(a: u32, t: u32, i: u32, s: u32) -> u32 {
    t * (a * STK) + i * STK + s
}

#[cfg(test)]
mod tests {
    use super::*;
    use scratchy_subtile::sdsc_abstract::dev_off;

    #[test]
    fn stage_write_equals_matmul_read() {
        // kernel out dims that occur (hd 128, kv 1024, hidden 4096, padded vocab 49664); a = in (hidden).
        for &(a, b) in &[(4096u32, 128u32), (4096, 1024), (4096, 4096), (4096, 49664)] {
            let tiles = b / STK;
            for t in 0..tiles {
                for i in [0u32, 1, a - 1] {
                    for s in [0u32, 1, 63] {
                        let dv = device_flat(a, t, i, s) as usize;
                        let rd = dev_off(
                            &[a as usize, b as usize],
                            1,
                            &[i as usize, (t * STK + s) as usize],
                        );
                        assert_eq!(
                            dv, rd,
                            "a={a} b={b} t={t} i={i} s={s}: stage write ≠ matmul read"
                        );
                    }
                }
            }
        }
    }
}

// ── Kani: staging's flat write == matmul's dev_off read, ∀ in-range coords (concrete b ⇒ cheap /64,%64). ──
#[cfg(kani)]
mod kani_retile {
    use super::*;
    use scratchy_subtile::sdsc_abstract::dev_off;

    fn stage_equals_read(a: u32, b: u32) {
        let tiles = b / STK;
        let t: u32 = kani::any();
        let i: u32 = kani::any();
        let s: u32 = kani::any();
        kani::assume(t < tiles && i < a && s < STK);
        let dv = device_flat(a, t, i, s) as usize;
        let rd = dev_off(
            &[a as usize, b as usize],
            1,
            &[i as usize, (t * STK + s) as usize],
        );
        assert!(dv == rd); // producer (staging flat) == consumer (matmul dev_off read)
    }
    #[kani::proof]
    fn retile_hd() {
        stage_equals_read(128, 128); // head_dim kernel (a bounded small)
    }
    #[kani::proof]
    fn retile_hidden() {
        stage_equals_read(256, 4096); // wide-out kernel (a kept small to stay in the seconds bar)
    }
}
