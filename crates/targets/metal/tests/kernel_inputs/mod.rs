// SPDX-License-Identifier: Apache-2.0
//! Deterministic kernel inputs, shared by `gemma4_p1_kernels_test` and
//! `rope_on_read_bench_test`.

/// `n` values in `[-scale, scale)` from an xorshift seeded with `seed`.
pub fn pseudo(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as u32 & 0x00FF_FFFF) as f32 / (1u32 << 23) as f32 - 1.0) * scale
        })
        .collect()
}

/// cos_sin table `[max_pos, rot_dim]` = `[cos(0..half); sin(0..half)]` per
/// row, matching `rope_append`'s layout (rope.metal). theta is arbitrary
/// here: the SAME table feeds both the kernel (buffer 7) and the CPU
/// reference, so the test isolates the rotation math/pairing/shuffle, not
/// the production cos_sin generation.
pub fn build_cos_sin(max_pos: usize, rot_dim: usize) -> Vec<f32> {
    let half = rot_dim / 2;
    let theta = 10000.0f32;
    let mut t = vec![0f32; max_pos * rot_dim];
    for pos in 0..max_pos {
        for d in 0..half {
            let inv_freq = 1.0 / theta.powf(2.0 * d as f32 / rot_dim as f32);
            let ang = pos as f32 * inv_freq;
            t[pos * rot_dim + d] = ang.cos();
            t[pos * rot_dim + half + d] = ang.sin();
        }
    }
    t
}
