// SPDX-License-Identifier: Apache-2.0
//! The GEMM-chain tapes, with no CUDA toolchain: each hand-written `static` is exactly what CUDA's
//! compiler lowers its canonical to, the numbers are the ones the layout rules give, and every
//! GEMM the kernel cannot run is refused at expansion, naming the value and the rule.

mod common {
    pub mod chains;
}

use common::chains::{
    THREE_GEMM_M256, TWO_GEMM_M128, TWO_GEMM_M256, assert_tape_is_lowered, chain_input, three_gemm,
    two_gemm, with_hidden,
};
use scratchy_subtile::fixtures::{front_end_lowered, gemm_m1_only_input, one_layer_input};
use scratchy_subtile::handoff::LoweredDecode;
use scratchy_subtile::lower::{AffineInt4, GemmWeight, GemmWeightKind};
use scratchy_subtile::subtile_ir::SubOp;
use scratchy_target_cuda::step::{CudaStep, GemmBf16};
use scratchy_target_cuda::tape::{CUDA_ARENA_ALIGN, CudaTape};
use scratchy_target_cuda_compiler::canonical::{
    ArenaRefusal, CanonicalAt, CanonicalRefusal, GemmShapeRule, lower_canonical,
};

fn lower(
    l: &LoweredDecode,
) -> Result<scratchy_target_cuda_compiler::canonical::CudaCanonical, CanonicalRefusal> {
    let at = CanonicalAt {
        stem: "fixture",
        m: u64::from(l.input.ops[0].m),
    };
    lower_canonical(l, at)
}

/// The refusal and its message, which must name every value in `names`.
fn refused(l: &LoweredDecode, names: &[&str]) -> CanonicalRefusal {
    let e = match lower(l) {
        Err(e) => e,
        Ok(c) => panic!("lowered a canonical the kernel cannot run: {c:?}"),
    };
    let msg = e.to_string();
    for name in names {
        assert!(msg.contains(name), "{msg:?} does not name {name:?}");
    }
    eprintln!("refused: {msg}");
    e
}

#[test]
fn each_static_is_what_its_canonical_lowers_to() {
    assert_tape_is_lowered("two_gemm(128)", &TWO_GEMM_M128, &two_gemm(128));
    assert_tape_is_lowered("two_gemm(256)", &TWO_GEMM_M256, &two_gemm(256));
    assert_tape_is_lowered("three_gemm", &THREE_GEMM_M256, &three_gemm());
}

fn gemm_steps(tape: &CudaTape) -> Vec<GemmBf16> {
    tape.steps
        .iter()
        .map(|s| match *s {
            CudaStep::Gemm(g) => g,
        })
        .collect()
}

#[test]
fn the_two_gemm_numbers() {
    for (tape, m) in [(&TWO_GEMM_M128, 128u64), (&TWO_GEMM_M256, 256)] {
        // x [m, 512], y1 [m, 1024], y2 [m, 512], bf16.
        let bytes = [m * 512 * 2, m * 1024 * 2, m * 512 * 2];
        assert_eq!(tape.arena.colour_bytes, bytes);
        assert_eq!(
            tape.arena.colour_offsets,
            [0, bytes[0], bytes[0] + bytes[1]]
        );
        assert_eq!(tape.arena.total_bytes, bytes.iter().sum::<u64>());
        assert_eq!((tape.input.offset.0, tape.input.bytes), (0, bytes[0]));
        assert_eq!(
            (tape.result.offset.0, tape.result.bytes),
            (bytes[0] + bytes[1], bytes[2])
        );
        assert_eq!(tape.weight_bytes, [0, 1048576, 1048576]);
        for &o in tape.arena.colour_offsets {
            assert_eq!(o % CUDA_ARENA_ALIGN, 0);
        }
    }
    assert_eq!(TWO_GEMM_M256.arena.total_bytes, 1048576);
    assert_eq!(TWO_GEMM_M128.arena.total_bytes, 524288);
    let s = gemm_steps(&TWO_GEMM_M256);
    assert_eq!(
        [
            (s[0].a.0, s[0].b.0, s[0].c.0),
            (s[1].a.0, s[1].b.0, s[1].c.0)
        ],
        [(0, 1, 262144), (262144, 2, 786432)]
    );
    assert_eq!(
        [(s[0].m, s[0].n, s[0].k), (s[1].m, s[1].n, s[1].k)],
        [(256, 1024, 512), (256, 512, 1024)]
    );
}

#[test]
fn the_three_gemm_chain_reuses_y1s_colour_for_y3() {
    let t = &THREE_GEMM_M256;
    let s = gemm_steps(t);
    assert_eq!(s[2].c, s[0].c, "y3 is written over y1's colour");
    assert_eq!(
        (s[2].a.0, s[2].b.0, s[2].c.0, s[2].m, s[2].n, s[2].k),
        (786432, 3, 262144, 256, 1024, 512)
    );
    assert_eq!(t.arena.colour_bytes, [262144, 524288, 262144]);
    assert_eq!(t.arena.colour_offsets, [0, 262144, 786432]);
    assert_eq!(t.arena.total_bytes, 1048576);
    // x, y1, y2 and y3 each in a buffer of its own.
    let per_value: u64 = 262144 + 524288 + 262144 + 524288;
    assert_eq!(per_value, 1572864);
    assert!(t.arena.total_bytes < per_value);
}

#[test]
fn a_decode_gemm_is_refused_naming_m_1() {
    let e = refused(
        &front_end_lowered(gemm_m1_only_input()),
        &["m=1", "decode GEMV"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            step: 0,
            m: 1,
            rule: GemmShapeRule::Decode,
            ..
        }
    ));
}

#[test]
fn a_gemm_off_the_tile_grid_is_refused_naming_the_dimension_and_its_tile() {
    let e = refused(
        &with_hidden(chain_input(256, 512, &[200, 512])),
        &["n = 200", "multiple of 256"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            step: 0,
            n: 200,
            rule: GemmShapeRule::OffTile { dim: "n", .. },
            ..
        }
    ));
    let e = refused(
        &with_hidden(chain_input(256, 100, &[1024, 512])),
        &["k = 100", "multiple of 64"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            step: 0,
            k: 100,
            rule: GemmShapeRule::OffTile { dim: "k", .. },
            ..
        }
    ));
}

#[test]
fn a_gemm_dimension_above_int_max_is_refused() {
    let n = 1u32 << 31;
    let e = refused(
        &with_hidden(chain_input(256, 512, &[n])),
        &["n = 2147483648", "2147483647"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            step: 0,
            rule: GemmShapeRule::AboveIntMax { dim: "n", .. },
            ..
        }
    ));
}

#[test]
fn an_affine_weight_is_refused() {
    let mut l = two_gemm(256);
    let affine = AffineInt4::mint(4, 64, 512).expect("4-bit, group 64, over k = 512");
    l.input.ops[0].op = SubOp::MatmulTile {
        n: 1024,
        weight: GemmWeight::Affine { affine },
    };
    let e = refused(&l, &["Affine", "Dense only"]);
    assert!(matches!(
        e,
        CanonicalRefusal::GemmWeight {
            step: 0,
            weight: GemmWeightKind::Affine
        }
    ));
}

#[test]
fn a_canonical_with_no_embedded_hidden_is_refused() {
    let e = refused(
        &front_end_lowered(chain_input(256, 512, &[1024, 512])),
        &["embedded hidden"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::Arena(ArenaRefusal::NoEmbeddedHidden)
    ));
}

#[test]
fn a_layer_is_still_refused_at_its_first_op() {
    let e = refused(&front_end_lowered(one_layer_input()), &["RmsNorm"]);
    assert!(matches!(
        e,
        CanonicalRefusal::NotLowered {
            step: 0,
            op: "RmsNorm"
        }
    ));
}
